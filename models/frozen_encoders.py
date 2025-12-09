from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
from torch.amp.autocast_mode import autocast
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel

try:
    import torchaudio.functional as F_audio
except ImportError:
    F_audio = None


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or (isinstance(device, str) and device == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_autocast_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@dataclass
class TokenPayload:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None


class FrozenTextEncoder:
    """Frozen Gemma sentence encoder with batched inference."""

    def __init__(self, device: str | torch.device | None = None, batch_size: int = 32) -> None:
        self.device = _resolve_device(device)
        self.batch_size = max(1, int(batch_size))
        self.model = SentenceTransformer("google/embeddinggemma-300m")
        self.model.eval()
        self.model.to(device=str(self.device))
        self.model.requires_grad_(False)
        dim = self.model.get_sentence_embedding_dimension()
        if dim is None or dim <= 0:
            raise ValueError("embeddinggemma reported an invalid embedding dimension")
        self.output_dim = int(dim)

    def encode_text(self, texts: Sequence[str], batch_size: int | None = None) -> torch.Tensor:
        if len(texts) == 0:
            return torch.empty((0, self.output_dim), device=self.device, dtype=torch.float32)
        effective_bs = max(1, batch_size or self.batch_size)
        with torch.inference_mode():
            embeddings = self.model.encode(
                list(texts),
                batch_size=effective_bs,
                convert_to_tensor=True,
                device=str(self.device),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        if embeddings.device != self.device:
            embeddings = embeddings.to(self.device, non_blocking=True)
        return embeddings.to(dtype=torch.float32)

    def train(self, mode: bool = False) -> "FrozenTextEncoder":  # noqa: D401 - mimic nn.Module API
        self.model.eval()
        return self

    def eval(self) -> "FrozenTextEncoder":  # noqa: D401 - mimic nn.Module API
        self.model.eval()
        return self


class FrozenAudioEncoder:
    """Frozen WavCoch + AuriStream stack that yields pooled embeddings."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        target_sample_rate: int = 16000,
    ) -> None:
        self.device = _resolve_device(device)
        self.target_sr = int(target_sample_rate)
        if self.target_sr <= 0:
            raise ValueError("target_sample_rate must be positive")

        self.quantizer = AutoModel.from_pretrained("TuKoResearch/WavCochV8192", trust_remote_code=True)
        self.quantizer.to(self.device)
        self.quantizer.eval()
        self.quantizer.requires_grad_(False)

        self.acoustic = AutoModel.from_pretrained(
            "TuKoResearch/AuriStream100M_RoPE_librilight",
            trust_remote_code=True,
        )
        self.acoustic.to(self.device)
        self.acoustic.eval()
        self.acoustic.requires_grad_(False)

        self.hidden_dim = int(getattr(self.acoustic.config, "n_embd", getattr(self.acoustic.config, "hidden_size", 0)))
        if self.hidden_dim <= 0:
            raise ValueError("AuriStream config did not expose a valid hidden size")
        self.output_dim = self.hidden_dim
        self.max_seq_len = int(getattr(self.acoustic.config, "seq_len", 1024))

        self.amp_enabled = self.device.type == "cuda"
        self.amp_dtype = _resolve_autocast_dtype()
        self._resample_fn = getattr(F_audio, "resample", None)

    def encode_audio(self, wave_batch: torch.Tensor | Sequence[torch.Tensor], sample_rates: Iterable[int] | int | torch.Tensor) -> torch.Tensor:
        waveforms = self._as_waveform_list(wave_batch)
        sr_list = self._expand_sample_rates(sample_rates, len(waveforms))
        processed = [self._preprocess_waveform(wave, sr) for wave, sr in zip(waveforms, sr_list)]
        payloads = self._quantize(processed)
        if not payloads:
            return torch.empty((0, self.output_dim), device=self.device, dtype=torch.float32)
        embeddings = self._tokens_to_embedding(payloads)
        return embeddings

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _as_waveform_list(self, wave_batch: torch.Tensor | Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if isinstance(wave_batch, torch.Tensor):
            if wave_batch.ndim == 1:
                return [wave_batch.detach().cpu()]
            if wave_batch.ndim == 2:
                return [wave_batch[index].detach().cpu() for index in range(wave_batch.size(0))]
            raise ValueError("Expected wave_batch to have shape [T] or [B, T]")
        return [wave.detach().cpu() for wave in wave_batch]

    def _expand_sample_rates(self, sample_rates: Iterable[int] | int | torch.Tensor, expected: int) -> List[int]:
        if isinstance(sample_rates, torch.Tensor):
            values = sample_rates.detach().cpu().tolist()
        elif isinstance(sample_rates, int):
            values = [sample_rates] * expected
        else:
            values = list(sample_rates)
        if len(values) != expected:
            raise ValueError(f"Expected {expected} sample rates, received {len(values)}")
        return [int(v) for v in values]

    def _preprocess_waveform(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        wav = waveform.to(dtype=torch.float32).contiguous()
        if wav.ndim != 1:
            wav = wav.view(-1)
        if sample_rate != self.target_sr:
            if self._resample_fn is None:
                raise RuntimeError(
                    "torchaudio is required to resample audio to the target sample rate, but it is not installed.",
                )
            wav = self._resample_fn(wav.unsqueeze(0), float(sample_rate), float(self.target_sr)).squeeze(0)
        peak = wav.abs().max()
        if torch.isfinite(peak) and peak > 0:
            wav = wav / peak
        return wav.clamp(-1.0, 1.0)

    def _quantize(self, waveforms: List[torch.Tensor]) -> List[TokenPayload]:
        if not waveforms:
            return []
        padded = pad_sequence(waveforms, batch_first=True)
        if self.device.type == "cuda":
            padded = padded.pin_memory()
        wav = padded.to(self.device, non_blocking=True).unsqueeze(1)

        autocast_ctx = contextlib.nullcontext()
        if self.amp_enabled:
            autocast_ctx = autocast(device_type='cuda', dtype=self.amp_dtype)

        with torch.inference_mode():
            with autocast_ctx:
                token_output = self.quantizer(wav)
        if "input_ids" not in token_output:
            raise KeyError("Quantizer output missing 'input_ids'")

        token_ids = token_output["input_ids"].detach().to("cpu")
        attention_mask = token_output.get("attention_mask")
        mask_cpu = attention_mask.detach().to("cpu") if isinstance(attention_mask, torch.Tensor) else None

        payloads: List[TokenPayload] = []
        for index in range(token_ids.size(0)):
            ids = token_ids[index].to(dtype=torch.long)
            mask_tensor = None if mask_cpu is None else mask_cpu[index].to(dtype=torch.float32)
            payloads.append(TokenPayload(input_ids=ids, attention_mask=mask_tensor))
        return payloads

    def _tokens_to_embedding(self, payloads: List[TokenPayload]) -> torch.Tensor:
        token_tensors = [payload.input_ids for payload in payloads]
        padded_tokens = pad_sequence(token_tensors, batch_first=True, padding_value=0)

        if any(payload.attention_mask is not None for payload in payloads):
            masks = [
                payload.attention_mask
                if payload.attention_mask is not None
                else torch.ones_like(payload.input_ids, dtype=torch.float32)
                for payload in payloads
            ]
            padded_mask = pad_sequence(masks, batch_first=True, padding_value=0.0)
        else:
            padded_mask = torch.zeros_like(padded_tokens, dtype=torch.float32)
            for row, payload in enumerate(payloads):
                length = payload.input_ids.size(0)
                padded_mask[row, :length] = 1.0

        tokens = padded_tokens.to(self.device, non_blocking=True)
        mask = padded_mask.to(self.device, non_blocking=True)

        vocab_size = getattr(self.acoustic.config, "vocab_size", None)
        if vocab_size is not None:
            tokens = tokens.clamp(min=0, max=int(vocab_size) - 1)

        batch_size = tokens.size(0)
        hidden_accum = torch.zeros((batch_size, self.hidden_dim), device=self.device, dtype=torch.float32)
        frame_counts = torch.zeros((batch_size, 1), device=self.device, dtype=torch.float32)

        autocast_ctx = contextlib.nullcontext()
        if self.amp_enabled:
            autocast_ctx = autocast(device_type='cuda', dtype=self.amp_dtype)

        with torch.inference_mode():
            for start in range(0, tokens.size(1), self.max_seq_len):
                window = tokens[:, start : start + self.max_seq_len]
                if window.numel() == 0:
                    continue
                with autocast_ctx:
                    outputs = self.acoustic(seq=window, output_hidden_states=True)
                hidden = outputs["hidden_states"][-1].to(dtype=torch.float32)
                mask_slice = mask[:, start : start + hidden.size(1)].unsqueeze(-1)
                hidden_accum = hidden_accum + (hidden * mask_slice).sum(dim=1)
                frame_counts = frame_counts + mask_slice.sum(dim=1)

        if torch.any(frame_counts == 0):
            raise RuntimeError("Audio encoder returned zero valid frames for at least one sample")

        mean_hidden = hidden_accum / frame_counts.clamp_min(1.0)
        normalized = F.normalize(mean_hidden, p=2, dim=-1)
        return normalized.detach()

    def train(self, mode: bool = False) -> "FrozenAudioEncoder":
        self.quantizer.eval()
        self.acoustic.eval()
        return self

    def eval(self) -> "FrozenAudioEncoder":
        self.quantizer.eval()
        self.acoustic.eval()
        return self
