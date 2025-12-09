from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, cast

import torch
import torch.nn as nn

from models.frozen_encoders import FrozenAudioEncoder, FrozenTextEncoder
from .slm_model import ProsodySLM

STAGE3_NUM_EMOTIONS = 36


def _load_stage3_state_dict(path: str) -> OrderedDict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint)
    cleaned: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value
    return cleaned


class MELDBaseModel(nn.Module):
    """Shared wiring for MELD models that reuse the frozen Stage-3 backbone."""

    def __init__(
        self,
        *,
        stage3_ckpt: str,
        clap_ckpt_path: str,
        in_dim: int,
        proj_dim: int,
        hidden_dim: int,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.txt_enc = FrozenTextEncoder(device=self.device)
        self.aud_enc = FrozenAudioEncoder(device=self.device)

        self.backbone = ProsodySLM(
            clap_ckpt_path=clap_ckpt_path,
            in_dim=in_dim,
            proj_dim=proj_dim,
            num_emotions=STAGE3_NUM_EMOTIONS,
            hidden_dim=hidden_dim,
        )
        state = _load_stage3_state_dict(stage3_ckpt)
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        if missing or unexpected:
            missing_fmt = ", ".join(missing)
            unexpected_fmt = ", ".join(unexpected)
            raise RuntimeError(
                "Stage-3 checkpoint mismatch: "
                + (f"missing keys: {missing_fmt} " if missing else "")
                + (f"unexpected keys: {unexpected_fmt}" if unexpected else ""),
            )
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.to(self.device).eval()

    def train(self, mode: bool = True) -> "MELDBaseModel":  # noqa: D401 - mimic nn.Module API
        super().train(mode)
        self.backbone.eval()
        self.txt_enc.eval()
        self.aud_enc.eval()
        return self

    def eval(self) -> "MELDBaseModel":  # noqa: D401 - mimic nn.Module API
        super().eval()
        self.backbone.eval()
        self.txt_enc.eval()
        self.aud_enc.eval()
        return self

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:  # pragma: no cover - implemented in subclasses
        raise NotImplementedError

    def trainable_state_dict(self) -> Dict[str, Any]:  # pragma: no cover - implemented in subclasses
        raise NotImplementedError

    def load_trainable_state_dict(self, state: Dict[str, Any]) -> None:  # pragma: no cover - implemented in subclasses
        raise NotImplementedError


class MELDModelOptionA(MELDBaseModel):
    """Maps fused Stage-3 features through a new MELD classifier head."""

    def __init__(
        self,
        *,
        stage3_ckpt: str,
        clap_ckpt_path: str,
        num_meld_labels: int = 7,
        in_dim: int = 768,
        proj_dim: int = 768,
        hidden_dim: int = 1024,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__(
            stage3_ckpt=stage3_ckpt,
            clap_ckpt_path=clap_ckpt_path,
            in_dim=in_dim,
            proj_dim=proj_dim,
            hidden_dim=hidden_dim,
            device=device,
        )

        fusion_dim = proj_dim * 5  # must match backbone.forward_features output

        # tiny MLP head: lets you express mixtures of fused features cleanly
        self.meld_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_meld_labels),
        )

        # init last layer nicely
        last_layer = cast(nn.Linear, self.meld_head[-1])
        nn.init.xavier_uniform_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

        self.meld_head.to(self.device)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        audio = batch["audio"]
        sample_rates = batch["sr"]
        texts = batch["text"]

        with torch.inference_mode():
            audio_embed = self.aud_enc.encode_audio(audio, sample_rates)
            text_embed = self.txt_enc.encode_text(texts)
            fused = self.backbone.forward_features(audio_embed, text_embed)

        fused = fused.clone()  # Detach from inference mode for autograd
        # fused is already on the right device via the backbone
        logits = self.meld_head(fused)
        return logits

    def trainable_state_dict(self) -> Dict[str, Any]:
        return {"meld_head": self.meld_head.state_dict()}

    def load_trainable_state_dict(self, state: Dict[str, Any]) -> None:
        head_state = state.get("meld_head")
        if head_state is None:
            raise KeyError("Missing 'meld_head' in state dict")
        self.meld_head.load_state_dict(head_state)



class MELDModelOptionB(MELDBaseModel):
    """Adapts frozen Stage-3 logits to MELD's 7-way label space."""

    def __init__(
        self,
        *,
        stage3_ckpt: str,
        clap_ckpt_path: str,
        num_meld_labels: int = 7,
        in_dim: int = 768,
        proj_dim: int = 768,
        hidden_dim: int = 1024,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__(
            stage3_ckpt=stage3_ckpt,
            clap_ckpt_path=clap_ckpt_path,
            in_dim=in_dim,
            proj_dim=proj_dim,
            hidden_dim=hidden_dim,
            device=device,
        )
        self.adapter = nn.Linear(STAGE3_NUM_EMOTIONS, num_meld_labels)
        nn.init.xavier_uniform_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)
        self.adapter.to(self.device)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        audio = batch["audio"]
        sample_rates = batch["sr"]
        texts = batch["text"]

        with torch.no_grad():
            audio_embed = self.aud_enc.encode_audio(audio, sample_rates)
            text_embed = self.txt_enc.encode_text(texts)
            logits36 = self.backbone(audio_embed, text_embed)
            feats = logits36 / logits36.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        logits7 = self.adapter(feats)
        return logits7

    def trainable_state_dict(self) -> Dict[str, Any]:
        return {"adapter": self.adapter.state_dict()}

    def load_trainable_state_dict(self, state: Dict[str, Any]) -> None:
        adapter_state = state.get("adapter")
        if adapter_state is None:
            raise KeyError("Missing 'adapter' in state dict")
        self.adapter.load_state_dict(adapter_state)
