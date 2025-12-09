import torch
from clap_model import CLAP

ckpt_path = "checkpoints/clap_proj_heads_final.pt"
out_path = "checkpoints/clap_projectors.pt"

device = "cpu"

model = CLAP(dim=768).to(device)
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt, strict=True)

projectors = {
    "audio_proj": model.audio_proj.state_dict(),
    "text_proj": model.text_proj.state_dict(),
}

torch.save(projectors, out_path)
print(f"saved {out_path}")
