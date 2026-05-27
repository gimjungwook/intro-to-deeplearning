"""ResNet-101 frozen encoder + .npy feature cache extractor."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from tqdm import tqdm

from .data import encoder_transform, load_samples
from .utils import DATA_DIR, pick_device


FEATURE_DIM = 2048
FEATURE_GRID = 14  # 14x14


class ResNet101Encoder(nn.Module):
    """Outputs [B, 14*14, 2048] from 224x224 input. All params frozen."""

    def __init__(self) -> None:
        super().__init__()
        weights = models.ResNet101_Weights.IMAGENET1K_V2
        net = models.resnet101(weights=weights)
        # strip avgpool + fc -> output is [B, 2048, 7, 7] with input 224
        # we want 14x14, so feed 224 then upsample? No: use modified net that keeps stride.
        # Simpler: use 224 -> 7x7 (49 patches, 2048). Show-Attend-Tell paper used 14x14 from VGG.
        # We'll keep ResNet 7x7 to stay close to standard. 49 patches still fine.
        modules = list(net.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)            # [B, 2048, 7, 7]
        b, c, h, w = feat.shape
        feat = feat.view(b, c, h * w).permute(0, 2, 1).contiguous()  # [B, 49, 2048]
        return feat


def extract_features(
    data_dir: Path,
    out_dir: Path,
    batch_size: int = 32,
    limit: int | None = None,
    device: torch.device | None = None,
) -> int:
    """Walk all unique images under data_dir, save [49, 2048] .npy per image."""
    device = device or pick_device()
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(data_dir)
    if limit:
        samples = samples[:limit]
    tfm = encoder_transform()
    enc = ResNet101Encoder().to(device)

    todo = [s for s in samples if not (out_dir / f"{s.image_id}.npy").exists()]
    print(f"[encoder] total {len(samples)} | cached {len(samples) - len(todo)} | todo {len(todo)}")

    n_written = 0
    for i in tqdm(range(0, len(todo), batch_size), desc="extract"):
        chunk = todo[i : i + batch_size]
        imgs = []
        for s in chunk:
            try:
                img = Image.open(s.image_path).convert("RGB")
            except Exception as e:
                print(f"[encoder] skip {s.image_id}: {e}", file=sys.stderr)
                imgs.append(None)
                continue
            imgs.append(tfm(img))
        valid = [(s, t) for s, t in zip(chunk, imgs) if t is not None]
        if not valid:
            continue
        batch = torch.stack([t for _, t in valid]).to(device)
        feats = enc(batch).cpu().numpy().astype(np.float32)
        for (s, _), f in zip(valid, feats):
            np.save(out_dir / f"{s.image_id}.npy", f)
            n_written += 1
    return n_written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR / "flickr8k"))
    ap.add_argument("--out-dir",  default=str(DATA_DIR / "features"))
    ap.add_argument("--batch",    type=int, default=32)
    ap.add_argument("--limit",    type=int, default=None)
    args = ap.parse_args()
    n = extract_features(Path(args.data_dir), Path(args.out_dir), args.batch, args.limit)
    print(f"[encoder] wrote {n} feature files to {args.out_dir}")
