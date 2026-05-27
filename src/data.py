"""Flickr8k loader: download, captions, tokenizer, splits, dataloader."""
from __future__ import annotations

import json
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .utils import DATA_DIR


PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]
MAX_CAPTION_LEN = 22


# ---------- Tokenizer ----------

class Vocab:
    def __init__(self, itos: list[str]) -> None:
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}
        for s in SPECIALS:
            assert s in self.stoi, f"missing special {s}"

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str], max_len: int) -> tuple[list[int], int]:
        ids = [self.stoi[BOS]]
        for t in tokens[: max_len - 2]:
            ids.append(self.stoi.get(t, self.stoi[UNK]))
        ids.append(self.stoi[EOS])
        length = len(ids)
        ids += [self.stoi[PAD]] * (max_len - length)
        return ids, length

    def decode(self, ids: list[int]) -> list[str]:
        out = []
        for i in ids:
            tok = self.itos[i]
            if tok == EOS:
                break
            if tok in (PAD, BOS):
                continue
            out.append(tok)
        return out

    @property
    def pad_id(self) -> int: return self.stoi[PAD]
    @property
    def bos_id(self) -> int: return self.stoi[BOS]
    @property
    def eos_id(self) -> int: return self.stoi[EOS]
    @property
    def unk_id(self) -> int: return self.stoi[UNK]


_word_re = re.compile(r"[a-zA-Z]+")


def tokenize(caption: str) -> list[str]:
    return _word_re.findall(caption.lower())


def build_vocab(captions: list[str], min_count: int = 5) -> Vocab:
    counter: Counter[str] = Counter()
    for c in captions:
        counter.update(tokenize(c))
    itos = list(SPECIALS)
    for w, n in counter.most_common():
        if n < min_count:
            break
        itos.append(w)
    return Vocab(itos)


# ---------- Captions / splits ----------

@dataclass
class Sample:
    image_id: str          # e.g. "1000268201_693b08cb0e"
    image_path: Path
    captions: list[str]    # up to 5 per image


def _read_captions_token_file(token_path: Path) -> dict[str, list[str]]:
    caps: dict[str, list[str]] = {}
    for line in token_path.read_text().splitlines():
        if not line.strip():
            continue
        # format: "1000268201_693b08cb0e.jpg#0\ta child ..."
        left, right = line.split("\t", 1)
        img = left.split("#", 1)[0].replace(".jpg", "")
        caps.setdefault(img, []).append(right.strip())
    return caps


def _read_captions_csv(csv_path: Path) -> dict[str, list[str]]:
    # Kaggle format: image,caption
    caps: dict[str, list[str]] = {}
    text = csv_path.read_text()
    lines = text.splitlines()
    if lines and lines[0].lower().startswith("image"):
        lines = lines[1:]
    for line in lines:
        if "," not in line:
            continue
        img, cap = line.split(",", 1)
        img_id = img.strip().replace(".jpg", "")
        caps.setdefault(img_id, []).append(cap.strip().strip('"'))
    return caps


def discover_captions(data_dir: Path) -> dict[str, list[str]]:
    """Find a Flickr8k caption file under data_dir."""
    candidates = [
        data_dir / "Flickr8k.token.txt",
        data_dir / "captions.txt",
        data_dir / "flickr8k" / "captions.txt",
        data_dir / "Flickr8k_text" / "Flickr8k.token.txt",
    ]
    for c in candidates:
        if c.exists():
            if c.suffix == ".txt" and "token" in c.name.lower():
                return _read_captions_token_file(c)
            return _read_captions_csv(c)
    raise FileNotFoundError(
        f"No caption file found in {data_dir}. Expected one of: {[str(c) for c in candidates]}"
    )


def discover_images_root(data_dir: Path) -> Path:
    for c in [
        data_dir / "Images",
        data_dir / "Flicker8k_Dataset",
        data_dir / "flickr8k" / "Images",
        data_dir / "Flickr8k_Dataset",
    ]:
        if c.exists() and c.is_dir():
            return c
    raise FileNotFoundError(f"No image folder found in {data_dir}")


def load_samples(data_dir: Path) -> list[Sample]:
    caps = discover_captions(data_dir)
    img_root = discover_images_root(data_dir)
    out: list[Sample] = []
    for img_id, cs in caps.items():
        p = img_root / f"{img_id}.jpg"
        if not p.exists():
            continue
        out.append(Sample(image_id=img_id, image_path=p, captions=cs[:5]))
    out.sort(key=lambda s: s.image_id)
    return out


def split_samples(samples: list[Sample], seed: int = 42) -> dict[str, list[Sample]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n = len(samples)
    n_train, n_val = int(n * 0.6), int(n * 0.2)
    parts = {
        "train": [samples[i] for i in idx[:n_train]],
        "val":   [samples[i] for i in idx[n_train : n_train + n_val]],
        "test":  [samples[i] for i in idx[n_train + n_val :]],
    }
    return parts


# ---------- Image transforms ----------

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


def encoder_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])


# ---------- Caption dataset (uses cached features) ----------

class CaptionDataset(Dataset):
    """Per-caption sample: returns (features, caption_ids, length, image_id)."""

    def __init__(
        self,
        samples: list[Sample],
        vocab: Vocab,
        features_dir: Path,
        max_len: int = MAX_CAPTION_LEN,
    ) -> None:
        self.vocab = vocab
        self.max_len = max_len
        self.features_dir = Path(features_dir)
        self.items: list[tuple[str, str, list[str]]] = []  # (img_id, caption, all_caps)
        for s in samples:
            for c in s.captions:
                self.items.append((s.image_id, c, s.captions))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        img_id, cap, all_caps = self.items[i]
        feat = np.load(self.features_dir / f"{img_id}.npy")
        ids, length = self.vocab.encode(tokenize(cap), self.max_len)
        return {
            "features": torch.from_numpy(feat),  # [L, D]
            "caption":  torch.tensor(ids, dtype=torch.long),
            "length":   length,
            "image_id": img_id,
            "all_caps": all_caps,
        }


def collate(batch: list[dict]) -> dict:
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "caption":  torch.stack([b["caption"]  for b in batch]),
        "lengths":  torch.tensor([b["length"]  for b in batch], dtype=torch.long),
        "image_ids": [b["image_id"] for b in batch],
        "all_caps":  [b["all_caps"] for b in batch],
    }


def make_loader(dataset: CaptionDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate,
        drop_last=False,
    )


# ---------- Eval dataset (one entry per image, ref = 5 caps) ----------

class EvalImageDataset(Dataset):
    def __init__(self, samples: list[Sample], features_dir: Path) -> None:
        self.samples = samples
        self.features_dir = Path(features_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        s = self.samples[i]
        feat = np.load(self.features_dir / f"{s.image_id}.npy")
        return {
            "features": torch.from_numpy(feat),
            "image_id": s.image_id,
            "image_path": str(s.image_path),
            "captions": s.captions,
        }


def eval_collate(batch: list[dict]) -> dict:
    return {
        "features":   torch.stack([b["features"] for b in batch]),
        "image_ids":  [b["image_id"] for b in batch],
        "image_paths":[b["image_path"] for b in batch],
        "captions":   [b["captions"] for b in batch],
    }
