"""Evaluation: BLEU, attention overlay, beam ablation."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from PIL import Image
from tqdm import tqdm

from .data import EvalImageDataset, MAX_CAPTION_LEN, Vocab, eval_collate, tokenize
from .decoder import AttentionDecoder
from .utils import pick_device


def _detok(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t]


def compute_bleu(refs_list: list[list[list[str]]], hyps: list[list[str]]) -> dict[str, float]:
    """refs_list[i] = list of reference token lists for image i; hyps[i] = hypothesis tokens."""
    smooth = SmoothingFunction().method1
    out: dict[str, float] = {}
    for n in (1, 2, 3, 4):
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        out[f"BLEU-{n}"] = corpus_bleu(refs_list, hyps, weights=weights, smoothing_function=smooth)
    return out


@torch.no_grad()
def evaluate(
    decoder: AttentionDecoder,
    samples,
    features_dir: Path,
    vocab: Vocab,
    device: torch.device,
    beam: int = 1,
    max_len: int = MAX_CAPTION_LEN,
    limit: int | None = None,
) -> tuple[dict[str, float], list[tuple[str, list[str], list[list[str]], torch.Tensor]]]:
    decoder.eval()
    ds = EvalImageDataset(samples[:limit] if limit else samples, features_dir)
    refs_list: list[list[list[str]]] = []
    hyps: list[list[str]] = []
    records: list[tuple[str, list[str], list[list[str]], torch.Tensor]] = []
    for i in tqdm(range(len(ds)), desc=f"eval beam={beam}"):
        item = ds[i]
        feats = item["features"].unsqueeze(0).to(device)
        refs_tok = [tokenize(c) for c in item["captions"]]
        if beam <= 1:
            toks, alphas = decoder.generate_greedy(feats, vocab.bos_id, vocab.eos_id, max_len=max_len)
            ids = toks[0].tolist()
            alpha_seq = alphas[0]
        else:
            ids, alpha_seq = decoder.generate_beam(feats, vocab.bos_id, vocab.eos_id, beam=beam, max_len=max_len)
        hyp_tok = vocab.decode(ids)
        refs_list.append(refs_tok)
        hyps.append(hyp_tok)
        records.append((item["image_id"], hyp_tok, refs_tok, alpha_seq.cpu()))
    bleu = compute_bleu(refs_list, hyps)
    return bleu, records


def save_attention_overlay(
    image_path: str,
    caption_tokens: list[str],
    alpha_seq: torch.Tensor,   # [T_gen, L]
    out_path: Path,
    grid: int = 7,
    upscale: int = 32,
) -> None:
    """Save a grid showing the original image with per-token attention heatmaps."""
    if len(caption_tokens) == 0:
        return
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_np = np.array(img) / 255.0
    n = min(len(caption_tokens), alpha_seq.size(0))
    cols = min(n, 6)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.expand_dims(axes, 0)
    elif cols == 1:
        axes = np.expand_dims(axes, 1)
    for k in range(rows * cols):
        r, cc = k // cols, k % cols
        ax = axes[r][cc]
        ax.axis("off")
        if k >= n:
            continue
        alpha = alpha_seq[k].numpy().reshape(grid, grid)
        alpha = np.kron(alpha, np.ones((upscale, upscale)))
        alpha = alpha / (alpha.max() + 1e-8)
        ax.imshow(img_np)
        ax.imshow(alpha, cmap="jet", alpha=0.5)
        ax.set_title(caption_tokens[k], fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


NOUN_POS_HINT = {  # cheap heuristic; real POS would need nltk pos tag
    "man", "woman", "child", "boy", "girl", "dog", "cat", "bird", "horse", "ball",
    "water", "snow", "grass", "tree", "rock", "beach", "field", "mountain", "bike",
    "shirt", "hat", "people", "person", "kid", "baby", "car", "truck", "bus",
    "table", "chair", "book", "phone", "camera", "guitar", "ball", "frisbee",
    "skateboard", "surfboard", "swing", "swing", "kite", "umbrella", "flower",
    "river", "ocean", "lake", "pool", "stage", "court", "track", "race", "soccer",
}


def attention_noun_peak_score(records, grid: int = 7) -> tuple[int, int]:
    """Heuristic alignment QC: count records where attention has a clear peak (max>2*mean) at any noun token."""
    ok = 0
    total = 0
    for img_id, hyp_tok, refs_tok, alphas in records:
        total += 1
        hit = False
        n = min(len(hyp_tok), alphas.size(0))
        for k in range(n):
            tok = hyp_tok[k]
            if tok not in NOUN_POS_HINT:
                continue
            a = alphas[k].numpy()
            if a.max() > 2.0 * a.mean():
                hit = True
                break
        if hit:
            ok += 1
    return ok, total
