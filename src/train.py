"""Training driver: builds vocab, runs epochs, writes run dir, evaluates at end."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import (
    CaptionDataset,
    MAX_CAPTION_LEN,
    build_vocab,
    load_samples,
    make_loader,
    split_samples,
    tokenize,
)
from .decoder import AttentionDecoder
from .eval import attention_noun_peak_score, evaluate, save_attention_overlay
from .report import append_failures, append_metric_row, init_run, write_loss_curve
from .utils import DATA_DIR, RUNS_DIR, git_hash, make_run_dir, new_run_id, pick_device, set_seed


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",     default=str(DATA_DIR / "flickr8k"))
    ap.add_argument("--features-dir", default=str(DATA_DIR / "features"))
    ap.add_argument("--run-id",       default=None)
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--batch",        type=int, default=32)
    ap.add_argument("--lr",           type=float, default=4e-4)
    ap.add_argument("--epochs",       type=int, default=30)
    ap.add_argument("--patience",     type=int, default=5)
    ap.add_argument("--vocab-min",    type=int, default=5)
    ap.add_argument("--embed",        type=int, default=512)
    ap.add_argument("--hidden",       type=int, default=512)
    ap.add_argument("--attn",         type=int, default=512)
    ap.add_argument("--dropout",      type=float, default=0.5)
    ap.add_argument("--ds-lambda",    type=float, default=1.0,
                    help="doubly stochastic regularization coefficient")
    ap.add_argument("--grad-clip",    type=float, default=5.0)
    ap.add_argument("--max-len",      type=int, default=MAX_CAPTION_LEN)
    ap.add_argument("--limit",        type=int, default=None,
                    help="cap number of images (smoke test)")
    ap.add_argument("--eval-limit",   type=int, default=None)
    ap.add_argument("--beams",        type=str, default="1,3,5")
    ap.add_argument("--note",         default="")
    return ap.parse_args()


def epoch_loop(
    decoder: AttentionDecoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    pad_id: int,
    ds_lambda: float,
    grad_clip: float,
    train: bool,
) -> tuple[float, float]:
    decoder.train(train)
    total_loss, total_tokens = 0.0, 0
    attn_entropy_sum, attn_entropy_n = 0.0, 0
    for batch in loader:
        feats = batch["features"].to(device, non_blocking=True)
        caps  = batch["caption"].to(device, non_blocking=True)
        lens  = batch["lengths"].to(device, non_blocking=True)

        out = decoder(feats, caps, lens)
        logits = out["logits"]           # [B, T-1, V]
        alphas = out["alphas"]           # [B, T-1, L]
        targets = caps[:, 1:]            # next-token

        # mask
        T_minus_1 = logits.size(1)
        ar = torch.arange(T_minus_1, device=device).unsqueeze(0)
        mask = (ar < (lens - 1).unsqueeze(1)).float()  # [B, T-1]

        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                             targets.reshape(-1),
                             reduction="none",
                             ignore_index=pad_id)
        ce = ce.view_as(targets) * mask
        loss_ce = ce.sum() / mask.sum().clamp_min(1.0)

        # doubly stochastic: each location's attention summed across time should be ~1
        alphas_masked = alphas * mask.unsqueeze(-1)
        ds_term = ((1.0 - alphas_masked.sum(dim=1)) ** 2).mean()
        loss = loss_ce + ds_lambda * ds_term

        # attention entropy (for logging)
        with torch.no_grad():
            ent = -(alphas_masked * (alphas_masked.clamp_min(1e-8)).log()).sum(dim=-1)
            ent = ent.sum() / mask.sum().clamp_min(1.0)
            attn_entropy_sum += float(ent.item()) * mask.sum().item()
            attn_entropy_n   += float(mask.sum().item())

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
            optimizer.step()

        total_loss   += float(loss.item()) * mask.sum().item()
        total_tokens += float(mask.sum().item())
    return (total_loss / max(total_tokens, 1.0),
            attn_entropy_sum / max(attn_entropy_n, 1.0))


def select_failures_and_samples(records, vocab, k_failures: int = 24, k_random: int = 12, k_hard: int = 12):
    # cheap: pick those with no overlap with any ref unigrams as failures
    scored = []
    for img_id, hyp, refs, alphas in records:
        ref_set = set(sum(refs, []))
        overlap = sum(1 for t in hyp if t in ref_set)
        scored.append((overlap, img_id, hyp, refs, alphas))
    scored.sort(key=lambda x: x[0])  # lowest overlap first
    failures = scored[:k_failures]
    rest = scored[k_failures:]
    rng = np.random.default_rng(0)
    rand_idx = rng.choice(len(rest), size=min(k_random, len(rest)), replace=False) if rest else []
    randoms = [rest[i] for i in rand_idx]
    hards = scored[k_failures : k_failures + k_hard]
    return failures, randoms, hards


def main() -> None:
    args = parse_args()
    device = pick_device()
    set_seed(args.seed)

    print(f"[train] device={device} | git={git_hash()}")

    samples = load_samples(Path(args.data_dir))
    if args.limit:
        samples = samples[: args.limit]
    parts = split_samples(samples, seed=args.seed)
    print(f"[train] images: train {len(parts['train'])} val {len(parts['val'])} test {len(parts['test'])}")

    train_caps = [c for s in parts["train"] for c in s.captions]
    vocab = build_vocab(train_caps, min_count=args.vocab_min)
    print(f"[train] vocab={len(vocab)} (min_count={args.vocab_min})")

    features_dir = Path(args.features_dir)
    train_ds = CaptionDataset(parts["train"], vocab, features_dir, max_len=args.max_len)
    val_ds   = CaptionDataset(parts["val"],   vocab, features_dir, max_len=args.max_len)
    train_loader = make_loader(train_ds, args.batch, shuffle=True)
    val_loader   = make_loader(val_ds,   args.batch, shuffle=False)

    decoder = AttentionDecoder(
        vocab_size=len(vocab),
        feat_dim=2048,
        embed_dim=args.embed,
        hidden_dim=args.hidden,
        attn_dim=args.attn,
        dropout=args.dropout,
        pad_id=vocab.pad_id,
    ).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    run_id = args.run_id or new_run_id()
    run_dir = make_run_dir(run_id)
    config = {
        "run_id": run_id,
        "seed": args.seed,
        "device": str(device),
        "git": git_hash(),
        "n_train_images": len(parts["train"]),
        "n_val_images":   len(parts["val"]),
        "n_test_images":  len(parts["test"]),
        "vocab_size": len(vocab),
        "vocab_min_count": args.vocab_min,
        "batch": args.batch,
        "lr": args.lr,
        "epochs": args.epochs,
        "patience": args.patience,
        "embed": args.embed,
        "hidden": args.hidden,
        "attn": args.attn,
        "dropout": args.dropout,
        "ds_lambda": args.ds_lambda,
        "grad_clip": args.grad_clip,
        "max_len": args.max_len,
        "limit": args.limit,
        "beams": args.beams,
        "note": args.note,
    }
    init_run(run_dir, config)
    # save vocab for reproducibility
    (run_dir / "vocab.json").write_text(json.dumps(vocab.itos, ensure_ascii=False))
    print(f"[train] run_dir={run_dir}")

    best_val = float("inf")
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_ent = epoch_loop(decoder, train_loader, optimizer, device,
                                     vocab.pad_id, args.ds_lambda, args.grad_clip, train=True)
        val_loss, _ = epoch_loop(decoder, val_loader, None, device,
                                 vocab.pad_id, args.ds_lambda, args.grad_clip, train=False)
        dt = time.time() - t0
        print(f"[train] ep {epoch:02d}  tr_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  ent={tr_ent:.2f}  t={dt:.1f}s")
        # write metric row (BLEU left blank during epochs to save time)
        append_metric_row(run_dir, [epoch, f"{tr_loss:.4f}", f"{val_loss:.4f}", "", "", "", "", f"{tr_ent:.3f}"])

        if val_loss < best_val - 1e-3:
            best_val = val_loss
            patience_left = args.patience
            torch.save(decoder.state_dict(), run_dir / "decoder_best.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[train] early stop at epoch {epoch}")
                break

    write_loss_curve(run_dir)

    # ---- final evaluation: greedy + beams on val (or test if val too small) ----
    print("[eval] loading best decoder")
    decoder.load_state_dict(torch.load(run_dir / "decoder_best.pt", map_location=device))
    eval_split = parts["val"] if len(parts["val"]) >= 50 else parts["test"]

    beam_results: dict[str, dict[str, float]] = {}
    primary_records = None
    for b in [int(x) for x in args.beams.split(",")]:
        bleu, records = evaluate(decoder, eval_split, features_dir, vocab, device,
                                 beam=b, max_len=args.max_len, limit=args.eval_limit)
        print(f"[eval] beam={b}  " + "  ".join(f"{k}={v:.4f}" for k, v in bleu.items()))
        beam_results[f"beam_{b}"] = bleu
        if b == 3 or primary_records is None:
            primary_records = records

    (run_dir / "beam_results.json").write_text(json.dumps(beam_results, indent=2))

    # attention noun-peak QC over primary (beam=3 or first available)
    ok, total = attention_noun_peak_score(primary_records or [])
    qc = {"noun_peak_ok": ok, "noun_peak_total": total}
    (run_dir / "attention_qc.json").write_text(json.dumps(qc, indent=2))
    print(f"[eval] attention noun-peak QC: {ok}/{total}")

    # write attention overlays for failures + randoms
    failures, randoms, hards = select_failures_and_samples(primary_records or [], vocab)
    samples_dir = run_dir / "samples"
    # image lookup
    img_path_lookup = {s.image_id: str(s.image_path) for s in samples}
    for kind, lst in [("fail", failures[:12]), ("rand", randoms[:12]), ("hard", hards[:12])]:
        for i, (_, img_id, hyp, refs, alphas) in enumerate(lst):
            img_path = img_path_lookup.get(img_id)
            if img_path is None:
                continue
            save_attention_overlay(img_path, hyp, alphas, samples_dir / f"{kind}_{i:02d}_{img_id}.png", grid=7)

    # failures.md content
    fail_records = [{
        "image_id": img_id,
        "hyp": hyp,
        "ref": refs,
        "cause": "TBD (auto: low unigram overlap with refs)",
    } for _, img_id, hyp, refs, _ in failures]
    append_failures(run_dir, fail_records)

    # ---- ACCEPT gate ----
    bleu4_target = 0.18
    ok_thresh, ok_total = qc["noun_peak_ok"], qc["noun_peak_total"]
    primary_bleu = beam_results.get("beam_3", beam_results.get("beam_1", {})).get("BLEU-4", 0.0)
    greedy_bleu  = beam_results.get("beam_1", {}).get("BLEU-4", 0.0)
    gate = {
        "BLEU4_>=0.18":      primary_bleu >= bleu4_target,
        "noun_peak_>=18/24": (ok_thresh >= 18 and ok_total >= 24),
        "beam3_vs_greedy_+0.01": (primary_bleu - greedy_bleu) >= 0.01,
        "no_divergence_NaN":  (not (np.isnan(best_val) or np.isinf(best_val))),
        "primary_BLEU4": primary_bleu,
        "greedy_BLEU4":  greedy_bleu,
    }
    (run_dir / "accept_gate.json").write_text(json.dumps(gate, indent=2))
    print(f"[gate] {gate}")

    # final summary append
    with (run_dir / "notes.md").open("a") as f:
        f.write(f"\n## Final BLEU\n")
        for k, v in beam_results.items():
            f.write(f"- {k}: {v}\n")
        f.write(f"\n## ACCEPT gate\n```json\n{json.dumps(gate, indent=2)}\n```\n")


if __name__ == "__main__":
    main()
