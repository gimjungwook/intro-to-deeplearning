"""Run-dir bookkeeping: config.json, metrics.csv, samples, failures, notes."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def init_run(run_dir: Path, config: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "samples").mkdir(exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    with (run_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "attn_entropy"])
    (run_dir / "notes.md").write_text(f"# Run notes\n\nconfig = `config.json`\n")
    (run_dir / "failures.md").write_text("# Failure cases (top wrong captions + likely cause)\n")


def append_metric_row(run_dir: Path, row: list[Any]) -> None:
    with (run_dir / "metrics.csv").open("a", newline="") as f:
        csv.writer(f).writerow(row)


def write_loss_curve(run_dir: Path) -> None:
    import pandas as pd
    p = run_dir / "metrics.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    if len(df) == 0:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(df["epoch"], df["train_loss"], label="train")
    if "val_loss" in df and df["val_loss"].notna().any():
        ax[0].plot(df["epoch"], df["val_loss"], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    bcols = [c for c in df.columns if c.startswith("BLEU")]
    for c in bcols:
        if df[c].notna().any():
            ax[1].plot(df["epoch"], df[c], label=c)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("BLEU"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=110)
    plt.close(fig)


def append_failures(run_dir: Path, failures: list[dict]) -> None:
    p = run_dir / "failures.md"
    with p.open("a") as f:
        for i, fitem in enumerate(failures, 1):
            f.write(f"\n## {i}. {fitem['image_id']}\n")
            f.write(f"- hyp: `{' '.join(fitem['hyp'])}`\n")
            f.write(f"- ref: `{' '.join(fitem['ref'][0])}`\n")
            f.write(f"- cause: {fitem.get('cause', 'TBD')}\n")
