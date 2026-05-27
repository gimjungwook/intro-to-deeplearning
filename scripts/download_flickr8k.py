"""Download jxie/flickr8k parquets and materialize to data/flickr8k/{Images,captions.txt}."""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image
import pandas as pd

REPO_ID = "jxie/flickr8k"
FILES = [
    "data/train-00000-of-00002-2f8f6bfa852eac4b.parquet",
    "data/train-00001-of-00002-2173151d8cd6c7fb.parquet",
    "data/validation-00000-of-00001-7025a2b596f14b7b.parquet",
    "data/test-00000-of-00001-42a2661d12c73e48.parquet",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data" / "flickr8k"))
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of images for smoke test")
    args = ap.parse_args()

    out_dir  = Path(args.out)
    img_dir  = out_dir / "Images"
    cap_path = out_dir / "captions.txt"
    img_dir.mkdir(parents=True, exist_ok=True)

    # captions: "imageid.jpg,caption"
    seen_ids: set[str] = set()
    cap_rows: list[str] = []
    n_written = 0

    for fname in FILES:
        print(f"[dl] {fname}")
        local = hf_hub_download(REPO_ID, fname, repo_type="dataset")
        df = pd.read_parquet(local)
        print(f"     rows={len(df)} cols={list(df.columns)}")
        for _, row in df.iterrows():
            if args.limit and n_written >= args.limit:
                break
            # detect image bytes column
            img_obj = row.get("image") if "image" in df.columns else row.get("img")
            if isinstance(img_obj, dict):
                bytes_ = img_obj.get("bytes")
                path   = img_obj.get("path") or ""
            else:
                bytes_ = img_obj
                path   = ""
            # id
            img_id = None
            if path:
                img_id = Path(path).stem
            if not img_id:
                img_id = f"img_{n_written:06d}"
            # caps
            caps = []
            for k in df.columns:
                if k.startswith("caption"):
                    v = row.get(k)
                    if isinstance(v, str) and v.strip():
                        caps.append(v.strip())
            if not caps:
                continue
            jpg = img_dir / f"{img_id}.jpg"
            if not jpg.exists():
                try:
                    im = Image.open(io.BytesIO(bytes_)).convert("RGB")
                    im.save(jpg, "JPEG", quality=90)
                except Exception as e:
                    print(f"     skip {img_id}: {e}", file=sys.stderr)
                    continue
            if img_id not in seen_ids:
                for c in caps:
                    cap_rows.append(f"{img_id}.jpg,{c}")
                seen_ids.add(img_id)
            n_written += 1
            if n_written % 500 == 0:
                print(f"     ... {n_written} images saved")
        if args.limit and n_written >= args.limit:
            break

    # write captions.txt with header
    cap_path.write_text("image,caption\n" + "\n".join(cap_rows) + "\n")
    print(f"[dl] DONE images={n_written} captions={len(cap_rows)} -> {out_dir}")


if __name__ == "__main__":
    main()
