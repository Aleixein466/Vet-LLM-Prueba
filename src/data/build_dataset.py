"""Paso 3b: tokeniza corpus a .bin (uint16, vocab 50257 < 65536) + split train/val."""
from __future__ import annotations
import argparse
import glob
from pathlib import Path
import numpy as np
from src.tokenizer.tok import get_enc, EOT


def main(corpus_glob="data/raw/*.txt", train_file="data/train/train.bin",
         val_file="data/validation/val.bin", train_split=0.9):
    files = sorted(glob.glob(corpus_glob))
    if not files:
        raise SystemExit(f"Sin corpus: {corpus_glob}")
    enc = get_enc()
    ids: list[int] = []
    for f in files:
        text = Path(f).read_text(encoding="utf-8")
        for chunk in text.split("\n\n"):
            chunk = chunk.strip()
            if not chunk:
                continue
            ids.extend(enc.encode(chunk) + [EOT])
    arr = np.array(ids, dtype=np.uint16)
    assert int(arr.max()) < 65536
    cut = int(len(arr) * train_split)
    Path(train_file).parent.mkdir(parents=True, exist_ok=True)
    Path(val_file).parent.mkdir(parents=True, exist_ok=True)
    arr[:cut].tofile(train_file)
    arr[cut:].tofile(val_file)
    print(f"tokens={len(arr)} train={cut} val={len(arr)-cut} max_id={int(arr.max())}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-glob", default="data/raw/*.txt")
    ap.add_argument("--train-file", default="data/train/train.bin")
    ap.add_argument("--val-file", default="data/validation/val.bin")
    ap.add_argument("--train-split", type=float, default=0.9)
    a = ap.parse_args()
    main(a.corpus_glob, a.train_file, a.val_file, a.train_split)
