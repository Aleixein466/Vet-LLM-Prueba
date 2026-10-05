"""Dataset con BPE propio 16k: corpus -> train_bpe.bin / val_bpe.bin (uint16).

Equivalente a build_dataset.py pero con data/tokenizer/bpe16k/tokenizer.json.
NO toca train.bin/val.bin (conviven). Requiere haber corrido train_bpe.py.

Uso:
  python src/data/build_dataset_bpe.py
"""
from __future__ import annotations
import argparse
import glob
from pathlib import Path
import numpy as np


def main(corpus_glob="data/raw/*.txt", tok_file="data/tokenizer/bpe16k/tokenizer.json",
         train_file="data/train/train_bpe.bin", val_file="data/validation/val_bpe.bin",
         train_split=0.9):
    from tokenizers import Tokenizer
    files = sorted(glob.glob(corpus_glob))
    if not files:
        raise SystemExit(f"Sin corpus: {corpus_glob}")
    tok = Tokenizer.from_file(tok_file)
    eos = tok.token_to_id("[EOS]")
    assert eos is not None and eos < 65536
    ids: list[int] = []
    for f in files:
        text = Path(f).read_text(encoding="utf-8")
        for chunk in text.split("\n\n"):
            chunk = chunk.strip()
            if not chunk:
                continue
            ids.extend(tok.encode(chunk).ids + [eos])
    arr = np.array(ids, dtype=np.uint16)
    cut = int(len(arr) * train_split)
    Path(train_file).parent.mkdir(parents=True, exist_ok=True)
    Path(val_file).parent.mkdir(parents=True, exist_ok=True)
    arr[:cut].tofile(train_file)
    arr[cut:].tofile(val_file)
    print(f"tokens={len(arr)} train={cut} val={len(arr)-cut} vocab_eos={eos} max_id={int(arr.max())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-glob", default="data/raw/*.txt")
    ap.add_argument("--tok-file", default="data/tokenizer/bpe16k/tokenizer.json")
    ap.add_argument("--train-file", default="data/train/train_bpe.bin")
    ap.add_argument("--val-file", default="data/validation/val_bpe.bin")
    ap.add_argument("--train-split", type=float, default=0.9)
    a = ap.parse_args()
    main(a.corpus_glob, a.tok_file, a.train_file, a.val_file, a.train_split)
