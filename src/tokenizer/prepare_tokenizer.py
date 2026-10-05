"""Paso 3: analiza el corpus con tiktoken (no requiere entrenar tokenizer)."""
from __future__ import annotations
import argparse
import glob
from pathlib import Path
from src.tokenizer.tok import get_enc, EOT


def main(corpus_glob="data/raw/*.txt"):
    files = sorted(glob.glob(corpus_glob))
    if not files:
        raise SystemExit(f"Sin corpus: {corpus_glob}. Ejecuta corpus_sintetico.py")
    enc = get_enc()
    n_docs, n_tok, n_chars = 0, 0, 0
    for f in files:
        text = Path(f).read_text(encoding="utf-8")
        for chunk in text.split("\n\n"):
            chunk = chunk.strip()
            if not chunk:
                continue
            n_docs += 1
            n_chars += len(chunk)
            n_tok += len(enc.encode(chunk)) + 1  # + EOS
    print(f"docs={n_docs} chars={n_chars} tokens~{n_tok} fertilidad={n_tok/max(1,n_chars):.2f} tok/char")
    print(f"encoding=tiktoken/gpt2 vocab={enc.n_vocab} eot={EOT}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-glob", default="data/raw/*.txt")
    main(ap.parse_args().corpus_glob)
