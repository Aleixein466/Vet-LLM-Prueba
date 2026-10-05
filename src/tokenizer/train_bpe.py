"""Entrena un tokenizer BPE propio (vocab 16000) sobre el corpus veterinario.

Usa la librería `tokenizers` (Hugging Face), pre-tokenizador ByteLevel
(robusto con tildes/ñ). Guarda en data/tokenizer/bpe16k/.

Uso:
  python src/tokenizer/train_bpe.py
  python src/tokenizer/train_bpe.py --vocab-size 16000
"""
from __future__ import annotations
import argparse
import glob
from pathlib import Path

VOCAB_SIZE = 16000
SPECIAL = ["[PAD]", "[UNK]", "[EOS]"]


def main(vocab_size=VOCAB_SIZE, out="data/tokenizer/bpe16k",
         corpus_glob="data/raw/*.txt", extra_glob="data/cleaned/docs_real.jsonl"):
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
    import json

    files = sorted(glob.glob(corpus_glob))
    tmp = Path(out) / "_train_corpus.txt"
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as fh:
        for f in files:
            fh.write(Path(f).read_text(encoding="utf-8") + "\n")
        # refuerza con documentos reales limpios si existen
        p = Path(extra_glob)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    fh.write(json.loads(line)["text"] + "\n")
                except (KeyError, json.JSONDecodeError):
                    pass
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL,
                         show_progress=True)
    tok.train([str(tmp)], trainer)
    tok.save(str(Path(out) / "tokenizer.json"))
    eos = tok.token_to_id("[EOS]")
    print(f"vocab={tok.get_vocab_size()} eos={eos} -> {out}/tokenizer.json")
    # prueba rápida
    ids = tok.encode("El perro presenta vómito y diarrea.").ids
    print("test_ids=", ids[:12], "decode=", tok.decode(ids))
    tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    ap.add_argument("--out", default="data/tokenizer/bpe16k")
    ap.add_argument("--corpus-glob", default="data/raw/*.txt")
    a = ap.parse_args()
    main(a.vocab_size, a.out, a.corpus_glob)
