"""RAG ligero: índice TF-IDF (numpy) sobre el corpus veterinario + recuperación top-k."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import numpy as np

STOP = set("""de la el en y a los del se las por un para con no una su al lo como más pero sus
  este esta estos estas ese esa eso ser son fue han hay tiene tienen
  mi tu su sus nos os me te se le les lo la los las un una unos unas al del ante bajo cabe sobre tras""".split())

TOK = re.compile(r"[a-záéíóúñü]+", re.I)

def _norm(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")

def tokens(text: str) -> list[str]:
    return [t for t in (_norm(w) for w in TOK.findall(text)) if len(t) > 2 and t not in _norm_stop()]

def _norm_stop() -> set[str]:
    import unicodedata
    return {"".join(c for c in unicodedata.normalize("NFD", w) if unicodedata.category(c) != "Mn") for w in STOP}

class TfidfIndex:
    def __init__(self, docs: list[str]):
        self.docs = docs
        df: dict[str, int] = {}
        tfs: list[dict[str, int]] = []
        for d in docs:
            c: dict[str, int] = {}
            for t in tokens(d):
                c[t] = c.get(t, 0) + 1
            tfs.append(c)
            for t in c:
                df[t] = df.get(t, 0) + 1
        self.vocab = {t: i for i, t in enumerate(sorted(df))}
        n, v = len(docs), len(self.vocab)
        idf = np.zeros(v)
        for t, i in self.vocab.items():
            idf[i] = np.log((1 + n) / (1 + df[t])) + 1.0
        self.idf = idf
        mat = np.zeros((n, v), dtype=np.float32)
        for di, c in enumerate(tfs):
            for t, f in c.items():
                mat[di, self.vocab[t]] = (1 + np.log(f)) * idf[self.vocab[t]]
        mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        self.mat = mat

    def query(self, text: str, k: int = 3) -> list[tuple[int, float]]:
        q = np.zeros(len(self.vocab), dtype=np.float32)
        c: dict[str, int] = {}
        for t in tokens(text):
            if t in self.vocab:
                c[t] = c.get(t, 0) + 1
        if not c:
            return []
        for t, f in c.items():
            q[self.vocab[t]] = (1 + np.log(f)) * self.idf[self.vocab[t]]
        q /= np.linalg.norm(q) + 1e-9
        sims = self.mat @ q
        top = np.argsort(sims)[::-1][:k]
        return [(int(i), float(sims[i])) for i in top if sims[i] > 0]

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / "tfidf.npz", mat=self.mat, idf=self.idf)
        (path / "vocab.json").write_text(json.dumps(self.vocab, ensure_ascii=False), encoding="utf-8")
        (path / "docs.json").write_text(json.dumps(self.docs, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "TfidfIndex":
        path = Path(path)
        z = np.load(path / "tfidf.npz")
        idx = cls.__new__(cls)
        idx.mat, idx.idf = z["mat"], z["idf"]
        idx.vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
        idx.docs = json.loads((path / "docs.json").read_text(encoding="utf-8"))
        return idx


def build(raw_glob="data/raw/*.txt", out="data/rag/index"):
    import glob
    docs = []
    for f in sorted(glob.glob(raw_glob)):
        docs += [c.strip() for c in Path(f).read_text(encoding="utf-8").split("\n\n") if c.strip()]
    docs = sorted(set(docs))
    idx = TfidfIndex(docs)
    idx.save(Path(out))
    print(f"pasajes={len(docs)} vocab={len(idx.vocab)} -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-glob", default="data/raw/*.txt")
    ap.add_argument("--out", default="data/rag/index")
    a = ap.parse_args()
    build(a.raw_glob, a.out)
