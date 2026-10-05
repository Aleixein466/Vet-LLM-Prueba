"""RAG híbrido: índice semántico (MiniLM multilingüe) como respaldo del TF-IDF.

El TF-IDF sigue siendo la vía principal (rápido, con guardrails validados).
Este índice se usa SOLO cuando TF-IDF rechaza (SIN_EVIDENCIA / FUERA_DE_AMBITO)
o para re-rankear candidatos. CPU-friendly, sin servidor vectorial.

Uso:
  python src/rag/embeddings.py --build     # construye data/rag/index/semantic.npz
  python src/rag/embeddings.py --ask "..."  # prueba una consulta
"""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # 384 dim, ES OK, ~470MB
MODEL_DIR = "C:/Users/alexi/AppData/Local/Temp/opencode/models/minilm-l12-v2"
# Calibrado 2026-10-05: rescates reales 0.62-0.77; saludos/OOD 0.32-0.50.
SEM_THRESHOLD = 0.55

_MODEL = None  # singleton en proceso (evita recargar 500MB por consulta)


def _load_model(model_name=MODEL_NAME):
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        # 1) snapshot local (offline OK) 2) nombre Hub (requiere red)
        try:
            _MODEL = SentenceTransformer(MODEL_DIR)
        except OSError:
            _MODEL = SentenceTransformer(model_name)
    return _MODEL  # cosine mínimo para aceptar respaldo semántico


def _passages(raw_glob="data/raw/*.txt") -> list[str]:
    docs = []
    for f in sorted(glob.glob(raw_glob)):
        docs += [c.strip() for c in Path(f).read_text(encoding="utf-8").split("\n\n") if c.strip()]
    return sorted(set(docs))


def build(raw_glob="data/raw/*.txt", out="data/rag/index", model_name=MODEL_NAME):
    import numpy as np
    docs = _passages(raw_glob)
    print(f"pasajes={len(docs)} modelo={model_name}", flush=True)
    st = _load_model(model_name)
    mat = st.encode(docs, batch_size=32, show_progress_bar=True,
                    convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outp / "semantic.npz", mat=mat)
    (outp / "semantic_docs.json").write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    (outp / "semantic_meta.json").write_text(json.dumps(
        {"model": model_name, "dim": int(mat.shape[1]), "n": len(docs)}), encoding="utf-8")
    print(f"OK dim={mat.shape[1]} n={len(docs)} -> {out}", flush=True)


def query(text: str, k: int = 2, index="data/rag/index", model_name=MODEL_NAME):
    import numpy as np
    idx = Path(index)
    z = np.load(idx / "semantic.npz")
    mat = z["mat"]
    docs = json.loads((idx / "semantic_docs.json").read_text(encoding="utf-8"))
    st = _load_model(model_name)
    q = st.encode([text], convert_to_numpy=True, normalize_embeddings=True).astype("float32")[0]
    sims = mat @ q
    top = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i]), docs[i]) for i in top]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--ask", default=None)
    ap.add_argument("-k", type=int, default=2)
    ap.add_argument("--raw-glob", default="data/raw/*.txt")
    ap.add_argument("--out", default="data/rag/index")
    a = ap.parse_args()
    if a.build:
        build(a.raw_glob, a.out)
    elif a.ask:
        for i, s, d in query(a.ask, a.k, a.out):
            print(f"{s:.4f} doc={i} :: {d[:160]!r}")
    else:
        ap.print_help()
