"""Sonda RAG temporal (solo retrieval, sin modelo): MODO B retrieval puro.
Replica EXACTAMENTE la logica actual de app/vet_chat.py::answer() hasta el threshold.
Uso: python scripts/rag_diagnose.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rag.retriever import TfidfIndex, tokens
from app.vet_chat import es_fuera_de_ambito, ESPECIES_MAP

IDX = TfidfIndex.load(Path("data/rag/index"))
K = 2
THRESHOLD = 0.20  # app/vet_chat.py:100

def probe(query, history=None):
    toks = tokens(query)
    n_tok = len(toks)
    q_ret = (history[-1][0] + " " + query) if (history and n_tok < 4) else query
    fuera = (not history or n_tok >= 4) and es_fuera_de_ambito(IDX, query)
    hits = IDX.query(q_ret, k=K + 2)
    qlow = q_ret.lower()
    esp = next((v for kw, v in ESPECIES_MAP.items() if kw in qlow), None)
    if esp:
        rescored = []
        for j, s in hits:
            if esp in IDX.docs[j].lower():
                s += 0.15
            rescored.append((j, s))
        rescored.sort(key=lambda t: t[1], reverse=True)
        hits = rescored[:K]
    else:
        hits = hits[:K]
    thr_ok = bool(hits) and hits[0][1] >= THRESHOLD
    if hits and hits[0][1] < THRESHOLD:
        hits = []
    return {
        "query": query,
        "tokens": toks,
        "overlap_vocab": sorted(set(toks) & set(IDX.vocab)),
        "q_ret": q_ret,
        "contaminada": q_ret != query,
        "especie_bonus": esp,
        "fuera_de_ambito": fuera,
        "threshold": THRESHOLD,
        "hits": [{"doc": j, "score": round(float(s), 4), "text": IDX.docs[j][:220]} for j, s in hits],
        "raw_top_score": round(float(hits[0][1]), 4) if hits else 0.0,
        "decision": "FUERA_DE_AMBITO" if fuera else ("SIN_EVIDENCIA" if not hits else "RECUPERA"),
    }

if __name__ == "__main__":
    qs = [
        "mi perro lo golpeo un carro en la pata y lo veo agitado",
        "cristobal colon comio perros",
        "los dinosaurios comian perros?",
        "¿qué es la ehrlichiosis canina?",
        "¿qué signos tiene un perro con ehrlichiosis?",
        "¿Qué signos presenta un perro con ehrlichiosis?",
        "¿Qué información se debe recopilar en un perro con vómito?",
        "¿Cuál es la capital de Francia?",
        "¿Quién fue Cristóbal Colón?",
    ]
    for q in qs:
        r = probe(q)
        print("=" * 78)
        print(f"QUERY: {r['query']}")
        print(f"TOKENS({len(r['tokens'])}): {r['tokens']}")
        print(f"OVERLAP_VOCAB({len(r['overlap_vocab'])}): {r['overlap_vocab']}")
        print(f"Q_RET: {r['q_ret']!r} contaminada={r['contaminada']} especie={r['especie_bonus']} fuera={r['fuera_de_ambito']}")
        for h in r["hits"]:
            print(f"  doc={h['doc']} score={h['score']:.4f} :: {h['text']!r}")
        if not r["hits"]:
            print("  (sin documentos: threshold o fuera_de_ambito)")
        print(f"THRESHOLD={r['threshold']} DECISION={r['decision']}")
    print("\n--- con historial (contaminacion) ---")
    hist = [("cristobal colon comio perros", "prev")]
    r = probe("los dinosaurios comian perros?", history=hist)
    print(r)
