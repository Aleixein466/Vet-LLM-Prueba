"""App mínima: chat veterinario con RAG (recupera contexto y genera con el tiny LLM)."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import yaml
from src.model.gpt import TinyGPT
from src.tokenizer.tok import get_enc, EOT
from src.rag.retriever import TfidfIndex
import re

_STOP = re.compile(r"\s+")

URGENCIAS = ["chocolate", "veneno", "envenen", "convulsi", "sangr", "atropell",
             "golpe de calor", "no respira", "inconsciente", "hinchad", "torsión",
             "quemad", "picadura", "anticongelante", "matarratas", "raticida",
             "xilitol", "orina con sangre", "no puede orinar", "parto"]

BANNER = ("URGENTE: si hay intoxicación, convulsiones, sangrado, "
          "dificultad respiratoria o decaimiento grave, acude al veterinario "
          "cuanto antes. No mediques por tu cuenta.")

ESPECIES_MAP = {"perro": "perro", "pug": "perro", "bulldog": "perro", "gato": "gato",
                "caballo": "caballo", "vaca": "vaca", "conejo": "conejo"}

def _kw(text: str) -> set[str]:
    return {w.lower().strip(".,;:¿?¡!()") for w in _STOP.split(text) if len(w) > 4}

def es_urgencia(text: str) -> bool:
    t = text.lower()
    return any(u in t for u in URGENCIAS)

def es_vaga(text: str) -> bool:
    from src.rag.retriever import tokens
    return len(tokens(text)) < 3

FUERA_DE_AMBITO = ("Eso está fuera de mi ámbito: solo respondo temas de salud animal "
                   "(vacunas, desparasitación, alimentación, constantes, urgencias). "
                   "Reformula tu pregunta sobre tu mascota.")

SIN_EVIDENCIA = ("No encontré información veterinaria relevante para responder esa "
                 "pregunta en mi base. Pregunta sobre vacunas, "
                 "desparasitación, alimentación, constantes o urgencias (intoxicaciones, "
                 "convulsiones, sangrados).")

# --- Configuración RAG (diagnóstico 2026-09-28; ver outputs/evaluation/rag_diagnostic_report.md) ---
RAG_ENABLED: bool = True    # MODO B (True) vs MODO A sin-RAG (False). Ver --no-rag.
DEBUG_RAG: bool = False     # Si True, imprime QUERY/TOKENS/SCORES/THRESHOLD/DECISION/MODE. Ver --debug-rag.
SIM_THRESHOLD: float = 0.18  # recalibrado 2026-10-05 con corpus ampliado
# (antes 0.20 con vocab chico; con 17k términos el coseno se diluye).
# Puertas del respaldo semántico (ver answer()): score + solape query∩vocab.
SEM_FALLBACK_MIN: float = 0.55
SEM_FALLBACK_MIN_OVERLAP: int = 4
SEM_FALLBACK_HIGH: float = 0.72
SEM_FALLBACK_HIGH_OVERLAP: int = 3  # Umbral cosine TF-IDF sobre score CRUDO (antes del bonus de especie).
MIN_SHARED_TERMS: int = 2    # Regla de relevancia: el top-doc debe compartir >=2 términos
                             # de la consulta (evita "perro" solo -> envenenamiento).
SPECIES_BONUS: float = 0.15  # Solo se aplica DESPUÉS de pasar threshold + shared-terms.
GENERIC_IDF_MAX: float = 3.0  # Tokens con idf < 3.0 (que, perro, acude, ...) son genéricos;
                              # no bastan como evidencia de dominio (solo informativo en debug).

# Último diagnóstico (para tests): answer() lo rellena en cada llamada.
LAST_DEBUG: dict = {}

def es_fuera_de_ambito(idx, text: str) -> bool:
    from src.rag.retriever import tokens
    return len(set(tokens(text)) & set(idx.vocab)) < 2


def retrieve_semantic(query: str, k: int = 2, index="data/rag/index"):
    """Respaldo semántico perezoso (MiniLM). None si no hay índice o no pasa umbral."""
    from pathlib import Path as _P
    if not (_P(index) / "semantic.npz").exists():
        return None
    try:
        from src.rag.embeddings import query as _sq, SEM_THRESHOLD
    except ImportError:
        return None
    try:
        res = _sq(query, k=k, index=index)
    except Exception:
        return None
    good = [(i, s, d) for i, s, d in res if s >= SEM_THRESHOLD]
    return good or None


def _shared_terms(idx, query: str, doc_id: int) -> set[str]:
    """Términos de la consulta presentes en el documento (tokens filtrados)."""
    from src.rag.retriever import tokens
    return set(tokens(query)) & set(tokens(idx.docs[doc_id]))


def _generic_tokens(idx) -> set[str]:
    """Tokens demasiado frecuentes para probar relevancia (idf bajo). Solo debug."""
    try:
        return {t for t, i in idx.vocab.items() if float(idx.idf[i]) < GENERIC_IDF_MAX}
    except Exception:
        return set()


def retrieve_debug(idx, query: str, k: int = 2) -> dict:
    """Recuperación con trazabilidad (no genera texto). Devuelve scores y decisión.

    Orden corregido: score crudo -> shared-terms -> threshold -> bonus especie.
    """
    from src.rag.retriever import tokens
    toks = tokens(query)
    overlap = sorted(set(toks) & set(idx.vocab))
    generic = _generic_tokens(idx)
    overlap_specific = sorted(set(overlap) - generic)
    raw = idx.query(query, k=k + 2)
    qlow = query.lower()
    esp = next((v for kw, v in ESPECIES_MAP.items() if kw in qlow), None)
    cands = []
    for j, s in raw:
        shared = sorted(_shared_terms(idx, query, j))
        cands.append({"doc": j, "raw_score": round(float(s), 4),
                      "shared": shared, "n_shared": len(shared),
                      "text": idx.docs[j][:220]})
    passing = [c for c in cands
               if c["raw_score"] >= SIM_THRESHOLD and c["n_shared"] >= MIN_SHARED_TERMS]
    rescored = []
    for c in passing:
        bonus = SPECIES_BONUS if (esp and esp in idx.docs[c["doc"]].lower()) else 0.0
        rescored.append({**c, "bonus": bonus, "score": round(c["raw_score"] + bonus, 4)})
    rescored.sort(key=lambda c: c["score"], reverse=True)
    top = rescored[:k]
    if len(overlap) < 2:
        decision, mode = "FUERA_DE_AMBITO", "RECHAZADO"
    elif not top:
        decision, mode = "SIN_EVIDENCIA", "RECHAZADO"
    else:
        decision, mode = "RECUPERA", "RAG"
    return {"query": query, "tokens": toks, "overlap_vocab": overlap,
            "overlap_specific": overlap_specific,
            "especie": esp, "top_k": k, "threshold": SIM_THRESHOLD,
            "min_shared_terms": MIN_SHARED_TERMS,
            "candidates": cands, "selected": top,
            "decision": decision, "mode": mode}


def build_prompt(enc, passages: list[str], question: str, block: int, history: list = None) -> list[int]:
    # Solo la parte de respuesta como contexto (el prefijo "Pregunta:" distrae al modelo tiny)
    answers = [p.split("Respuesta:", 1)[1].strip() if "Respuesta:" in p else p for p in passages]
    ctx = "\n".join(f"- {a}" for a in answers)
    hist = ""
    if history:
        pq, pa = history[-1]
        hist = f"Anterior: {pq} / {pa[:120]}\n"
    head = f"Contexto:\n{ctx}\n\n{hist}Pregunta: {question}\nRespuesta:"
    ids = enc.encode(head)
    if len(ids) > block - 1:
        ids = ids[-(block - 1):]
    return ids


class _BpeAdapter:
    """Adapta tokenizers.Tokenizer a la interfaz .encode/.decode usada aquí."""
    def __init__(self, tok):
        self._tok = tok
    def encode(self, s: str) -> list[int]:
        return self._tok.encode(s).ids
    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)


def main(cfg="configs/tiny-18m.yaml", index="data/rag/index", ask=None, k=2,
         rag_enabled: bool = True, debug: bool = False):
    global EOT
    c = yaml.safe_load(Path(cfg).read_text(encoding="utf-8"))
    m = c["model"]
    if c.get("tokenizer", {}).get("name") == "bpe-propio":
        from src.tokenizer.tok_bpe import get_bpe_tok, eot_id
        enc = _BpeAdapter(get_bpe_tok(c["tokenizer"]["file"]))
        EOT = eot_id()
        print(f"tokenizer: BPE propio ({c['tokenizer']['file']})", flush=True)
    else:
        enc = get_enc()
    idx = TfidfIndex.load(Path(index))
    model = TinyGPT(m["vocab_size"], m["block_size"], m["n_layer"], m["n_head"],
                    m["n_embd"], m["dropout"], m["bias"], m["tie_weights"])
    ckpt = Path(c["inference"]["checkpoint"])
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()
    i = c["inference"]
    global RAG_ENABLED, DEBUG_RAG
    RAG_ENABLED = rag_enabled
    DEBUG_RAG = debug

    def _generate_with_passages(q: str, passages: list, history: list = None):
        ids = build_prompt(enc, passages, q, m["block_size"], history)
        x = torch.tensor([ids], dtype=torch.long)
        out = model.generate(x, max_new_tokens=i["max_new_tokens"], temperature=i["temperature"],
                             top_k=i["top_k"], top_p=i["top_p"],
                             repetition_penalty=i.get("repetition_penalty", 1.0))
        gen = out[0].tolist()[len(ids):]
        if EOT in gen:
            gen = gen[:gen.index(EOT)]
        return enc.decode(gen).strip()

    def answer(q: str, history: list = None, use_rag: bool | None = None,
               debug: bool | None = None) -> tuple[str, list[str]]:
        dbg = DEBUG_RAG if debug is None else debug
        rag = RAG_ENABLED if use_rag is None else use_rag
        LAST_DEBUG.clear()
        if es_vaga(q) and not history:
            LAST_DEBUG.update({"query": q, "mode": "RECHAZADO", "decision": "VAGA",
                               "threshold": SIM_THRESHOLD})
            if dbg:
                print(f"QUERY: {q}\nDECISION: VAGA (pide concretar)\nRESPONSE MODE: RECHAZADO")
            return ("¿Puedes concretar? Por ejemplo: especie (perro, gato...), síntoma "
                    "y desde cuándo. Temas: vacunas, desparasitación, alimentación, urgencias.", [])
        if not rag:
            # MODO A: sin retrieval, solo modelo generativo (para pruebas).
            ids0 = enc.encode(f"Pregunta: {q}\nRespuesta:")
            if len(ids0) > m["block_size"] - 1:
                ids0 = ids0[-(m["block_size"] - 1):]
            x = torch.tensor([ids0], dtype=torch.long)
            out = model.generate(x, max_new_tokens=i["max_new_tokens"], temperature=i["temperature"],
                                 top_k=i["top_k"], top_p=i["top_p"],
                                 repetition_penalty=i.get("repetition_penalty", 1.0))
            gen = out[0].tolist()[len(ids0):]
            if EOT in gen:
                gen = gen[:gen.index(EOT)]
            text = enc.decode(gen).strip()
            LAST_DEBUG.update({"query": q, "mode": "GENERATIVO_SIN_RAG",
                               "decision": "RAG_DESACTIVADO", "threshold": SIM_THRESHOLD})
            if dbg:
                print(f"QUERY: {q}\nEMBEDDING MODEL: (ninguno, RAG desactivado)"
                      f"\nRESPONSE MODE: GENERATIVO_SIN_RAG")
            return text, []
        # MODO B (RAG): la consulta va SOLA; el historial solo da contexto al prompt,
        # nunca se concatena para retrieval (evita contaminación entre temas).
        info = retrieve_debug(idx, q, k=k)
        LAST_DEBUG.update(info)
        if dbg:
            print(f"QUERY:\n{info['query']}")
            print(f"EMBEDDING MODEL: TF-IDF numpy (vocab={len(idx.vocab)}, docs={len(idx.docs)})")
            print(f"TOP_K: {info['top_k']}  SIMILARITY: cosine  THRESHOLD: {info['threshold']} "
                  f"(crudo, pre-bonus)  MIN_SHARED_TERMS: {info['min_shared_terms']}")
            print(f"TOKENS: {info['tokens']}")
            print(f"OVERLAP_VOCAB({len(info['overlap_vocab'])}): {info['overlap_vocab']}")
            print(f"OVERLAP_ESPECIFICO({len(info['overlap_specific'])}): {info['overlap_specific']}")
            for n, c in enumerate(info["candidates"][:k + 2], 1):
                print(f"RESULT {n}: doc={c['doc']} raw_score={c['raw_score']:.4f} "
                      f"shared({c['n_shared']})={c['shared']}\n  text: {c['text']!r}")
            print(f"ESPECIE: {info['especie']} (bonus +{SPECIES_BONUS} solo post-filtro)")
            for s in info["selected"]:
                print(f"SELECTED: doc={s['doc']} {s['raw_score']:.4f}+{s['bonus']:.2f}={s['score']:.4f}")
            print(f"DECISION: {info['decision']}\nRESPONSE MODE: {info['mode']}")
        if info["decision"] in ("FUERA_DE_AMBITO", "SIN_EVIDENCIA"):
            # Respaldo semántico: si TF-IDF rechaza, MiniLM puede rescatar
            # paráfrasis/sinónimos. Doble puerta anti-OOD (calibrado 2026-10-05):
            # el rescate exige score alto Y solape real con el vocabulario;
            # sin esto, "cristóbal colón" (0.68) o "dinosaurios" (0.55)
            # colaban como veterinaria. Si tampoco pasa, se rechaza igual.
            sem = retrieve_semantic(q, k=k, index=index) if rag else None
            n_over = len(info["overlap_vocab"])
            sem_ok = False
            if sem:
                s_top = sem[0][1]
                sem_ok = ((s_top >= SEM_FALLBACK_MIN and n_over >= SEM_FALLBACK_MIN_OVERLAP)
                          or (s_top >= SEM_FALLBACK_HIGH and n_over >= SEM_FALLBACK_HIGH_OVERLAP))
                if dbg:
                    print(f"SEMANTIC scores={[round(s,4) for _,s,_ in sem]} overlap={n_over} ok={sem_ok}")
            if sem_ok:
                passages = [d for _, _, d in sem]
                LAST_DEBUG.update({"semantic_fallback": True,
                                   "semantic_scores": [round(s, 4) for _, s, _ in sem],
                                   "mode": "RAG_SEMANTICO"})
            else:
                LAST_DEBUG.update({"semantic_fallback": False})
                if info["decision"] == "FUERA_DE_AMBITO":
                    return (FUERA_DE_AMBITO, [])
                # NO usar el fragmento más parecido: rechazar en vez de inventar.
                return (SIN_EVIDENCIA, [])
        else:
            passages = [idx.docs[s["doc"]] for s in info["selected"]]
        text = _generate_with_passages(q, passages, history)
        # Fallback extractivo: si la generación no se apoya en la fuente, devuelve la fuente
        top = passages[0] if passages else ""
        kw_top = _kw(top)
        rec = len(_kw(text) & kw_top) / max(1, len(kw_top))
        mode = "GENERATIVO_RAG"
        if passages and rec < 0.3:
            text = top.split("Respuesta:", 1)[1].strip() if "Respuesta:" in top else top
            text += " [extractivo]"
            mode = "EXTRACTIVO"
        LAST_DEBUG["rec"] = round(rec, 4)
        LAST_DEBUG["mode"] = mode
        if dbg:
            print(f"RECALL_LEXICO vs top-doc: {rec:.3f} -> {mode}")
        if es_urgencia(q):
            text = BANNER + "\n\n" + text
        return text, passages

    if ask:
        r, passages = answer(ask, debug=DEBUG_RAG)
        print("Fuentes:")
        for p in passages:
            print("-", p[:150])
        print("\nVet:", r)
        return
    print("Asistente veterinario con RAG (escribe 'salir' para terminar)")
    history: list = []
    while True:
        try:
            q = input("\nTú: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("salir", "exit", "quit"):
            break
        if not q:
            continue
        r, _ = answer(q, history)
        print("Vet:", r)
        history.append((q, r.replace(BANNER + "\n\n", "")))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/tiny-18m.yaml")
    ap.add_argument("--index", default="data/rag/index")
    ap.add_argument("--ask", default=None)
    ap.add_argument("-k", type=int, default=2)
    ap.add_argument("--no-rag", action="store_true", help="MODO A: genera sin retrieval")
    ap.add_argument("--debug-rag", action="store_true", help="Muestra QUERY/SCORES/THRESHOLD/DECISION")
    a = ap.parse_args()
    main(a.cfg, a.index, a.ask, a.k, rag_enabled=not a.no_rag, debug=a.debug_rag)
