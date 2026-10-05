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

def es_fuera_de_ambito(idx, text: str) -> bool:
    from src.rag.retriever import tokens
    return len(set(tokens(text)) & set(idx.vocab)) < 2


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


def main(cfg="configs/tiny-18m.yaml", index="data/rag/index", ask=None, k=2):
    c = yaml.safe_load(Path(cfg).read_text(encoding="utf-8"))
    m = c["model"]
    enc = get_enc()
    idx = TfidfIndex.load(Path(index))
    model = TinyGPT(m["vocab_size"], m["block_size"], m["n_layer"], m["n_head"],
                    m["n_embd"], m["dropout"], m["bias"], m["tie_weights"])
    ckpt = Path(c["inference"]["checkpoint"])
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()
    i = c["inference"]

    def answer(q: str, history: list = None) -> tuple[str, list[str]]:
        if es_vaga(q) and not history:
            return ("¿Puedes concretar? Por ejemplo: especie (perro, gato...), síntoma "
                    "y desde cuándo. Temas: vacunas, desparasitación, alimentación, urgencias.", [])
        # Recuperación con memoria SOLO si la pregunta actual es corta (seguimiento);
        # si es completa, va sola para no contaminar con el tema anterior
        from src.rag.retriever import tokens as _t
        n_tok = len(_t(q))
        q_ret = (history[-1][0] + " " + history[-1][1][:200] + " " + q) if (history and n_tok < 4) else q
        # Fuera de ámbito: <2 términos de dominio, salvo seguimientos cortos con historial
        if (not history or n_tok >= 4) and es_fuera_de_ambito(idx, q):
            return (FUERA_DE_AMBITO, [])
        hits = idx.query(q_ret, k=k + 2)
        # Preferencia de especie: +0.15 si el pasaje menciona la especie preguntada
        qlow = q_ret.lower()
        esp = next((v for kw, v in ESPECIES_MAP.items() if kw in qlow), None)
        if esp:
            rescored = []
            for j, s in hits:
                if esp in idx.docs[j].lower():
                    s += 0.15
                rescored.append((j, s))
            rescored.sort(key=lambda t: t[1], reverse=True)
            hits = rescored[:k]
        else:
            hits = hits[:k]
        if hits and hits[0][1] < 0.20:
            hits = []  # sin coincidencia suficiente: no inventar con un pasaje aleatorio
        passages = [idx.docs[j] for j, _ in hits]
        if not passages:
            return ("No encontré información sobre eso en mi base. Pregunta sobre vacunas, "
                    "desparasitación, alimentación, constantes o urgencias (intoxicaciones, "
                    "convulsiones, sangrados).", [])
        ids = build_prompt(enc, passages, q, m["block_size"], history)
        x = torch.tensor([ids], dtype=torch.long)
        out = model.generate(x, max_new_tokens=i["max_new_tokens"], temperature=i["temperature"],
                             top_k=i["top_k"], top_p=i["top_p"],
                             repetition_penalty=i.get("repetition_penalty", 1.0))
        gen = out[0].tolist()[len(ids):]
        if EOT in gen:
            gen = gen[:gen.index(EOT)]
        text = enc.decode(gen).strip()
        # Fallback extractivo: si la generación no se apoya en la fuente, devuelve la fuente
        top = passages[0] if passages else ""
        kw_top = _kw(top)
        rec = len(_kw(text) & kw_top) / max(1, len(kw_top))
        if passages and rec < 0.3:
            text = top.split("Respuesta:", 1)[1].strip() if "Respuesta:" in top else top
            text += " [extractivo]"
        if es_urgencia(q):
            text = BANNER + "\n\n" + text
        return text, passages

    if ask:
        r, passages = answer(ask)
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
    a = ap.parse_args()
    main(a.cfg, a.index, a.ask, a.k)
