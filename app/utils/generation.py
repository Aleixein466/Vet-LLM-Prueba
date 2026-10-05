"""Generación con el modelo propio (greedy si temperature<=0) + corte en EOS."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from src.tokenizer.tok import get_enc, EOT

ADVERTENCIA = ("Este sistema es experimental y educativo. "
               "No sustituye la evaluación de un médico veterinario.")
URG = ["chocolate", "veneno", "envenen", "convulsi", "sangr", "atropell",
       "golpe de calor", "no respira", "inconsciente", "hinchad"]


def generar(model, prompt: str, temperature=0.0, top_k=40, top_p=0.9,
            repetition_penalty=1.1, max_new_tokens=80) -> str:
    enc = get_enc()
    ids = enc.encode(prompt)[-model.block_size:]
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature,
                             top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)
    gen = out[0].tolist()[len(ids):]
    if EOT in gen:
        gen = gen[:gen.index(EOT)]
    return enc.decode(gen).strip()


def es_urgente(texto: str) -> bool:
    t = texto.lower()
    return any(u in t for u in URG)


def stream_palabras(texto: str):
    import time
    for w in texto.split(" "):
        yield w + " "
        time.sleep(0.02)
