"""Wrapper de tokenización con tiktoken (encoding gpt2, sin entrenamiento)."""
from __future__ import annotations
import tiktoken

ENCODING = "gpt2"

_enc = None

def get_enc():
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding(ENCODING)
    return _enc

def encode(text: str) -> list[int]:
    return get_enc().encode(text)

def decode(ids: list[int]) -> str:
    return get_enc().decode(ids)

EOT = 50256  # <|endoftext|>
VOCAB = 50257


def encode_with_prompt_mask(prompt: str, completion: str):
    """Codifica prompt+completion JUNTOS (igual que en pretraining) y enmascara el prompt.

    Evita la costura OOD de concatenar enc(prompt)+enc(" "+completion),
    que tiktoken segmenta distinto a enc(prompt+completion).
    """
    completion = completion.lstrip(" ")
    ids = get_enc().encode(prompt + completion)
    k = len(get_enc().encode(prompt))
    if get_enc().decode(ids[:k]) != prompt:
        k = None
        for i in range(1, len(ids) + 1):
            if get_enc().decode(ids[:i]) == prompt:
                k = i
                break
        if k is None:
            raise ValueError("No se pudo alinear prompt en la codificación conjunta")
    labels = [-100] * k + ids[k:]
    return ids, labels


def encode_pref_masked(prompt: str, completion: str):
    ids, labels = encode_with_prompt_mask(prompt, completion)
    mask = [0 if l == -100 else 1 for l in labels]
    return ids, mask
