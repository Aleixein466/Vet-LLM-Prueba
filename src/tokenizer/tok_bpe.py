"""Wrapper del BPE propio 16k con la misma API mínima que tok.py.

Usado por sft_bpe.py y eval_bpe.py. El modelo BPE vive en un vocabulario
distinto al tiktoken: NO mezclar ids entre tokenizadores.
"""
from __future__ import annotations

TOK_FILE = "data/tokenizer/bpe16k/tokenizer.json"

_tok = None


def get_bpe_tok(path: str = TOK_FILE):
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(path)
    return _tok


def eot_id(tok=None) -> int:
    tok = tok or get_bpe_tok()
    eid = tok.token_to_id("[EOS]")
    assert eid is not None
    return eid


def encode_with_prompt_mask_bpe(tok, prompt: str, completion: str):
    """Igual que tok.encode_with_prompt_mask pero con tokenizers API."""
    completion = completion.lstrip(" ")
    ids = tok.encode(prompt + completion).ids
    k = len(tok.encode(prompt).ids)
    if tok.decode(ids[:k]) != prompt:
        k = None
        for i in range(1, len(ids) + 1):
            if tok.decode(ids[:i]) == prompt:
                k = i
                break
        if k is None:
            raise ValueError("No se pudo alinear prompt (BPE)")
    labels = [-100] * k + ids[k:]
    return ids, labels
