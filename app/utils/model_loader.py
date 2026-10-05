"""Carga del checkpoint VET-TINY-GPT una sola vez (cache Streamlit)."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml


def listar_checkpoints() -> list[str]:
    cks = sorted((ROOT / "checkpoints").rglob("*.pt"))
    return [str(p.relative_to(ROOT)) for p in cks]


def cargar_modelo(ckpt_rel: str):
    from src.model.gpt import TinyGPT
    cfg = yaml.safe_load((ROOT / "configs" / "tiny-18m.yaml").read_text(encoding="utf-8"))
    m = cfg["model"]
    model = TinyGPT(m["vocab_size"], m["block_size"], m["n_layer"], m["n_head"],
                    m["n_embd"], m["dropout"], m["bias"], m["tie_weights"])
    sd = torch.load(ROOT / ckpt_rel, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()
    n = model.count_params()
    return model, {"params_M": round(n / 1e6, 2), **m, "checkpoint": ckpt_rel,
                   "infer": cfg["inference"]}
