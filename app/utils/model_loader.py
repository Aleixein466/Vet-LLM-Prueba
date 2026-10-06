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
    """Lista checkpoints compatibles con tiktoken (vocab 50257), excluyendo BPE."""
    cks = []
    for p in sorted((ROOT / "checkpoints").rglob("*.pt")):
        # Skip BPE checkpoints (vocab 16000)
        if "bpe" in str(p).lower():
            continue
        cks.append(str(p.relative_to(ROOT)))
    return cks


def _detect_vocab_size(ckpt_path: Path) -> int:
    """Detecta el vocab_size del checkpoint sin cargar todo el modelo."""
    sd = torch.load(ckpt_path, map_location="cpu")
    state = sd["model"] if "model" in sd else sd
    wte = state.get("wte.weight")
    if wte is None:
        wte = state.get("transformer.wte.weight")
    if wte is not None:
        return wte.shape[0]
    return 50257  # default


def cargar_modelo(ckpt_rel: str):
    from src.model.gpt import TinyGPT
    
    ckpt_path = ROOT / ckpt_rel
    vocab_size = _detect_vocab_size(ckpt_path)
    
    # Seleccionar config según vocab_size
    if vocab_size == 16000:
        cfg = yaml.safe_load((ROOT / "configs" / "tiny-18m-bpe.yaml").read_text(encoding="utf-8"))
    else:
        cfg = yaml.safe_load((ROOT / "configs" / "tiny-18m.yaml").read_text(encoding="utf-8"))
    
    m = cfg["model"]
    # Override vocab_size from checkpoint
    m = {**m, "vocab_size": vocab_size}
    
    model = TinyGPT(m["vocab_size"], m["block_size"], m["n_layer"], m["n_head"],
                    m["n_embd"], m["dropout"], m["bias"], m["tie_weights"])
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()
    n = model.count_params()
    return model, {"params_M": round(n / 1e6, 2), **m, "checkpoint": ckpt_rel,
                   "infer": cfg.get("inference", {})}


def get_default_checkpoint() -> str:
    """Retorna el mejor checkpoint por defecto (reentrenado step-6000 o pretraining final)."""
    cks = listar_checkpoints()
    # Preferir step-6000 (reentrenado final)
    for ckpt in cks:
        if "step-6000" in ckpt:
            return ckpt
    # Fallback: pretraining final
    for ckpt in cks:
        if "pretraining/final/model.pt" in ckpt and "backup" not in ckpt:
            return ckpt
    # Último recurso: primer checkpoint válido
    return cks[0] if cks else ""
