"""Auto-preguntas de consistencia: misma pregunta con distinta forma → misma respuesta.

Agrupa sft_val por (especie, tema) reconstruyendo el tema desde la respuesta,
genera con el checkpoint dado y mide Jaccard medio entre pares de paráfrasis.
Guarda outputs/evaluation/consistency.json
"""
from __future__ import annotations
import argparse
import json
import re
from itertools import combinations
from pathlib import Path
import torch
import yaml
from src.model.gpt import TinyGPT
from src.tokenizer.tok import get_enc, EOT
from src.data.corpus_sintetico import QA_BASE, QA_EMERG, ESPECIES

_STOP = re.compile(r"\s+")

def kw(text: str) -> set[str]:
    return {w.lower().strip(".,;:¿?¡!()") for w in _STOP.split(text) if len(w) > 4}

RESP2TEMA = {}
for _topic, _q, _a in list(QA_BASE) + list(QA_EMERG):
    for _esp, _info in ESPECIES.items():
        RESP2TEMA[_a.format(esp=_esp, art=_info["art"], peso=_info["peso"], t=_info["t"]).strip()] = (_topic, _esp)

def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))

def main(cfg_path="configs/tiny-18m.yaml", ckpt=None, out="outputs/evaluation/consistency.json"):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    m, ig = cfg["model"], cfg["inference"]
    enc = get_enc()
    model = TinyGPT(m["vocab_size"], m["block_size"], m["n_layer"], m["n_head"],
                    m["n_embd"], m["dropout"], m["bias"], m["tie_weights"])
    ck = Path(ckpt or ig["checkpoint"])
    sd = torch.load(ck, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()
    val = [json.loads(l) for l in Path(cfg["data"]["sft_val"]).read_text(encoding="utf-8").splitlines()]
    grupos: dict[tuple, list[str]] = {}
    for v in val:
        tema = RESP2TEMA.get(v["response"].strip())
        if tema:
            grupos.setdefault(tema, []).append(v["prompt"])
    res, tot = [], 0
    for (tema, esp), prompts in sorted(grupos.items()):
        if len(prompts) < 2:
            continue
        outs = []
        for p in prompts:
            ids = enc.encode(p)
            x = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad():
                o = model.generate(x, max_new_tokens=ig["max_new_tokens"], temperature=0.0,
                                   repetition_penalty=ig.get("repetition_penalty", 1.0))
            g = o[0].tolist()[len(ids):]
            if EOT in g:
                g = g[:g.index(EOT)]
            outs.append(enc.decode(g))
        pares = [jaccard(kw(a), kw(b)) for a, b in combinations(outs, 2)]
        j = sum(pares) / len(pares)
        res.append({"tema": tema, "especie": esp, "n": len(prompts), "jaccard_medio": round(j, 3)})
        tot += 1
    rep = {"grupos": res, "jaccard_global": round(sum(r["jaccard_medio"] for r in res) / max(1, len(res)), 3),
           "n_grupos": tot, "checkpoint": str(ck)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/tiny-18m.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="outputs/evaluation/consistency.json")
    a = ap.parse_args()
    main(a.cfg, a.ckpt, a.out)
