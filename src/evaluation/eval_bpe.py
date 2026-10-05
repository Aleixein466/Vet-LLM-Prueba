"""Evaluación del modelo BPE-16k: ppl en val_bpe + QA con keywords (BPE).

Espejo de eval.py con tok_bpe. Salida: outputs/evaluation/bpe_sft_eval.json
(si checkpoint es el SFT-BPE) o bpe_base_eval.json.

Uso:
  python src/evaluation/eval_bpe.py --ckpt checkpoints/tiny-18m-bpe/last.pt --out outputs/evaluation/bpe_base_eval.json --out-md outputs/evaluation/bpe_base_ejemplos.md
"""
from __future__ import annotations
import argparse
import json
import math
import re
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from src.model.gpt import TinyGPT
from src.tokenizer.tok_bpe import get_bpe_tok, eot_id

STOP = re.compile(r"\s+")


def keywords(text: str) -> set[str]:
    return {w.lower().strip(".,;:¿?¡!()") for w in STOP.split(text) if len(w) > 4}


def main(cfg_path="configs/tiny-18m-bpe.yaml", ckpt=None, out_metrics=None, out_md=None, qa_max=42):
    t0 = time.time()
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    m = cfg["model"]
    tok = get_bpe_tok(cfg["tokenizer"]["file"])
    eot = eot_id(tok)
    model = TinyGPT(m["vocab_size"], m["block_size"], m["n_layer"], m["n_head"],
                    m["n_embd"], m["dropout"], m["bias"], m["tie_weights"])
    ck = Path(ckpt or cfg["evaluation"]["checkpoint"])
    sd = torch.load(ck, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()

    val = np.fromfile(cfg["data"]["val_file"], dtype=np.uint16)
    bs, T = 16, m["block_size"]
    losses = []
    for _ in range(50):
        ix = torch.randint(0, len(val) - T - 1, (bs,))
        x = torch.stack([torch.from_numpy(val[i:i + T].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(val[i + 1:i + T + 1].astype(np.int64)) for i in ix])
        with torch.no_grad():
            _, loss = model(x, y)
        losses.append(loss.item())
    ppl = math.exp(sum(losses) / len(losses))

    qa = [json.loads(l) for l in Path(cfg["data"]["sft_val"]).read_text(encoding="utf-8").splitlines()][:qa_max]
    recs, examples = [], []
    for q in qa:
        ids = tok.encode(q["prompt"]).ids
        if len(ids) > m["block_size"] - 1:
            ids = ids[-(m["block_size"] - 1):]
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            out = model.generate(x, max_new_tokens=80, temperature=0.0,
                                 top_k=40, top_p=0.9, repetition_penalty=1.1)
        gen = tok.decode(out[0].tolist()[len(ids):])
        kw = keywords(q["response"])
        rec = len(keywords(gen) & kw) / max(1, len(kw))
        recs.append(rec)
        examples.append({"prompt": q["prompt"], "esperado": q["response"].strip(),
                         "generado": gen.strip(), "recall": round(rec, 3)})
    acc = sum(r >= 0.4 for r in recs) / max(1, len(recs))
    metrics = {"perplejidad_val": round(ppl, 2), "qa_recall_medio": round(sum(recs) / max(1, len(recs)), 3),
               "qa_accuracy": round(acc, 3), "qa_n": len(recs),
               "params_M": round(model.count_params() / 1e6, 2),
               "minutos": round((time.time() - t0) / 60, 1)}
    out_metrics = out_metrics or "outputs/evaluation/bpe_eval.json"
    out_md = out_md or "outputs/evaluation/bpe_ejemplos.md"
    Path(out_metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(out_metrics).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# Ejemplos QA (BPE)\n"]
    for ex in examples[:10]:
        md.append(f"**{ex['prompt']}**\n\n- Esperado: {ex['esperado']}\n- Generado: {ex['generado']}\n- recall={ex['recall']}\n")
    Path(out_md).write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/tiny-18m-bpe.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--qa-max", type=int, default=42)
    a = ap.parse_args()
    main(a.cfg, a.ckpt, a.out, a.out_md, a.qa_max)
