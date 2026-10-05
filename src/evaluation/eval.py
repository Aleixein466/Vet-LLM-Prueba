"""Paso 7c: evaluación — perplejidad en val + QA con keywords. Guarda outputs/metricas.json."""
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
from src.tokenizer.tok import get_enc

STOP = re.compile(r"\s+")

def keywords(text: str) -> set[str]:
    return {w.lower().strip(".,;:¿?¡!()") for w in STOP.split(text) if len(w) > 4}

def main(cfg_path="configs/tiny-18m.yaml", qa_max=None):
    t0 = time.time()
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    e = cfg["evaluation"]
    if qa_max is not None:
        e["qa_max"] = qa_max
    torch.manual_seed(0)
    enc = get_enc()
    m = cfg["model"]
    model = TinyGPT(m["vocab_size"], m["block_size"], m["n_layer"], m["n_head"],
                    m["n_embd"], m["dropout"], m["bias"], m["tie_weights"])
    ckpt = Path(e["checkpoint"])
    if not ckpt.exists():
        raise SystemExit(f"Falta checkpoint SFT: {ckpt}")
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()

    # 1) Perplejidad en val (pretraining)
    val = np.fromfile(cfg["data"]["val_file"], dtype=np.uint16)
    bs, T = cfg["training"]["batch_size"], m["block_size"]
    losses = []
    for _ in range(e["ppl_batches"]):
        ix = torch.randint(0, len(val) - T - 1, (bs,))
        x = torch.stack([torch.from_numpy(val[i:i + T].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(val[i + 1:i + T + 1].astype(np.int64)) for i in ix])
        with torch.no_grad():
            _, loss = model(x, y)
        losses.append(loss.item())
    ppl = math.exp(sum(losses) / len(losses))

    # 2) QA held-out: recall de keywords
    qa = [json.loads(l) for l in Path(cfg["data"]["sft_val"]).read_text(encoding="utf-8").splitlines()][:e["qa_max"]]
    i_cfg = cfg["inference"]
    recs, examples = [], []
    for q in qa:
        ids = enc.encode(q["prompt"])
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            out = model.generate(x, max_new_tokens=i_cfg["max_new_tokens"], temperature=i_cfg["temperature"],
                                 top_k=i_cfg["top_k"], top_p=i_cfg["top_p"],
                                 repetition_penalty=i_cfg.get("repetition_penalty", 1.0))
        gen = enc.decode(out[0].tolist()[len(ids):])
        kw = keywords(q["response"])
        hit = keywords(gen) & kw
        rec = len(hit) / max(1, len(kw))
        recs.append(rec)
        examples.append({"prompt": q["prompt"], "esperado": q["response"].strip(),
                         "generado": gen.strip(), "recall": round(rec, 3)})
    acc = sum(r >= 0.4 for r in recs) / max(1, len(recs))
    metrics = {"perplejidad_val": round(ppl, 2), "qa_recall_medio": round(sum(recs)/max(1,len(recs)), 3),
               "qa_accuracy": round(acc, 3), "qa_n": len(recs),
               "params_M": round(model.count_params()/1e6, 2),
               "minutos": round((time.time()-t0)/60, 1)}
    Path(e["out_metrics"]).parent.mkdir(parents=True, exist_ok=True)
    Path(e["out_metrics"]).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# Ejemplos QA\n"]
    for ex in examples[:10]:
        md.append(f"**{ex['prompt']}**\n\n- Esperado: {ex['esperado']}\n- Generado: {ex['generado']}\n- recall={ex['recall']}\n")
    Path(e["out_examples"]).write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/tiny-18m.yaml")
    ap.add_argument("--qa-max", type=int, default=None)
    a = ap.parse_args()
    main(a.cfg, a.qa_max)
