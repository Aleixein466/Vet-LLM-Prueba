"""SFT para el modelo BPE-16k (40 pasos, lote 4 sint + 2 replay + 2 real).

Espejo de sft.py con tok_bpe. Base: checkpoints/tiny-18m-bpe/last.pt.
Salida: checkpoints/tiny-18m-bpe-sft/last.pt.

Uso:
  python src/training/sft_bpe.py
"""
from __future__ import annotations
import json
import math
import random
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import yaml
from src.model.gpt import TinyGPT
from src.tokenizer.tok_bpe import get_bpe_tok, eot_id, encode_with_prompt_mask_bpe

SKIPPED = 0


def encode_pair(tok, eot, prompt: str, response: str, block: int):
    global SKIPPED
    try:
        ids, labels = encode_with_prompt_mask_bpe(tok, prompt, response)
    except ValueError:
        SKIPPED += 1
        return None
    ids = ids + [eot]
    labels = labels + [eot]
    if len(ids) > block:
        ids = ids[-block:]
        labels = labels[-block:]
    return ids, labels


def collate(tok, eot, batch, block: int):
    ok = [r for r in (encode_pair(tok, eot, b["prompt"], b["response"], block) for b in batch) if r]
    if not ok:
        return None, None
    ids, labs = zip(*ok)
    n = max(map(len, ids))
    x = torch.full((len(ids), n), eot, dtype=torch.long)
    y = torch.full((len(ids), n), -100, dtype=torch.long)
    for i, (a, b) in enumerate(zip(ids, labs)):
        x[i, :len(a)] = torch.tensor(a)
        y[i, :len(b)] = torch.tensor(b)
    return x, y


def pt_batch(train: np.ndarray, n: int, block: int):
    ix = torch.randint(0, len(train) - block - 1, (n,))
    x = torch.stack([torch.from_numpy(train[i:i + block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(train[i + 1:i + block + 1].astype(np.int64)) for i in ix])
    return x, y


def main(cfg_path="configs/tiny-18m-bpe.yaml", max_steps=None):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    s = cfg["sft"]
    if max_steps is not None:
        s["max_steps"] = max_steps
    torch.manual_seed(1234)
    random.seed(1234)
    tok = get_bpe_tok(cfg["tokenizer"]["file"])
    eot = eot_id(tok)
    m = cfg["model"]
    model = TinyGPT(m["vocab_size"], m["block_size"], m["n_layer"], m["n_head"],
                    m["n_embd"], m["dropout"], m["bias"], m["tie_weights"])
    base = Path(s["base_checkpoint"])
    if not base.exists():
        raise SystemExit(f"Falta pretraining BPE: {base}")
    sd = torch.load(base, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    print(f"params={model.count_params()/1e6:.2f}M base={base} eot={eot}")
    opt = torch.optim.AdamW(model.parameters(), lr=s["lr"])
    data = [json.loads(l) for l in Path(cfg["data"]["sft_train"]).read_text(encoding="utf-8").splitlines()]
    extra_p = cfg["data"].get("sft_extra")
    # sft_extra apunta al JSONL tiktoken; para BPE se reutiliza el texto
    # (los ids se recalculan con el BPE). Si no existe, solo sintético.
    extra = []
    if extra_p and Path(extra_p).exists():
        extra = [json.loads(l) for l in Path(extra_p).read_text(encoding="utf-8").splitlines()]
    train = np.fromfile(cfg["data"]["train_file"], dtype=np.uint16)
    print(f"sft_train={len(data)} sft_real={len(extra)} replay_tokens={len(train)}")

    def lr_at(t):
        w, T = s["warmup_steps"], s["max_steps"]
        if t < w:
            return s["lr"] * (t + 1) / w
        return s["lr"] * 0.5 * (1 + math.cos(math.pi * (t - w) / max(1, T - w)))

    model.train()
    step = 0
    n_syn, n_pt = 4, 2
    n_real = 2 if extra else 0
    while step < s["max_steps"]:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x_sft, y_sft = collate(tok, eot, random.sample(data, n_syn), m["block_size"])
        if x_sft is None:
            continue
        x_pt, y_pt = pt_batch(train, n_pt, m["block_size"])
        logits_sft, _ = model(x_sft)
        loss_sft = F.cross_entropy(logits_sft.view(-1, logits_sft.size(-1)), y_sft.view(-1), ignore_index=-100)
        _, loss_pt = model(x_pt, y_pt)
        if extra:
            x_r, y_r = collate(tok, eot, random.sample(extra, n_real), m["block_size"])
            if x_r is None:
                continue
            logits_r, _ = model(x_r)
            loss_real = F.cross_entropy(logits_r.view(-1, logits_r.size(-1)), y_r.view(-1), ignore_index=-100)
            loss = (loss_sft + loss_pt + loss_real) / 3.0
        else:
            loss = 0.5 * (loss_sft + loss_pt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), s["grad_clip"])
        opt.step()
        step += 1
        if step % 10 == 0 or step == s["max_steps"]:
            det = f"(sft={loss_sft.item():.3f} pt={loss_pt.item():.3f})"
            if extra:
                det = f"(sft={loss_sft.item():.3f} pt={loss_pt.item():.3f} real={loss_real.item():.3f})"
            print(f"[sft-bpe] step={step} loss={loss.item():.3f} {det}", flush=True)
    out = Path(s["checkpoint_dir"])
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "step": step}, out / "last.pt")
    print(f"sft-bpe done -> {out / 'last.pt'} (omitidos: {SKIPPED})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/tiny-18m-bpe.yaml")
    ap.add_argument("--max-steps", type=int, default=None)
    a = ap.parse_args()
    main(a.cfg, a.max_steps)
