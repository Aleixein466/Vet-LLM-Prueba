"""Paso 7b: DPO mínimo (policy vs referencia congelada) sobre prefs.jsonl."""
from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path
import torch
import torch.nn.functional as F
import yaml
from src.model.gpt import TinyGPT
from src.tokenizer.tok import get_enc, EOT, encode_pref_masked


def load_cfg(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def seq_logp(model, x, mask):
    """Suma de log-probs en posiciones mask==1 (mask sobre tokens objetivo)."""
    logits, _ = model(x)
    logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
    tgt = x[:, 1:]
    m = mask[:, 1:].bool()
    lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return (lp * m).sum(dim=1)


def encode_pref(enc, prompt, completion, block):
    ids, mask = encode_pref_masked(prompt, completion)
    ids = ids + [EOT]
    mask = mask + [1]
    if len(ids) > block:
        ids = ids[-block:]
        mask = mask[-block:]
    return ids, mask


def collate(enc, batch, block):
    xs, ms = [], []
    for b in batch:
        for key in ("chosen", "rejected"):
            ids, mask = encode_pref(enc, b["prompt"], b[key], block)
            xs.append(ids)
            ms.append(mask)
    n = max(map(len, xs))
    x = torch.full((len(xs), n), EOT, dtype=torch.long)
    mk = torch.zeros((len(xs), n), dtype=torch.long)
    for i, (a, m_) in enumerate(zip(xs, ms)):
        x[i, :len(a)] = torch.tensor(a)
        mk[i, :len(m_)] = torch.tensor(m_)
    return x, mk


def main(cfg_path="configs/tiny-18m.yaml", max_steps=None):
    cfg = load_cfg(cfg_path)
    d = cfg["dpo"]
    if max_steps is not None:
        d["max_steps"] = max_steps
    torch.manual_seed(7)
    random.seed(7)
    enc = get_enc()
    m = cfg["model"]
    kw = dict(vocab_size=m["vocab_size"], block_size=m["block_size"], n_layer=m["n_layer"],
              n_head=m["n_head"], n_embd=m["n_embd"], dropout=m["dropout"],
              bias=m["bias"], tie_weights=m["tie_weights"])
    policy = TinyGPT(**kw)
    ref = TinyGPT(**kw)
    base = Path(d["base_checkpoint"])
    if not base.exists():
        raise SystemExit(f"Falta SFT: {base}. Ejecuta sft.py primero.")
    sd = torch.load(base, map_location="cpu")
    state = sd["model"] if "model" in sd else sd
    policy.load_state_dict(state)
    ref.load_state_dict(state)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=d["lr"])
    data = [json.loads(l) for l in Path(cfg["data"]["prefs_file"]).read_text(encoding="utf-8").splitlines()]
    print(f"prefs={len(data)} beta={d['beta']}")
    out = Path(d["checkpoint_dir"])
    out.mkdir(parents=True, exist_ok=True)

    def lr_at(t):
        w, T = d["warmup_steps"], d["max_steps"]
        if t < w:
            return d["lr"] * (t + 1) / w
        return d["lr"] * 0.5 * (1 + math.cos(math.pi * (t - w) / max(1, T - w)))

    policy.train()
    step = 0
    while step < d["max_steps"]:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        batch = random.sample(data, d["batch_size"])
        x, mk = collate(enc, batch, m["block_size"])
        lp = seq_logp(policy, x, mk)
        with torch.no_grad():
            lp_ref = seq_logp(ref, x, mk)
        B = len(batch)
        # pares (chosen, rejected) intercalados
        lp_c, lp_r = lp[0::2], lp[1::2]
        rf_c, rf_r = lp_ref[0::2], lp_ref[1::2]
        adv = (lp_c - lp_r) - (rf_c - rf_r)
        loss = -F.logsigmoid(d["beta"] * adv).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 10 == 0 or step == d["max_steps"]:
            print(f"[dpo] step={step} loss={loss.item():.3f} adv={adv.mean().item():+.3f}", flush=True)
    torch.save({"model": policy.state_dict(), "step": step}, out / "last.pt")
    print(f"dpo done -> {out / 'last.pt'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/tiny-18m.yaml")
    ap.add_argument("--max-steps", type=int, default=None)
    a = ap.parse_args()
    main(a.cfg, a.max_steps)
