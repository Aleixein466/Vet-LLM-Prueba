"""Paso 7a: SFT con máscara de loss solo en la respuesta. Loss: Cross-Entropy (ignore_index=-100)."""
from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import yaml
from src.model.gpt import TinyGPT
from src.tokenizer.tok import get_enc, EOT, encode_with_prompt_mask


def load_cfg(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


SKIPPED_PAIRS = 0  # pares omitidos por no alinear en codificación conjunta


def encode_pair(enc, prompt: str, response: str, block: int):
    global SKIPPED_PAIRS
    try:
        ids, labels = encode_with_prompt_mask(prompt, response)
    except ValueError:
        SKIPPED_PAIRS += 1
        return None
    ids = ids + [EOT]
    labels = labels + [EOT]
    if len(ids) > block:
        ids = ids[-block:]
        labels = labels[-block:]
    return ids, labels


def collate(enc, batch, block: int):
    ok = [r for r in (encode_pair(enc, b["prompt"], b["response"], block) for b in batch) if r]
    if not ok:
        return None, None
    ids, labs = zip(*ok)
    n = max(map(len, ids))
    x = torch.full((len(ids), n), EOT, dtype=torch.long)
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


def main(cfg_path="configs/tiny-18m.yaml", max_steps=None):
    cfg = load_cfg(cfg_path)
    s = cfg["sft"]
    if max_steps is not None:
        s["max_steps"] = max_steps
    torch.manual_seed(1234)
    random.seed(1234)
    enc = get_enc()
    m = cfg["model"]
    model = TinyGPT(m["vocab_size"], m["block_size"], m["n_layer"], m["n_head"],
                    m["n_embd"], m["dropout"], m["bias"], m["tie_weights"])
    base = Path(s["base_checkpoint"])
    if not base.exists():
        raise SystemExit(f"Falta pretraining: {base}. Ejecuta train.py primero.")
    sd = torch.load(base, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    print(f"params={model.count_params()/1e6:.2f}M base={base}")
    opt = torch.optim.AdamW(model.parameters(), lr=s["lr"])  # AdamW
    data = [json.loads(l) for l in Path(cfg["data"]["sft_train"]).read_text(encoding="utf-8").splitlines()]
    extra_p = cfg["data"].get("sft_extra")
    extra = []
    if extra_p and Path(extra_p).exists():
        extra = [json.loads(l) for l in Path(extra_p).read_text(encoding="utf-8").splitlines()]
    train = np.fromfile(cfg["data"]["train_file"], dtype=np.uint16)
    print(f"sft_train={len(data)} sft_real={len(extra)} replay_tokens={len(train)}")
    out = Path(s["checkpoint_dir"])
    out.mkdir(parents=True, exist_ok=True)

    def lr_at(t):
        w, T = s["warmup_steps"], s["max_steps"]
        if t < w:
            return s["lr"] * (t + 1) / w
        return s["lr"] * 0.5 * (1 + math.cos(math.pi * (t - w) / max(1, T - w)))

    model.train()
    step = 0
    half = max(1, s["batch_size"] // 2)
    use_extra = len(extra) > 0
    # Con corpus real: mitad SFT sintético + 1/4 replay PT + 1/4 SFT real.
    # Sin extra: comportamiento original 50/50.
    n_syn = half
    n_pt = s["batch_size"] // 4 if use_extra else half
    n_real = s["batch_size"] - n_syn - n_pt if use_extra else 0
    while step < s["max_steps"]:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        # mitad SFT (loss enmascarado) + mitad replay pretraining (loss completo)
        batch = random.sample(data, n_syn)
        x_sft, y_sft = collate(enc, batch, m["block_size"])
        if x_sft is None:
            continue  # lote no-alineable (raro); reintenta sin contar paso
        x_pt, y_pt = pt_batch(train, n_pt, m["block_size"])
        logits_sft, _ = model(x_sft)
        loss_sft = F.cross_entropy(logits_sft.view(-1, logits_sft.size(-1)), y_sft.view(-1), ignore_index=-100)
        _, loss_pt = model(x_pt, y_pt)
        if use_extra:
            batch_r = random.sample(extra, n_real)
            x_r, y_r = collate(enc, batch_r, m["block_size"])
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
            detail = f"(sft={loss_sft.item():.3f} pt={loss_pt.item():.3f})"
            if use_extra:
                detail = f"(sft={loss_sft.item():.3f} pt={loss_pt.item():.3f} real={loss_real.item():.3f})"
            print(f"[sft] step={step} loss={loss.item():.3f} {detail}", flush=True)
    torch.save({"model": model.state_dict(), "step": step}, out / "last.pt")
    print(f"sft done -> {out / 'last.pt'} (pares omitidos por no-alineación: {SKIPPED_PAIRS})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/tiny-18m.yaml")
    ap.add_argument("--max-steps", type=int, default=None)
    a = ap.parse_args()
    main(a.cfg, a.max_steps)
