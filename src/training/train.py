"""Loop de entrenamiento CPU con checkpoints reanudables."""
from __future__ import annotations
import argparse
import math
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from src.model.gpt import TinyGPT


def load_cfg(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def get_batch(data: np.ndarray, batch_size: int, block_size: int):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + block_size + 1].astype(np.int64)) for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, train, val, cfg):
    model.eval()
    out = {}
    for name, data in (("train", train), ("val", val)):
        losses = []
        for _ in range(20):
            x, y = get_batch(data, cfg["training"]["batch_size"], cfg["model"]["block_size"])
            _, loss = model(x, y)
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def main(cfg_path="configs/tiny-18m.yaml", max_steps=None):
    cfg = load_cfg(cfg_path)
    if max_steps is not None:
        cfg["training"]["max_steps"] = max_steps
    torch.manual_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])

    train = np.fromfile(cfg["data"]["train_file"], dtype=np.uint16)
    val = np.fromfile(cfg["data"]["val_file"], dtype=np.uint16)
    print(f"train_tokens={len(train)} val_tokens={len(val)}")

    m = cfg["model"]
    model = TinyGPT(m["vocab_size"], m["block_size"], m["n_layer"], m["n_head"], m["n_embd"],
                    m["dropout"], m["bias"], m["tie_weights"])
    print(f"params={model.count_params()/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                            weight_decay=cfg["training"]["weight_decay"])

    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    last = ckpt_dir / "last.pt"
    if cfg["training"]["resume"] and last.exists():
        sd = torch.load(last, map_location="cpu")
        model.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        step = sd["step"]
        print(f"resume step={step} loss={sd.get('loss')}")

    def lr_at(s):
        w = cfg["training"]["warmup_steps"]
        T = cfg["training"]["max_steps"]
        base = cfg["training"]["lr"]
        if s < w:
            return base * (s + 1) / w
        return base * 0.5 * (1 + math.cos(math.pi * (s - w) / max(1, T - w)))

    model.train()
    t0 = time.time()
    while step < cfg["training"]["max_steps"]:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch(train, cfg["training"]["batch_size"], m["block_size"])
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
        opt.step()
        step += 1
        if step % 50 == 0:
            print(f"step={step} loss={loss.item():.3f} lr={opt.param_groups[0]['lr']:.2e} {(time.time()-t0)/60:.1f}min", flush=True)
        if step % cfg["training"]["eval_every"] == 0:
            e = estimate_loss(model, train, val, cfg)
            print(f"[eval] step={step} train={e['train']:.3f} val={e['val']:.3f}", flush=True)
        if step % cfg["training"]["save_every"] == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "loss": loss.item()}, last)
            torch.save({"model": model.state_dict(), "step": step}, ckpt_dir / f"step-{step}.pt")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "loss": loss.item()}, last)
    print(f"done step={step}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/tiny-18m.yaml")
    ap.add_argument("--max-steps", type=int, default=None)
    a = ap.parse_args()
    main(a.cfg, a.max_steps)
