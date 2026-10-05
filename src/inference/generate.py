"""Generación única con checkpoint (tiktoken)."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import yaml
from src.model.gpt import TinyGPT
from src.tokenizer.tok import get_enc, EOT


def main(cfg="configs/tiny-18m.yaml", prompt="Pregunta: ¿Cada cuánto desparasito a mi perro?\nRespuesta:", max_new_tokens=80):
    c = yaml.safe_load(Path(cfg).read_text(encoding="utf-8"))
    m = c["model"]
    enc = get_enc()
    model = TinyGPT(m["vocab_size"], m["block_size"], m["n_layer"], m["n_head"],
                    m["n_embd"], m["dropout"], m["bias"], m["tie_weights"])
    ckpt = Path(c["inference"]["checkpoint"])
    if not ckpt.exists():
        raise SystemExit(f"Sin checkpoint {ckpt}. Entrena primero.")
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    ids = enc.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long)
    out = model.generate(x, max_new_tokens=max_new_tokens, temperature=c["inference"]["temperature"],
                         top_k=c["inference"]["top_k"], top_p=c["inference"]["top_p"],
                         repetition_penalty=c["inference"].get("repetition_penalty", 1.0))
    gen = out[0].tolist()[len(ids):]
    if EOT in gen:
        gen = gen[:gen.index(EOT)]
    print(enc.decode(ids + gen))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/tiny-18m.yaml")
    ap.add_argument("--prompt", default="Pregunta: ¿Cada cuánto desparasito a mi perro?\nRespuesta:")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    a = ap.parse_args()
    main(a.cfg, a.prompt, a.max_new_tokens)
