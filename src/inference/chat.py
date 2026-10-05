"""Paso 7d: chat final por terminal con el asistente veterinario."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import yaml
from src.model.gpt import TinyGPT
from src.tokenizer.tok import get_enc, EOT


def main(cfg="configs/tiny-18m.yaml", checkpoint=None):
    c = yaml.safe_load(Path(cfg).read_text(encoding="utf-8"))
    m = c["model"]
    enc = get_enc()
    model = TinyGPT(m["vocab_size"], m["block_size"], m["n_layer"], m["n_head"],
                    m["n_embd"], m["dropout"], m["bias"], m["tie_weights"])
    ckpt = Path(checkpoint or c["inference"]["checkpoint"])
    if not ckpt.exists():
        raise SystemExit(f"Sin checkpoint {ckpt}. Completa el pipeline primero.")
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()
    i = c["inference"]
    print("Asistente veterinario (escribe 'salir' para terminar)")
    while True:
        try:
            q = input("\nTú: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("salir", "exit", "quit"):
            break
        if not q:
            continue
        ids = enc.encode(f"Pregunta: {q}\nRespuesta:")
        x = torch.tensor([ids[-m['block_size']:]], dtype=torch.long)
        out = model.generate(x, max_new_tokens=i["max_new_tokens"], temperature=i["temperature"],
                             top_k=i["top_k"], top_p=i["top_p"],
                             repetition_penalty=i.get("repetition_penalty", 1.0))
        gen = out[0].tolist()[len(ids):]
        if EOT in gen:
            gen = gen[:gen.index(EOT)]
        print("Vet:", enc.decode(gen).strip())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/tiny-18m.yaml")
    ap.add_argument("--checkpoint", default=None)
    a = ap.parse_args()
    main(a.cfg, a.checkpoint)
