"""Monitor de progreso VET-TINY-GPT: reescribe outputs/progreso.html cada 60 s.
Lee el avance del pretraining del log + outputs/fases.json (fases del pipeline).
Consumo mínimo. Termina cuando el pretraining acaba o tras 6 h."""
from __future__ import annotations
import json
import re
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("D:/VET-TINY-LLM")
OUT = ROOT / "outputs" / "progreso.html"
FASES_JSON = ROOT / "outputs" / "fases.json"
TRAIN_LOG = Path(r"C:\Users\alexi\AppData\Local\Temp\opencode")
STEP_RE = re.compile(r"step=(\d+)\s+loss=([\d.]+)")
EVAL_RE = re.compile(r"\[eval\] step=(\d+) train=([\d.]+) val=([\d.]+)")
MAX_STEPS = 3000

FASES_BASE = [
    ("pretraining", "1. Pretraining (3000 pasos, CPU)"),
    ("verificacion", "2. Verificación checkpoint final"),
    ("eval_base", "3. Evaluación modelo base"),
    ("decision", "4. Decisión más pretraining"),
    ("sft", "5. SFT veterinario"),
    ("eval_sft", "6. Evaluación post-SFT"),
    ("generaciones", "7. Generaciones finales"),
    ("web", "8. Entorno web Streamlit"),
    ("pruebas", "9. Pruebas web"),
    ("docs", "10. Documentación final"),
]

CSS = """body{font-family:Segoe UI,Arial;background:#0f172a;color:#e2e8f0;margin:0;padding:32px}
.card{max-width:760px;margin:auto;background:#1e293b;border-radius:16px;padding:28px;box-shadow:0 8px 30px #0008}
h1{margin:0 0 4px;font-size:24px}.sub{color:#94a3b8;margin-bottom:18px}
.bar{height:22px;background:#334155;border-radius:11px;overflow:hidden;margin:8px 0 4px}
.fill{height:100%;background:linear-gradient(90deg,#22d3ee,#34d399);width:0%}
ul{list-style:none;padding:0}li{padding:7px 10px;border-radius:8px;margin:4px 0;background:#0f172a}
.ok{border-left:5px solid #34d399}.run{border-left:5px solid #22d3ee}.todo{border-left:5px solid #475569;color:#94a3b8}
.meta{color:#94a3b8;font-size:13px}.spin{display:inline-block;animation:s 1.2s linear infinite}@keyframes s{to{transform:rotate(360deg)}}
pre{background:#0f172a;padding:10px;border-radius:8px;overflow:auto;font-size:12px}"""

def parse_train():
    step, loss, ev = 0, None, None
    for f in TRAIN_LOG.rglob("sh_*.out"):
        try:
            txt = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "params=17.62M" not in txt and "train_tokens=367289" not in txt:
            continue
        for m in STEP_RE.finditer(txt):
            step, loss = int(m.group(1)), float(m.group(2))
        for m in EVAL_RE.finditer(txt):
            ev = (int(m.group(1)), float(m.group(2)), float(m.group(3)))
        if "done step=3000" in txt:
            return 3000, loss, ev, True
    return step, loss, ev, False

def main():
    t0 = time.time()
    while time.time() - t0 < 6 * 3600:
        step, loss, ev, done = parse_train()
        try:
            fases = json.loads(FASES_JSON.read_text(encoding="utf-8"))
        except OSError:
            fases = {}
        pct = min(100, step / MAX_STEPS * 100)
        items = []
        for key, label in FASES_BASE:
            st = fases.get(key, "run" if key == "pretraining" and not done else ("ok" if done and key == "pretraining" else "todo"))
            if key == "pretraining" and not done:
                st = "run"
            if key == "pretraining" and done:
                st = "ok"
            items.append((label, st))
        ev_txt = f"train={ev[1]:.3f} val={ev[2]:.3f} (paso {ev[0]})" if ev else "—"
        html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="30"><title>VET-TINY-GPT · Progreso</title>
<style>{CSS}</style></head><body><div class="card">
<h1><span class="spin">⚙️</span> VET-TINY-GPT · Progreso del proyecto</h1>
<div class="sub">Actualizado: {datetime.now():%H:%M:%S} · se recarga cada 30 s</div>
<div class="meta">Pretraining: paso {step}/{MAX_STEPS} ({pct:.1f}%) · loss={loss} · eval: {ev_txt}</div>
<div class="bar"><div class="fill" style="width:{pct}%"></div></div>
<ul>{''.join(f'<li class="{st}">{lab}</li>' for lab, st in items)}</ul>
<div class="meta">Fases 2–10 avanzan automáticamente al terminar el pretraining. No cerrar el equipo.</div>
</div></body></html>"""
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(html, encoding="utf-8")
        if done:
            break
        time.sleep(60)

if __name__ == "__main__":
    main()
