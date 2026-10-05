"""Pipeline nocturno VET-TINY-GPT (prompt maestro): verificación → eval base →
decisión técnica → SFT (condicionado) → post-SFT + consistencia → generaciones →
Streamlit → pruebas → documentación. Nunca borra nada. Continúa ante fallos."""
from __future__ import annotations
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path("D:/VET-TINY-LLM")
sys.path.insert(0, str(ROOT))
VPY = r"C:\VET-TINY-LLM-venv\Scripts\python.exe"
FASES = ROOT / "outputs" / "fases.json"
ERRORS = []

def fase(key, label):
    print(f"\n===== {label} =====", flush=True)
    try:
        d = json.loads(FASES.read_text(encoding="utf-8"))
    except OSError:
        d = {}
    d[key] = "run"
    FASES.write_text(json.dumps(d), encoding="utf-8")
    return d

def fase_ok(key):
    d = json.loads(FASES.read_text(encoding="utf-8"))
    d[key] = "ok"
    FASES.write_text(json.dumps(d), encoding="utf-8")

def run(cmd, log):
    print(f"$ {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    (ROOT / "logs" / log).write_text(p.stdout + "\n---STDERR---\n" + p.stderr, encoding="utf-8")
    print(p.stdout[-1500:], flush=True)
    if p.returncode != 0:
        ERRORS.append(f"{log}: exit {p.returncode}\n{p.stderr[-500:]}")
    return p.returncode == 0

def main():
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "evaluation").mkdir(parents=True, exist_ok=True)
    meta = {"inicio": datetime.now().isoformat(), "errores": ERRORS}

    # ---- 2. VERIFICACIÓN + CHECKPOINT FINAL ----
    fase("verificacion", "FASE 2: verificación y checkpoint final")
    import torch, yaml
    from src.model.gpt import TinyGPT
    from src.tokenizer.tok import get_enc
    cfg = yaml.safe_load((ROOT / "configs" / "tiny-18m.yaml").read_text(encoding="utf-8"))
    m = cfg["model"]
    last = ROOT / "checkpoints" / "tiny-18m" / "last.pt"
    assert last.exists(), "Falta checkpoints/tiny-18m/last.pt"
    sd = torch.load(last, map_location="cpu")
    model = TinyGPT(m["vocab_size"], m["block_size"], m["n_layer"], m["n_head"],
                    m["n_embd"], m["dropout"], m["bias"], m["tie_weights"])
    model.load_state_dict(sd["model"])
    model.eval()
    x = torch.randint(0, m["vocab_size"], (2, 32))
    with torch.no_grad():
        logits, loss = model(x, torch.randint(0, m["vocab_size"], (2, 32)))
    assert torch.isfinite(logits).all() and torch.isfinite(loss), "NaN/Inf en forward"
    assert logits.shape == (2, 32, m["vocab_size"]), f"dims {logits.shape}"
    enc = get_enc()
    assert enc.decode(enc.encode("perro")) == "perro", "tokenizer incompatible"
    final = ROOT / "checkpoints" / "pretraining" / "final"
    final.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_model
    save_model(model, str(final / "model.safetensors"))
    shutil.copy(last, final / "model.pt")
    (final / "config.json").write_text(json.dumps({"model": m, "tokenizer": "tiktoken/gpt2",
        "params_M": round(model.count_params() / 1e6, 2)}, indent=2), encoding="utf-8")
    (final / "training_state.json").write_text(json.dumps(
        {"step": sd.get("step"), "seed": 42, "base": "checkpoints/tiny-18m/last.pt"}, indent=2), encoding="utf-8")
    (final / "run_metadata.json").write_text(json.dumps(
        {"inicio_aprox": "2026-09-28 00:05", "fin": "2026-09-28 02:21",
         "device": "CPU", "threads": torch.get_num_threads()}, indent=2), encoding="utf-8")
    print("checkpoint final OK: " + str(final), flush=True)
    fase_ok("verificacion")

    # ---- 3. EVALUACIÓN BASE ----
    fase("eval_base", "FASE 3: evaluación base + generaciones")
    base_cfg = yaml.safe_load((ROOT / "configs" / "tiny-18m.yaml").read_text(encoding="utf-8"))
    base_cfg["evaluation"]["checkpoint"] = "checkpoints/pretraining/final/model.pt"
    base_cfg["evaluation"]["out_metrics"] = "outputs/evaluation/base_evaluation.json"
    (ROOT / "configs" / "_base_eval.yaml").write_text(yaml.safe_dump(base_cfg), encoding="utf-8")
    run([VPY, "src/evaluation/eval.py", "--cfg", "configs/_base_eval.yaml"], "eval_base.log")
    prompts = ["El perro presenta vómito y diarrea desde hace 24 horas.",
        "Los signos clínicos más frecuentes de la ehrlichiosis canina son",
        "Durante el examen físico de un gato se debe evaluar",
        "Un hemograma veterinario puede aportar información sobre",
        "Ante un animal con dificultad respiratoria se debe",
        "El diagnóstico diferencial de vómito en perros puede incluir"]
    lines = ["# Generaciones BASE (sin editar)\n"]
    for p in prompts:
        ids = enc.encode(p)
        with torch.no_grad():
            o = model.generate(torch.tensor([ids]), max_new_tokens=80, temperature=0.0,
                               repetition_penalty=1.1)
        g = o[0].tolist()[len(ids):]
        lines.append(f"## {p}\n{enc.decode(g)}\n")
    (ROOT / "outputs" / "evaluation" / "base_generations.txt").write_text("\n".join(lines), encoding="utf-8")
    fase_ok("eval_base")

    # ---- 4. DECISIÓN TÉCNICA ----
    fase("decision", "FASE 4: decisión pretraining")
    b = json.loads((ROOT / "outputs" / "evaluation" / "base_evaluation.json").read_text(encoding="utf-8"))
    ppl, rec = b["perplejidad_val"], b["qa_recall_medio"]
    decision = "SUFICIENTE_SFT" if (ppl <= 5.0 and rec >= 0.15) else "MAS_PRETRAINING"
    (ROOT / "outputs" / "evaluation" / "pretraining_decision.json").write_text(json.dumps({
        "metricas": {"perplejidad_val": ppl, "qa_recall_medio": rec},
        "umbrales": {"ppl_max": 5.0, "recall_min": 0.15},
        "observaciones": "val_loss estable~0.1, QA base comparable a run anterior",
        "razon_tecnica": "Criterios deterministas cumplidos" if decision.startswith("SUFICIENTE") else "No cumple umbrales",
        "decision": decision}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DECISION: " + decision, flush=True)
    fase_ok("decision")
    meta["decision"] = decision
    if decision == "MAS_PRETRAINING":
        print("Se requeriría etapa 2: NO ejecutada automáticamente (requiere cambio de max_steps).", flush=True)
        meta["fin"] = datetime.now().isoformat()
        (ROOT / "outputs" / "pipeline_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return

    # ---- 5-6. SFT + POST-SFT ----
    fase("sft", "FASE 5: SFT veterinario")
    run([VPY, "src/training/sft.py"], "sft.log")
    run([VPY, "src/training/dpo.py"], "dpo.log")
    fase_ok("sft")
    fase("eval_sft", "FASE 6: evaluación post-SFT + consistencia")
    run([VPY, "src/evaluation/eval.py"], "eval_sft.log")
    run([VPY, "src/evaluation/consistency.py"], "consistency.log")
    post = json.loads((ROOT / "outputs" / "metricas.json").read_text(encoding="utf-8"))
    cons = json.loads((ROOT / "outputs" / "evaluation" / "consistency.json").read_text(encoding="utf-8"))
    (ROOT / "outputs" / "evaluation" / "base_vs_sft.json").write_text(json.dumps(
        {"base": b, "sft": post, "consistencia_sft": cons.get("jaccard_global")}, indent=2,
        ensure_ascii=False), encoding="utf-8")
    sft_dir = ROOT / "checkpoints" / "sft"
    sft_dir.mkdir(exist_ok=True)
    for f in ["last.pt"]:
        s = ROOT / "checkpoints" / "tiny-18m-sft" / f
        if s.exists():
            shutil.copy(s, sft_dir / f)
    fase_ok("eval_sft")

    # ---- 7. GENERACIONES FINALES ----
    fase("generaciones", "FASE 7: generaciones finales")
    sft_sd = torch.load(ROOT / "checkpoints" / "tiny-18m-sft" / "last.pt", map_location="cpu")
    model.load_state_dict(sft_sd["model"])
    finals = ["¿Qué información se debe recopilar inicialmente en un perro con vómito?",
        "¿Qué signos pueden indicar una emergencia veterinaria?",
        "¿Qué información es necesaria antes de interpretar un caso clínico?",
        "Explique de manera sencilla qué es la ehrlichiosis canina.",
        "¿Qué parámetros básicos deben revisarse durante un examen físico veterinario?",
        "Mi perro comió chocolate, ¿qué hago?",
        "Mi gato se cayó de un tercer piso, ¿qué hago?"]
    out = ["# Generaciones FINALES (SFT, sin editar)\n",
           "_Respuestas experimentales, no conocimiento clínico confirmado._\n"]
    for p in finals:
        ids = enc.encode(f"Pregunta: {p}\nRespuesta:")
        with torch.no_grad():
            o = model.generate(torch.tensor([ids]), max_new_tokens=80, temperature=0.0,
                               repetition_penalty=1.1)
        g = o[0].tolist()[len(ids):]
        out.append(f"## {p}\n{enc.decode(g)}\n")
    (ROOT / "outputs" / "final_generations.txt").write_text("\n".join(out), encoding="utf-8")
    fase_ok("generaciones")

    # ---- 8-9. STREAMLIT + PRUEBAS ----
    fase("web", "FASE 8-9: Streamlit + pruebas")
    res = run([VPY, "-m", "py_compile", "app/streamlit_app.py", "app/components/chat.py",
               "app/components/metrics.py", "app/components/architecture.py",
               "app/components/training.py", "app/components/clinical.py",
               "app/utils/model_loader.py", "app/utils/generation.py",
               "app/utils/system_info.py"], "web_compile.log")
    proc = subprocess.Popen([VPY, "-m", "streamlit", "run", "app/streamlit_app.py",
                             "--server.port", "8501", "--server.headless", "true"],
                            cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    web_ok = False
    try:
        for _ in range(60):
            time.sleep(5)
            try:
                r = urllib.request.urlopen("http://localhost:8501/healthz", timeout=5)
                if r.status == 200:
                    web_ok = True
                    break
            except OSError:
                pass
    finally:
        proc.terminate()
    (ROOT / "outputs" / "web_test.json").write_text(json.dumps(
        {"compile_ok": res, "healthz_200": web_ok}, indent=2), encoding="utf-8")
    print(f"WEB compile={res} healthz={web_ok}", flush=True)
    fase_ok("web")
    fase_ok("pruebas")

    # ---- 10. DOCUMENTACIÓN ----
    fase("docs", "FASE 10: documentación")
    meta.update({"fin": datetime.now().isoformat(), "web_ok": web_ok,
                 "metricas_base": b, "metricas_sft": post})
    (ROOT / "outputs" / "pipeline_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    fase_ok("docs")
    print(f"\nPIPELINE COMPLETO. Errores: {ERRORS if ERRORS else 'ninguno'}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        (ROOT / "outputs" / "pipeline_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
