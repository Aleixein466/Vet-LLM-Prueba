"""Componente: métricas + gráficas desde archivos reales (o NO DISPONIBLE)."""
import json
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

def _load(rel):
    p = ROOT / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except OSError:
        return None


def render():
    st.header("📊 Evaluación")
    m = _load("outputs/metricas.json")
    b = _load("outputs/evaluation/base_evaluation.json")
    d = _load("outputs/evaluation/pretraining_decision.json")
    c1, c2, c3 = st.columns(3)
    if m:
        c1.metric("Perplejidad val", m.get("perplejidad_val", "NO DISPONIBLE"))
        c2.metric("QA accuracy", m.get("qa_accuracy", "NO DISPONIBLE"))
        c3.metric("Parámetros (M)", m.get("params_M", "NO DISPONIBLE"))
    else:
        st.warning("outputs/metricas.json NO DISPONIBLE")
    if b:
        st.subheader("Evaluación base")
        st.json(b)
    if d:
        st.subheader("Decisión pretraining")
        st.json(d)
    curva = ROOT / "outputs" / "evaluation" / "curva_loss.png"
    if curva.exists():
        st.subheader("Curva de loss")
        st.image(str(curva))
    else:
        st.info("Gráfica de loss NO DISPONIBLE")
