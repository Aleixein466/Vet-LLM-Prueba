"""Componente: estado del entrenamiento desde logs reales."""
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]


def render():
    st.header("🏋️ Training")
    rep = ROOT / "outputs" / "night_training_report.txt"
    if rep.exists():
        st.text(rep.read_text(encoding="utf-8"))
    else:
        st.info("Informe nocturno NO DISPONIBLE")
    curva = ROOT / "outputs" / "evaluation" / "curva_loss.png"
    if curva.exists():
        st.image(str(curva))
