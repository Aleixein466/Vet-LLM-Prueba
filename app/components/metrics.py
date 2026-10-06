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
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="margin: 0; font-size: 1.5rem; font-weight: 700;">📊 Evaluación del Modelo</h2>
        <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0;">Métricas de rendimiento y resultados de evaluación</p>
    </div>
    """, unsafe_allow_html=True)
    
    m = _load("outputs/metricas.json")
    b = _load("outputs/evaluation/base_evaluation.json")
    d = _load("outputs/evaluation/pretraining_decision.json")
    
    # Main metrics cards
    if m:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Perplejidad (val)",
                m.get("perplejidad_val", "NO DISPONIBLE"),
                help="Menor es mejor. Mide qué tan bien el modelo predice el siguiente token."
            )
        with col2:
            st.metric(
                "QA Accuracy",
                f"{m.get('qa_accuracy', 'NO DISPONIBLE')}%" if isinstance(m.get('qa_accuracy'), (int, float)) else m.get('qa_accuracy', 'NO DISPONIBLE'),
                help="Precisión en preguntas y respuestas veterinarias."
            )
        with col3:
            st.metric(
                "Parámetros",
                f"{m.get('params_M', 'NO DISPONIBLE')}M" if isinstance(m.get('params_M'), (int, float)) else m.get('params_M', 'NO DISPONIBLE'),
                help="Número de parámetros del modelo en millones."
            )
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📊</div>
            <div class="empty-state-title">Métricas no disponibles</div>
            <div class="empty-state-text">El archivo <code>outputs/metricas.json</code> no existe. Ejecuta la evaluación para generar métricas.</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed metrics in expandable sections
    col_left, col_right = st.columns(2)
    
    with col_left:
        if b:
            with st.expander("📈 Evaluación Base", expanded=True):
                st.json(b)
        else:
            with st.expander("📈 Evaluación Base", expanded=False):
                st.info("Archivo <code>outputs/evaluation/base_evaluation.json</code> no disponible", icon="ℹ️")
    
    with col_right:
        if d:
            with st.expander("🎯 Decisión Pretraining", expanded=True):
                st.json(d)
        else:
            with st.expander("🎯 Decisión Pretraining", expanded=False):
                st.info("Archivo <code>outputs/evaluation/pretraining_decision.json</code> no disponible", icon="ℹ️")
    
    # Loss curve
    st.markdown("---")
    curva = ROOT / "outputs" / "evaluation" / "curva_loss.png"
    if curva.exists():
        st.markdown("### 📉 Curva de Loss")
        st.image(str(curva), use_container_width=True)
    else:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <div class="empty-state-icon">📈</div>
            <div class="empty-state-title">Gráfica de loss no disponible</div>
            <div class="empty-state-text">Ejecuta el entrenamiento o evaluación para generar <code>outputs/evaluation/curva_loss.png</code></div>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional metrics files if they exist
    additional_files = [
        ("outputs/metricas_sft40.json", "SFT 40 pasos"),
        ("outputs/metricas_sft120.json", "SFT 120 pasos"),
    ]
    
    for filepath, label in additional_files:
        data = _load(filepath)
        if data:
            with st.expander(f"📋 {label}", expanded=False):
                st.json(data)