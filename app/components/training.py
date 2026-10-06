"""Componente: estado del entrenamiento desde logs reales."""
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]


def render():
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="margin: 0; font-size: 1.5rem; font-weight: 700;">🏋️ Entrenamiento</h2>
        <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0;">Estado y logs del proceso de entrenamiento</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Night training report
    rep = ROOT / "outputs" / "night_training_report.txt"
    if rep.exists():
        with st.expander("📋 Informe de Entrenamiento Nocturno", expanded=True):
            content = rep.read_text(encoding="utf-8")
            st.code(content, language=None)
    else:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <div class="empty-state-icon">📋</div>
            <div class="empty-state-title">Informe nocturno no disponible</div>
            <div class="empty-state-text">Ejecuta el entrenamiento nocturno para generar <code>outputs/night_training_report.txt</code></div>
        </div>
        """, unsafe_allow_html=True)
    
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
            <div class="empty-state-text">Ejecuta la evaluación para generar <code>outputs/evaluation/curva_loss.png</code></div>
        </div>
        """, unsafe_allow_html=True)
    
    # Training samples
    st.markdown("---")
    samples = ROOT / "outputs" / "night_training_samples.txt"
    if samples.exists():
        with st.expander("📝 Muestras de Entrenamiento", expanded=False):
            content = samples.read_text(encoding="utf-8")
            st.code(content, language=None)
    
    # Pipeline meta
    st.markdown("---")
    meta_file = ROOT / "outputs" / "pipeline_meta.json"
    if meta_file.exists():
        import json
        with st.expander("⚙️ Metadatos del Pipeline", expanded=False):
            st.json(json.loads(meta_file.read_text(encoding="utf-8")))
    
    # Checkpoints available
    st.markdown("---")
    st.markdown("### 💾 Checkpoints Disponibles")
    ckpts = sorted((ROOT / "checkpoints").rglob("*.pt"))
    if ckpts:
        for ckpt in ckpts:
            rel = ckpt.relative_to(ROOT)
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            st.markdown(f"""
            <div class="card" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <div>
                    <div style="font-weight: 600; font-family: var(--font-mono); font-size: 0.85rem;">{rel}</div>
                    <div style="font-size: 0.75rem; color: var(--text-muted);">{size_mb:.1f} MB</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No hay checkpoints en la carpeta checkpoints/")