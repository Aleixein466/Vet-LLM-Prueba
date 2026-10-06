"""Componente: arquitectura REAL del modelo (fines educativos, sin inventar)."""
import streamlit as st


BLOQUES = [
    ("🔤", "Tokenizer", "tiktoken/gpt2 (BPE, 50257 tokens)"),
    ("📐", "Token Embeddings + Positional Embeddings", "Learned embeddings, tied weights"),
    ("🧱", "6× Transformer Block", "Pre-LN architecture"),
    ("⚖️", "LayerNorm", "NO usa RMSNorm (usa LayerNorm estándar)"),
    ("👁️", "Self-Attention MHA Causal", "Multi-head attention, NO GQA"),
    ("🔄", "MLP GELU", "Feed-forward con GELU, NO SwiGLU"),
    ("🔗", "Residual Connections", "Pre-norm residuals"),
    ("📏", "LayerNorm Final + LM Head", "Tied weights con token embeddings"),
    ("➡️", "Next Token Prediction", "Causal language modeling"),
]


def render(meta):
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="margin: 0; font-size: 1.5rem; font-weight: 700;">🧠 Arquitectura del Modelo</h2>
        <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0;">Descripción real de la arquitectura basada en el checkpoint cargado</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("Parámetros leídos del checkpoint cargado. Se indica lo ausente para no inducir error.")
    
    # Architecture blocks as cards
    for i, (icon, title, desc) in enumerate(BLOQUES):
        st.markdown(f"""
        <div class="card" style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.75rem;">
            <div style="font-size: 1.5rem;">{icon}</div>
            <div style="flex: 1;">
                <div style="font-weight: 600; font-size: 0.95rem; color: var(--text-primary);">{title}</div>
                <div style="font-size: 0.8rem; color: var(--text-secondary);">{desc}</div>
            </div>
            <div style="font-size: 0.7rem; color: var(--text-muted); font-family: var(--font-mono);">Bloque {i+1}/{len(BLOQUES)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Hyperparameters table
    st.markdown("### ⚙️ Hiperparámetros")
    
    # Filter and format meta
    fields = [
        ("params_M", "Parámetros (M)", "M"),
        ("vocab_size", "Vocabulario", ""),
        ("block_size", "Tamaño de contexto", " tokens"),
        ("n_layer", "Capas", ""),
        ("n_head", "Cabezas de atención", ""),
        ("n_embd", "Dimensión embedding", ""),
        ("dropout", "Dropout", ""),
        ("bias", "Bias", ""),
        ("tie_weights", "Pesos atados", ""),
        ("checkpoint", "Checkpoint", ""),
    ]
    
    table_data = []
    for key, label, suffix in fields:
        if key in meta:
            value = meta[key]
            if isinstance(value, float):
                value = f"{value:.2f}"
            elif isinstance(value, bool):
                value = "✅ Sí" if value else "❌ No"
            table_data.append({"Parámetro": label, "Valor": f"{value}{suffix}"})
    
    if table_data:
        st.table(table_data)
    
    # Inference config if available
    if "infer" in meta:
        st.markdown("---")
        st.markdown("### 🔧 Configuración de Inferencia")
        infer_data = [{"Parámetro": k, "Valor": v} for k, v in meta["infer"].items()]
        st.table(infer_data)
    
    # Model size visualization
    st.markdown("---")
    st.markdown("### 📏 Tamaño del Modelo")
    
    params_m = meta.get("params_M", 0)
    if params_m:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Parámetros totales", f"{params_m:.2f}M")
        with col2:
            # Estimate memory
            mem_mb = params_m * 4 / 1024 * 1024  # FP32
            mem_mb_fp16 = params_m * 2 / 1024 * 1024  # FP16
            st.metric("Memoria FP32", f"{mem_mb:.0f} MB")
        with col3:
            st.metric("Memoria FP16", f"{mem_mb_fp16:.0f} MB")