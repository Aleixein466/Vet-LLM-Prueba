"""Componente: arquitectura REAL del modelo (fines educativos, sin inventar)."""
import streamlit as st

BLOQUES = ["Tokenizer (tiktoken/gpt2)", "Token Embeddings + Positional Embeddings",
           "6× Transformer Block", "RMSNorm — NO presente (usa LayerNorm)",
           "Self-Attention MHA causal (GQA — NO presente)", "MLP GELU (SwiGLU — NO presente)",
           "Residual Connections", "LayerNorm final + LM Head (tied)", "Next Token"]


def render(meta):
    st.header("🧠 Arquitectura (real)")
    st.caption("Parámetros leídos del checkpoint cargado. Se indica lo ausente para no inducir error.")
    for b in BLOQUES:
        st.markdown(f"- {b}")
    st.subheader("Hiperparámetros")
    st.table([{"campo": k, "valor": v} for k, v in meta.items()
              if k in ("params_M", "vocab_size", "block_size", "n_layer", "n_head",
                       "n_embd", "dropout", "bias", "tie_weights", "checkpoint")])
