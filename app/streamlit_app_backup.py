"""VET-TINY-GPT — interfaz web local (Streamlit, CPU). Solo usa checkpoints existentes."""
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.model_loader import listar_checkpoints, cargar_modelo
from app.utils.system_info import info
from app.utils.generation import generar, ADVERTENCIA
from app.components import chat, metrics, architecture, training, clinical

st.set_page_config(page_title="VET-TINY-GPT", layout="wide")
st.title("VET-TINY-GPT")
st.subheader("Asistente experimental de lenguaje veterinario")
st.caption(ADVERTENCIA)

with st.sidebar:
    st.header("Modelo")
    cks = listar_checkpoints()
    ckpt = st.selectbox("Checkpoint", cks, index=len(cks) - 1 if cks else 0)
    st.header("Generación")
    st.session_state["temp"] = st.slider("Temperature", 0.0, 1.5, 0.0, 0.05)
    st.session_state["top_p"] = st.slider("Top P", 0.1, 1.0, 0.9, 0.05)
    st.session_state["mnt"] = st.slider("Max new tokens", 20, 150, 80, 5)
    st.session_state["rp"] = st.slider("Repetition penalty", 1.0, 1.5, 1.1, 0.05)
    st.header("Sistema")
    st.json(info())

if not cks:
    st.error("Sin checkpoints en checkpoints/. Entrena primero.")
    st.stop()

@st.cache_resource(show_spinner="Cargando modelo…")
def _modelo(ckpt_rel: str):
    return cargar_modelo(ckpt_rel)

model, meta = _modelo(ckpt)
st.sidebar.caption(f"{meta['params_M']}M params · {ckpt}")

pag = st.sidebar.radio("Página", ["Chat", "Generación", "Evaluación", "Arquitectura",
                                  "Training", "Historia clínica"])
if pag == "Chat":
    chat.render(model, meta)
elif pag == "Generación":
    st.header("✍️ Generación")
    p = st.text_area("Prompt", "El examen físico de un perro incluye")
    if st.button("Generar"):
        st.write(generar(model, p, temperature=st.session_state["temp"],
                         top_p=st.session_state["top_p"],
                         max_new_tokens=st.session_state["mnt"],
                         repetition_penalty=st.session_state["rp"]))
elif pag == "Evaluación":
    metrics.render()
elif pag == "Arquitectura":
    architecture.render(meta)
elif pag == "Training":
    training.render()
else:
    clinical.render(model)
