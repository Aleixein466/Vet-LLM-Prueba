"""Componente: chat principal con historial + especie + banner de urgencia."""
import streamlit as st
from app.utils.generation import generar, es_urgente, ADVERTENCIA, stream_palabras

ESPECIES = ["Perro", "Gato", "Bovino", "Equino", "Otra"]


def render(model, meta):
    st.header("💬 Chat")
    st.caption(ADVERTENCIA)
    especie = st.selectbox("Especie", ESPECIES)
    if "hist" not in st.session_state:
        st.session_state.hist = []
    for q, a in st.session_state.hist:
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)
    q = st.chat_input("Describe el caso o pregunta…")
    if q:
        ctx = ""
        if st.session_state.hist:
            pq, pa = st.session_state.hist[-1]
            ctx = f"Anterior: {pq} / {pa[:120]}\n"
        prompt = f"{ctx}Pregunta sobre {especie.lower()}: {q}\nRespuesta:"
        r = generar(model, prompt, temperature=st.session_state.get("temp", 0.0),
                    top_p=st.session_state.get("top_p", 0.9),
                    max_new_tokens=st.session_state.get("mnt", 80),
                    repetition_penalty=st.session_state.get("rp", 1.1))
        if es_urgente(q + " " + r):
            r = "⚠️ Posible URGENCIA: se requiere evaluación profesional cuanto antes.\n\n" + r
        st.session_state.hist.append((q, r))
        st.chat_message("user").write(q)
        st.chat_message("assistant").write_stream(stream_palabras(r))
