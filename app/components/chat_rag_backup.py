"""Componente: chat principal con RAG + historial + especie + banner de urgencia."""
import streamlit as st
from pathlib import Path
from app.utils.generation import generar, es_urgente, ADVERTENCIA, stream_palabras
from app.vet_chat import retrieve_debug, FUERA_DE_AMBITO, SIN_EVIDENCIA, _kw

ESPECIES = ["Perro", "Gato", "Bovino", "Equino", "Otra"]
K = 2


@st.cache_resource(show_spinner="Cargando base veterinaria…")
def _indice():
    from src.rag.retriever import TfidfIndex
    root = Path(__file__).resolve().parents[2]
    return TfidfIndex.load(root / "data" / "rag" / "index")


def _prompt_con_contexto(passages: list[str], especie: str, q: str, hist) -> str:
    answers = [p.split("Respuesta:", 1)[1].strip() if "Respuesta:" in p else p
               for p in passages]
    ctx = "\n".join(f"- {a}" for a in answers)
    anterior = ""
    if hist:
        pq, pa = hist[-1]
        anterior = f"Anterior: {pq} / {pa[:120]}\n"
    return f"Contexto:\n{ctx}\n\n{anterior}Pregunta sobre {especie.lower()}: {q}\nRespuesta:"


def render(model, meta):
    st.header("💬 Chat")
    st.caption(ADVERTENCIA)
    especie = st.selectbox("Especie", ESPECIES)
    usar_rag = st.toggle("Usar base veterinaria (RAG)", value=True)
    ver_debug = st.toggle("Ver detalle de recuperación", value=False)
    if "hist" not in st.session_state:
        st.session_state.hist = []
    for q, a in st.session_state.hist:
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)
    q = st.chat_input("Describe el caso o pregunta…")
    if q:
        if usar_rag:
            idx = _indice()
            info = retrieve_debug(idx, q, k=K)
            if ver_debug:
                with st.expander("🔍 Recuperación", expanded=True):
                    st.write(f"Decisión: **{info['decision']}** · "
                             f"threshold: {info['threshold']} · "
                             f"tokens: {info['tokens']} · "
                             f"solape: {info['overlap_vocab']}")
                    for c in info["candidates"][:K + 2]:
                        st.write(f"doc={c['doc']} score={c['raw_score']:.4f} "
                                 f"compartidos={c['shared']}")
                        st.caption(c["text"][:220])
            if info["decision"] == "FUERA_DE_AMBITO":
                r = FUERA_DE_AMBITO
                pasajes = []
            elif info["decision"] == "SIN_EVIDENCIA":
                r = SIN_EVIDENCIA
                pasajes = []
            else:
                pasajes = [idx.docs[s["doc"]] for s in info["selected"]]
                prompt = _prompt_con_contexto(pasajes, especie, q, st.session_state.hist)
                r = generar(model, prompt, temperature=st.session_state.get("temp", 0.0),
                            top_p=st.session_state.get("top_p", 0.9),
                            max_new_tokens=st.session_state.get("mnt", 80),
                            repetition_penalty=st.session_state.get("rp", 1.1))
                top = pasajes[0]
                kw_top = _kw(top)
                rec = len(_kw(r) & kw_top) / max(1, len(kw_top))
                if rec < 0.3:
                    r = (top.split("Respuesta:", 1)[1].strip()
                         if "Respuesta:" in top else top) + " [extractivo]"
                with st.expander("📚 Fuentes"):
                    for p in pasajes:
                        st.caption(p[:300])
        else:
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
