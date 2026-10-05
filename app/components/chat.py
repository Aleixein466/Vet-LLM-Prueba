"""Componente: chat principal con RAG + historial multi-chat + especie + banner de urgencia."""
import streamlit as st
from pathlib import Path
from app.utils.generation import generar, es_urgente, ADVERTENCIA, stream_palabras
from app.utils import chats
from app.vet_chat import retrieve_debug, FUERA_DE_AMBITO, SIN_EVIDENCIA, _kw

ESPECIES = ["Perro", "Gato", "Bovino", "Equino", "Exóticos", "Otra"]
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
    chats.store()
    chat = chats.activo()
    hist = chat["messages"]
    cid = st.session_state.active

    st.markdown('# VET-TINY-GPT <span class="badge-exp">Asistente experimental</span>',
                unsafe_allow_html=True)
    st.caption(ADVERTENCIA)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        especie = st.selectbox("Especie", ESPECIES,
                               index=ESPECIES.index(chat.get("species", "Perro"))
                               if chat.get("species") in ESPECIES else 0,
                               key=f"esp_{cid}")
        chat["species"] = especie
    with c2:
        usar_rag = st.toggle("Usar RAG (Base Veterinaria)", value=True)
    with c3:
        ver_debug = st.toggle("Ver detalles de recuperación", value=False)

    for q, a in hist:
        st.chat_message("user", avatar="🧑‍⚕️").write(q)
        st.chat_message("assistant", avatar="🐾").write(a)
    q = st.chat_input("Describe el caso o pregunta…  (Enter para enviar)")
    if q:
        chats.titular(q)
        if usar_rag:
            idx = _indice()
            info = retrieve_debug(idx, q, k=K)
            if info["decision"] == "FUERA_DE_AMBITO":
                r, pasajes = FUERA_DE_AMBITO, []
            elif info["decision"] == "SIN_EVIDENCIA":
                r, pasajes = SIN_EVIDENCIA, []
            else:
                pasajes = [idx.docs[s["doc"]] for s in info["selected"]]
                prompt = _prompt_con_contexto(pasajes, especie, q, hist)
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
        else:
            ctx = ""
            if hist:
                pq, pa = hist[-1]
                ctx = f"Anterior: {pq} / {pa[:120]}\n"
            prompt = f"{ctx}Pregunta sobre {especie.lower()}: {q}\nRespuesta:"
            r = generar(model, prompt, temperature=st.session_state.get("temp", 0.0),
                        top_p=st.session_state.get("top_p", 0.9),
                        max_new_tokens=st.session_state.get("mnt", 80),
                        repetition_penalty=st.session_state.get("rp", 1.1))
        if es_urgente(q + " " + r):
            r = "⚠️ Posible URGENCIA: se requiere evaluación profesional cuanto antes.\n\n" + r
        hist.append((q, r))
        st.chat_message("user", avatar="🧑‍⚕️").write(q)
        st.chat_message("assistant", avatar="🐾").write_stream(stream_palabras(r))
    st.caption("🔒 " + ADVERTENCIA)
