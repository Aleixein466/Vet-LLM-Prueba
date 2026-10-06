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
    chat_data = chats.activo()
    hist = chat_data["messages"]
    cid = st.session_state.active

    # Header with chat title
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
        <div>
            <h1 style="margin: 0; font-size: 1.5rem; font-weight: 800; background: linear-gradient(135deg, #0EA5E9, #34D399); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">VET-TINY-GPT</h1>
            <p style="margin: 0.25rem 0 0 0; color: var(--text-secondary); font-size: 0.9rem;">Consulta: {chat_data['title']}</p>
        </div>
        <span class="badge-exp">Asistente experimental</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption(ADVERTENCIA)

    # Controls row
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        especie = st.selectbox(
            "Especie", ESPECIES,
            index=ESPECIES.index(chat_data.get("species", "Perro"))
            if chat_data.get("species") in ESPECIES else 0,
            key=f"esp_{cid}"
        )
        chat_data["species"] = especie
    with col2:
        usar_rag = st.toggle("Usar RAG (Base Veterinaria)", value=True)
    with col3:
        ver_debug = st.toggle("Ver detalles de recuperación", value=False)

    # Chat history display
    chat_container = st.container()
    with chat_container:
        for q, a in hist:
            st.chat_message("user", avatar="🧑‍⚕️").write(q)
            st.chat_message("assistant", avatar="🐾").write(a)

    # Chat input
    q = st.chat_input("Describe el caso o pregunta…  (Enter para enviar)")
    
    if q:
        chats.titular(q)
        
        # Show user message immediately
        st.chat_message("user", avatar="🧑‍⚕️").write(q)
        
        # Generate response
        with st.chat_message("assistant", avatar="🐾"):
            response_placeholder = st.empty()
            
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
                    r = generar(
                        model, prompt,
                        temperature=st.session_state.get("temp", 0.0),
                        top_p=st.session_state.get("top_p", 0.9),
                        max_new_tokens=st.session_state.get("mnt", 80),
                        repetition_penalty=st.session_state.get("rp", 1.1)
                    )
                    top = pasajes[0]
                    kw_top = _kw(top)
                    rec = len(_kw(r) & kw_top) / max(1, len(kw_top))
                    if rec < 0.3:
                        r = (top.split("Respuesta:", 1)[1].strip()
                             if "Respuesta:" in top else top) + " [extractivo]"
                    
                    # Sources expander
                    with st.expander("📚 Fuentes", expanded=False):
                        for i, p in enumerate(pasajes):
                            st.markdown(f"""
                            <div class="card" style="margin-bottom: 0.5rem;">
                                <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.25rem;">Fuente {i+1}</div>
                                <div style="font-size: 0.85rem; color: var(--text-secondary);">{p[:400]}...</div>
                            </div>
                            """, unsafe_allow_html=True)
                
                if ver_debug:
                    with st.expander("🔍 Detalles de Recuperación", expanded=True):
                        st.markdown(f"""
                        <div style="font-family: var(--font-mono); font-size: 0.8rem;">
                            <div><strong>Decisión:</strong> {info['decision']}</div>
                            <div><strong>Threshold:</strong> {info['threshold']}</div>
                            <div><strong>Tokens:</strong> {info['tokens']}</div>
                            <div><strong>Solapamiento vocab:</strong> {info['overlap_vocab']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        for c in info["candidates"][:K + 2]:
                            st.markdown(f"""
                            <div class="card" style="margin-bottom: 0.5rem; padding: 0.75rem;">
                                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-muted);">
                                    <span>doc={c['doc']}</span>
                                    <span>score={c['raw_score']:.4f}</span>
                                    <span>compartidos={c['shared']}</span>
                                </div>
                                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.5rem;">{c['text'][:220]}...</div>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                ctx = ""
                if hist:
                    pq, pa = hist[-1]
                    ctx = f"Anterior: {pq} / {pa[:120]}\n"
                prompt = f"{ctx}Pregunta sobre {especie.lower()}: {q}\nRespuesta:"
                r = generar(
                    model, prompt,
                    temperature=st.session_state.get("temp", 0.0),
                    top_p=st.session_state.get("top_p", 0.9),
                    max_new_tokens=st.session_state.get("mnt", 80),
                    repetition_penalty=st.session_state.get("rp", 1.1)
                )
            
            # Check for urgency
            if es_urgente(q + " " + r):
                r = "⚠️ **Posible URGENCIA**: se requiere evaluación profesional cuanto antes.\n\n" + r
            
            # Stream the response
            response_placeholder.write_stream(stream_palabras(r))
        
        # Save to history
        hist.append((q, r))
        st.rerun()
    
    # Footer warning
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; color: var(--text-muted); font-size: 0.75rem; border-top: 1px solid var(--border-color); margin-top: 1.5rem;">
        🔒 {ADVERTENCIA}
    </div>
    """, unsafe_allow_html=True)