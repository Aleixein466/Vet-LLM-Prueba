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
from app.utils import chats
from app.components import chat, metrics, architecture, training, clinical

st.set_page_config(page_title="VET-TINY-GPT", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
section[data-testid="stSidebar"] { background: #121826; }
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #0EA5E9, #6366F1);
    border: none; font-weight: 600; width: 100%;
}
div[data-testid="stChatMessage"] { border-radius: 16px; }
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { margin-left: 12%; }
.badge-exp { font-size: 11px; padding: 2px 10px; border-radius: 999px;
    background: rgba(99,102,241,.12); color: #818CF8;
    border: 1px solid rgba(99,102,241,.4); font-weight: 600; }
.hist-btn > button { text-align: left; font-size: 13px; }
.side-label { font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
    color: #64748B; font-weight: 700; margin: 10px 0 4px; }
.profile-box { display: flex; align-items: center; gap: 8px; padding-top: 8px; }
.profile-avatar { width: 32px; height: 32px; border-radius: 50%;
    background: linear-gradient(135deg, #34D399, #0EA5E9);
    display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 800; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    if "_page" not in st.session_state:
        st.session_state._page = "Chat"
    if st.button("＋ Nueva Consulta", type="primary"):
        chats.nuevo()
        st.session_state._page = "Chat"
        st.session_state.pop("renaming", None)
        st.rerun()
    st.markdown('<div class="side-label">Historial</div>', unsafe_allow_html=True)
    store = chats.store()
    active_id = st.session_state.active
    orden = sorted(store.items(), key=lambda kv: (not kv[1]["pinned"], kv[1]["title"]))
    grupos: dict[str, list] = {}
    for cid, c in orden:
        grupos.setdefault("📌 Fijados" if c["pinned"] else "Recientes", []).append((cid, c))
    for g, items in grupos.items():
        st.caption(g)
        for cid, c in items:
            if st.session_state.get("renaming") == cid:
                nt = st.text_input("Nuevo nombre", value=c["title"], key=f"rn_{cid}")
                b1, b2 = st.columns(2)
                if b1.button("Guardar", key=f"sv_{cid}", type="primary"):
                    chats.renombrar(cid, nt)
                    st.session_state.pop("renaming", None)
                    st.rerun()
                if b2.button("Cancelar", key=f"cx_{cid}"):
                    st.session_state.pop("renaming", None)
                    st.rerun()
                continue
            ncols = [1, 0.15, 0.15, 0.15] if cid == active_id else [1, 0.15, 0.15]
            cols = st.columns(ncols)
            with cols[0]:
                if st.button(("📌 " if c["pinned"] else "") + c["title"],
                             key=f"open_{cid}", help="Abrir chat"):
                    chats.abrir(cid)
                    st.session_state._page = "Chat"
                    st.session_state.pop("renaming", None)
                    st.rerun()
            with cols[1]:
                if st.button("📍" if c["pinned"] else "📌", key=f"pin_{cid}",
                             help="Anclar / soltar"):
                    chats.fijar(cid)
                    st.rerun()
            with cols[2]:
                if st.button("🗑️", key=f"del_{cid}", help="Borrar chat"):
                    chats.borrar(cid)
                    st.session_state.pop("renaming", None)
                    st.rerun()
            if cid == active_id:
                with cols[3]:
                    if st.button("✏️", key=f"rnbtn_{cid}", help="Renombrar"):
                        st.session_state["renaming"] = cid
                        st.rerun()

    st.markdown('<div class="side-label">Sistema</div>', unsafe_allow_html=True)
    pag = st.radio("Página", ["Chat", "Generación", "Evaluación", "Arquitectura",
                              "Training", "Historia clínica"],
                   key="_page", label_visibility="collapsed")

    with st.expander("⚙️ Técnica / Ajustes"):
        cks = listar_checkpoints()
        ckpt = st.selectbox("Checkpoint", cks, index=len(cks) - 1 if cks else 0)
        st.session_state["temp"] = st.slider("Temperature", 0.0, 1.5, 0.0, 0.05)
        st.session_state["top_p"] = st.slider("Top P", 0.1, 1.0, 0.9, 0.05)
        st.session_state["mnt"] = st.slider("Max new tokens", 20, 150, 80, 5)
        st.session_state["rp"] = st.slider("Repetition penalty", 1.0, 1.5, 1.1, 0.05)
        st.json(info())

    st.markdown('<div class="side-label">Perfil</div>', unsafe_allow_html=True)
    meta_txt = st.session_state.get("_meta_txt", "…M params")
    st.markdown(f"""<div class="profile-box"><div class="profile-avatar">V</div>
        <div><div style="font-size:13px;font-weight:600">Veterinario</div>
        <div style="font-size:11px;color:#64748B">{meta_txt}</div></div></div>""",
                unsafe_allow_html=True)

if not cks:
    st.error("Sin checkpoints en checkpoints/. Entrena primero.")
    st.stop()

@st.cache_resource(show_spinner="Cargando modelo…")
def _modelo(ckpt_rel: str):
    return cargar_modelo(ckpt_rel)

model, meta = _modelo(ckpt)
st.session_state["_meta_txt"] = f"{meta['params_M']}M params"
st.sidebar.caption(f"{meta['params_M']}M params · {ckpt}")

if pag == "Chat":
    chat.render(model, meta)
elif pag == "Generación":
    st.header("✍️ Generación")
    p = st.text_area("Prompt", "El examen físico de un perro incluye")
    if st.button("Generar"):
        st.write(generar(model, p, temperature=st.session_state["temp"],
                         top_p=st.session_state.get("top_p", 0.9),
                         max_new_tokens=st.session_state.get("mnt", 80),
                         repetition_penalty=st.session_state.get("rp", 1.1)))
elif pag == "Evaluación":
    metrics.render()
elif pag == "Arquitectura":
    architecture.render(meta)
elif pag == "Training":
    training.render()
else:
    clinical.render(model)
