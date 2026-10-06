"""VET-TINY-GPT — Interfaz web local (Streamlit, CPU). Solo usa checkpoints existentes."""
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.model_loader import listar_checkpoints, cargar_modelo, get_default_checkpoint
from app.utils.system_info import info
from app.utils.generation import generar, ADVERTENCIA
from app.utils import chats
from app.components import chat, metrics, architecture, training, clinical, proceso

# Page configuration
st.set_page_config(
    page_title="VET-TINY-GPT",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global Styles ──────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
/* ── CSS Variables ── */
:root {
    --color-primary: #0EA5E9;           /* Sky 500 */
    --color-primary-dark: #0284C7;      /* Sky 600 */
    --color-primary-light: #38BDF8;     /* Sky 400 */
    --color-secondary: #6366F1;         /* Indigo 500 */
    --color-secondary-dark: #4F46E5;    /* Indigo 600 */
    --color-accent: #34D399;            /* Emerald 400 */
    --color-accent-dark: #10B981;       /* Emerald 500 */
    --color-warning: #F59E0B;           /* Amber 500 */
    --color-danger: #EF4444;            /* Red 500 */
    --color-danger-light: #FCA5A5;      /* Red 300 */
    
    --bg-primary: #0F172A;              /* Slate 950 */
    --bg-secondary: #1E293B;            /* Slate 800 */
    --bg-tertiary: #334155;             /* Slate 700 */
    --bg-card: #1E293B;                 /* Slate 800 */
    --bg-card-hover: #334155;           /* Slate 700 */
    --bg-input: #0F172A;                /* Slate 950 */
    
    --text-primary: #F1F5F9;            /* Slate 100 */
    --text-secondary: #94A3B8;          /* Slate 400 */
    --text-muted: #64748B;              /* Slate 500 */
    --text-inverse: #0F172A;            /* Slate 950 */
    
    --border-color: #334155;            /* Slate 700 */
    --border-light: #475569;            /* Slate 600 */
    
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 24px;
    --radius-full: 9999px;
    
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.4);
    --shadow-lg: 0 8px 24px rgba(0,0,0,0.5);
    --shadow-glow: 0 0 20px rgba(14, 165, 233, 0.3);
    
    --transition-fast: 150ms ease;
    --transition-normal: 250ms ease;
    --transition-slow: 350ms ease;
    
    --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
}

/* ── Base Styles ── */
html, body, [class*="css"] {
    font-family: var(--font-sans) !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-color) !important;
    padding-top: 1.5rem !important;
}

section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--color-primary), var(--color-secondary)) !important;
    border: none !important;
    font-weight: 600 !important;
    width: 100% !important;
    border-radius: var(--radius-md) !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.95rem !important;
    transition: all var(--transition-fast) !important;
    box-shadow: var(--shadow-md) !important;
}

section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-lg), var(--shadow-glow) !important;
}

section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-md) !important;
    transition: all var(--transition-fast) !important;
}

section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
    background: var(--bg-card-hover) !important;
    border-color: var(--border-light) !important;
}

/* ── Typography ── */
.side-label {
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--text-muted) !important;
    font-weight: 700 !important;
    margin: 1.5rem 0 0.75rem 0 !important;
    display: block !important;
}

.side-label:first-of-type {
    margin-top: 0.5rem !important;
}

/* ── Chat Messages ── */
div[data-testid="stChatMessage"] {
    border-radius: var(--radius-lg) !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 0.75rem !important;
    box-shadow: var(--shadow-sm) !important;
    border: 1px solid var(--border-color) !important;
    animation: fadeInUp 0.3s ease forwards !important;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.1), rgba(99, 102, 241, 0.1)) !important;
    border-color: rgba(14, 165, 233, 0.2) !important;
    margin-left: 8% !important;
}

div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: linear-gradient(135deg, rgba(52, 211, 153, 0.1), rgba(16, 185, 129, 0.1)) !important;
    border-color: rgba(52, 211, 153, 0.2) !important;
    margin-right: 8% !important;
}

/* ── Badges ── */
.badge-exp {
    font-size: 0.65rem !important;
    padding: 0.25rem 0.75rem !important;
    border-radius: var(--radius-full) !important;
    background: rgba(99, 102, 241, 0.15) !important;
    color: #818CF8 !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
    font-weight: 600 !important;
    display: inline-block !important;
    vertical-align: middle !important;
    margin-left: 0.5rem !important;
}

.badge-warning {
    background: rgba(245, 158, 11, 0.15) !important;
    color: #FBBF24 !important;
    border-color: rgba(245, 158, 11, 0.4) !important;
}

.badge-danger {
    background: rgba(239, 68, 68, 0.15) !important;
    color: #FCA5A5 !important;
    border-color: rgba(239, 68, 68, 0.4) !important;
}

.badge-success {
    background: rgba(16, 185, 129, 0.15) !important;
    color: #34D399 !important;
    border-color: rgba(16, 185, 129, 0.4) !important;
}

/* ── Chat History Buttons ── */
.hist-btn > button {
    text-align: left !important;
    font-size: 0.85rem !important;
    padding: 0.625rem 0.75rem !important;
    border-radius: var(--radius-md) !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    color: var(--text-primary) !important;
    transition: all var(--transition-fast) !important;
    width: 100% !important;
    justify-content: flex-start !important;
}

.hist-btn > button:hover {
    background: var(--bg-tertiary) !important;
    border-color: var(--border-color) !important;
}

.hist-btn > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.2), rgba(99, 102, 241, 0.2)) !important;
    border-color: rgba(14, 165, 233, 0.3) !important;
    font-weight: 600 !important;
}

/* ── Profile Box ── */
.profile-box {
    display: flex !important;
    align-items: center !important;
    gap: 0.75rem !important;
    padding: 0.75rem !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    margin-top: 0.5rem !important;
}

.profile-avatar {
    width: 40px !important;
    height: 40px !important;
    border-radius: var(--radius-full) !important;
    background: linear-gradient(135deg, var(--color-accent), var(--color-primary)) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    color: var(--text-inverse) !important;
    font-weight: 800 !important;
    font-size: 1.1rem !important;
    box-shadow: var(--shadow-md) !important;
}

/* ── Input Styling ── */
.stTextArea textarea, .stTextInput input, .stSelectbox > div > div {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
    transition: all var(--transition-fast) !important;
}

.stTextArea textarea:focus, .stTextInput input:focus, .stSelectbox > div > div:focus-within {
    border-color: var(--color-primary) !important;
    box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.2) !important;
    outline: none !important;
}

.stSlider > div > div > div[role="slider"] {
    background: var(--color-primary) !important;
}

.stSlider > div > div > div[data-baseweb="slider"] {
    background: var(--border-color) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    font-weight: 500 !important;
    color: var(--text-primary) !important;
}

.streamlit-expanderContent {
    background: var(--bg-primary) !important;
    border: 1px solid var(--border-color) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
    padding: 1rem !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem !important;
    background: transparent !important;
    border-bottom: 1px solid var(--border-color) !important;
    padding-bottom: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: var(--radius-md) var(--radius-md) 0 0 !important;
    padding: 0.75rem 1.25rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    transition: all var(--transition-fast) !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    background: var(--bg-tertiary) !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--color-primary) !important;
    background: var(--bg-card) !important;
    border-bottom: 2px solid var(--color-primary) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1.25rem !important;
    transition: all var(--transition-normal) !important;
}

[data-testid="stMetric"]:hover {
    border-color: var(--border-light) !important;
    box-shadow: var(--shadow-md) !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    font-size: 1.75rem !important;
    font-family: var(--font-mono) !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: var(--radius-md) !important;
    font-weight: 500 !important;
    transition: all var(--transition-fast) !important;
    border: none !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--color-primary), var(--color-secondary)) !important;
    color: white !important;
    box-shadow: var(--shadow-md) !important;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-lg), var(--shadow-glow) !important;
}

.stButton > button[kind="secondary"] {
    background: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-color) !important;
}

.stButton > button[kind="secondary"]:hover {
    background: var(--bg-card-hover) !important;
    border-color: var(--border-light) !important;
}

/* ── Radio ── */
.stRadio > div {
    gap: 0.5rem !important;
}

.stRadio label {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.75rem 1rem !important;
    cursor: pointer !important;
    transition: all var(--transition-fast) !important;
}

.stRadio label:hover {
    border-color: var(--border-light) !important;
    background: var(--bg-card-hover) !important;
}

.stRadio input:checked + div {
    border-color: var(--color-primary) !important;
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.1), rgba(99, 102, 241, 0.1)) !important;
}

/* ── Toggle ── */
.stToggle > label {
    background: var(--bg-tertiary) !important;
    border-radius: var(--radius-full) !important;
}

.stToggle input:checked + div > div {
    background: var(--color-primary) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: var(--bg-input) !important;
}

.stSelectbox [data-baseweb="popover"] {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
}

/* ── Dataframe / Table ── */
.stDataFrame, .stTable {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
}

/* ── Code ── */
code, pre {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-mono) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-primary);
}

::-webkit-scrollbar-thumb {
    background: var(--bg-tertiary);
    border-radius: var(--radius-full);
}

::-webkit-scrollbar-thumb:hover {
    background: var(--border-light);
}

/* ── Header Styles ── */
.main-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--border-color);
}

.main-header h1 {
    font-size: 1.75rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--color-primary), var(--color-accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}

.page-subtitle {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin: 0.25rem 0 0 0;
}

/* ── Card ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    transition: all var(--transition-normal);
}

.card:hover {
    border-color: var(--border-light);
    box-shadow: var(--shadow-md);
}

.card-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 0.5rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.card-text {
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 1.6;
    margin: 0;
}

/* ── Empty State ── */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: var(--text-muted);
}

.empty-state-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.5;
}

.empty-state-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}

.empty-state-text {
    font-size: 0.9rem;
    max-width: 300px;
    margin: 0 auto;
}

/* ── Status Indicators ── */
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 0.5rem;
}

.status-online { background: var(--color-accent); box-shadow: 0 0 8px var(--color-accent); }
.status-warning { background: var(--color-warning); box-shadow: 0 0 8px var(--color-warning); }
.status-offline { background: var(--color-danger); box-shadow: 0 0 8px var(--color-danger); }

/* ── Footer ── */
.app-footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border-color);
    text-align: center;
    color: var(--text-muted);
    font-size: 0.75rem;
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .main-header h1 { font-size: 1.5rem; }
    .card { padding: 1rem; }
    div[data-testid="stChatMessage"] { margin-left: 0 !important; margin-right: 0 !important; }
}
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Session State Initialization ──────────────────────────────────────────
if "_page" not in st.session_state:
    st.session_state._page = "Chat"
if "chats" not in st.session_state:
    st.session_state.chats = {}
    st.session_state.active = None

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    # New Chat Button
    if st.button("＋ Nueva Consulta", type="primary", use_container_width=True):
        chats.nuevo()
        st.session_state._page = "Chat"
        st.session_state.pop("renaming", None)
        st.rerun()
    
    # Chat History
    st.markdown('<span class="side-label">Historial</span>', unsafe_allow_html=True)
    store = chats.store()
    active_id = st.session_state.active
    
    # Sort: pinned first, then by title
    orden = sorted(store.items(), key=lambda kv: (not kv[1]["pinned"], kv[1]["title"].lower()))
    grupos: dict[str, list] = {}
    for cid, c in orden:
        grupos.setdefault("📌 Fijados" if c["pinned"] else "Recientes", []).append((cid, c))
    
    for g, items in grupos.items():
        st.caption(g)
        for cid, c in items:
            is_active = cid == active_id
            
            if st.session_state.get("renaming") == cid:
                # Rename mode
                nt = st.text_input("Nuevo nombre", value=c["title"], key=f"rn_{cid}", label_visibility="collapsed")
                col_save, col_cancel = st.columns(2)
                with col_save:
                    if st.button("Guardar", key=f"sv_{cid}", type="primary", use_container_width=True):
                        chats.renombrar(cid, nt)
                        st.session_state.pop("renaming", None)
                        st.rerun()
                with col_cancel:
                    if st.button("Cancelar", key=f"cx_{cid}", use_container_width=True):
                        st.session_state.pop("renaming", None)
                        st.rerun()
                continue
            
            # Chat item with action buttons
            cols = st.columns([1, 0.2, 0.2, 0.2] if is_active else [1, 0.2, 0.2])
            
            with cols[0]:
                label = c["title"]
                if c["pinned"]:
                    label = "📌 " + label
                if is_active:
                    label = "▸ " + label
                
                btn_type = "primary" if is_active else "secondary"
                if st.button(label, key=f"open_{cid}", help="Abrir chat", type=btn_type, use_container_width=True):
                    chats.abrir(cid)
                    st.session_state._page = "Chat"
                    st.session_state.pop("renaming", None)
                    st.rerun()
            
            with cols[1]:
                pin_icon = "📍" if c["pinned"] else "📌"
                pin_help = "Soltar" if c["pinned"] else "Anclar"
                if st.button(pin_icon, key=f"pin_{cid}", help=pin_help, use_container_width=True):
                    chats.fijar(cid)
                    st.rerun()
            
            with cols[2]:
                if st.button("🗑️", key=f"del_{cid}", help="Borrar chat", use_container_width=True):
                    chats.borrar(cid)
                    st.session_state.pop("renaming", None)
                    st.rerun()
            
            if is_active:
                with cols[3]:
                    if st.button("✏️", key=f"rnbtn_{cid}", help="Renombrar", use_container_width=True):
                        st.session_state["renaming"] = cid
                        st.rerun()
    
    # Navigation
    st.markdown('<span class="side-label">Navegación</span>', unsafe_allow_html=True)
    pag = st.radio(
        "Página",
        ["💬 Chat", "✍️ Generación", "📊 Evaluación", "🧠 Arquitectura", "🏋️ Training", "📋 Historia Clínica", "🛠️ Proceso"],
        key="_page",
        label_visibility="collapsed"
    )
    # Extract page name without emoji
    page_map = {
        "💬 Chat": "Chat",
        "✍️ Generación": "Generación",
        "📊 Evaluación": "Evaluación",
        "🧠 Arquitectura": "Arquitectura",
        "🏋️ Training": "Training",
        "📋 Historia Clínica": "Historia clínica",
        "🛠️ Proceso": "Proceso"
    }
    current_page = page_map.get(pag, "Chat")
    
    # Technical Settings
    with st.expander("⚙️ Ajustes Técnicos", expanded=False):
        cks = listar_checkpoints()
        default_ckpt = get_default_checkpoint()
        default_idx = cks.index(default_ckpt) if default_ckpt in cks else 0
        if cks:
            ckpt = st.selectbox("Checkpoint", cks, index=default_idx, label_visibility="collapsed")
        else:
            ckpt = None
            st.warning("No hay checkpoints disponibles")
        
        st.markdown("**Parámetros de Generación**")
        st.session_state["temp"] = st.slider("Temperature", 0.0, 1.5, 0.0, 0.05, help="Controla la aleatoriedad (0 = determinístico)")
        st.session_state["top_p"] = st.slider("Top P", 0.1, 1.0, 0.9, 0.05, help="Núcleo de probabilidad acumulada")
        st.session_state["mnt"] = st.slider("Max New Tokens", 20, 500, 120, 10, help="Longitud máxima de respuesta")
        st.session_state["rp"] = st.slider("Repetition Penalty", 1.0, 1.5, 1.1, 0.05, help="Penaliza repeticiones")
        
        st.markdown("---")
        st.markdown("**Información del Sistema**")
        sys_info = info()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CPU", f"{sys_info['cpu_logicos']} núcleos")
            st.metric("RAM", sys_info['ram'].split(" ")[0] + " GB")
        with col2:
            st.metric("Disco libre", f"{sys_info['disco_D_libre_GB']} GB")
            st.metric("PyTorch", sys_info['torch'])
    
    # Profile
    st.markdown('<span class="side-label">Perfil</span>', unsafe_allow_html=True)
    meta_txt = st.session_state.get("_meta_txt", "…M params")
    st.markdown(f"""
    <div class="profile-box">
        <div class="profile-avatar">V</div>
        <div>
            <div style="font-size:0.9rem;font-weight:600;color:var(--text-primary)">Veterinario</div>
            <div style="font-size:0.75rem;color:var(--text-muted)">{meta_txt}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Checkpoint Validation ─────────────────────────────────────────────────
if not cks:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">📦</div>
        <div class="empty-state-title">Sin checkpoints disponibles</div>
        <div class="empty-state-text">Entrena primero un modelo para poder usar la interfaz. Los checkpoints deben estar en la carpeta <code>checkpoints/</code>.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Model Loading ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelo…")
def _modelo(ckpt_rel: str):
    return cargar_modelo(ckpt_rel)

model, meta = _modelo(ckpt)
st.session_state["_meta_txt"] = f"{meta['params_M']}M params"

# Update sidebar caption
st.sidebar.caption(f"{meta['params_M']}M params · {ckpt.split('/')[-1]}")

# ── Main Content ──────────────────────────────────────────────────────────
# Page Header
st.markdown(f"""
<div class="main-header">
    <div>
        <h1>VET-TINY-GPT</h1>
        <p class="page-subtitle">Asistente veterinario experimental · {meta['params_M']}M parámetros</p>
    </div>
    <span class="badge-exp">Experimental</span>
</div>
""", unsafe_allow_html=True)

# Warning Banner
st.warning(ADVERTENCIA, icon="⚠️")

# ── Generation Page ───────────────────────────────────────────────────────
def render_generation_page(model):
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="margin: 0; font-size: 1.5rem; font-weight: 700;">✍️ Generación de Texto</h2>
        <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0;">Genera completaciones usando el modelo entrenado</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Prompt templates
        templates = {
            "Examen físico": "El examen físico de un perro incluye",
            "Historia clínica": "Historia clínica: Perro, 5 años, presenta vómitos y diarrea desde hace 2 días.",
            "Diagnóstico diferencial": "Diagnósticos diferenciales para cojera en perro joven:",
            "Tratamiento": "Protocolo de tratamiento para parvovirus canino:",
            "Personalizado": ""
        }
        
        template_choice = st.selectbox("Plantilla", list(templates.keys()), label_visibility="collapsed")
        default_prompt = templates[template_choice]
        
        prompt = st.text_area(
            "Prompt",
            value=default_prompt,
            height=150,
            placeholder="Escribe tu prompt aquí...",
            label_visibility="collapsed"
        )
        
        col_gen, col_clear = st.columns([1, 1])
        with col_gen:
            generate_btn = st.button("Generar", type="primary", use_container_width=True)
        with col_clear:
            if st.button("Limpiar", use_container_width=True):
                st.rerun()
        
        if generate_btn and prompt.strip():
            with st.spinner("Generando..."):
                result = generar(
                    model, prompt,
                    temperature=st.session_state.get("temp", 0.0),
                    top_p=st.session_state.get("top_p", 0.9),
                    max_new_tokens=st.session_state.get("mnt", 120),
                    repetition_penalty=st.session_state.get("rp", 1.1)
                )
            
            st.markdown("### Resultado")
            st.markdown(f"""
            <div class="card" style="white-space: pre-wrap; font-family: var(--font-sans); line-height: 1.7;">
                {prompt}<span style="color: var(--color-primary); font-weight: 600;">{result}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Copy button
            st.code(prompt + result, language=None)
    
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">💡 Consejos</div>
            <ul style="color: var(--text-secondary); font-size: 0.85rem; line-height: 1.8; margin: 0; padding-left: 1.25rem;">
                <li>Usa <strong>Temperature = 0</strong> para respuestas determinísticas</li>
                <li>Aumenta <strong>Temperature</strong> para más creatividad</li>
                <li><strong>Top P</strong> controla la diversidad del vocabulario</li>
                <li><strong>Repetition Penalty</strong> evita bucles</li>
                <li>El modelo funciona mejor en español veterinario</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card" style="margin-top: 1rem;">
            <div class="card-title">⚡ Parámetros Actuales</div>
        """, unsafe_allow_html=True)
        
        params = [
            ("Temperature", st.session_state.get("temp", 0.0)),
            ("Top P", st.session_state.get("top_p", 0.9)),
            ("Max Tokens", st.session_state.get("mnt", 120)),
            ("Rep. Penalty", st.session_state.get("rp", 1.1)),
        ]
        
        for name, value in params:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid var(--border-color);">
                <span style="color: var(--text-secondary);">{name}</span>
                <span style="font-family: var(--font-mono); font-weight: 600; color: var(--color-primary);">{value}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)


# ── Page Routing ────────────────────────────────────────────────────────────
if current_page == "Chat":
    chat.render(model, meta)
elif current_page == "Generación":
    render_generation_page(model)
elif current_page == "Evaluación":
    metrics.render()
elif current_page == "Arquitectura":
    architecture.render(meta)
elif current_page == "Training":
    training.render()
elif current_page == "Historia clínica":
    clinical.render(model)
elif current_page == "Proceso":
    proceso.render()

# Footer
st.markdown("""
<div class="app-footer">
    VET-TINY-GPT · Modelo experimental para uso educativo · No sustituye criterio veterinario profesional
</div>
""", unsafe_allow_html=True)