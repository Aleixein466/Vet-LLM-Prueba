"""Estado multi-chat estilo ChatGPT (sidebar con historial, anclar, renombrar, borrar)."""
from __future__ import annotations
import time
import streamlit as st


def store() -> dict:
    if "chats" not in st.session_state:
        st.session_state.chats = {}
        st.session_state.active = None
    if "hist" in st.session_state and st.session_state.hist and not st.session_state.chats:
        cid = _new("Consulta recuperada")
        st.session_state.chats[cid]["messages"] = list(st.session_state.hist)
        st.session_state.active = cid
        del st.session_state["hist"]
    if not st.session_state.chats:
        st.session_state.active = _new("Nueva consulta")
    if st.session_state.active not in st.session_state.chats:
        st.session_state.active = next(iter(st.session_state.chats))
    return st.session_state.chats


def _new(title: str) -> str:
    cid = f"c{int(time.time() * 1000)}"
    st.session_state.chats[cid] = {"title": title, "messages": [],
                                   "pinned": False, "species": "Perro"}
    return cid


def nuevo() -> str:
    store()
    cid = _new("Nueva consulta")
    st.session_state.active = cid
    return cid


def abrir(cid: str):
    if cid in st.session_state.chats:
        st.session_state.active = cid


def borrar(cid: str):
    store()
    st.session_state.chats.pop(cid, None)
    if not st.session_state.chats:
        nuevo()
    elif st.session_state.active == cid:
        st.session_state.active = next(iter(st.session_state.chats))


def fijar(cid: str):
    if cid in st.session_state.chats:
        st.session_state.chats[cid]["pinned"] = not st.session_state.chats[cid]["pinned"]


def renombrar(cid: str, title: str):
    if cid in st.session_state.chats and title.strip():
        st.session_state.chats[cid]["title"] = title.strip()[:60]


def activo() -> dict:
    store()
    return st.session_state.chats[st.session_state.active]


def titular(q: str):
    c = activo()
    if c["title"] in ("Nueva consulta", "Consulta recuperada") and q.strip():
        c["title"] = q.strip()[:42]
