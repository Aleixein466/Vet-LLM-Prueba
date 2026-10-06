"""VET-TINY-GPT FastAPI backend for Vercel deployment."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from contextlib import asynccontextmanager

# Import project modules
from app.utils.model_loader import listar_checkpoints, cargar_modelo, get_default_checkpoint
from app.utils.generation import generar, es_urgente, ADVERTENCIA
from app.utils import chats
from src.rag.retriever import TfidfIndex

# ── Global state ────────────────────────────────────────────────────────────
_model = None
_meta = None
_rag_index = None


def load_resources():
    """Carga modelo y RAG index una sola vez."""
    global _model, _meta, _rag_index
    if _model is None:
        ckpt = get_default_checkpoint()
        _model, _meta = cargar_modelo(ckpt)
    if _rag_index is None:
        _rag_index = TfidfIndex.load(ROOT / "data" / "rag" / "index")


# ── Pydantic Models ─────────────────────────────────────────────────────────
class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    temperature: float = Field(0.0, ge=0.0, le=1.5)
    top_p: float = Field(0.9, ge=0.1, le=1.0)
    max_new_tokens: int = Field(120, ge=20, le=500)
    repetition_penalty: float = Field(1.1, ge=1.0, le=1.5)


class GenerationResponse(BaseModel):
    text: str
    urgent: bool
    warning: str


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_length=1)
    species: str = Field("Perro", pattern="^(Perro|Gato|Bovino|Equino|Exóticos|Otra)$")
    use_rag: bool = True
    temperature: float = Field(0.0, ge=0.0, le=1.5)
    top_p: float = Field(0.9, ge=0.1, le=1.0)
    max_new_tokens: int = Field(80, ge=20, le=200)
    repetition_penalty: float = Field(1.1, ge=1.0, le=1.5)


class ChatResponse(BaseModel):
    response: str
    urgent: bool
    sources: Optional[List[str]] = None
    warning: str


class ClinicalRequest(BaseModel):
    especie: str
    raza: Optional[str] = ""
    edad: Optional[str] = ""
    sexo: Optional[str] = ""
    peso: Optional[str] = ""
    motivo: str
    antecedentes: Optional[str] = ""
    signos: Optional[str] = ""
    temperatura: Optional[str] = ""
    fc: Optional[str] = ""
    fr: Optional[str] = ""
    laboratorio: Optional[str] = ""
    tratamientos: Optional[str] = ""


class ClinicalResponse(BaseModel):
    resumen: dict
    alarmas: List[str]
    modelo_texto: Optional[str] = None
    reporte_completo: str
    warning: str


class HealthResponse(BaseModel):
    status: str
    model_params: str
    checkpoint: str
    rag_docs: int


# ── RAG Helpers ─────────────────────────────────────────────────────────────
ESPECIES = ["Perro", "Gato", "Bovino", "Equino", "Exóticos", "Otra"]
K = 2


def _kw(text: str) -> set[str]:
    import re
    TOK = re.compile(r"[a-záéíóúñü]+", re.I)
    STOP = set("""de la el en y a los del se las por un para con no una su al lo como más pero sus
      este esta estos estas ese esa eso ser son fue han hay tiene tienen
      mi tu su sus nos os me te se le les lo la los las un una unos unas al del ante bajo cabe sobre tras""".split())
    def _norm(s: str) -> str:
        import unicodedata
        return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")
    return {t for t in (_norm(w) for w in TOK.findall(text)) if len(t) > 2 and t not in STOP}


def _prompt_con_contexto(passages: list[str], especie: str, q: str, hist) -> str:
    answers = [p.split("Respuesta:", 1)[1].strip() if "Respuesta:" in p else p for p in passages]
    ctx = "\n".join(f"- {a}" for a in answers)
    anterior = ""
    if hist:
        pq, pa = hist[-1]
        anterior = f"Anterior: {pq} / {pa[:120]}\n"
    return f"Contexto:\n{ctx}\n\n{anterior}Pregunta sobre {especie.lower()}: {q}\nRespuesta:"


# ── FastAPI App ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_resources()
    yield
    # Shutdown (cleanup if needed)


app = FastAPI(
    title="VET-TINY-GPT API",
    version="0.1.0",
    description="Veterinary LLM API for clinical assistance (experimental)",
    lifespan=lifespan
)

# CORS for local Streamlit frontend
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En prod: ["http://localhost:8501"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    load_resources()
    return HealthResponse(
        status="ok",
        model_params=f"{_meta['params_M']}M",
        checkpoint=_meta['checkpoint'],
        rag_docs=len(_rag_index.docs)
    )


@app.post("/generate", response_model=GenerationResponse)
async def generate(req: GenerationRequest):
    load_resources()
    try:
        text = generar(
            _model, req.prompt,
            temperature=req.temperature,
            top_p=req.top_p,
            max_new_tokens=req.max_new_tokens,
            repetition_penalty=req.repetition_penalty
        )
        urgent = es_urgente(req.prompt + " " + text)
        return GenerationResponse(
            text=text,
            urgent=urgent,
            warning=ADVERTENCIA
        )
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    load_resources()
    try:
        # Convert messages to history tuples
        hist = []
        user_msg = ""
        for msg in req.messages:
            if msg.role == "user":
                user_msg = msg.content
            elif msg.role == "assistant" and user_msg:
                hist.append((user_msg, msg.content))
                user_msg = ""
        
        # Current question
        q = req.messages[-1].content if req.messages else ""
        if not q:
            raise HTTPException(400, "No user message found")
        
        sources = None
        if req.use_rag:
            info = _rag_index.query(q, k=K)
            if info:
                pasajes = [_rag_index.docs[r[0]] for r in info]
                prompt = _prompt_con_contexto(pasajes, req.species, q, hist)
                sources = [p[:300] for p in pasajes]
            else:
                prompt = f"Pregunta sobre {req.species.lower()}: {q}\nRespuesta:"
        else:
            ctx = ""
            if hist:
                pq, pa = hist[-1]
                ctx = f"Anterior: {pq} / {pa[:120]}\n"
            prompt = f"{ctx}Pregunta sobre {req.species.lower()}: {q}\nRespuesta:"
        
        text = generar(
            _model, prompt,
            temperature=req.temperature,
            top_p=req.top_p,
            max_new_tokens=req.max_new_tokens,
            repetition_penalty=req.repetition_penalty
        )
        
        urgent = es_urgente(q + " " + text)
        if urgent:
            text = "⚠️ Posible URGENCIA: se requiere evaluación profesional cuanto antes.\n\n" + text
        
        return ChatResponse(
            response=text,
            urgent=urgent,
            sources=sources,
            warning=ADVERTENCIA
        )
    except Exception as e:
        raise HTTPException(500, f"Chat failed: {e}")


@app.post("/clinical", response_model=ClinicalResponse)
async def clinical(req: ClinicalRequest):
    load_resources()
    try:
        ALARMAS = ["sangr", "convulsi", "no respira", "inconsciente", "hinchad", "vómito",
                   "diarrea", "cojea", "fiebre", "chocolate", "veneno", "atropell"]
        
        campos = {
            "Especie": req.especie, "Raza": req.raza, "Edad": req.edad, "Sexo": req.sexo,
            "Peso": req.peso, "Motivo": req.motivo, "Antecedentes": req.antecendentes,
            "Signos": req.signos, "Temperatura": req.temperatura, "FC": req.fc,
            "FR": req.fr, "Laboratorio": req.laboratorio, "Tratamientos": req.tratamientos
        }
        
        faltan = [k for k, v in campos.items() if not v.strip()]
        presentes = {k: v for k, v in campos.items() if v.strip()}
        alarmas = sorted({a for a in ALARMAS if a in (req.motivo + " " + req.signos).lower()})
        
        modelo_texto = None
        if req.motivo.strip():
            modelo_texto = generar(
                _model,
                f"Pregunta sobre {req.especie.lower()}: {req.motivo}\nRespuesta:",
                temperature=0.0, max_new_tokens=120
            )
        
        # Build full report
        lineas = ["HISTORIA CLÍNICA VETERINARIA", "=" * 40, ""]
        for k, v in presentes.items():
            lineas.append(f"{k}: {v}")
        lineas += ["", "ALARMAS DETECTADAS:", ", ".join(alarmas) if alarmas else "Ninguna"]
        if modelo_texto:
            lineas += ["", "MODELO (experimental):", modelo_texto]
        lineas += ["", "---", ADVERTENCIA]
        reporte = "\n".join(lineas)
        
        return ClinicalResponse(
            resumen=presentes,
            alarmas=alarmas,
            modelo_texto=modelo_texto,
            reporte_completo=reporte,
            warning=ADVERTENCIA
        )
    except Exception as e:
        raise HTTPException(500, f"Clinical analysis failed: {e}")


@app.get("/models")
async def models():
    load_resources()
    cks = listar_checkpoints()
    return {"checkpoints": cks, "current": _meta['checkpoint'], "params_M": _meta['params_M']}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)