"""Ingesta de PDFs reales -> corpus limpio + JSON para RAG/SFT (CPU, anonimiza historias).

Lee:
  Datos/Documentos/*.pdf
  Datos/HISTORIAS CLINICAS/**/*.pdf
Produce:
  data/cleaned/chunks_real.jsonl   (chunk + metadata, para RAG)
  data/cleaned/docs_real.jsonl     (documento completo limpio + metadata)
  data/cleaned/ingest_report.json  (estadisticas)
  data/raw/vet-real-limpio.txt     (texto plano para build_dataset.py / retriever.py)
  data/sft/sft_real.jsonl          (pares prompt/response extractivos anonimizados)

Uso:
  C:\\VET-TINY-LLM-venv\\Scripts\\python.exe src/data/ingest_pdfs.py
  C:\\VET-TINY-LLM-venv\\Scripts\\python.exe src/data/ingest_pdfs.py --no-sft
"""
from __future__ import annotations
import argparse
import json
import re
import unicodedata
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

# ---------------- limpieza ----------------

def norm_unicode(s: str) -> str:
    s = s.replace("\x00", " ")
    s = unicodedata.normalize("NFC", s)
    # ligaduras comunes
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff")
    return s

def clean_page(text: str) -> str:
    t = norm_unicode(text or "")
    # PDFs con ToUnicode roto dejan U+FFFD: borrar para no envenenar TF-IDF/modelo
    t = t.replace("�", "").replace("\u00ad", "")
    # desguionar fin de linea: "veteri-\nnaria" -> "veterinaria"
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    t = t.replace("\r", "\n")
    # puntos de indice "Resumen....5" -> "Resumen 5"
    t = re.sub(r"\.{3,}", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    # lineas basura tipicas (solo numero de pagina)
    lines = []
    for ln in t.split("\n"):
        s = ln.strip()
        if not s:
            continue
        if re.fullmatch(r"\d{1,4}", s):
            continue
        if len(s) < 3:
            continue
        lines.append(s)
    return "\n".join(lines).strip()

def is_poor_page(text: str, min_chars: int = 200) -> bool:
    return len(text.strip()) < min_chars

# ---------------- anonimizado historias ----------------
# Solo para HISTORIAS CLINICAS. Documentos tecnicos NO se anonimizan.

EMAIL = re.compile(r"[\w.\-+]+@[\w.\-]+\.\w+")
CEL = re.compile(r"(\+?57[\s\-.]*)?3\d{2}[\s\-.]*\d{3}[\s\-.]*\d{4}")
CC_CTX = re.compile(r"(?i)(c[eé]dula|CC|C\.C\.?|ID|identificaci[oó]n|chip)[^\d]{0,20}(\d[\d.\s\-]{6,14}\d)")
FECHA_HORA = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
DIRECCION = re.compile(r"(?i)\b(cra|carrera|calle|cll|avenida|av\.?|transversal|diagonal|barrio|vereda)\b[^\n]{0,60}")

def anonymize_historia(text: str) -> tuple[str, int]:
    n = 0
    def sub(rx, repl, t):
        nonlocal_n = [0]
        def f(m):
            nonlocal_n[0] += 1
            return repl
        t2 = rx.sub(f, t)
        return t2, nonlocal_n[0]
    t = text
    t, c = sub(EMAIL, "[EMAIL]", t); n += c
    t, c = sub(CEL, "[TEL]", t); n += c
    t, c = sub(CC_CTX, r"\1 [ID]", t); n += c
    # "Yo NOMBRE identificado" -> "Yo [PROPIETARIO] identificado"
    rx_yo = re.compile(r"(?i)\bYo\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,3})\s+(identificado)")
    t, c = sub(rx_yo, r"Yo [PROPIETARIO] \2", t); n += c
    # "Nombre | Name XXX" / "Propietario ...: XXX"
    rx_owner = re.compile(r"(?i)(Nombre\s*\|\s*Name|Propietario[^\n:]{0,20}:)\s*([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑa-záéíóúñ\s]{2,40}?)(?=\n|C[eé]dula|Celular|Correo|Direcci|$)")
    t, c = sub(rx_owner, r"\1 [PROPIETARIO]", t); n += c
    t, c = sub(DIRECCION, "[DIRECCIÓN]", t); n += c
    # correos "No Registra" se deja (no es PII)
    return t, n

ESPECIE_RX = re.compile(r"(?i)\bEspecie\s*\|\s*Species\s*([A-Za-záéíóúñ]+)")
RAZA_RX = re.compile(r"(?i)\bRaza\s*\|\s*Breed\s*([A-Za-záéíóúñ\s/]+?)(?:\s*Edad|\n|$)")
MOTIVO_RX = re.compile(r"(?i)(Detalle\s*Consul[^\n]*\n)(.{0,300})", re.S)

# ---------------- chunking ----------------

def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    # une parrafos cortos hasta tamano
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= size:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
                # overlap: cola del anterior
                cur = cur[-overlap:] + "\n\n" + p if overlap else p
                if len(cur) > size * 1.5:
                    # corta por frases
                    parts = re.split(r"(?<=[.!?])\s+", cur)
                    buf = ""
                    for s in parts:
                        if len(buf) + len(s) + 1 <= size:
                            buf = (buf + " " + s).strip()
                        else:
                            if buf:
                                chunks.append(buf)
                            buf = s
                    cur = buf
            else:
                # parrafo gigante: corta por frases
                parts = re.split(r"(?<=[.!?])\s+", p)
                buf = ""
                for s in parts:
                    if len(buf) + len(s) + 1 <= size:
                        buf = (buf + " " + s).strip()
                    else:
                        if buf:
                            chunks.append(buf)
                        buf = s
                cur = buf
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if len(c) >= 120]

# ---------------- extraccion ----------------

def extract_pdf(path: Path, max_pages: int = 0) -> tuple[list[str], int]:
    r = PdfReader(str(path))
    pages = r.pages[:max_pages] if max_pages else r.pages
    out = []
    for pg in pages:
        try:
            t = pg.extract_text() or ""
        except Exception:
            t = ""
        out.append(clean_page(t))
    return out, len(r.pages)

def doc_kind(path: Path) -> str:
    s = str(path).lower()
    if "historia" in s or "animal happy" in s or "kyron" in s:
        return "historia_clinica"
    return "documento_tecnico"

def make_sft_pairs(chunks: list[dict]) -> list[dict]:
    """Pares extractivos: prompt pregunta por el tema, response = chunk.
    Formato compatible con data/sft/*.jsonl (prompt/response).
    Filtra chunks con mojibake residual (>2% de chars raros) para no ensenar ruido."""
    pairs = []
    for c in chunks:
        txt = c["text"]
        if len(txt) < 200:
            continue
        raros = sum(1 for ch in txt if ch in "�\uFFFD\x00" or ord(ch) > 0x2500)
        if raros / max(1, len(txt)) > 0.02:
            continue
        first = txt.split(".")[0].strip()[:160]
        if c["type"] == "historia_clinica":
            esp = "paciente"
            m = ESPECIE_RX.search(txt)
            if m:
                esp = m.group(1).strip().lower()
            pairs.append({
                "prompt": f"Pregunta: Describe un caso clínico veterinario de {esp} según la historia registrada.\nRespuesta:",
                "response": " " + txt[:1200],
                "source": c["source"],
            })
        else:
            pairs.append({
                "prompt": f"Pregunta: Explica el siguiente tema veterinario: {first}.\nRespuesta:",
                "response": " " + txt[:1200],
                "source": c["source"],
            })
    return pairs

def main(datos="Datos", out_cleaned="data/cleaned", out_raw="data/raw/vet-real-limpio.txt",
         out_sft="data/sft/sft_real.jsonl", chunk=900, overlap=150, no_sft=False):
    if PdfReader is None:
        raise SystemExit("Falta pypdf: instala con  pip install pypdf")
    root = Path(datos)
    if not root.exists():
        raise SystemExit(f"No existe {datos}")
    pdfs = sorted(root.rglob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"Sin PDFs en {datos}")
    out_c = Path(out_cleaned); out_c.mkdir(parents=True, exist_ok=True)

    docs, chunks = [], []
    rep = {"pdfs": 0, "paginas": 0, "paginas_pobres": 0, "chars_limpios": 0,
           "pii_reemplazos": 0, "por_tipo": {}, "omitidos_escaneado": []}
    for pdf in pdfs:
        kind = doc_kind(pdf)
        try:
            pages, npages = extract_pdf(pdf)
        except Exception as e:
            rep["omitidos_escaneado"].append({"pdf": pdf.name, "error": str(e)[:120]})
            continue
        rep["pdfs"] += 1
        rep["paginas"] += npages
        rep["por_tipo"][kind] = rep["por_tipo"].get(kind, 0) + 1
        limpias = []
        for pg in pages:
            if is_poor_page(pg):
                rep["paginas_pobres"] += 1
                continue
            limpias.append(pg)
        full = "\n\n".join(limpias).strip()
        if kind == "historia_clinica":
            full, nrep = anonymize_historia(full)
            rep["pii_reemplazos"] += nrep
        if len(full) < 500:
            rep["omitidos_escaneado"].append({"pdf": pdf.name, "motivo": f"solo {len(full)} chars (posible escaneado, necesita OCR)"})
            continue
        rep["chars_limpios"] += len(full)
        did = f"{kind}::{pdf.parent.name}/{pdf.name}" if pdf.parent.name not in ("Documentos", "Datos") else f"{kind}::{pdf.name}"
        docs.append({"id": did, "source": pdf.name, "type": kind, "pages": npages, "chars": len(full), "text": full})
        for i, ch in enumerate(chunk_text(full, chunk, overlap)):
            chunks.append({"id": f"{did}#c{i}", "source": pdf.name, "type": kind, "chunk": i, "text": ch})

    (out_c / "docs_real.jsonl").write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in docs), encoding="utf-8")
    (out_c / "chunks_real.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks), encoding="utf-8")
    Path(out_raw).parent.mkdir(parents=True, exist_ok=True)
    Path(out_raw).write_text("\n\n".join(c["text"] for c in chunks), encoding="utf-8")

    n_sft = 0
    if not no_sft:
        pairs = make_sft_pairs(chunks)
        Path(out_sft).parent.mkdir(parents=True, exist_ok=True)
        Path(out_sft).write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs), encoding="utf-8")
        n_sft = len(pairs)

    rep.update({"docs": len(docs), "chunks": len(chunks), "sft_real": n_sft,
                "out_raw": out_raw, "avg_chars_chunk": round(sum(len(c['text']) for c in chunks) / max(1, len(chunks)))})
    (out_c / "ingest_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "omitidos_escaneado"}, ensure_ascii=False, indent=2))
    print(f"omitidos={len(rep['omitidos_escaneado'])}")
    for o in rep["omitidos_escaneado"][:10]:
        print("  -", o)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default="Datos")
    ap.add_argument("--out-cleaned", default="data/cleaned")
    ap.add_argument("--out-raw", default="data/raw/vet-real-limpio.txt")
    ap.add_argument("--out-sft", default="data/sft/sft_real.jsonl")
    ap.add_argument("--chunk", type=int, default=900)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--no-sft", action="store_true")
    a = ap.parse_args()
    main(a.datos, a.out_cleaned, a.out_raw, a.out_sft, a.chunk, a.overlap, a.no_sft)
