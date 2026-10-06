"""VET-TINY-GPT — Página informativa del proceso de construcción.
Standalone: no carga el modelo ni necesita checkpoints.
Puerto sugerido: 8502 (la app principal usa 8501).
Uso: streamlit run app/pagina_proceso.py --server.port 8502
"""
import streamlit as st

st.set_page_config(
    page_title="VET-TINY-GPT · Proceso de construcción",
    page_icon="🛠️",
    layout="wide",
)

CSS = """
<style>
:root { --pri:#0EA5E9; --acc:#34D399; --bg:#0F172A; --card:#1E293B; --bd:#334155; --mut:#94A3B8; }
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.hero { background: linear-gradient(135deg, rgba(14,165,233,.18), rgba(99,102,241,.18));
  border:1px solid var(--bd); border-radius:20px; padding:2.2rem 2rem; margin-bottom:1.5rem; }
.hero h1 { margin:0; font-size:2rem; font-weight:800;
  background:linear-gradient(135deg,#0EA5E9,#34D399); -webkit-background-clip:text;
  -webkit-text-fill-color:transparent; background-clip:text; }
.hero p { color:var(--mut); margin:.5rem 0 0 0; font-size:1rem; }
.card { background:var(--card); border:1px solid var(--bd); border-radius:14px; padding:1.2rem 1.3rem; margin-bottom:.8rem; }
.card h4 { margin:0 0 .4rem 0; }
.card p, .card li { color:#CBD5E1; font-size:.92rem; line-height:1.65; }
.step { border-left:4px solid var(--pri); }
.fail { border-left:4px solid #EF4444; }
.win { border-left:4px solid var(--acc); }
.kpi { text-align:center; }
.kpi .v { font-size:1.9rem; font-weight:800; color:var(--pri); font-family:monospace; }
.kpi .l { color:var(--mut); font-size:.8rem; text-transform:uppercase; letter-spacing:.05em; }
.badge { display:inline-block; font-size:.72rem; font-weight:700; padding:.2rem .7rem; border-radius:9999px;
  background:rgba(14,165,233,.15); color:#38BDF8; border:1px solid rgba(14,165,233,.4); margin-right:.4rem; }
.footer { text-align:center; color:var(--mut); font-size:.78rem; margin-top:2rem; padding-top:1rem; border-top:1px solid var(--bd); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Hero ──
st.markdown("""
<div class="hero">
  <h1>🛠️ VET-TINY-GPT — Cómo se construyó</h1>
  <p>Asistente veterinario en español entrenado <b>desde cero en CPU</b> · 17.62M parámetros ·
  <span class="badge">Pretraining</span><span class="badge">SFT</span><span class="badge">DPO</span>
  <span class="badge">RAG híbrido</span><span class="badge">Streamlit</span></p>
  <p style="margin-top:.6rem;">📅 Sep 2026 → Oct 2026 · 🖥️ Sin GPU (Intel UHD, 11.8 GB RAM) · 📦 Modelo desplegado:
  <code>checkpoints/pretraining/final/model.pt</code></p>
</div>
""", unsafe_allow_html=True)

# ── KPIs ──
k1, k2, k3, k4, k5 = st.columns(5)
for col, v, l in [
    (k1, "17.6M", "parámetros"),
    (k2, "860K", "tokens mixtos"),
    (k3, "172×", "mejora ppl real"),
    (k4, "7/7", "tests RAG PASS"),
    (k5, "353", "datos PII anonimizados"),
]:
    col.markdown(f'<div class="card kpi"><div class="v">{v}</div><div class="l">{l}</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ── Tabs ──
t1, t2, t3, t4, t5 = st.tabs(["🗺️ Etapas", "❌ Fallas", "🏆 Éxitos", "🧰 Tecnologías", "▶️ Cómo replicarlo"])

with t1:
    st.subheader("Línea de tiempo")
    etapas = [
        ("1️⃣ Corpus sintético — 2026-09-28",
         "Generador propio <code>src/data/corpus_sintetico.py</code>: 5 especies × 15 temas + 12 protocolos de urgencia + QA. "
         "6 518 documentos (1.1 MB) → 408 099 tokens. Produce texto base + SFT train (238) / val (42) + preferencias DPO (120). "
         "Valor: respuestas verificadas (FC, T°, desparasitación). Límite: plantillas → el modelo aprende estilo, no razonamiento."),
        ("2️⃣ Tokenización + dataset",
         "tiktoken GPT-2 (50 257 tokens, EOS <code>&lt;|endoftext|&gt;</code>) + empaquetado binario <code>uint16</code> "
         "(<code>build_dataset.py</code>): se concatena todo en un array plano, 90/10 train/val, carga con <code>np.fromfile</code>. "
         "Reserva uint16: vocab &lt; 65 535 (verificado con assert)."),
        ("3️⃣ Modelo — 17.62M params",
         "<code>src/model/gpt.py</code>: GPT causal escrito a mano — embeddings + 6 bloques (masked self-attention 4 cabezas + MLP 256→1024→256 GELU) "
         "+ LayerNorm pre-norm + posicionales aprendidas 128×256 + <b>tied weights</b> (ahorra 12.87M, el 73% son embeddings). "
         "Lo NO implementado se declara: sin GQA/RoPE/RMSNorm/SwiGLU/FlashAttention."),
        ("4️⃣ Pretraining — 3 000 pasos (~2h19 CPU)",
         "Batch 16×128, AdamW lr 3e-4 + coseno/warmup 200, clip 1.0, ventanas aleatorias. Loss 0.09/0.10, ppl 1.11, QA 0.762. "
         "Checkpoints cada 500 + <code>last.pt</code> reanudable (modelo+optimizador)."),
        ("5️⃣ SFT + DPO + Evaluación",
         "SFT: misma cross-entropy pero con <b>máscara -100 en el prompt</b> (solo la respuesta cuenta) + lote mixto con replay. "
         "DPO: política vs referencia congelada, β=0.1, 30 pasos, loss→0.000 (demostrativo). "
         "Triple métrica: perplejidad + QA retenido (42, greedy, recall≥0.4) + consistencia Jaccard por reformulación."),
        ("6️⃣ Corpus real + RAG híbrido — 2026-10-05 ⭐ punto de inflexión",
         "<code>ingest_pdfs.py</code>: 70 PDFs (15 técnicos + 55 historias, 669 págs) → limpieza (NFC, ToUnicode roto, desguionado) + "
         "anonimización regex (353 reemplazos) → 1 639 chunks → 453k tokens. Dataset mixto: <b>860 629 tokens</b>. "
         "Reentreno 3000→6000 (~3.3h, resume): ppl real <b>29 540 → 172</b>, QA intacto 0.762. "
         "RAG: TF-IDF propio (17 278 términos, 2 061 pasajes) + rescate MiniLM multilingüe 384d con doble puerta."),
        ("7️⃣ Variante BPE 16k + Web",
         "BPE propio (<code>tokenizers</code>, ByteLevel, 16k+3 especiales): 8.85M params, 40% menos tokens, QA 0.952 — "
         "pero val 6.97→7.43 con sobreajuste → <b>experimental, no desplegado</b> (<code>--cfg tiny-18m-bpe.yaml</code>). "
         "Web Streamlit 6+1 páginas en puerto 8501."),
    ]
    for titulo, desc in etapas:
        st.markdown(f'<div class="card step"><h4>{titulo}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

    st.subheader("Flujo del sistema final")
    st.code("PDFs / plantillas → limpieza + anonimización → chunks (900/150) → tokens → GPT causal 17.6M → RAG (TF-IDF + MiniLM) → respuesta con fuente… o RECHAZO", language=None)
    st.markdown('<div class="card"><h4>💡 Idea clave</h4><p>Con 17M el modelo <b>memoriza, no razona</b> (estudiante de primer año con los apuntes de memoria). '
                'El RAG terminó siendo más importante que el entrenamiento: en vez de memorizar 1 639 fragmentos, se entrega el relevante como contexto — '
                'y <b>si no hay evidencia, se rechaza antes que inventar</b>.</p></div>', unsafe_allow_html=True)

with t2:
    st.subheader("Fallas encontradas → soluciones aplicadas")
    fallas = [
        ("🐌 venv en USB a 1.6 MB/s, inutilizable", "Mover el venv a <b>C:</b> (NVMe). Script <code>env-fast.ps1</code>."),
        ("🧠 ppl 29 540 sobre texto real", "El modelo solo-sintético veía PDFs reales como azar. <b>Reentreno mixto 3000→6000</b>: 29 540 → 172 (172×), QA intacto."),
        ("💀 SFT sin replay: ppl 1.1 → 252", "Olvido catastrófico. Lote <b>4 sint + 2 replay + 2 real</b>, pérdida promediada. Aun así el SFT extractivo se descartó (recall 0.018)."),
        ("🧵 Costura de tokenización en SFT", "<code>encode(p)+encode(r)</code> parte distinto que el pretraining. Fix: <b>codificación conjunta + verificación del borde</b> (<code>tok.py:25-43</code>); no-alineados se descartan y cuentan."),
        ("☠️ RAG envenenado: dinosaurios / Colón con contenido clínico", "4 defectos: umbral tras bonus, sin regla de relevancia, historial en la búsqueda, fallback incondicional. Fix: <b>score≥0.18 crudo + ≥2 términos, umbral antes del bonus, historial fuera de búsqueda</b>."),
        ("🔢 Ningún umbral separa relevante/irrelevante", "Medido: dino OOD 0.2859 &gt; vet legítima 0.2849. Solución <b>estructural (conteo de términos)</b>, no numérica."),
        ("🌀 MiniLM colaba OOD (Colón 0.68, dinos 0.55)", "<b>Doble puerta</b>: ≥0.55+solape≥4, o ≥0.72+solape≥3. Solo rescata si TF-IDF rechazó."),
        ("🎲 Muestreo = ensalada de tokens", "top-k/top-p rompe composición en 17M. Decisión: <b>voraz temp 0 + repetition penalty 1.1</b>."),
        ("📄 Tildes corruptas � en PDFs", "ToUnicode roto. Limpieza NFC + borrar U+FFFD + desguionado + <b>índice sin acentos</b> y términos ASCII."),
        ("📉 DPO satura (loss 0.000, ventaja +155/+182)", "Prefs triviales (correcta vs 'dale chocolate al perro'). Queda <b>demostrativo</b>: faltan prefs cercanas de matiz."),
        ("📦 BPE-16k sobreajusta (val 6.97→7.43)", "Mejor QA sintético 0.952 pero sin eval real → <b>experimental, no desplegado</b>, aislado en sus directorios."),
    ]
    for titulo, desc in fallas:
        st.markdown(f'<div class="card fail"><h4>❌ {titulo}</h4><p>✅ {desc}</p></div>', unsafe_allow_html=True)

with t3:
    st.subheader("Éxitos medidos (con evidencia)")
    st.table({
        "Modelo": ["Base vieja (sintética)", "Base reentrenada ✅", "SFT real ❌", "Base BPE 16k 🧪", "SFT BPE ❌", "DPO"],
        "Perplejidad": ["29 540 (en real)", "172", "300", "1 773*", "2 418*", "—"],
        "QA precisión": ["0.762", "0.762", "0.0", "0.952", "0.143", "demostrativo"],
        "Decisión": ["superada", "desplegada", "descartado", "experimental", "descartado", "sin efecto"],
    })
    st.caption("* ppl entre tokenizadores distintos no son comparables (vocabulario diferente).")
    exitos = [
        "⚡ <b>6 000 pasos estables en CPU</b> sin divergir (AdamW + coseno/warmup + clip; resume sin repetir 3h).",
        "🎯 <b>Despliegue correcto gracias a triple métrica</b>: vieja y nueva empatan en QA (0.762); la ppl real reveló la diferencia.",
        "🛡️ <b>Sistema que prefiere callar antes que alucinar</b>: fuera-de-ámbito, sin-evidencia, vaga, banner de urgencia, especie preferida.",
        "🔎 <b>Rescate semántico real</b>: 'mi can vomita sangre' → parvovirosis 0.71 (TEST 7, lo que TF-IDF solo rechazaba).",
        "🔎 <b>Batería RAG 7/7 PASS</b> incluyendo rechazos ('capital de Francia', 'Colón', 'dinosaurios').",
        "🔒 <b>70 PDFs → 2 061 pasajes</b> con privacidad (353 reemplazos PII) y diagnóstico honesto de límites (sin OCR, sin stemming).",
    ]
    for e in exitos:
        st.markdown(f'<div class="card win"><p>🏆 {e}</p></div>', unsafe_allow_html=True)

with t4:
    st.subheader("Stack por peso real en el resultado")
    st.markdown("**Nivel 1 — explican por qué funciona**")
    st.table({
        "Tecnología": ["Transformer decoder-only + atención causal", "BPE (tiktoken / propio 16k)", "Next-token + cross-entropy",
                        "AdamW + coseno/warmup + clip", "SFT con máscara + replay", "Triple evaluación", "RAG TF-IDF + MiniLM"],
        "Rol": ["Sin máscara causal no hay LM", "Longitud, costo, compatibilidad ckpts", "Objetivo único (SFT/DPO = reformulaciones)",
                "6 000 pasos estables en CPU", "Enseña a responder; replay evita olvido", "ppl+QA+Jaccard → despliegue correcto",
                "Evidencia que 17M no memorizan"],
    })
    st.markdown("**Nivel 2 — reproducibilidad**")
    st.table({
        "Tecnología": ["Checkpoints reanudables", "Binario uint16 + np.fromfile", "YAML declarativo", "Codificación conjunta", "DPO con referencia congelada"],
        "Rol": ["Reentreno sin repetir", "1.5 MB en memoria", "Variante BPE en pocas líneas", "Corrige bug de segmentación", "Alineación sin reward model"],
    })
    st.markdown("**Nivel 3 — apoyo**")
    st.code("Python 3.11 · torch 2.14+cpu · tiktoken 0.14 · tokenizers 0.23 · sentence-transformers 6.1 (MiniLM 384d) · pypdf 6.19 · numpy 2.4 · Streamlit 1.30+ · safetensors · TF-IDF propio en numpy (sin sklearn)", language=None)
    st.markdown("**❎ No implementado**")
    st.markdown('<div class="card"><p>RoPE / GQA / RMSNorm / SwiGLU (GPT-2 clásico deliberado) · FlashAttention (innecesario a 128 en CPU) · '
                'PPO/GRPO/reward model (DPO los evita) · OCR (escaneos se omiten) · Stemming ("golpeo"≠"golpe") · CUDA/AMP/LoRA/distribuido (sin GPU).</p></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown('<div class="card"><h4>⚠️ Limitaciones</h4><p>17.6M + 860k tokens: memoriza &gt; generaliza. QA de 42 sintéticas: fidelidad al estilo, no criterio clínico. Sin GPU para escalar.</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="card"><h4>🚀 Siguientes pasos</h4><p>Corpus con licencia · GQA/RoPE + reentrenar · PPO/GRPO · BPE por defecto · eval real · OCR + stemming.</p></div>', unsafe_allow_html=True)

with t5:
    st.subheader("Replicar el pipeline")
    st.code(
        "pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "pip install -r requirements.txt\n\n"
        "python src/data/corpus_sintetico.py --n-docs 4000\n"
        "python src/data/ingest_pdfs.py\n"
        "python src/data/build_dataset.py\n"
        "python src/training/train.py\n"
        "python src/training/sft.py\n"
        "python src/training/dpo.py\n"
        "python src/evaluation/eval.py\n"
        "python src/rag/retriever.py\n"
        "python src/rag/embeddings.py --build\n"
        "python app/vet_chat.py --ask \"¿Cada cuánto desparasito a mi gato?\"",
        language="powershell",
    )
    st.markdown('<div class="card"><h4>📚 Fuentes</h4><p><code>DOCUMENTACION.md</code> · <code>PROJECT_REPORT.md</code> · '
                '<code>docs/PROCESO_ENTRENAMIENTO.md</code> · <code>outputs/evaluation/</code> · <code>logs/</code></p></div>', unsafe_allow_html=True)

st.markdown('<div class="footer">VET-TINY-GPT · Página informativa standalone (puerto 8502) · No sustituye criterio veterinario profesional</div>', unsafe_allow_html=True)
