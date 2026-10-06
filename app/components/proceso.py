"""Componente: Proceso de construcción — fallas, éxitos y tecnologías."""
import streamlit as st


def render():
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="margin: 0; font-size: 1.5rem; font-weight: 700;">🛠️ Proceso de Construcción</h2>
        <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0;">Cómo se construyó VET-TINY-GPT: etapas, fallas, éxitos y tecnologías</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Etapas", "❌ Fallas → ✅ Soluciones", "🏆 Éxitos", "🧰 Tecnologías"])

    # ── TAB 1: Etapas ──
    with tab1:
        st.markdown("### Línea de tiempo del proyecto")
        etapas = [
            ("1️⃣ Corpus sintético", "2026-09-28", "Generador propio <code>corpus_sintetico.py</code>: 5 especies × 15 temas + 12 protocolos de urgencia + QA. 6 518 docs → 408 099 tokens. Salidas: texto base + SFT (238/42) + prefs DPO (120)."),
            ("2️⃣ Tokenización + dataset", "Sep 2026", "tiktoken GPT-2 (50 257 tokens) + empaquetado binario <code>uint16</code> con <code>build_dataset.py</code>. 90/10 train/val. Carga con <code>np.fromfile</code>, sin parsear JSON."),
            ("3️⃣ Modelo", "Sep 2026", "GPT decoder-only propio en <code>src/model/gpt.py</code>: 6 capas, 4 cabezas, embd 256, ventana 128, tied weights. <b>17.62M params</b> (73% son embeddings)."),
            ("4️⃣ Pretraining", "Sep 2026", "3 000 pasos, batch 16×128, AdamW lr 3e-4 + coseno/warmup 200, clip 1.0. ~2h19 en CPU. Loss 0.09/0.10, ppl 1.11, QA 0.762."),
            ("5️⃣ SFT + DPO + Eval", "Sep–Oct 2026", "SFT con máscara en respuesta + replay. DPO 30 pasos vs referencia congelada. Triple métrica: perplejidad + QA retenido (42) + consistencia Jaccard."),
            ("6️⃣ Corpus real + RAG", "2026-10-05", "<code>ingest_pdfs.py</code>: 70 PDFs (15 técnicos + 55 historias, 353 reemplazos PII) → 1 639 chunks, 453k tokens. Dataset mixto 860 629 tokens. Reentreno 3000→6000 (~3.3h). RAG TF-IDF + rescate semántico MiniLM."),
            ("7️⃣ BPE propio + Web", "Oct 2026", "BPE 16k con <code>tokenizers</code> (8.85M params, QA 0.952, experimental). Web Streamlit de 6 páginas + esta página de proceso."),
        ]
        for titulo, fecha, desc in etapas:
            st.markdown(f"""
            <div class="card" style="margin-bottom: 0.75rem;">
                <div class="card-title">{titulo} <span style="font-size:0.75rem;color:var(--text-muted);font-weight:400;">{fecha}</span></div>
                <p class="card-text">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Flujo del sistema final")
        st.code("PDFs / plantillas → limpieza + anonimización → chunks → tokens → GPT causal → RAG (TF-IDF + MiniLM) → respuesta con fuente o rechazo", language=None)
        st.markdown("""
        <div class="card">
            <div class="card-title">💡 Idea clave del proyecto</div>
            <p class="card-text">Con 17M de parámetros el modelo <b>memoriza, no razona</b> (analogía: estudiante de primer año que se sabe los apuntes de memoria).
            Por eso el RAG terminó siendo más importante que el propio entrenamiento: en vez de pedirle que memorice 1 639 fragmentos clínicos,
            se le entrega el fragmento relevante como contexto en cada pregunta — y si no hay evidencia, <b>se rechaza antes que inventar</b>.</p>
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 2: Fallas ──
    with tab2:
        st.markdown("### Fallas encontradas y cómo se resolvieron")
        fallas = [
            ("🐌 venv en USB a 1.6 MB/s, inutilizable", "Mover el entorno virtual a <code>C:</code> (NVMe). Script <code>env-fast.ps1</code>. Decisión de entorno que condicionó todo."),
            ("🧠 ppl 29 540 sobre texto real", "El modelo solo sintético veía los PDFs reales como texto aleatorio. <b>Reentreno mixto 3000→6000</b>: ppl real 29 540 → 172 (mejora 172×), QA intacto 0.762."),
            ("💀 SFT sin replay: ppl 1.1 → 252", "Olvido catastrófico. Lote mixto <b>4 sint + 2 replay + 2 real</b> con pérdida promediada. Aun así el SFT extractivo se descartó (recall 0.018)."),
            ("🧵 Costura de tokenización en SFT", "<code>encode(prompt)+encode(resp)</code> parte los tokens distinto que el pretraining. Solución: <b>codificación conjunta + verificación del borde</b> en <code>tok.py:25-43</code>; pares no-alineados se descartan y se cuentan."),
            ("☠️ RAG envenenado: dinosaurios / Colón", "4 defectos: umbral tras bonus de especie, sin regla de relevancia, historial contaminando búsqueda, fallback extractivo incondicional. Fix: <b>score ≥ 0.18 crudo + ≥2 términos compartidos, umbral antes del bonus, historial fuera de la búsqueda</b>."),
            ("🔢 Ningún umbral separa relevante/irrelevante", "Medido: dino OOD 0.2859 &gt; vet legítima 0.2849. Por eso la solución fue <b>estructural (conteo de términos)</b>, no subir el número a ciegas."),
            ("🌀 MiniLM colaba OOD (Colón 0.68, dinos 0.55)", "<b>Doble puerta semántica</b>: score ≥ 0.55 y solape ≥ 4, o score ≥ 0.72 y solape ≥ 3. Solo rescata si TF-IDF rechazó."),
            ("🎲 Muestreo = ensalada de tokens", "Con 17M, top-k/top-p introduce errores que el modelo no corrige. Decisión: <b>decodificación voraz (temp 0) + penalización 1.1</b>."),
            ("📄 Tildes corruptas � en PDFs", "ToUnicode roto en algunas fuentes. Limpieza: NFC, borrar U+FFFD, desguionado, y <b>normalización de acentos en el índice</b> + apoyo en términos ASCII."),
            ("📉 DPO satura en loss 0.000", "Prefs triviales (correcta vs 'dale chocolate al perro'). El modelo las separa con margen +155/+182. Queda como etapa <b>demostrativa</b>: faltan prefs cercanas de matiz."),
            ("📦 BPE 16k con sobreajuste (val 6.97→7.43)", "Mejor QA sintético (0.952) pero val subiendo + nunca evaluado en preguntas reales. Veredicto: <b>experimental, no desplegado</b>. Vive en <code>--cfg tiny-18m-bpe.yaml</code>."),
        ]
        for titulo, desc in fallas:
            st.markdown(f"""
            <div class="card" style="margin-bottom: 0.75rem; border-left: 3px solid var(--color-danger);">
                <div class="card-title">❌ {titulo}</div>
                <p class="card-text">✅ {desc}</p>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 3: Éxitos ──
    with tab3:
        st.markdown("### Éxitos medidos (con evidencia)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ppl texto real", "29 540 → 172", "172× mejor")
        c2.metric("QA precisión", "0.762", "base reentrenada")
        c3.metric("RAG tests", "7 / 7 PASS", "con rechazos OOD")
        c4.metric("PII anonimizados", "353", "55 historias")

        st.markdown("---")
        exitos = [
            ("⚡ 6 000 pasos en CPU sin divergir", "AdamW 3e-4 + coseno/warmup 200 + clip 1.0. Checkpoints reanudables cada 500 pasos: el reentreno continuó sin repetir 3h."),
            ("🎯 Decisión de despliegue correcta", "Triple métrica evitó el error: base vieja y nueva empatan en QA (0.762), pero la ppl real (29 540 vs 172) reveló la diferencia. Desplegada la <b>base reentrenada</b>, SFT descartado dos veces con datos."),
            ("🛡️ Sistema que prefiere callar antes que alucinar", "Guardrails: fuera-de-ámbito (&lt;2 términos), sin-evidencia, consulta vaga, banner de urgencia, especie preferida, sin contaminación de historial."),
            ("🔎 Rescate semántico real", "Paráfrasis 'mi can vomita sangre' → parvovirosis 0.71. Caso que TF-IDF solo rechazaba, documentado en TEST 7."),
            ("🗜️ BPE propio funcional", "40% menos tokens, 8.85M params, QA 0.952. No desplegado por honestidad evaluativa, pero usable y separado sin tocar producción."),
            ("🔒 Ingesta real con privacidad", "70 PDFs, 669 páginas, 1 639 chunks, 2 061 pasajes indexados (vocab 17 278). Historias anonimizadas con regex ([EMAIL], [TEL], [ID]…). Sin OCR: limitación declarada."),
        ]
        for titulo, desc in exitos:
            st.markdown(f"""
            <div class="card" style="margin-bottom: 0.75rem; border-left: 3px solid var(--color-accent);">
                <div class="card-title">🏆 {titulo}</div>
                <p class="card-text">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Tabla comparativa de candidatos")
        st.table({
            "Modelo": ["Base vieja (solo sintético)", "Base reentrenada ✅", "SFT real ❌", "Base BPE 16k 🧪", "SFT BPE ❌", "DPO"],
            "Perplejidad": ["29 540 (en real)", "172", "300", "1 773*", "2 418*", "—"],
            "QA precisión": ["0.762", "0.762", "0.0", "0.952", "0.143", "demostrativo"],
            "Decisión": ["superada", "desplegada", "descartado", "experimental", "descartado", "sin efecto"],
        })
        st.caption("* ppl entre tokenizadores distintos no son comparables (vocabulario diferente).")

    # ── TAB 4: Tecnologías ──
    with tab4:
        st.markdown("### Stack ordenado por peso real en el resultado")
        st.markdown("#### Nivel 1 — explican por qué el sistema funciona")
        st.table({
            "Tecnología": ["Transformer decoder-only + atención causal", "BPE (tiktoken / propio 16k)", "Next-token + entropía cruzada", "AdamW + coseno/warmup + clip", "SFT con máscara + replay", "Triple evaluación", "RAG TF-IDF + MiniLM"],
            "Qué resuelve": [
                "Sin máscara causal no hay modelo de lenguaje",
                "Define longitud, costo y compatibilidad de checkpoints",
                "Objetivo único del que SFT/DPO son reformulaciones",
                "6 000 pasos estables en CPU",
                "Enseña el formato respuesta; replay evita olvido",
                "ppl + QA + consistencia → despliegue correcto",
                "La evidencia que 17M no pueden memorizar",
            ],
        })
        st.markdown("#### Nivel 2 — reproducibilidad")
        st.table({
            "Tecnología": ["Checkpoints reanudables", "Binario uint16 + np.fromfile", "YAML declarativo", "Codificación conjunta con máscara", "DPO con referencia congelada"],
            "Para qué": [
                "Reentreno 3000→6000 sin repetir",
                "1.5 MB en memoria sin parsear",
                "Variante BPE cambiando pocas líneas",
                "Corrige bug real de segmentación",
                "Alineación sin reward model (viable en CPU)",
            ],
        })
        st.markdown("#### Nivel 3 — apoyo")
        st.code("Python 3.11 · PyTorch 2.14+cpu · tiktoken 0.14 · tokenizers 0.23 · sentence-transformers 6.1 (MiniLM 384d) · pypdf 6.19 · numpy 2.4 · Streamlit ≥1.30 · safetensors · scikit no usado (TF-IDF propio en numpy)", language=None)

        st.markdown("---")
        st.markdown("### ❎ No implementado (y por qué importa)")
        st.markdown("""
        <div class="card">
        <ul style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.9; margin: 0; padding-left: 1.25rem;">
            <li><b>RoPE / GQA / RMSNorm / SwiGLU</b> — arquitectura deliberadamente GPT-2 clásica; añadirlos exige reentrenar desde cero.</li>
            <li><b>FlashAttention / kernels</b> — innecesario a ventana 128 en CPU.</li>
            <li><b>PPO / GRPO / reward model</b> — documentados como extensión; DPO los evita por diseño.</li>
            <li><b>OCR</b> — PDFs escaneados sin capa de texto se omiten.</li>
            <li><b>Stemming</b> — "golpeo" ≠ "golpe" para el índice (limita relevancia, no corrección).</li>
            <li><b>CUDA / AMP / LoRA / distribuido</b> — sin GPU: restricción del entorno, no decisión.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Limitaciones honestas y siguientes pasos")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div class="card">
                <div class="card-title">⚠️ Limitaciones</div>
                <p class="card-text">17.6M + 860k tokens: memoriza más de lo que generaliza.<br>
                QA de 42 preguntas sintéticas: mide fidelidad al estilo, no criterio clínico.<br>
                DPO sin señal útil · RAG léxico sin stemming · sin GPU para escalar.</p>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown("""
            <div class="card">
                <div class="card-title">🚀 Mejoras propuestas</div>
                <p class="card-text">Corpus real con licencia · GQA/RoPE/RMSNorm/SwiGLU + reentrenar ·<br>
                PPO/GRPO · tokenizer propio por defecto · más eval real ·<br>
                OCR + stemming en el recuperador.</p>
            </div>
            """, unsafe_allow_html=True)

        st.caption("Fuentes: DOCUMENTACION.md · PROJECT_REPORT.md · docs/PROCESO_ENTRENAMIENTO.md · outputs/evaluation/ · logs/")
