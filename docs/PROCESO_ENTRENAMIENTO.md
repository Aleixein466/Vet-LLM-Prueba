# Proceso de entrenamiento — tecnologías y detalle

Documento de referencia del pipeline completo: qué hace cada etapa, con qué tecnología,
por qué se eligió esa tecnología y qué se midió. Todas las cifras salen de
`logs/*.log`, `outputs/evaluation/*.json` y `data/cleaned/ingest_report.json`.

---

## 0. Resumen del pipeline

```
PDF + corpus sintético
        │  pypdf + limpieza + anonimización + chunking
        ▼
data/raw/*.txt  ──► tokenización ──► train.bin / val.bin (uint16)
        │                              (tiktoken gpt2  |  BPE propio 16k)
        ▼
TinyGPT (decoder-only, 17.62M | 8.85M)  ◄── embeddings + self-attention causal
        │
        ├─ 1) PRETRAINING  train.py     next-token + CE + AdamW/cosine   6000 pasos
        ├─ 2) SFT          sft.py        CE con ignore_index=-100 + replay  40 pasos
        ├─ 3) DPO          dpo.py        policy vs referencia congelada      30 pasos
        │
        ├─ 4) EVALUACIÓN   eval.py       perplejidad + QA recall/accuracy
        │                  consistency.py  Jaccard entre paráfrasis
        │
        └─ 5) INFERENCIA + RAG  chat.py / vet_chat.py / Streamlit
                                    TF-IDF (principal) + MiniLM (rescate)
```

Decisión de despliegue respaldada en evidencia: se despliega la **base reentrenada**
(perplejidad 172, QA 0.762); SFT y DPO se descartan por medición, no por intuición.

---

## 1. Etapa 0 — Datos (tecnologías: pypdf, regex, JSONL)

### 1.1 Corpus sintético (`src/data/corpus_sintetico.py`)
Generador propio de plantillas veterinarias en español: 6518 documentos (1 097 883
caracteres útiles) a partir de 5 especies × 15 temas + 12 protocolos de urgencia +
QA (6 general + 6 emergencias). Genera tres artefactos:

| Archivo | Contenido | Uso |
|---|---|---|
| `data/raw/vet-es-sintetico.txt` | 1.15 MB de texto | pretraining |
| `data/sft/{sft_train,sft_val}.jsonl` | 238 / 42 pares `prompt`/`response` | SFT y eval QA |
| `data/sft/prefs.jsonl` | 120 pares `chosen`/`rejected` | DPO |

Es la única fuente de las respuestas *correctas y verificadas* (intervalos de
frecuencia cardiaca, temperatura,indowisos de desparasitación). Limitación honesta:
son plantillas, el modelo memoriza el estilo en vez de razonar.

### 1.2 Corpus real (`src/data/ingest_pdfs.py`) — etapa añadida el 2026-10-05
70 PDFs de `Datos/` (15 técnicos + 55 historias clínicas), 669 páginas:

- **Extracción**: `pypdf.PdfReader().extract_text()`.
- **Limpieza**: normalización Unicode NFC, ligaduras tipográficas, borrado de
  U+FFFD (los PDFs con `ToUnicode` roto envenenarían el corpus), desguionado de
  fin de línea (`veteri-\nnaria` → `veterinaria`), colapso de puntos de índice,
  descarte de páginas pobres (<200 chars, 43 páginas) y de páginas que solo tienen
  número.
- **Anonimización** (solo historias clínicas): regex para emails, celulares
 Colombia, cédulas, fechas-hora, direcciones y nombres del propietario →
  353 reemplazos por `[EMAIL]`, `[TEL]`, `[ID]`, `[DIRECCIÓN]`, `[PROPIETARIO]`.
- **Chunking**: 900 caracteres con 150 de solapamiento, re-corte por frases si un
  párrafo excede el tamaño → 1639 chunks (727 chars promedio).
- **Salidas**: `docs_real.jsonl`, `chunks_real.jsonl`, `vet-real-limpio.txt`
  (453 529 tokens) y `sft_real.jsonl` (1574 pares extractivos).

El impacto fue directo: la base vieja tenía perplejidad **29 540** sobre el corpus
real; tras reentrenar, **172**. La corpus real es la que hace que el modelo no
alucine.

### 1.3 Empaquetado binario (`build_dataset.py`, `build_dataset_bpe.py`)
El texto se trocea por `\n\n`, se tokeniza, se le añade el token EOS y se
concatenan los ids en un `numpy.uint16` plano que se escribe con `.tofile()`. Split
90/10. Sin formato JSONL ni overhead: `np.fromfile` lo lee directo a RAM.

| Corpus | Tokenizador | Tokens | train / val |
|---|---|---|---|
| sintético | tiktoken gpt2 | 408 099 | 367 289 / 40 810 |
| mixto (sintético + real) | tiktoken gpt2 | 860 629 | 774 566 / 86 063 |
| mixto | BPE propio 16k | 519 809 | 467 828 / 51 981 |

El BPE propioNeeds 40 % menos tokens que tiktoken para el mismo texto: el vocabulario
de 50k tokens de GPT-2 desperdicia fragmentos en español veterinario.

---

## 2. Etapa 1 — Tokenización (tecnologías: tiktoken, tokenizers/BPE)

### 2.1 tiktoken gpt2 (desplegado) — `src/tokenizer/tok.py`
`tiktoken.get_encoding("gpt2")`, |V| = 50257, EOS = `<|endoftext|>` (50256).
**No se entrena**: se reutiliza el vocabulario ya entrenado por OpenAI. Ventaja:
segmentación consistente y probada. Coste: 40 % de tokens "desperdiciados" en un
dominio estrecho, y `uint16` obliga a vigilar `max_id < 65536`.

**Detalle crítico — codificación conjunta con máscara**
(`encode_with_prompt_mask`, `tok.py:25-43`). En SFT y DPO no se hace
`enc(prompt) + enc(" " + respuesta)`, sino `enc(prompt + respuesta)` y luego se
localiza el borde por longitud, con verificación de que
`decode(ids[:k]) == prompt`. Motivo (bug real encontrado): tiktoken segmenta distinto
al concatenar, así que el modelo veía en SFT una costura OOD que nunca encuentra en
pretraining. Si el borde no es reproducible, se lanza `ValueError` y el par se
cuenta como omitido (2/1574 en el último run) en vez de colar etiquetas mal
alineadas.

### 2.2 BPE propio 16k (experimental) — `src/tokenizer/train_bpe.py`
Librería `tokenizers` de Hugging Face: modelo `BPE`, pre-tokenizador **ByteLevel**
(robusto con tildes y ñ, resuelve el problema de bytes desconocidos),
`BpeTrainer(vocab_size=16000, special_tokens=[PAD, UNK, EOS])` sobre el corpus
veterinario + documentos reales. Se guarda `tokenizer.json` (1.13 MB) y se usa con
el wrapper `tok_bpe.py` (EOS = id 2).

Cambiar el vocabulario **invalida todos los checkpoints**; por eso BPE vive en
directorios separados (`tiny-18m-bpe*`) y no toca los del modelo desplegado.

---

## 3. Etapa 2 — Embeddings

Dos tablas sumadas en `TinyGPT.forward` (`gpt.py:87`):

- `wte`: embedding de token, |V| × 256 = 12.87 M parámetros (tiktoken) o 4.10 M (BPE).
- `wpe`: embedding de **posición absoluta aprendida**, 128 × 256.

Se eligieron posicionales aprendidos en vez de **RoPE** porque con `block_size=128`
la tabla completa son 32 768 parámetros (irrelevante en 17.6M) y no requiere
reescalar frecuencias ni reordenar el cómputo de atención. Es una decisión
consciente de alcance: RoPE/RMSNorm/SwiGLU/GQA **no** están implementados, y añadir
cualquiera de ellos exige reentreno desde cero.

Inicialización `N(0, 0.02)` en todas las `Linear` y `Embedding` (`gpt.py:75-82`), que es
la escala estándar para redes pre-LN.

---

## 4. Etapa 3 — Arquitectura (`src/model/gpt.py`, `TinyGPT`)

GPT decoder-only estilo GPT-2, escrito a mano en PyTorch. 6 bloques, 4 cabezas,
embd 256, block_size 128, dropout 0.0, sin bias, `tie_weights=True`.

### 4.1 Self-Attention causal (`CausalSelfAttention`)
1. `c_attn` proyecta 256 → 3×256 en una sola pasada (Q, K, V juntos, luego `permute`).
2. `att = (q @ kᵀ) / sqrt(head_dim)` con head_dim = 64.
3. **Máscara causal** precalculada como buffer `torch.tril` y aplicada con
   `masked_fill(-inf)` → cada posición solo ve el pasado.
4. `softmax` por fila, salida `att @ v` reensamblada, `c_proj` de vuelta a 256.

Sin `FlashAttention` ni kernel fusionado: a `T=128` la versión ingenua es más rápida en
CPU que una implementación tiled, y el perfil entero es CPU. Es la pieza que más
impacta el costo por paso (O(T²) en memoria y cómputo).

### 4.2 MLP
`c_fc` 256 → 1024, **GELU**, `c_proj` 1024 → 256 (ratio 4×). GELU en lugar de ReLU
por su derivada suave en el centro: con 6 capas y gradients pequeños, ReLU muere más
fácil.

### 4.3 Normalización y conexiones
**Pre-norm**: cada bloque hace `x = x + attn(ln1(x))` y `x = x + mlp(ln2(x))`.
Pre-LN (no post-LN como en GPT-2 original) porque hace el entrenamiento estable sin
warmup largo — relevante con solo 6 capas y batches de 16×128 en CPU.
`ln_f` final, luego `lm_head`.

### 4.4 Pesos atados (tied weights)
`lm_head.weight = wte.weight`. Ahorra 12.87 M parámetros en la versión tiktoken y, en
la práctica, actúa como regularizador: la salida y la entrada comparten representación
y el modelo se entrena con dos tareas a la vez.

### 4.5 Desglose de parámetros (verificado con `TinyGPT(...).count_params()`)

| Componente | Params | % |
|---|---|---|
| `wte` (50257×256) | 12 865 792 | 73.0 |
| 6 bloques | 4 721 664 | 26.8 |
| ↳ `c_attn` + `c_proj` por bloque | 262 144 | |
| ↳ `c_fc` + `c_proj` del MLP por bloque | 524 288 | |
| ↳ 2 LayerNorm por bloque | 512 | |
| `wpe` (128×256) | 32 768 | 0.19 |
| `ln_f` | 256 | ~0 |
| `lm_head` | 0 (pesos atados) | 0 |
| **total** | **17 620 480** | |

Con BPE 16k: 8 850 688 (**8.85 M**), misma estructura. Ambas variantes caen dentro
del rango 12M–77M pedido para la versión tiktoken; BPE queda por debajo pero con la
mitad de parámetros y mejor QA sintético.

---

## 5. Etapa 4 — Pretraining (`src/training/train.py`)

### 5.1 Objetivo
Predicción del siguiente token: entrada `x` y objetivo `y = x[1:]` (desplazada).
Loss = **Cross-Entropy** sobre los logits aplanados (`gpt.py:95`).

### 5.2 Mini-batching (`get_batch`)
Ventanas aleatorias de 128 tokens: `ix = torch.randint(0, len(data) - block_size - 1,
(16,))`, luego dos slices solapados por posición. Con 774 566 tokens de train y
16×128 = 2048 tokens por paso, 6000 pasos = **12.3 M tokens procesados ≈ 16 épocas**
sobre el corpus. Es repetition-heavy a propósito: el corpus es pequeño y no cabe el
riesgo de dejar sin ver tokens.

### 5.3 Optimizador y schedule
- **AdamW** lr 3e-4, weight_decay 0.01, betas (0.9, 0.999) — desacopla el decaimiento
  del peso de la actualización de gradiente, lo que evita que el wd se comporte como
  un término de gradiente acoplado en tensores de baja magnitud como LayerNorm.
- **Cosine con warmup** (`lr_at`, `train.py:68-74`): 200 pasos de calentamiento lineal,
  luego decaimiento coseno hasta 0 en 6000. El warmup evita el pico de loss inicial;
  el coseno aterriza el lr en ~0 para que el final del run no esté "en movimiento".
- **Gradient clipping** `clip_grad_norm_(1.0)`: PyTorch en CPU puede dar picos de
  gradiente en batches con secciones de texto muy distintas.

### 5.4 Entrenamiento, checkpoints y reanudabilidad
- `loss.backward()` → `opt.step()`, un paso por batch (sin accumulation, sin AMP, sin
  CUDA: `torch 2.14.0+cpu`, 4 threads).
- Checkpoint cada 500 pasos + `last.pt` con `{model, opt, step, loss}`; `resume: true`
  recarga modelo **y** optimizador, así que el reentreno 3000→6000 continúa donde
  quedó en vez de empezar de cero (loss 0.10 → 1.68/5.08).
- `estimate_loss` con 20 lotes de train y val cada 250 pasos, en `torch.no_grad()` +
  `model.eval()`.

### 5.5 Evidencia de ejecución

| Run | Pasos | Tiempo | train loss | val loss |
|---|---|---|---|---|
| sintético (2026-09-28) | 3000 | 139 min | 0.093 | 0.095 |
| mixto reentreno (2026-10-05) | 3000→6000 | ~3.3 h (resume) | 1.677 | 5.081 |
| BPE 16k (2026-10-05) | 3000 | 100 min | 1.939 | 7.431 |

El salto de val 0.095 → 5.081 no es "empeorar" en calidad de respuesta: es el cambio
de corpus de validación. El 0.095 era contra las plantillas sintéticas que el modelo
memorizaba (perplejidad 1.11 sobre ese mismo set). La comparación útil es contra el
mismo corpus real: la base vieja daba **29 540** y la reentrenada **172**. BPE sí
muestra sobreajuste medible dentro de su propio run: val 6.97 → 7.43 desde ~750
pasos (≈13 épocas sobre 468k tokens).

---

## 6. Etapa 5 — SFT (`src/training/sft.py`)

Misma Cross-Entropy pero con `ignore_index=-100`: los labels del prompt son -100 y
**solo la respuesta cuenta** en el loss. Es el único cambio conceptual respecto al
pretraining y es lo que convierte el modelo en algo que responde a preguntas.

### 6.1 Composición del lote (batch 8)
- 4 pares SFT sintéticos (respuestas correctas, estilo QA)
- 2 ventanas de replay de pretraining (loss completo, sin máscara)
- 2 pares SFT reales (extractivos de `sft_real.jsonl`)

El **replay** no es cosmético: sin él el modelo olvidaba el corpus (perplejidad
1.1 → 252, olvido catastrófico). Es el mismo criterio de los trabajos de
"pre-training mix"/"replay" para ajustar sin destruir la base.

Loss combinado `(loss_sft + loss_pt + loss_real)/3`, lr 2e-5, 40 pasos, warmup 10,
clip 1.0. Observado: 6.523 → 3.786 (sft 8.13→4.43, real 9.24→6.45).

### 6.2 Por qué SFT se descartó (dos veces)
Post-SFT: perplejidad 300 (vs 172) y **QA accuracy 0.0** (recall 0.018). El modo de
fallo es consistente en las tres variantes: los pares extractivos (respuesta = chunk
de 1200 caracteres) sacan al modelo tiny de la distribución de QA y produce
respuestas de formato extractivo en lugar de una respuesta corta y accionable. El
problema no es la técnica, es la **desalineación de datos** para un modelo de 17M con
40 pasos de presupuesto.

La corrección se hizo en inferencia, no en entrenamiento: RAG con fallback extractivo
(`app/vet_chat.py`), que sí aprovecha los documentos reales sin destruir el modelo.

---

## 7. Etapa 6 — DPO (`src/training/dpo.py`)

Implementación mínima de Direct Preference Optimization sin modelo de recompensa:

1. Dos `TinyGPT` en memoria: `policy` (entrenable) y `ref` (misma checkpoint,
   `requires_grad_(False)`, `eval()`).
2. `seq_logp` (`dpo.py:19-26`): log-probabilidad de la secuencia, sumando solo las
   posiciones donde la máscara vale 1 (la completion).
3. Ventaja implícita: `adv = (lp_c − lp_r) − (ref_c − ref_r)`; la referencia se resta
   para que el objetivo sea el **cambio** relativo, no la preferencia absoluta.
4. Loss `−logsigmoid(β · adv)` con β = 0.1. lr 5e-5, 30 pasos, 120 pares.

Resultado observado: `adv ≈ +155 a +182` y loss `0.000` desde el paso 10. El modelo
ya separa las respuestas elegidas de las rechazadas por margen tan grande que el
gradiente es numéricamente nulo. Es una etapa **demostrativa**: la señal existe, pero
las preferencias (plantilla buena vs absurdo clínico) son triviales y no dejan margen
para aprender. PPO/GRPO están documentados como extensión, no implementados.

---

## 8. Etapa 7 — Evaluación (`eval.py`, `eval_bpe.py`, `consistency.py`)

Tres métricas, porque una sola no alcanza en un modelo pequeño:

**1. Perplejidad** — `exp(media de CE sobre 50 lotes de val)`. Mide ajuste al texto, no
capacidad de responder. Solo comparable dentro del mismo vocabulario + mismo set.

**2. QA held-out** — 42 prompts de `sft_val` (nunca vistos en entrenamiento), generación
greedy de 80 tokens, `recall` = |keywords(generado) ∩ keywords(esperado)| /
|keywords(esperado)| y `accuracy` = fracción con recall ≥ 0.4. Las keywords son
palabras de >4 caracteres tras normalizar puntuación.

**3. Consistencia** (`consistency.py`) — agrupa `sft_val` por (tema, especie),
genera con cada paráfrasis y mide **Jaccard** medio de keywords entre respuestas.
Detecta si el modelo cambia de respuesta a la misma pregunta reformulada (base 0.609
vs SFT 0.15).

Resultados de todos los candidatos:

| Modelo | ppl val | QA recall | QA acc | Jaccard | Decisión |
|---|---|---|---|---|---|
| base vieja (sintética) sobre val nuevo | 29 540 | 0.72 | 0.762 | — | superada |
| **base reentrenada (tiktoken)** | **172** | 0.705 | **0.762** | 0.609 | **desplegada** |
| SFT real (tiktoken) | 300 | 0.018 | 0.0 | 0.15 | descartado |
| base BPE 16k (8.85M) | 1 773* | 0.881 | **0.952** | — | experimental |
| SFT BPE 16k | 2 418* | 0.119 | 0.143 | — | descartado |
| DPO (tiktoken) | — | — | — | — | demostrativo |

\* no comparable entre vocabularios y sets distintos.

Decisión con evidencia, no intuición: `inference.checkpoint` =
`checkpoints/pretraining/final/model.pt`, con respaldo del anterior en
`model_sintetico_backup.pt`.

---

## 9. Inferencia y RAG (`chat.py`, `app/vet_chat.py`)

- **Generación** (`TinyGPT.generate`): soporta temperature, top-k, top-p y
  repetition_penalty, pero la config usa **greedy** (`temperature: 0.0`) +
  `repetition_penalty: 1.1`. Hallazgo medido: en modelos tiny el muestreo top-k/top-p
  produce "ensalada de tokens"; greedy es más estable. El penalty evita bucles de
  repetición en 80 tokens.
- **TF-IDF** (`src/rag/retriever.py`): implementación propia con numpy, sin sklearn ni
  servidor vectorial. TF `(1 + log f)`, IDF `log((1+n)/(1+df)) + 1`, matriz de 2061
  pasajes normalizada, similitud coseno por producto punto. Normalización de acentos
  con `unicodedata` (NFD, quitando diacríticos) porque los PDFs traen `ToUnicode`
  roto. Umbral recalibrado 0.20 → **0.18** al ampliar el corpus (más documentos
  diluyen el coseno).
- **Rescate semántico** (`src/rag/embeddings.py`): cuando TF-IDF rechaza,
  `paraphrase-multilingual-MiniLM-L12-v2` (384 dim, snapshot local, singleton en
  proceso para no recargar 470 MB por consulta) recupera paráfrasis.
  **Doble puerta** para no colar OOD: score ≥ 0.55 con ≥4 términos en común, o
  score ≥ 0.72 con ≥3. Sin esa segunda puerta, "Colón" (0.68) y "dinosaurios" (0.55)
  se aceptaban como consulta veterinaria.
- **Guardrails**: fuera-de-ámbito (respuesta fija, no alucina), vaga (pide
  concretar), memoria conversacional, banner de urgencia determinista, preferencia de
  especie (+0.15). Batería `scripts/run_rag_tests.py`: 7/7 PASS.

---

## 10. Tecnologías, por peso real en el resultado

Ordenadas por cuánto cambiaron el comportamiento del sistema, no por popularidad.

### Tier 1 — explican el resultado
1. **Transformer decoder-only con self-attention causal** — sin la máscara causal no
   hay modelo de lenguaje; sin el bloque de atención no hay relación entre tokens.
2. **Tokenización (BPE)** - decide la longitud de secuencia, el costo por paso y la
   separación entre train/val. Cambiar de gpt2 a BPE 16k bajó los tokens 40 % y cambió
   los parámetros de 17.6M a 8.85M.
3. **Next-token prediction + Cross-Entropy** — el único objetivo; SFT y DPO son
   reformulaciones suyas.
4. **AdamW + cosine/warmup + grad clip** - lo que hizo entrenable de forma estable
   los 6000 pasos en CPU sin divergir ni un solo NaN en el log.
5. **SFT con `ignore_index=-100` + replay** — define qué es "responder"; el replay
   evita el olvido catastrófico (ppl 1.1 → 252 sin él).
6. **Evaluación múltiple (ppl + QA recall/acc + Jaccard)** — sin las tres, la
   decisión de despliegue habría sido errónea: la base vieja y la reentrenada tienen el
   mismo QA 0.762 pero perplejidad 29 540 vs 172.
7. **RAG (TF-IDF + MiniLM con doble puerta)** — aporta la evidencia que un modelo de
   17M no puede memorizar; es la tecnología que más valor clínico aporta al final.

### Tier 2 — relevantes para la reproducibilidad
8. **Checkpoints reanudables** (`last.pt` con estado del optimizador) - permite el
   reentreno 3000→6000 sin repetir 3 horas.
9. **Empaquetado `uint16` con `np.fromfile`** — 1.5 MB de train se cargan en RAM sin
   parseo; `uint16` vale porque 50257 < 65536.
10. **Configuración YAML declarativa** — un archivo gobierna modelo, datos, training,
    SFT, DPO, inference y evaluación; permite el fork BPE cambiando solo
    `vocab_size` y rutas.
11. **Mascarado conjunto prompt+respuesta** — fix de bug real de segmentación.
12. **DPO (policy vs referencia congelada)** — implementado y medido; sin efecto por
    preferencias triviales.

### Tier 3 — de apoyo, no decisivas
13. **pypdf + regex de anonimización** — habilita los 70 PDFs reales; sin esto el
    modelo solo tenía plantillas.
14. **Streamlit** — interfaz (chat, generación, evaluación, arquitectura, training,
    historia clínica). No toca el modelo.
15. **safetensors** — formato de checkpoint para la web.
16. **CPU-only torch 2.14+cpu** — restricción del entorno, no decisión. Sin
    CUDA/AMP/mixed precision, sin distributed, sin LoRA/peft, sin
    cuantización.

### No implementado (y por qué importa)
- **RoPE, GQA, RMSNorm, SwiGLU** — la arquitectura es deliberadamente GPT-2 clásica.
  Añadir cualquiera obliga a reentrenar desde cero (los checkpoints no son
  transferibles).
- **FlashAttention / kernels fused** — innecesario a T=128 en CPU.
- **PPO / GRPO** — documentados como extensión; no implementados.
- **Modelos de recompensa** — DPO los evita por diseño.

---

## 11. Limitaciones honestas

- 17.6M parámetros, CPU, corpus de 860k tokens: memoriza más de lo que generaliza.
- El QA eval (42 prompts) viene del generador sintético: mide fidelidad al estilo del
  corpus, no juicio clínico. El BPE saca 0.952 ahí y aun así no se despliega, porque
  su validación (7.43 con sobreajuste creciente) y su QA real son flojos.
- Perplejidades entre tokenizadores distintos no son comparables.
- El DPO no aporta señal con preferencias triviales.
- Los PDFs escaneados sin capa de texto quedarían fuera; en esta corrida ninguno
  fue omitido (`omitidos_escaneado: []`), pero el pipeline no tiene OCR.

---

## 12. Cómo reproducir

```powershell
# 0) entorno (venv en C:, no en el USB D:)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 1) datos
python src/data/corpus_sintetico.py --n-docs 4000
python src/data/ingest_pdfs.py
python src/tokenizer/prepare_tokenizer.py

# 2) tokenización + dataset
python src/data/build_dataset.py

# 3) pretraining
python src/training/train.py                    # 3000 → 6000 pasos, reanuda solo

# 4) post-entrenamiento
python src/training/sft.py
python src/training/dpo.py

# 5) evaluación
python src/evaluation/eval.py
python src/evaluation/consistency.py

# 6) variante BPE (directorios separados)
python src/tokenizer/train_bpe.py
python src/data/build_dataset_bpe.py
python src/training/train.py --cfg configs/tiny-18m-bpe.yaml

# 7) RAG + app
python src/rag/retriever.py
python src/rag/embeddings.py --build
python app/vet_chat.py --ask "¿Cada cuánto desparasito a mi gato?"
```
