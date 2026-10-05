# Guía completa del proceso — VET-TINY-LLM (2026-10-04/05)

De la recolección de información al modelo desplegado, con tecnologías y resultados.

## 0. Punto de partida
Proyecto existente `VET-TINY-LLM`: mini-LLM veterinario en español desde cero
(GPT 17.6M, CPU-only), entrenado solo con corpus **sintético** (408k tokens),
RAG TF-IDF y app Streamlit/CLI. Alucinaba por falta de datos reales.

## 1. Recolección y análisis de información
- **PDF guía** `Creando un LLM desde cero.pdf` (6.9 MB): 28 infografías en 4 partes
  (visión general, stack pedagógico, técnica Win11, caso académico). Solo imágenes.
- **Repo referencia** `FareedKhan-dev/train-llm-from-scratch`: stack pedagógico
  PyTorch puro + tiktoken + SFT/DPO/PPO/GRPO + Streamlit (documentado en
  `Tecnologías del repositorio Train LLM.pdf`).
- **Carpeta `Datos/`**: 15 PDFs técnicos veterinarios + 55 historias clínicas
  (ANIMAL HAPPY/KYRON) con PII (nombres, CC, teléfonos).
- **Decisiones del usuario**: mejorar tiny CPU+RAG (no Qwen/GPU), anonimizar
  historias, construir pipeline PDF→JSON.

## 2. Ingesta: PDF crudo → corpus limpio (`src/data/ingest_pdfs.py`)
- Extracción con **pypdf**, limpieza (unicode, desguionado, páginas pobres),
  **anonimización de historias** (353 reemplazos: emails, celulares, CC, nombres,
  direcciones), chunking 900/150.
- Salidas: `data/cleaned/{docs,chunks}_real.jsonl` (70 docs, 1639 chunks),
  `data/raw/vet-real-limpio.txt` (1.19M chars, 453 529 tokens),
  `data/sft/sft_real.jsonl` (1574 pares), `ingest_report.json`.
- Verificación: 0 U+FFFD, 0 emails en historias, RAG ya recupera casos reales
  (hepatozoon/Ehrlichia con citas).

## 3. RAG híbrido (TF-IDF + MiniLM semántico)
- TF-IDF existente como vía principal; añadido `src/rag/embeddings.py` con
  **paraphrase-multilingual-MiniLM-L12-v2** (384 dim, snapshot local en C:,
  singleton en proceso) + respaldo en `app/vet_chat.py:retrieve_semantic`.
- Calibración con datos reales: umbral TF-IDF 0.20→**0.18** (el corpus grande
  diluye el coseno); umbral semántico **0.55**; **doble puerta** score+solape
  (0.55&≥4 ó 0.72&≥3) tras detectar que "Colón" (0.68) y "dinosaurios" (0.55)
  colaban como veterinaria.
- Batería `scripts/run_rag_tests.py`: **7/7 PASS** (TEST 7 = rescate semántico).

## 4. SFT cableado al corpus real
- `sft.py`: lote 4 sintético + 2 replay + 2 real (`sft_extra` en config).
- Fix: 36/1574 pares con borde no-alineable → se omiten con conteo (no tumban).
- 40 pasos: loss 6.5→3.8.

## 5. Reentreno (modo automático, reportes c/5 min)
- Respaldo previo + `build_dataset.py`: **860 629 tokens** (train 774 566).
- `train.py` pasos 3000→6000 (~3.3 h CPU, resume desde loss 0.10).
- Evolución: spike inicial 3.69 → train 1.68 / val 5.08.
- DPO 30 pasos: saturado (demostrativo, como antes).

## 6. Evaluación, decisión y despliegue (con evidencia)
| Modelo | ppl val nuevo | QA acc | Decisión |
|---|---|---|---|
| Base vieja | 29 540 | 0.762 | superada |
| **Base reentrenada** | **172** (172×) | 0.762 intacto | **desplegada** |
| SFT nuevo | 300 | 0.0 | descartado |
- Despliegue verificado por hash: `checkpoints/pretraining/final/model.pt`
  (anterior en `model_sintetico_backup.pt`). Chat final OK.

## 7. Variante BPE 16k (experimental)
- `train_bpe.py` (vocab 16000) + `build_dataset_bpe.py` (519 809 tok) +
  pretrain 100 min (train 1.94/val 7.43, sobreajuste) + `sft_bpe.py` +
  `eval_bpe.py` + chat integrado (`--cfg tiny-18m-bpe.yaml`).
- Base-BPE: QA **0.952** (8.85M); SFT-BPE colapsa (0.143). Veredicto: no
  desplegar por defecto; alternativa usable. Ver `docs/PLAN_BPE16K.md`.

## 8. Tecnologías usadas
Python 3.11.6, torch 2.14+cpu, tiktoken 0.14/gpt2, tokenizers 0.23 (BPE),
sentence-transformers 6.1 (MiniLM), pypdf 6.19, numpy, safetensors, Streamlit.
Modelo: Transformer decoder-only propio (6 capas, 4 cabezas, embd 256).
Entrenamiento: Cross-Entropy + AdamW + cosine/warmup; SFT enmascarado; DPO.
RAG: TF-IDF numpy + MiniLM, sin servidor vectorial. Todo CPU, sin CUDA.

## 9. Cómo usarlo ahora
- Chat: `app/vet_chat.py --ask "..."` (o `--cfg configs/tiny-18m-bpe.yaml`).
- Web: `run_web.bat` → http://localhost:8501.
- Métricas: `outputs/evaluation/retrain_{base,sft}_eval.json`, `bpe_*_eval.json`.
- Detalle técnico: `PROJECT_REPORT.md`. Láminas: `README.md` §9-11.
