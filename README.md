# VET-TINY-LLM — LLM veterinario en español desde cero (CPU)

Pipeline según especificación (pasos 1–7):
Texto → Tokens (tiktoken) → Embeddings → Transformer Decoder-Only con Self-Attention
→ Entrenamiento (Pretraining, SFT, DPO) → Asistente (chat).

Tecnologías: Python, PyTorch, tiktoken, Embeddings, Transformer Decoder-Only,
Self-Attention, Cross-Entropy Loss, AdamW, Pretraining, SFT, DPO.
(PPO/GRPO documentados como extensión; DPO implementado. CUDA/GPU no disponible
en este equipo → build CPU.)

Detalle de cada etapa del entrenamiento y por qué se eligió cada tecnología:
`docs/PROCESO_ENTRENAMIENTO.md`.

**Documentación completa y vigente: `DOCUMENTACION.md`** (qué se hizo, qué tecnología se
usó en cada parte y por qué, con las métricas reales).

## 1. Entorno
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 2. Dataset reducido
```
python src/data/corpus_sintetico.py --n-docs 4000
```
Genera `data/raw/vet-es-sintetico.txt` + `data/sft/{sft_train,sft_val,prefs}.jsonl`.

## 3. Tokenizar y preparar datos
```
python src/tokenizer/prepare_tokenizer.py
python src/data/build_dataset.py
```

## 4. Modelo (~17.6M params, rango 12M–77M)
`src/model/gpt.py` — GPT causal: embeddings + 6 bloques (masked self-attention + MLP) + head con tied weights.
Ver `configs/tiny-18m.yaml`.

## 5. Pretraining
```
python src/training/train.py            # 3000 pasos, checkpoints en checkpoints/tiny-18m/
```

## 6. Checkpoints
`checkpoints/tiny-18m/last.pt` (+ `step-N.pt` cada 500). Reanudable (`resume: true`).

## 7. Ajuste y evaluación + chat final
```
python src/training/sft.py              # SFT con máscara en respuesta
python src/training/dpo.py              # DPO vs referencia congelada
python src/evaluation/eval.py           # perplejidad + QA -> outputs/metricas.json
python src/inference/chat.py            # chat final
```

## 8. RAG + app (recuperación + generación con fuentes)
```
python src/rag/retriever.py             # índice TF-IDF -> data/rag/index/
python app/vet_chat.py --ask "¿Cada cuánto desparasito a mi gato?"
python app/vet_chat.py                  # interactivo
```

## 9. Corpus real desde PDFs (Datos/ -> limpio -> JSON)
```
python src/data/ingest_pdfs.py          # 70 PDFs (15 técnicos + 55 historias anonimizadas)
```
Lee `Datos/Documentos/*.pdf` + `Datos/HISTORIAS CLINICAS/**/*.pdf`, limpia,
anonimiza PII de historias (nombres, CC, teléfonos, emails, direcciones; 353
reemplazos) y genera `data/cleaned/{docs_real,chunks_real}.jsonl` (70 docs,
1639 chunks), `data/raw/vet-real-limpio.txt` (~1.19M chars, ~453k tokens) y
`data/sft/sft_real.jsonl` (1574 pares). Luego reconstruye el índice:
`python src/rag/retriever.py` (2061 pasajes, vocab 17278).
Nota: algunos PDFs traen fuentes con ToUnicode roto (tildes se ven como � en
consola con codepage latin, pero el archivo UTF-8 es correcto); el RAG se apoya
en términos ASCII (hepatozoon, garrapata, ehrlichia) y funciona.
Reentrenar con el corpus ampliado (~868k tokens total): `python src/data/build_dataset.py`
seguido de `python src/training/train.py` (~2h en CPU).
Reentreno ejecutado 2026-10-05: 860 629 tokens, pasos 3000→6000 (~3.3 h),
eval train 1.68 / val 5.08; QA intacto (0.762); SFT descartado (recall 0.018);
desplegada la base reentrenada. Ver `PROJECT_REPORT.md`.

## 10. RAG híbrido (TF-IDF + semántico MiniLM)
```
python src/rag/embeddings.py --build   # índice semántico -> data/rag/index/semantic.npz
```
TF-IDF sigue como vía principal con sus guardrails (umbral 0.18 recalibrado con
corpus ampliado); si rechaza (`SIN_EVIDENCIA`/`FUERA_DE_AMBITO`), MiniLM
multilingüe rescata paráfrasis con DOBLE puerta (score≥0.55 y solape≥4, o
score≥0.72 y solape≥3) para no colar OOD como "Colón" o "dinosaurios".
Batería `scripts/run_rag_tests.py`: 7/7 PASS (TEST 7 = rescate semántico).
Ver `src/rag/embeddings.py` y respaldo en `app/vet_chat.py:retrieve_semantic`.

## 11. Variante BPE 16k (experimental, usable)
```
python app/vet_chat.py --cfg configs/tiny-18m-bpe.yaml --ask "..."
```
Modelo 8.85M con BPE propio; mejor QA sintético (0.952) pero no desplegado por
defecto (veredicto en `docs/PLAN_BPE16K.md`).
