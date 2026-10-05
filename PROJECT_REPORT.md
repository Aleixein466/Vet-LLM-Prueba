# PROJECT_REPORT.md — VET-TINY-GPT
Informe final con métricas reales. Lo no medido se marca NO DISPONIBLE.

## 1-5. Hardware y entorno
- CPU-only, sin GPU NVIDIA/CUDA. Intel UHD (gráfica), 11.8 GB RAM total.
- Windows. `D:` = USB ~58.6 GB (proyecto; escritura ~1.6 MB/s) — venv en `C:` NVMe.
- Python 3.11.6, torch 2.14.0+cpu, tiktoken 0.14, 4 threads torch.

## 6-9. Arquitectura y tokenizer
- GPT decoder-only propio (`src/model/gpt.py`): 17.62M params, block 128, 6 capas,
  4 cabezas MHA causal, embd 256, LayerNorm, GELU, posicionales aprendidas, tied head.
- **Brecha honesta**: NO implementa GQA / RoPE / RMSNorm / SwiGLU (la página web de
  arquitectura lo indica). Añadirlos exige reentrenar desde cero.
- Tokenizer: tiktoken/gpt2 (externo, 50257), EOS `<|endoftext|>`. NO es BPE propio.

## 10-11. Dataset y tokens
- Corpus sintético ES veterinario + 12 protocolos de urgencia + QA (6 general + 6 emergencias).
- 6518 docs (1 097 883 chars) → 408 099 tokens (train 367 289 / val 40 810).
- SFT: 238 train / 42 val (`prompt`/`response` = instruction/output). Prefs DPO: 120.
- Fuente: generador propio `src/data/corpus_sintetico.py`, 2026-09-28. Licencia: sintético propio.

## 12-18. Pretraining
- 3000 pasos, batch 16×128, AdamW lr 3e-4 cosine/warmup 200. Duración ~02:16 (00:05→02:21).
- Tokens procesados: 3000×16×128 = 6.14M (con repetición; corpus único 367k).
- Tokens/sec ~750 (derivado). Loss final train ~0.09, val ~0.10 (último visto 0.12@950).
- Checkpoints: `checkpoints/tiny-18m/` (7) + `checkpoints/pretraining/final/`
  (model.safetensors, model.pt, config.json, training_state.json, run_metadata.json).

## 19-21. Evaluación y generaciones
- Base: ppl 1.11, QA recall 0.72, accuracy 0.762 (n=42).
- Generaciones base: `outputs/evaluation/base_generations.txt` (6 prompts, sin editar).
- Decisión (`pretraining_decision.json`): SUFICIENTE_SFT (ppl≤5 y recall≥0.15).

## SFT y post-SFT
- SFT 40 pasos lr 2e-5 replay 50/50, loss respuesta enmascarada (ver `tok.encode_with_prompt_mask`).
- Post-SFT(40): ppl 1.13, recall 0.227, accuracy 0.238. SFT(120): recall 0.026, acc 0.024.
- Consistencia: base 0.609 vs SFT 0.15.
- **Despliegue (evidencia)**: base gana en QA held-out (0.762) y consistencia.
  `inference.checkpoint` = `checkpoints/pretraining/final/model.pt`.
  Ver `outputs/evaluation/base_vs_sft.json`.
- Experimento SFT-120 + consistencia base: ver `outputs/metricas_sft120.json` y
  `outputs/evaluation/consistency{,_base}.json`. Decisión de checkpoint en §Mejoras.
- DPO 30 pasos (prefs triviales, satura loss 0.000): demostrativo, sin efecto medible.

## Reentreno 2026-10-05 (corpus real + RAG híbrido + SFT cableado)
- Ingesta: `src/data/ingest_pdfs.py` — 70 PDFs (15 técnicos + 55 historias
  anonimizadas, 353 reemplazos PII) → 1639 chunks, `data/raw/vet-real-limpio.txt`
  (453 529 tokens). Dataset reconstruido: 860 629 tokens (train 774 566 / val 86 063).
- Pretraining 3000→6000 pasos (~3.3 h CPU, resume desde loss 0.10): eval final
  train 1.68 / val 5.08. Modelo viejo en val nuevo: ppl 29 540 (perdido en texto
  real); modelo reentrenado: ppl 172 con QA intacto (mejora 172× en texto real).
- SFT cableado a `sft_real.jsonl` (`sft.py`: lote 4 sint + 2 replay + 2 real;
  36/1574 pares omitidos por no-alineación + 2 en entreno): loss 6.5→3.8.
  Post-SFT: ppl 300, recall 0.018, acc 0.0 → SFT descartado de nuevo.
- DPO 30 pasos: saturado (loss 0.000), demostrativo.
- **Despliegue (evidencia)**: base reentrenada (`checkpoints/tiny-18m/last.pt` →
  `checkpoints/pretraining/final/model.pt`; anterior en `model_sintetico_backup.pt`).
  Ver `outputs/evaluation/retrain_{base,sft}_eval.json`, `oldbase_newval_eval.json`.
- RAG híbrido: TF-IDF + MiniLM (`src/rag/embeddings.py`, `SEM_THRESHOLD=0.55`,
  índice `semantic.npz` 2061×384). Rescata paráfrasis que TF-IDF rechaza
  (ej. "mi can vomita sangre" → parvovirosis 0.71). Requiere
  `sentence-transformers` (ver `requirements.txt`).
- BPE propio 16k: EJECUTADO completo (tokenizer + dataset 519 809 tok + pretrain
  100 min + SFT-BPE + eval-BPE + chat integrado vía `--cfg tiny-18m-bpe.yaml`).
  Base-BPE: QA 0.881/0.952 (mejor QA de todos, 8.85M params); SFT-BPE colapsa
  (0.119/0.143, mismo patrón). Veredicto: NO desplegar por defecto (val 7.43 con
  sobreajuste + QA solo estilo-sintético); queda alternativa experimental usable.
  Ver `docs/PLAN_BPE16K.md`.

## 22-27. Limitaciones, web, ejecución, problemas, mejoras
- Educativo, 17.6M params, CPU, sintético: memoriza, generaliza poco, puede alucinar.
- Web Streamlit `http://localhost:8501` (chat, generación, evaluación, arquitectura real,
  training, historia clínica) — `run_web.bat`, `README_WEB.md`. Solo usa checkpoints.
- Ejecución: ver README.md (verificación) y `run_web.bat` (web).
- Problemas→soluciones: venv USB lento→venv C:; SFT sin replay (ppl 252)→replay;
  costura tokenización OOD→codificación conjunta; muestreo→greedy; SFT largo→corto;
  RAG: fallback extractivo, puertas OOD/vaga, memoria, banner urgencia, umbral 0.20,
  preferencia de especie.
- Mejoras: corpus real con licencia, GQA/RoPE/RMSNorm/SwiGLU + reentrenar, PPO/GRPO,
  tokenizer propio, más eval.
