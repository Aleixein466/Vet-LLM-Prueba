# PLAN BPE-16k — BPE propio + reentreno completo (puerta de decisión)

## Objetivo
Sustituir tiktoken/gpt2 (50257) por BPE propio de 16000 entrenado en el corpus
veterinario, según láminas Parte 2-5/8 y Parte 3-4/8, y reentrenar el modelo.

## Precondición (ya lista)
- `src/tokenizer/train_bpe.py`, `src/data/build_dataset_bpe.py`,
  `configs/tiny-18m-bpe.yaml` (este plan). Requiere `tokenizers` (instalado 0.23.2).

## Pasos (NO ejecutados; cada uno pide confirmación)
1. `python src/tokenizer/train_bpe.py` (~1-2 min CPU). Verifica `vocab=16000`.
2. `python src/data/build_dataset_bpe.py` (~2-5 min). Verifica tokens train/val.
3. `python src/training/train.py --cfg configs/tiny-18m-bpe.yaml` (~2h CPU, 3000
   pasos). Checkpoints en `checkpoints/tiny-18m-bpe/` (no toca `tiny-18m/`).
4. `python src/training/sft.py --cfg configs/tiny-18m-bpe.yaml` + `dpo.py`.
   NOTA: `sft.py`/`dpo.py`/`eval.py` usan `src/tokenizer/tok.py` (tiktoken).
   Antes del paso 4 hay que adaptarlos a BPE (cargar `tokenizer.json`,
   EOS=`[EOS]`). Trabajo estimado: 30-60 min de código + pruebas.
5. `python src/evaluation/eval.py --cfg configs/tiny-18m-bpe.yaml`.

## Veredicto 2026-10-05 (ejecutado pasos 1-3, 100 min CPU)
- BPE-16k entrenado OK + dataset 519 809 tokens (mejor compresión que tiktoken).
- Pretraining 3000 pasos: train 1.94, **val 7.43 con deriva alcista desde ~750
  (6.97→7.43) = sobreajuste** (~13 épocas en 468k tokens).
- Smoke test genera español vet coherente (sí aprendió).
- **Decisión: NO desplegar por defecto.** SFT-BPE colapsa igual (QA 0.119/0.143);
  base-BPE tiene el mejor QA sintético (0.881/0.952, 8.85M) pero val 7.43 con
  sobreajuste y sin eval en QA real. Se mantiene el tiktoken desplegado.
  BPE queda alternativa experimental USABLE (`app/vet_chat.py --cfg
  configs/tiny-18m-bpe.yaml`, con `_BpeAdapter` + EOT=2).
  BPE queda experimental en `checkpoints/tiny-18m-bpe/`.

## Puerta de decisión (obligatoria antes de desplegar) — aplicada arriba
- Comparar `outputs/metricas_bpe.json` vs base (`ppl<=5`, `recall>=0.15`).
- Solo si gana se cambia `inference.checkpoint`. Si no, se conserva el modelo
  tiktoken desplegado. Rollback = no hacer nada (directorios separados).

## Riesgos
- Vocab nuevo invalida TODOS los checkpoints actuales (no reutilizables).
- SFT/DPO/eval/chat/RAG hoy asumen tiktoken: hay que migrar `tok.py`,
  `vet_chat.py` y `retriever.py` (este último es agnóstico, solo texto).
- Costo: ~2.5-3h CPU + adaptación de código.
