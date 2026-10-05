# VET-TINY-LLM — estado del proyecto
Fecha: 2026-09-28 (pipeline completo y verificado por ejecución)

## Dónde se quedó (Fase 0)
Solo esqueleto de carpetas vacías, venv Python 3.11.6 sin dependencias,
sin código ni datos. CPU-only (12.7 GB RAM, Intel UHD, sin CUDA).

## Pipeline completo (Fase 1+2, verificado)
- Paso 1: venv en `C:\VET-TINY-LLM-venv` (torch 2.14 CPU + tiktoken 0.14 + yaml/tqdm/numpy).
  `D:` es USB a 1.6 MB/s → el venv original de `D:` quedó descartado (eliminado).
  Usar `env-fast.ps1` (venv C: + caches en C:).
- Paso 2: `src/data/corpus_sintetico.py` — 5353 docs ES + 136 SFT train / 24 val + 120 prefs DPO.
- Paso 3: tiktoken/gpt2 (`src/tokenizer/tok.py`, `prepare_tokenizer.py`,
  `src/data/build_dataset.py` → `.bin` uint16): 328407 tokens
  (train 295566 / val 32841), fertilidad 0.37 tok/char.
- Paso 4: `src/model/gpt.py` — `TinyGPT` decoder-only 17.62M
  (block 128, 6 capas, 4 cabezas, embd 256, tied). Config `configs/tiny-18m.yaml`.
- Paso 5: `src/training/train.py` (Cross-Entropy + AdamW + cosine/warmup):
  3000 pasos (~139 min CPU) → train 0.093 / val 0.095.
- Paso 6: `checkpoints/tiny-18m/last.pt` (212 MB con optimizador) + `step-N.pt` (71 MB).
- Paso 7: `src/training/sft.py` (30 pasos, replay 50/50, codificación conjunta con
  máscara verificada), `src/training/dpo.py` (30 pasos),
  `src/evaluation/eval.py` → **ppl 1.12, QA recall 0.499, accuracy 0.542**
  (`outputs/metricas.json`, `outputs/ejemplos.md`),
  chat final `src/inference/chat.py` verificado (greedy + corte en EOS).

## Mapeo a la especificación (imágenes)
| Pedido | Implementación |
|---|---|
| Python, PyTorch | venv 3.11 + torch CPU |
| Tiktoken | `tiktoken/get_encoding("gpt2")`, sin entrenador externo |
| Embeddings | `wte` + `wpe` en `TinyGPT` |
| Decoder-Only + Self-Attention | `CausalSelfAttention` con máscara causal |
| Cross-Entropy + AdamW | `F.cross_entropy` + `AdamW` en pretrain/SFT/DPO |
| Pretraining | `train.py` (siguiente token) |
| SFT | `sft.py` |
| DPO/PPO/GRPO | `dpo.py` implementado; PPO/GRPO como extensión futura |
| CUDA/GPU | no disponible en este equipo; build CPU |
| Modelo 12M–77M | 17.62M |
| Chat final | `chat.py` |

## Lecciones (bugs reales encontrados y corregidos)
1. SFT sin replay → olvido catastrófico (ppl 1.1 → 252). Fix: replay 50% pretraining.
2. SFT concatenando `enc(prompt)+enc(" "+resp)` crea costura OOD (tiktoken segmenta
   distinto que `enc(prompt+resp)`). Fix: codificación conjunta + máscara verificada
   (`tok.encode_with_prompt_mask`), también en DPO.
3. SFT largo (400 pasos/23 épocas, 120 pasos) sobreentrena; 30 pasos bastan.
4. Muestreo top-k/p en modelo tiny → ensalada de tokens; greedy + repetition_penalty 1.1.
5. DPO satura (loss 0.000) porque las preferencias son triviales (elegida memorizada
   vs absurda): etapa demostrativa, sin efecto medible.

## Limitaciones honestas
- Corpus sintético de plantillas: el modelo memoriza, generaliza poco a
  formulaciones no vistas (precisión QA 0.54 en parafraseos held-out).
- 24 pares de eval: intervalo de confianza amplio.
- Sin GPU: escalar a corpus real grande requeriría más cómputo.

## Extensiones futuras
- Mezclar corpus real con licencia (p. ej. Wikipedia-ES veterinaria) + dedup.
- PPO/GRPO con modelo de recompensa.

## RAG + app (implementado y verificado)
- `src/rag/retriever.py`: índice TF-IDF (numpy, sin deps nuevas) sobre 115 pasajes
  (`data/rag/index/`). Recuperación top-k verificada 3/3.
- `app/vet_chat.py`: chat con RAG — recupera fuentes, genera con el tiny LLM y
  aplica **fallback extractivo** (si la generación no se apoya en la fuente,
  devuelve la fuente marcada `[extractivo]`).
  Uso: `python app/vet_chat.py --ask "..." [-k 2]` o interactivo `python app/vet_chat.py`.
- Hallazgo: el modelo de 17M memoriza tan fuerte que a veces ignora el contexto
  (in-context learning limitado); el fallback extractivo lo compensa y las 3
  consultas de prueba responden correcto con fuentes citadas.
- Puertas de la app (verificadas): fuera-de-ámbito (<2 términos de dominio en pregunta
  completa → "fuera de mi ámbito", no alucina), vaga (<3 términos → pide concretar),
  memoria conversacional (pregunta+respuesta previa, sin banner), banner de urgencia
  determinista, umbral TF-IDF 0.20, preferencia de especie (+0.15), tokens sin acentos.
- Contenido RAG: `data/raw/vet-protocolos-2.txt` (sangre en heces, dolor abdominal,
  caída de altura, traumatismo, braquicéfalos, jadeo) — la app lo usa sin reentrenar.
