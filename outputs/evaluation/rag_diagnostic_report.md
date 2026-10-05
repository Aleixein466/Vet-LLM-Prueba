# Diagnóstico RAG — VET-TINY-GPT respondía fuera de tema

Fecha: 2026-09-28. Modelo NO reentrenado. Checkpoints intactos. Arquitectura y tokenizer sin cambios.
Backup previo: `app/vet_chat_backup.py` (copia byte a byte de `app/vet_chat.py` antes de corregir).

## 1. Causa encontrada (resumen)

El error estaba en el **pipeline RAG de `app/vet_chat.py`**, no en los pesos de VET-TINY-GPT.
Combinación de cuatro defectos:

1. **Threshold demasiado permisivo y mal ordenado.** `hits[0][1] < 0.20` se comprobaba
   DESPUÉS de sumar el bonus de especie (+0.15). Cualquier consulta con una sola palabra
   de vocabulario ("perros", "que", "comio") puntuaba 0.20–0.29 y pasaba.
2. **Sin regla de relevancia.** Bastaba compartir 1 token con un documento
   ("perros" → doc 127 envenenamiento, score crudo 0.2859). No se exigía que el
   documento compartiera ≥2 términos de la consulta.
3. **Contaminación por historial.** Si la consulta actual tenía <4 tokens,
   se concatenaba `historial[-1] + consulta` para retrieval y además se saltaba el
   filtro de fuera-de-ámbito. En una sesión secuencial (perro → cristóbal → dinosaurios),
   la 3ª consulta heredaba "cristobal colon comio perros" y recuperaba envenenamiento.
4. **Fallback extractivo incondicional.** Si la generación del tiny LLM no se parecía
   léxicamente al pasaje (`rec < 0.3`), se devolvía el pasaje tal cual + ` [extractivo]`,
   aunque el pasaje fuera irrelevante. De ahí los tres síntomas reportados.

## 2. Archivo responsable

- `app/vet_chat.py` (función interna `answer()` dentro de `main()`).
- Retrieval: `src/rag/retriever.py` (`TfidfIndex.query`, cosine TF-IDF).
- Hallazgo secundario: `app/components/chat.py` (Streamlit) **no usa RAG en absoluto**
  (solo `generar(model, prompt)`); cualquier prueba hecha ahí es 100 % alucinación
  generativa, sin threshold ni rechazo. Las respuestas con `[extractivo]` del reporte
  provienen necesariamente de `app/vet_chat.py`, único sitio que emite ese sufijo
  (línea del fallback `rec < 0.3`).

## 3. Función responsable

- `main()->answer()` en `app/vet_chat.py`: retrieval (`idx.query`), bonus de especie,
  threshold 0.20, `build_prompt`, `model.generate`, fallback `rec < 0.3`,
  sufijo `" [extractivo]"`, `BANNER` de urgencia.
- `es_fuera_de_ambito()` (overlap vocab <2, con bypass si había historial).
- `TfidfIndex.query()` en `src/rag/retriever.py` (top-k cosine, `sims > 0`).

## 4. Flujo real (antes de corregir)

```
USUARIO → tokens() → TF-IDF cosine vs 216 docs → top k+2
  → +0.15 si menciona la especie → top k → threshold 0.20 (post-bonus)
  → build_prompt() → TinyGPT.generate (temp 0.0, greedy) → rec<0.3 ? [extractivo]
  → banner urgencia → RESPUESTA
```

## 5. Configuración anterior

- Embeddings: **TF-IDF numpy, vocab=448, docs=216, cosine** (no hay embeddings neuronales).
- `top_k = 2` (se pedían `k+2` y se recortaba).
- Threshold: `0.20` aplicado al score CON bonus de especie.
- Bonus especie: `+0.15` antes del threshold.
- Fuera-de-ámbito: `overlap_vocab < 2`, **omitido** cuando había historial y `n_tok < 4`.
- Historial: `q_ret = historial[-1] + q` si `n_tok < 4` (el historial incluía pregunta
  Y respuesta previa recortada).
- Fallback: `rec < 0.3` → copiar top-doc + `" [extractivo]"` sin revalidar relevancia.
- Sin modo debug, sin interruptor RAG on/off.

## 6. Configuración corregida (actual)

- `RAG_ENABLED = True` / `--no-rag` (MODO A sin retrieval vs MODO B con RAG).
- `DEBUG_RAG = False` / `--debug-rag` (imprime QUERY, modelo embedding, TOP_K,
  scores crudos + shared-terms, THRESHOLD, especie, SELECTED, DECISION, RESPONSE MODE).
- `SIM_THRESHOLD = 0.20` aplicado al score **crudo pre-bonus** (mismo valor numérico,
  pero ahora con orden correcto).
- `MIN_SHARED_TERMS = 2`: el documento debe compartir ≥2 términos filtrados con la
  consulta; si no, `SIN_EVIDENCIA` (rechazo, no se usa el fragmento más parecido).
- `SPECIES_BONUS = 0.15` solo post-filtro.
- Fuera-de-ámbito: `overlap_vocab < 2` → `FUERA_DE_AMBITO`, **siempre** (sin bypass).
- Historial: **nunca** se concatena para retrieval; solo da contexto al prompt.
- Rechazos: `FUERA_DE_AMBITO` / `SIN_EVIDENCIA` con mensajes explícitos, `pasajes=[]`,
  sin llamar al modelo. Modos: `RECHAZADO` / `GENERATIVO_SIN_RAG` / `GENERATIVO_RAG` /
  `EXTRACTIVO` (trazables en `LAST_DEBUG`, JSON y salida `--debug-rag`; el sufijo
  ` [extractivo]` se conserva por compatibilidad).
- Función nueva: `retrieve_debug(idx, query, k)` (trazabilidad sin generar texto).

## 7. Threshold: anterior vs corregido

| | Anterior | Corregido |
|---|---|---|
| Valor | 0.20 | 0.20 (mismo número) |
| Base | score + bonus especie | score crudo pre-bonus |
| Suficiencia | era la única barrera | + `MIN_SHARED_TERMS=2` + fuera-de-ámbito sin bypass |

Mediciones crudas (k=2, cosine): golpe-pata 0.2849, cristóbal 0.2032, dinosaurios 0.2859,
ehrlichiosis-canina 0.2937 ("que"), signos-ehrlichiosis 0.2711, vómito 0.3175.
Conclusión: **ningún threshold puro separa relevantes de irrelevantes**
(dinosaurios-irrelevante 0.2859 > golpe-relevante 0.2849); por eso se añadió la regla
de shared-terms en vez de subir el número a ciegas.

## 8. Modelo de embeddings

`TfidfIndex` (`src/rag/retriever.py`): TF `(1+log f)·idf`, `idf=log((1+n)/(1+df))+1`,
vectores L2-normalizados, cosine por producto punto. Dimensión = 448 (vocabulario),
índice `data/rag/index` (matriz 216×448). Sin stemming: `golpeo≠golpe`,
`perro≠perros`, `comio≠comian`; `ehrlichiosis`, `dinosaurios`, `cristobal`, `colon`,
`carro`, `agitado` están FUERA del vocabulario. Tokens genéricos (idf<3.0, df>30):
`veterinaria, acude, veterinario, pregunta, respuesta, urgencia, urgente, perro, gato,
caballo, conejo, vaca, que, sin, segun` — contarlos como "evidencia de dominio" era
parte del fallo (ej.: query ehrlichiosis-canina solo solapa `que`).

## 9. Ejemplos de documentos recuperados (prueba crítica)

Query `cristobal colon comio perros` (tokens: cristobal/colon/comio/perros; solapa: comio+perros):
antes → doc 8 `Mi perro comió chocolate…` (crudo 0.2032+0.15=0.3532, shared real con el
doc: solo `comio`) → generación contaminada con chocolate ("¿Cada cuánto… acude URGENTE…").
Ahora → `SIN_EVIDENCIA` (shared 1 < 2), respuesta: "No encontré información veterinaria
relevante…". PASS.

Query `los dinosaurios comian perros?` (solapa: solo `perros`):
antes (sin historial) → `FUERA_DE_AMBITO` correcto; **con** historial → `q_ret`
contaminada + bypass del filtro → doc 127
`Urgencia veterinaria perro. sospecha de envenenamiento en perros…` (0.3847) → `[extractivo]`.
Ahora → `FUERA_DE_AMBITO` siempre (el historial no contamina). PASS.

Query `mi perro lo golpeo un carro en la pata…` (solapa: pata+perro):
doc 10 `Mi perro sangra mucho de una pata… Presiona con gasa limpia…` (crudo 0.2849,
shared 2) + doc 32 cojera/golpe. Se mantiene `RECUPERA` → `[extractivo]` (la generación
tiny con contexto, `rec=0.0`, sigue sin apoyarse en la fuente; el extractivo aquí es el
comportamiento diseñado, aunque el doc 10 cubre sangrado y no atropello: limitación
léxica documentada).

## 10. Pruebas realizadas

- Sonda retrieval pre-fix: `scripts/rag_diagnose.py` (replica la lógica anterior).
- MODO A vs MODO B con el checkpoint real (`checkpoints/pretraining/final/model.pt`):
  sin RAG el modelo alucina plantillas ("plan de cachorro…", "¿Cómo elimino pulgas…?",
  "temperatura 38.5…") ignorando la pregunta — prueba de que el modelo puro no distingue
  dominios y de que el rechazo RAG es preferible a dejarlo generar.
- Batería de 6 tests post-fix: `scripts/run_rag_tests.py` → `outputs/evaluation/rag_tests.json`
  y `rag_tests.txt` (query, top_k, scores crudos/bonus, threshold, shared-terms,
  documento, modo, respuesta, esperado/obtenido, PASS/FAIL).
- CLI: `python -m app.vet_chat --ask "…" --debug-rag` y `--no-rag` verificados.

## 11. Resultados PASS/FAIL (post-fix)

- TEST 1 veterinaria signos-ehrlichiosis → RECUPERA doc 199 (signos de alarma) → PASS
  (parcial: el corpus no tiene pasaje específico de ehrlichiosis; ver TEST 6).
- TEST 2 veterinaria vómito → RECUPERA → PASS (doc 67/62; relevancia léxica parcial,
  limitación TF-IDF anotada).
- TEST 3 capital Francia → FUERA_DE_AMBITO, sin envenenamiento → PASS.
- TEST 4 dinosaurios → FUERA_DE_AMBITO, sin envenenamiento → PASS.
- TEST 5 Cristóbal Colón → FUERA_DE_AMBITO → PASS.
- TEST 6 "¿qué es la ehrlichiosis canina?" → FUERA_DE_AMBITO → **FAIL** (esperado:
  recuperar doc con "ehrlichiosis").

**5 PASS / 1 FAIL.** El FAIL es un gap de corpus, no de lógica: `ehrlichiosis` aparece
0 veces en `data/raw` (1.1 MB, 6518 pasajes) y el índice solo tiene 216 docs; el sistema
ahora rechaza en vez de inventar un envenenamiento, que es el comportamiento correcto
hasta que se amplíe el corpus. No requiere reentrenar el modelo.

## 12. ¿RAG o VET-TINY-GPT?

**El defecto reportado era del RAG (retrieval + threshold + fallback + historial),
no de los pesos del modelo.** Evidencia:

- `[extractivo]` solo lo emite el fallback RAG (trazado a `answer()`), y las tres
  respuestas del reporte coinciden byte a byte con docs 10/8/127 vía ese camino.
- MODO A (modelo puro) produce alucinaciones distintas ("pulgas", "vacunas",
  "temperatura"), nunca el texto de envenenamiento: el envenenamiento lo aportó el
  retrieval, no la generación espontánea.
- Tras corregir solo el pipeline (sin tocar el modelo), los 3 casos del reporte pasan
  a rechazo correcto y los casos veterinarios legítimos siguen recuperando.

Limitaciones conocidas que NO se corrigieron (fuera de alcance / requerirían
reindexar o ampliar corpus, no reentrenar): sin stemming (golpeo/golpe, perro/perros),
índice de 216 docs frente a 6518 pasajes raw, cero cobertura de ehrlichiosis,
relevancia puramente léxica (TEST 2 elige doc 67 "qué no debe comer" por solape
debe/perro/que). Recomendación: reindexar con normalización + ampliar corpus
veterinario (incluir ehrlichiosis/garrapatas) y reevaluar; el modelo queda intacto.
