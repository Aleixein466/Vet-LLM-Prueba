# VET-TINY-LLM — Documentación técnica

Documento único y vigente del proyecto: qué se construyó, qué tecnología se usó en cada
parte, por qué se eligió esa tecnología y qué se midió para aceptarla o descartarla.

**Estado: 2026-10-05.** Modelo desplegado: `checkpoints/pretraining/final/model.pt`
(17.62 M parámetros).

Complementarios: `docs/PROCESO_ENTRENAMIENTO.md` (detalle por etapa),
`docs/GUIA_PROCESO_COMPLETO.md` (bitácora de decisiones),
`outputs/evaluation/rag_diagnostic_report.md` (diagnóstico del RAG),
`PROJECT_REPORT.md` (informe por secciones de la especificación).

---

## 1. Qué es este proyecto, en una página

Un **asistente veterinario en español construido desde cero**: un modelo de lenguaje
completo, sin partir de ningún modelo preentrenado, entrenado en un equipo sin tarjeta
gráfica.

"No usar modelos preentrenados" es la condición del proyecto. Eso obliga a escribir a
mano la arquitectura del Transformer, el bucle de entrenamiento, la elección de
tokenizador y las etapas de ajuste y alineación. También implica que el modelo
resultante es diminuto: **17.6 millones de parámetros**, frente a los miles de millones
de un modelo comercial. La analogía correcta no es un médico, es un **estudiante de
primer año que se ha leído todos los apuntes de una asignatura y se los sabe de
memoria**: responde bien lo que estudió y se inventa cosas cuando le preguntas otra
cosa.

Esa limitación no se resolvió agrandando el modelo (imposible sin GPU), sino
**cambiando la arquitectura del sistema**: en lugar de pedirle al modelo que memorice
documentos clínicos, se le entrega el fragmento relevante en el momento de la pregunta.
Eso es Retrieval-Augmented Generation (RAG), y al final resultó ser la pieza más
importante del proyecto, por encima del propio entrenamiento.

### El problema que había que resolver

La primera versión (2026-09-28) se entrenó únicamente con texto generado por plantillas:
6518 documentos del mismo estilo repetido. El modelo aprendió bien ese estilo y nada
más. Con los documentos clínicos reales, la perplejidad llegó a **29 540**: es decir,
para el modelo esos documentos eran tan improbables como texto aleatorio. En la
práctica respondía inventando.

La corrección llegó el 2026-10-05 y tuvo dos partes:

1. **Incorporar los documentos reales al entrenamiento** (70 PDFs veterinarios).
2. **Añadir RAG con rechazo**, para que el sistema prefiera no responder antes que
   inventar.

Resultado: la perplejidad sobre texto real pasó de **29 540 a 172**, una mejora de
172×, sin perder respuesta a preguntas.

---

## 2. El punto de partida y las restricciones

| Elemento | Valor |
|---|---|
| Hardware | Intel UHD (sin GPU), 11.8 GB RAM, 4 threads de PyTorch |
| Sistema | Windows, unidad `D:` es USB (~1.6 MB/s de escritura) |
| Python | 3.11.6 |
| PyTorch | 2.14.0+**cpu** (build sin CUDA) |
| tiktoken | 0.14.0 |
| tokenizers | 0.23.2 |
| sentence-transformers | 6.1.0 |
| pypdf | 6.19.0 |
| numpy | 2.4.6 |
| Interfaz | Streamlit ≥ 1.30 |

Dos decisiones de entorno que condicionaron todo lo demás:

- **El entorno virtual se instaló en `C:`**, no en `D:`. El USB escribía a 1.6 MB/s y
  el venv en `D:` quedaba inutilizable. Script de arranque: `env-fast.ps1`.
- **Sin GPU, y eso tiene consecuencias técnicas.** No hay CUDA, ni AMP (precisión
  mixta), ni entrenamiento distribuido en varias máquinas, ni LoRA/PEFT. Todo el
  entrenamiento es CPU. Esto no es una preferencia: es lo que había.

---

## 3. El recorrido de los datos

Un modelo de lenguaje no aprende de documentos: aprende de una lista de números. Esta
etapa convierte PDFs y plantillas en tensores que se puedan leer rápido.

### 3.1 Corpus sintético (`src/data/corpus_sintetico.py`)

Generador propio de texto veterinario en español. Cruza **5 especies** (perro, gato,
caballo, vaca, conejo) con **15 temas clínicos**, más **12 protocolos de urgencia** y
un bloque de preguntas frecuentes. Produce 6518 documentos (1 097 883 caracteres
útiles) y cuatro archivos de supervisión:

| Archivo | Registros | Para qué sirve |
|---|---|---|
| `data/raw/vet-es-sintetico.txt` | 1.15 MB | texto para el entrenamiento base |
| `data/sft/sft_train.jsonl` | 238 pares pregunta/respuesta | ajuste fino |
| `data/sft/sft_val.jsonl` | 42 pares | evaluación (nunca vistos al entrenar) |
| `data/sft/prefs.jsonl` | 120 pares elegido/rechazado | alineación DPO |

Su valor real: contiene las respuestas **verificadas** (frecuencia cardiaca normal de
cada especie, temperatura, intervalos de desparasitación). Su limitación: al ser
plantillas, el modelo aprende el estilo, no el razonamiento.

### 3.2 Corpus real (`src/data/ingest_pdfs.py`)

70 PDFs de la carpeta `Datos/`: 15 documentos técnicos y 55 historias clínicas, 669
páginas en total. Tratamiento por bloques:

**Extracción.** `pypdf` lee el texto de cada página.

**Limpieza.** El PDF es un formato de impresión, no de datos, así que trae ruido:

- Normalización Unicode (NFC) y ligaduras tipográficas (`ﬁ` → `fi`).
- Borrado del carácter de reemplazo Unicode. Algunos PDFs tienen la tabla `ToUnicode`
  rota y devuelven `�` en cada tilde; dejarlo envenenaría el corpus y el índice de
  búsqueda.
- Desguionado de fin de línea: `veteri-\nnaria` → `veterinaria`.
- Colapso de puntos de índice (`Resumen....5` → `Resumen 5`).
- Descarte de páginas con menos de 200 caracteres (43 páginas) y de las que solo
  contienen un número de página.

**Anonimización.** Solo sobre las 55 historias clínicas, que sí tienen datos
personales. Expresiones regulares para correo electrónico, teléfono (formato
colombiano), cédula, fecha-hora, dirección y nombre del propietario, reemplazados por
marcadores: `[EMAIL]`, `[TEL]`, `[ID]`, `[DIRECCIÓN]`, `[PROPIETARIO]`. Resultado: **353
reemplazos**. Los documentos técnicos no se tocan: no contienen datos de pacientes y su
valor es la referencia.

**Fragmentado (chunking).** Se corta en bloques de 900 caracteres con 150 de
solapamiento, y si un párrafo excede el tamaño se recorta por frases. El solapamiento
existe para que una idea que cae justo en el corte no se pierda. Resultado: **1639
fragmentos** de 727 caracteres de media.

**Salidas.** `docs_real.jsonl`, `chunks_real.jsonl`, el texto plano
`vet-real-limpio.txt` (453 529 tokens) y `sft_real.jsonl` con 1574 pares
pregunta/respuesta de tipo extractivo.

> Nota técnica: el pipeline **no tiene OCR**. Si un PDF fuera un escaneo sin capa de
> texto, se omitió. En esta corrida no se omitió ninguno (`omitidos_escaneado: []`),
> pero es una limitación real para futuras ingesta.

### 3.3 Empaquetado binario (`build_dataset.py`, `build_dataset_bpe.py`)

El texto se trocea por párrafos, se tokeniza, se le añade el token de fin de documento
y se concatenan todos los identificadores en un array plano de enteros de 16 bits que
se escribe con `.tofile()`. Se reserva el último 10 % para validación.

El motivo es práctico: 1.5 MB se cargan en memoria con `np.fromfile`, sin analizar
JSON. Y `uint16` alcanza porque el vocabulario mayor (50257) cabe en 16 bits (máximo
65535); el código lo verifica con una aserción.

| Corpus | Tokenizador | Tokens totales | train / val |
|---|---|---|---|
| Solo sintético | tiktoken gpt2 | 408 099 | 367 289 / 40 810 |
| Mixto (sintético + real) | tiktoken gpt2 | **860 629** | 774 566 / 86 063 |
| Mixto | BPE propio 16k | 519 809 | 467 828 / 51 981 |

El BPE propio necesita **40 % menos tokens** para el mismo texto: el vocabulario de
50 257 tokens de GPT-2 está lleno de fragmentos en inglés que en español veterinario no
se usan, así que cada palabra se parte en más trozos.

---

## 4. Tokenización: la decisión que más condicionó el proyecto

Tokenizar es convertir texto en números. La elección del vocabulario determina la
longitud de las secuencias, el costo de cada paso de entrenamiento y, en última
instancia, la cantidad de parámetros del modelo. Es la decisión con más consecuencias
de todo el proyecto.

### 4.1 tiktoken gpt2 — el desplegado

`tiktoken.get_encoding("gpt2")`, 50 257 tokens, fin de documento `<|endoftext|>` (50256).

**Por qué:** reutilizar un vocabulario ya entrenado por OpenAI significa heredar una
segmentación probada a gran escala. No hubo que entrenar nada ni invertir tiempo en
ello.

**Coste:** 40 % de tokens innecesarios en este dominio, y una restricción técnica: al
usar `uint16` para el almacenamiento, el vocabulario no puede pasar de 65 535.

### 4.2 BPE propio de 16 000 — la variante experimental

Librería `tokenizers` de Hugging Face: modelo BPE, pre-tokenizador **ByteLevel**,
entrenador `BpeTrainer` con 16 000 entradas más `[PAD]`, `[UNK]` y `[EOS]`, alimentado
con el corpus veterinario más los documentos reales limpios.

**Por qué ByteLevel:** convierte cada carácter en una secuencia de bytes antes de
aplicar BPE. Eso resuelve el problema clásico de los vocabularios cerrados: si una palabra
contiene una letra o un símbolo que el vocabulario nunca vio, en lugar de marcarla como
desconocida (`[UNK]`) la reconstruye byte a byte. Con tildes, ñ y símbolos de grado
Celsius, esto es obligatorio.

**Resultado:** mismo esqueleto de red, pero con 8 850 688 parámetros en vez de
17 620 480, y el mejor resultado en preguntas del conjunto sintético (0.952 de
precisión frente a 0.762). Aun así **no se desplegó**: su validación mostraba
sobreajuste creciente y nunca se evaluó contra preguntas reales. Queda disponible con
`--cfg configs/tiny-18m-bpe.yaml`.

**Consecuencia práctica:** cambiar el vocabulario **invalida todos los checkpoints**.
Por eso la variante BPE vive en directorios separados (`checkpoints/tiny-18m-bpe*`) y
no toca el modelo en producción. Era la alternativa de menor riesgo.

### 4.3 El detalle que costó un bug: codificación conjunta con máscara

Al preparar datos de ajuste fino, la forma ingenua sería codificar por separado:

```
encode(prompt) + encode(" " + respuesta)
```

Eso está mal, y el motivo es sutil. BPE fusiona tokens **según el texto que ve a su
alrededor**. La palabra "urgente" al final de un prompt y la palabra "urgente" seguida
de un espacio seguido de "acude" no se parten igual. Al concatenar dos codificaciones
independientes se introduce una **costura** que el modelo nunca vio durante el
entrenamiento base, y esa costura no es ruido menor: es texto que el modelo no sabe
interpretar.

La solución en `src/tokenizer/tok.py:25-43` es codificar **prompt y respuesta juntos**,
igual que en el entrenamiento base, y luego localizar el borde verificando que al
decodificar los primeros *k* tokens se recupera exactamente el prompt. Si el borde no
es reproducible, el par se descarta y se cuenta (2 de 1574 en la última corrida) en
lugar de enviar etiquetas desalineadas.

---

## 5. La arquitectura del modelo (`src/model/gpt.py`)

Un Transformer decoder-only (solo decodificador) al estilo GPT-2, escrito a mano en
PyTorch. Configuración: **6 capas, 4 cabezas de atención, dimensión 256, ventana de 128
tokens, dropout 0, sin sesgos, pesos atados**. 17 620 480 parámetros, cifra verificada
ejecutando `TinyGPT(...).count_params()`.

### 5.1 Self-Attention causal: el corazón

Es el mecanismo que permite que la posición 57 del texto mire a la posición 12. Tres
pasos:

1. **Proyección.** Una sola capa densa transforma 256 dimensiones en 768 (tres veces
   256: consulta, clave y valor, que se calculan juntas y luego se separan).
2. **Puntuaciones.** Se calcula el producto escalar entre consultas y claves, dividido
   por la raíz de la dimensión de cabeza (evita que los valores crezcan sin control).
3. **Máscara causal.** Se aplica una matriz triangular (`torch.tril`) que pone `-infinito`
   en las posiciones futuras, de modo que **cada token solo puede atender a los que le
   preceden**. Esto es lo que convierte un transformador en un modelo de lenguaje: no
   puede "ver" la respuesta mientras la genera.

La máscara se precalcula una vez y se guarda como buffer, así que no se recalcula en
cada paso.

**Lo que no se implementó y por qué importa:** sin FlashAttention ni kernels
optimizados. A una ventana de 128 tokens y en CPU, la versión directa es más rápida que
una versión troceada: la optimización tiene sentido cuando el costo domina, y aquí el
costo real está en la cantidad de pasos, no en la velocidad de cada atención. La
atención es, eso sí, la operación más cara del modelo: crece con el cuadrado de la
longitud de secuencia.

### 5.2 La red feed-forward

Dos capas densas: 256 → 1024 → 256, con activación **GELU** en el medio. El factor 4 es
el estándar. GELU en lugar de ReLU porque su derivada es continua en el origen, lo que
importa cuando se entrenan 6 capas con lotes pequeños: ReLU se apaga con facilidad en
este régimen.

### 5.3 Normalización y conexiones residuales

Cada bloque trabaja en orden **pre-normalización**:

```
x = x + atencion(normalizar(x))
x = x + feedforward(normalizar(x))
```

Es decir, se normaliza **antes** de cada subcapa y se suma la entrada. GPT-2 original
usaba normalización después; la versión previa resulta más estable sin necesidad de
calentamiento largo, lo cual importa con solo 6 capas y lotes de 16×128 en CPU. La
conexión residual es lo que permite apilar profundidad sin que el gradiente desaparezca.

### 5.4 Pesos atados

La capa de salida comparte matriz con la tabla de embeddings: `lm_head.weight =
wte.weight`.

**Por qué:** ahorra 12.87 M parámetros (el 73 % del modelo son embeddings) y, más
interesante, actúa como regularizador: entrada y salida comparten representación, así
que el modelo entrena dos tareas a la vez con los mismos pesos.

### 5.5 Desglose de parámetros

| Componente | Parámetros | % del total |
|---|---|---|
| Embeddings de token (50257×256) | 12 865 792 | 73.0 |
| Los 6 bloques | 4 721 664 | 26.8 |
| ↳ atención por bloque | 262 144 | |
| ↳ feed-forward por bloque | 524 288 | |
| ↳ 2 normalizaciones por bloque | 512 | |
| Embeddings de posición (128×256) | 32 768 | 0.19 |
| Normalización final | 256 | ~0 |
| Capa de salida | 0 (pesos atados) | 0 |
| **Total** | **17 620 480** | |

Con el BPE de 16k: **8 850 688** con la misma estructura.

### 5.6 Embeddings de posición, y lo que no se hizo

La posición se codifica con una tabla aprendida de 128×256. La alternativa moderna
(rotaciones, RoPE) codifica la posición en el propio vector en lugar de en una tabla.
Con una ventana de 128 la tabla completa son 32 768 parámetros: irrelevante en un
modelo de 17.6 M. Es una decisión de alcance, no de calidad.

Conviene ser explícito porque la web del proyecto menciona RoPE, GQA, RMSNorm y
SwiGLU: **ninguno está implementado**. La arquitectura es deliberadamente la clásica de
GPT-2. Añadir cualquiera de ellos obliga a reentrenar desde cero, porque los pesos no
son compatibles.

---

## 6. Entrenamiento base: predecir la siguiente palabra

`src/training/train.py`. Objetivo: dado un fragmento de texto, predecir qué palabra
viene después. Es la única forma de aprender lenguaje sin intervención humana, y la
base sobre la que se apoyan las dos etapas siguientes.

### 6.1 El objetivo

Se toma una ventana de texto, se desplaza una posición y se mide el error de la
predicción con **entropía cruzada** (`cross_entropy` en `gpt.py:95`). La entropía
cruzada penaliza con más fuerza equivocarse entre opciones parecidas que entre
opciones lejanas, que es exactamente la estructura del lenguaje.

### 6.2 Cómo se forman los lotes

Se extraen 16 ventanas aleatorias de 128 tokens de todo el corpus, con solapamiento de
una posición entre entrada y objetivo. No se usa un orden fijo: las ventanas aleatorias
evitan que el modelo vea el corpus en la misma secuencia dos veces, que en un corpus
pequeño es la causa clásica de sobreajuste.

Con 774 566 tokens de entrenamiento y 2048 tokens por paso, 6000 pasos procesan
**12.3 millones de tokens**, unas 16 pasadas sobre el corpus. Es repetición
deliberada: el corpus es pequeño y cada token debe verse suficientes veces.

### 6.3 El optimizador y el plan de aprendizaje

**AdamW** con tasa 3e-4 y decaimiento de peso 0.01. AdamW se eligió sobre Adam porque
**desacopla** el decaimiento de la actualización por gradiente. En un modelo con
muchas magnitudes pequeñas (las normalizaciones), el decaimiento acoplado se comporta
como un gradiente extra y distorsiona la escala.

**Plan coseno con calentamiento** (`lr_at`, `train.py:68-74`): los primeros 200 pasos
suben la tasa de forma lineal desde cero, y después decae siguiendo una coseno hasta
llegar a cero. El calentamiento evita el pico de pérdida del inicio, cuando los pesos
aleatorios producen gradientes desorientados.

**Recorte de gradientes** a norma 1.0: en CPU, un lote que mezcla secciones de texto muy
distintas puede producir un pico de gradiente que arruine el paso.

### 6.4 Checkpoints reanudables

Cada 500 pasos se guarda el estado; `last.pt` incluye modelo, optimizador, paso y
pérdida. Con `resume: true`, la corrida reentrena **continuando** donde estaba, con el
optimizador cargado: el reentrenamiento de 3000 a 6000 pasos no repitió las 3 horas
previas.

### 6.5 Qué dio cada corrida

| Corrida | Pasos | Tiempo | Pérdida train | Pérdida val | Par tokens |
|---|---|---|---|---|---|
| Solo sintético (28-09) | 3000 | 139 min | 0.093 | 0.095 | 17.62 M |
| Mixto, reentreno (05-10) | 3000 → 6000 | ~3.3 h (reanudado) | 1.677 | 5.081 | 17.62 M |
| BPE 16k (05-10) | 3000 | 100 min | 1.939 | 7.431 | 8.85 M |

Cómo leer esto sin engañarse: la pérdida de validación del 0.095 al 5.081 **no es una
regresión**. Son dos problemas distintos. El 0.095 se medía sobre las plantillas sintéticas
que el modelo se había memorizado (perplejidad 1.11: acertaba casi todas las palabras).
La comparación legítima es contra el mismo corpus real, y ahí el resultado es 29 540
contra **172**.

La variante BPE sí muestra sobreajuste real dentro de su propia corrida: su validación
sube de 6.97 a 7.43 a partir de unos 750 pasos, unas 13 pasadas sobre 468 k tokens.

---

## 7. Ajuste fino supervisado (SFT)

`src/training/sft.py`. Un modelo entrenado solo con "continúa el texto" no sabe
responder a una pregunta: solo sabe continuar. SFT es lo que le enseña la forma de una
respuesta.

### 7.1 El cambio conceptual: la máscara

El ajuste usa la misma entropía cruzada, con una modificación en los labels: las
posiciones del prompt se marcan con `-100` y PyTorch las ignora. **Solo la respuesta
cuenta.** Es el único cambio conceptual respecto al entrenamiento base y es lo que
convierte el modelo en algo que responde.

### 7.2 La composición del lote, y por qué importa

Cada lote de 8 ejemplos mezcla tres fuentes:

- **4 pares sintéticos** (respuestas verificadas, formato de pregunta y respuesta).
- **2 ventanas de replay** del corpus de entrenamiento base, con pérdida completa.
- **2 pares reales** extraídos de los fragmentos clínicos.

El **replay** no es un detalle: sin él el modelo olvidaba el material base y la
perplejidad saltaba de 1.1 a 252 (olvido catastrófico). Es el mismo principio que
mantiene una capa de datos anteriores cuando ajustas con datos nuevos.

La pérdida final es el promedio simple de los tres términos,
`(síntesis + replay + real)/3` (`sft.py:121`). Tasa 2e-5, 40 pasos. Observado:
6.52 → 3.79.

### 7.3 Por qué se descartó, y por qué eso es un resultado válido

Después del ajuste: perplejidad 300 (frente a 172) y **precisión en preguntas de 0.0**.

El diagnóstico es consistente en las tres variantes probadas: los pares extractivos
(la respuesta es literalmente un fragmento de 1200 caracteres) sacan al modelo de 17 M
de la distribución de preguntas cortas y accionables. Empieza a devolver bloques de
texto en lugar de una respuesta directa.

La conclusión honesta: **la técnica es correcta, los datos no encajaban con el modelo.**
Con 40 pasos de presupuesto y 17 M de parámetros, ajustar con pares extractivos
destruye más de lo que construye. Se descartó y no se desplegó.

Lo interesante es la reacción: el problema se resolvió **en la capa de inferencia, no
reentrenando**. Si el modelo no puede sostener una respuesta, se le entrega la respuesta
correcta ya redactada desde el documento (mecanismo extractivo). Es un cambio de
arquitectura barato y reversible frente a un reentrenamiento de horas.

---

## 8. Alineación por preferencias (DPO)

`src/training/dpo.py`. Implementación mínima de Direct Preference Optimization, sin
modelo de recompensa.

**Cómo funciona.** Se cargan dos copias del modelo: la **política** (entrenable) y una
**referencia** (congelada, misma checkpoint). Para cada par (respuesta elegida,
respuesta rechazada) se calcula la log-probabilidad de cada respuesta bajo ambas. La
ventaja es la diferencia entre ambas, menos la misma diferencia bajo la referencia:

```
ventaja = (logp_elegida − logp_rechazada) − (logp_elegida_ref − logp_rechazada_ref)
```

Restar la referencia es lo que convierte el objetivo en "cambia respecto a donde
estabas" en lugar de "sé bueno en términos absolutos". La pérdida es
`-logsigmoid(β · ventaja)`
con β = 0.1, sobre 120 pares, 30 pasos.

**Qué pasó.** La ventaja llegó a **+155 a +182** y la pérdida a **0.000** desde el paso
10. Es decir: el modelo separa las respuestas buenas de las malas con un margen tan
grande que el gradiente es numéricamente nulo. No le queda nada que aprender.

El motivo está en los datos: las preferencias sintéticas contraponen una respuesta
correcta contra una absurdamente peligrosa ("dale chocolate, es bueno para el perro").
Un modelo ya entrenado las distingue sin dificultad. Faltan preferencias **cercanas**,
donde dos respuestas sean ambas plausibles y la diferencia sea de matiz. Es una etapa
demostrativa: la señal funciona, los datos no la ofrecen.

PPO y GRPO quedaron documentados como extensión, no implementados. DPO evita justamente
el modelo de recompensa que PPO necesitaría, que era la opción viable en CPU.

---

## 9. Evaluación: por qué una sola métrica no basta

Tres medidas complementarias, porque cada una ve un fallo distinto.

**1. Perplejidad.** `exp(pérdida media)` sobre 50 lotes de validación. Mide
qué tan bien el modelo predice texto. Solo comparable dentro del mismo vocabulario y
el mismo conjunto: las perplejidades de tiktoken y BPE no se pueden comparar.

**2. Precisión en preguntas retenidas.** 42 preguntas de `sft_val` que el modelo nunca
vio, generadas con decodificación voraz. Se mide el **recall**: qué fracción de las
palabras clave de la respuesta esperada aparece en la generada. Y la precisión: fracción
de preguntas con recall ≥ 0.4.

**3. Consistencia** (`consistency.py`). Agrupa las preguntas por tema y especie,
genera cada reformulación y mide el **Jaccard** (solapamiento de palabras clave) entre
respuestas a la misma pregunta dicha de otra forma. Detecta si el modelo se contradice
consigo mismo. Resultado: 0.609 en la base, 0.15 tras SFT.

### 9.1 Resultados de todos los candidatos

| Modelo | Perplejidad | Recall | Precisión | Jaccard | Decisión |
|---|---|---|---|---|---|
| Base vieja sobre datos reales | 29 540 | 0.72 | 0.762 | — | superada |
| **Base reentrenada (tiktoken)** | **172** | 0.705 | **0.762** | 0.609 | **desplegada** |
| SFT sobre datos reales | 300 | 0.018 | 0.0 | 0.15 | descartado |
| Base BPE 16k (8.85 M) | 1 773* | 0.881 | 0.952 | — | experimental |
| SFT BPE 16k | 2 418* | 0.119 | 0.143 | — | descartado |
| DPO | — | — | — | — | demostrativo |

\* no comparable con las de tiktoken: vocabulario y tokenización distintos.

**Por qué este diseño de métricas evitó un error.** La base vieja y la reentrenada
tienen exactamente la misma precisión (0.762). Si esa hubiera sido la única métrica, la
decisión habría sido indiferente y probablemente se habría conservado el modelo viejo,
que sobre texto real tiene perplejidad 29 540. La perplejidad sobre el corpus real es
lo que reveló la diferencia.

---

## 10. RAG: donde el sistema realmente resuelve el problema

Un modelo de 17 M puede memorizar mucho, pero no puede memorizar 1639 fragmentos
clínicos y acertar cuál necesita en cada momento. RAG (Retrieval-Augmented Generation)
resuelve exactamente eso: **se busca el fragmento relevante y se lo entrega al modelo
como contexto antes de que responda.**

El detalle importante: en este proyecto RAG no es un añadido, es **el mecanismo que
sustituye al ajuste fino**. Fue la respuesta a que SFT destruyera el modelo.

### 10.1 Búsqueda léxica: TF-IDF

`src/rag/retriever.py`. Implementación propia sobre numpy, sin scikit-learn ni base de
datos vectorial, en un solo archivo.

- **Peso del término:** `1 + log(frecuencia)`. Un término que aparece 100 veces no
  aporta 100 veces más información que uno que aparece una.
- **Peso inverso:** `log((1+n)/(1+df)) + 1`. Un término presente en todos los
  documentos no sirve para distinguirlos.
- Los vectores se normalizan y la similitud es el producto punto, que con vectores
  normalizados equivale al coseno.

**Por qué normalizar acentos:** se usa `unicodedata` para quitar los diacríticos
("atención" se indexa como "atencion"), porque los PDFs traen
la tabla `ToUnicode` rota y las tildes llegan corruptas. Un índice que no normaliza
hace fallar consultas por una tilde.

Estado actual: vocabulario de **17 278 términos**, **2061 pasajes**.

### 10.2 El diagnóstico que mejoró todo el RAG

En septiembre de 2026 el sistema respondió con contenido de envenenamiento a
preguntas sobre dinosaurios y sobre Cristóbal Colón. El informe completo está en
`outputs/evaluation/rag_diagnostic_report.md`; la causa fue el **pipeline de
recuperación, no los pesos del modelo**. Cuatro defectos combinados:

1. **El umbral se comprobaba después de sumar el bonus de especie.** Cualquier consulta
   con una sola palabra de vocabulario pasaba el filtro.
2. **No existía regla de relevancia.** Bastaba compartir un token con un documento.
3. **El historial contaminaba la búsqueda.** Si la consulta era corta, se le pegaba la
   respuesta anterior y se saltaba el filtro de ámbito. La tercera pregunta de una
   conversación heredaba el contexto equivocado.
4. **El respaldo extractivo era incondicional.** Si la generación no se apoyaba en la
   fuente, se devolvía la fuente... aunque la fuente fuera irrelevante.

El hallazgo más útil del diagnóstico: **ningún umbral numérico separa lo relevante de
lo irrelevante**. Midiendo los puntajes crudos, la consulta irrelevante sobre
dinosaurios (0.2859) puntuaba **más alto** que una consulta veterinaria legítima
(0.2849). Por eso subir el número a ciegas no funcionaba y la solución fue una regla
estructural: el documento debe compartir al menos 2 términos con la consulta, y el
umbral se aplica **antes** del bonus de especie.

Configuración actual y el orden de las comprobaciones, que es lo que corrigió el
problema:

```
tokens → ¿vaga? → fuera de ámbito (<2 términos en vocabulario) → RECHAZAR
      → recuperación TF-IDF, k=2
      → filtro: score ≥ 0.18 (crudo) Y ≥ 2 términos compartidos
      → bonus de especie +0.15 (solo después de pasar el filtro)
      → generar con el fragmento como contexto
      → ¿la respuesta se apoya en la fuente? (recall léxico ≥ 0.3)
         no → devolver la fuente marcada [extractivo]
```

### 10.3 Rescate semántico con doble puerta

TF-IDF solo encuentra coincidencias literales. Si el usuario escribe "mi can vomita
sangre" y el documento dice "vómitos con sangre", no hay coincidencia.

`src/rag/embeddings.py` añade un modelo de embeddings
`paraphrase-multilingual-MiniLM-L12-v2` (384 dimensiones, multilingüe), que convierte
texto en vectores donde el significado cercano implica distancia corta. Se usa
**solo cuando TF-IDF rechaza**.

**Por qué una segunda puerta.** El rescate semántico tenía su propio problema: sin
restricción, "Colón" puntúa 0.68 y "dinosaurios" 0.55, y ambos pasaban como consulta
veterinaria. La corrección exige **score alto Y solapamiento real con el vocabulario**:

- score ≥ 0.55 con ≥ 4 términos en común, **o**
- score ≥ 0.72 con ≥ 3 términos en común.

Un término compartido es letra y palabra, no solo semántica.

### 10.4 Comportamientos de seguridad

- **Fuera de ámbito**: menos de 2 términos del vocabulario veterinario → respuesta fija
  de fuera de alcance, sin llamar al modelo. Nunca se inventa.
- **Sin evidencia**: hay vocabulario pero ningún fragmento pasa los filtros → lo dice
  explícitamente. La decisión de diseño es **rechazar en lugar de devolver el fragmento
  más parecido**, porque ese fragmento es la fuente de las alucinaciones originales.
- **Consulta vaga**: pide concretar especie, síntoma y desde cuándo.
- **Sin evidencia con respaldo**: si la búsqueda semántica tampoco pasa, se rechaza igual.
- **Emergencias**: banner determinista de urgencia en consultas de intoxicación,
  convulsiones, atropellos, etc.
- **Sin contaminación de historial**: el historial da contexto al prompt pero nunca se
  concatena para la búsqueda.

### 10.5 Resultado de las pruebas

Batería `scripts/run_rag_tests.py`, con comparación explícita entre el modelo solo y el
modelo con RAG: **7 de 7 correctas**.

Las pruebas que importan no son las veterinarias (esas ya pasaban), sino las que
verifican que el sistema **rechaza**:

- "¿Cuál es la capital de Francia?" → sin evidencia, sin envenenamiento.
- "¿Quién fue Cristóbal Colón?" → fuera de ámbito.
- "¿Los dinosaurios comían perros?" → fuera de ámbito.
- "Mi can vomita sangre y tiene diarrea" → recupera; **es el caso que el TF-IDF solo
  rechazaba y el rescate semántico resuelve** (parvovirosis, 0.71).

La prueba que documenta el problema original sigue siendo válida: el modelo **sin** RAG
responde con plantillas ("plan de cachorro…", "¿Cómo elimino pulgas…?") ignorando la
pregunta. Rechazar es mejor que eso.

---

## 11. Inferencia y decisiones de generación

`TinyGPT.generate` soporta temperatura, top-k, top-p y penalización por repetición. La
configuración usa **decodificación voraz** (temperatura 0) con penalización 1.1.

**Por qué voraz y no muestreo:** hallazgo medido en este proyecto. Con un modelo de
17 M, muestrear con top-k o top-p produce "ensalada de tokens": extraer una muestra de
una distribución que apenas diferencia entre opciones plausibles introduce errores de
composición que el modelo no puede corregir. La penalización por repetición evita que
se atasque en un bucle dentro de los 80 tokens generados.

El contexto se recorta a los últimos 127 tokens (la ventana del modelo es 128 y hace
falta una posición para predecir). Cuando aparece el token de fin, la generación se
corta ahí.

---

## 12. La aplicación web

`run_web.bat` → `http://localhost:8501`, sobre Streamlit.

**Seis páginas:** Chat (con RAG, historial multi-conversación, selección de especie),
Generación libre, Evaluación (lee los archivos reales de métricas en `outputs/`, no
números escritos a mano), Arquitectura (muestra lo que el modelo tiene y lo que no
tiene), Training (informe de la corrida real y curva de pérdida) e Historia clínica
(formulario con apoyo educativo, explícitamente sin diagnósticos).

Controles técnicos: selector de checkpoint, temperatura, top-p, tokens máximos y
penalización por repetición. Límite de 150 tokens en CPU. El modelo se carga una sola
vez y se cachea entre páginas.

---

## 13. Tecnologías, ordenadas por peso real en el resultado

No por popularidad, sino por cuánto cambió el comportamiento del sistema.

### Nivel 1 — explican por qué el sistema funciona

| Tecnología | Qué resuelve |
|---|---|
| Transformer decoder-only con atención causal | Sin máscara causal no hay modelo de lenguaje; sin atención no hay relación entre palabras. |
| Tokenización BPE | Define la longitud de las secuencias, el costo por paso y la separación entre entrenamiento y validación. Cambiarla invalida los checkpoints. |
| Predicción de siguiente token + entropía cruzada | El objetivo único del que SFT y DPO son reformulaciones. |
| AdamW + coseno con calentamiento + recorte | Lo que permitió 6000 pasos en CPU sin divergir. |
| SFT con máscara y replay | Define qué significa "responder"; el replay evita el olvido catastrófico. |
| Evaluación múltiple | Sin perplejidad + precisión + consistencia, la decisión de despliegue habría sido errónea. |
| RAG con TF-IDF y rescate semántico | Aporta la evidencia que un modelo de 17 M no puede memorizar. Es la tecnología que más valor aporta al final. |

### Nivel 2 — relevantes para la reproducibilidad

| Tecnología | Para qué |
|---|---|
| Checkpoints reanudables | Permite el reentreno 3000→6000 sin repetir 3 horas. |
| Empaquetado binario `uint16` | 1.5 MB se cargan en memoria sin analizar nada. |
| Configuración YAML declarativa | Un archivo gobierna modelo, datos, entrenamiento, SFT, DPO, inferencia y evaluación; permite la variante BPE cambiando pocas líneas. |
| Codificación conjunta con máscara | Corrección de un bug real de segmentación de tokens. |
| DPO con referencia congelada | Implementado y medido; sin efecto por datos triviales. |

### Nivel 3 — de apoyo

| Tecnología | Para qué |
|---|---|
| pypdf + expresiones regulares de anonimización | Habilita los 70 PDFs reales y protege los datos personales. |
| Streamlit | Interfaz; no toca el modelo. |
| safetensors | Formato de checkpoint para la web. |
| CPU-only (torch 2.14+cpu) | Restricción del entorno, no decisión. Sin CUDA, AMP, sin entrenamiento distribuido, sin LoRA/PEFT. |

### No implementado, y por qué importa

- **RoPE, GQA, RMSNorm, SwiGLU** — la arquitectura es deliberadamente GPT-2 clásica.
  Añadir cualquiera obliga a reentrenar desde cero.
- **FlashAttention y kernels optimizados** — innecesario a 128 tokens en CPU.
- **PPO / GRPO** — documentados como extensión.
- **Modelos de recompensa** — DPO los evita por diseño; era la opción viable en CPU.
- **OCR** — los PDFs sin capa de texto se omiten en la ingesta.
- **Stemming en el recuperador** — "golpeo" y "golpe" son términos distintos para el
  índice. Limitación conocida que afecta la relevancia, no la corrección.

---

## 14. Limitaciones honestas

- **17.6 M parámetros y 860 k tokens** es un modelo que memoriza mucho más de lo que
  generaliza. Funciona en su dominio, se inventa fuera de él.
- **La evaluación de preguntas viene del generador sintético** (42 preguntas). Mide
  fidelidad al estilo del corpus, no criterio clínico. Por eso la variante BPE saca 0.952
  y aun así no se despliega: su validación mostraba sobreajuste y nunca se probó
  contra preguntas reales.
- **Perplejidades entre tokenizadores distintos no son comparables.**
- **DPO no aporta señal** con preferencias triviales.
- **Relevancia puramente léxica** en el recuperador: sin stemming, y con el riesgo
  documentado de elegir un documento por coincidencia de términos genéricos.
- **Sin GPU**: escalar a un corpus real grande requiere cómputo que no está disponible
  aquí.

---

## 15. Cómo ejecutarlo

```powershell
# 0) entorno (el venv va en C:, no en el USB D:)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 1) datos
python src/data/corpus_sintetico.py --n-docs 4000
python src/data/ingest_pdfs.py
python src/tokenizer/prepare_tokenizer.py

# 2) tokenización y dataset
python src/data/build_dataset.py

# 3) entrenamiento base
python src/training/train.py                    # reanuda solo si el checkpoint existe

# 4) ajuste y alineación
python src/training/sft.py
python src/training/dpo.py

# 5) evaluación
python src/evaluation/eval.py
python src/evaluation/consistency.py

# 6) variante BPE (directorios separados, no toca el desplegado)
python src/tokenizer/train_bpe.py
python src/data/build_dataset_bpe.py
python src/training/train.py --cfg configs/tiny-18m-bpe.yaml

# 7) RAG y aplicación
python src/rag/retriever.py
python src/rag/embeddings.py --build
python app/vet_chat.py --ask "¿Cada cuánto desparasito a mi gato?"

# web
.\run_web.bat                                   # http://localhost:8501
```

---

## 16. Mapa del código

| Ruta | Contenido |
|---|---|
| `src/data/` | Generación de corpus, ingesta de PDFs, construcción de dataset |
| `src/tokenizer/` | Wrapper de tiktoken, entrenamiento del BPE propio, codificación con máscara |
| `src/model/gpt.py` | Arquitectura completa del Transformer |
| `src/training/` | Entrenamiento base, SFT, DPO y sus variantes con BPE |
| `src/evaluation/` | Perplejidad, preguntas de retención, consistencia |
| `src/rag/` | Índice TF-IDF e índice semántico |
| `src/inference/` | Generación y chat por terminal |
| `app/` | Aplicación de terminal con RAG, aplicación web Streamlit y utilidades |
| `scripts/` | Pruebas de RAG, diagnóstico, pipeline nocturno, monitor |
| `configs/` | Configuración YAML de las dos variantes |
| `checkpoints/` | Pesos: base, SFT, DPO, variante BPE y el punto de despliegue |
| `outputs/` | Métricas, informes y ejemplos de generación reales |
| `logs/` | Salida de consola de cada corrida (evidencia) |
