# Altur Challenge — Detección de voz sintética en llamadas telefónicas

Sistema que, dada una llamada estéreo 8 kHz (canal 0 = llamante, canal 1 = agente),
decide si el llamante es humano o una voz sintética, y expone `POST /detect`.

## Guía rápida (de cero a corriendo)

Los modelos ya entrenados (`model.pkl`, `weights/neural_cnn.pt`) vienen incluidos
en el repo, así que **no hace falta el dataset ni reentrenar** para levantar el
sistema. Elige una de las dos rutas:

### 0) Clonar el repositorio

```bash
git clone https://github.com/ChemiRC/retoAltur.git
cd retoAltur
git checkout integracion-backend
```

### Opción A — Con Docker (recomendado, no instala nada más)

```bash
docker compose up --build
```

- API: http://localhost:8000 (probar con `curl http://localhost:8000/health`)
- Dashboard: http://localhost:8501

Para bajar los contenedores: `docker compose down` (Ctrl+C también funciona si
lo dejaste en primer plano).

### Opción B — Local, sin Docker

```bash
# 1) Entorno virtual
python -m venv .venv
.venv\Scripts\activate        # Windows (usar `source .venv/bin/activate` en Linux/Mac)

# 2) Dependencias
pip install -r requirements.txt

# 3) Levantar la API (desde la raíz del repo)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 4) (en otra terminal, opcional) Levantar el dashboard
streamlit run frontend.py
```

- API: http://localhost:8000
- Dashboard: http://localhost:8501

### Verificar que funciona

```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true,"neural_loaded":true,"threshold":0.5}
```

### Solo si vas a reentrenar

Necesitas el dataset de audio (~640 MB, no está en git — ver [Setup](#setup)
más abajo) y correr los scripts de `scripts/` — ver [Entrenamiento](#entrenamiento).

---

## Cómo funciona y por qué lo hicimos así

En una frase: **dos modelos que se equivocan de forma distinta, votando juntos.**
Cada llamada pasa por dos clasificadores independientes y sus probabilidades se
promedian (50/50) para dar el veredicto final:

1. **Un GBM (`HistGradientBoosting`)** mira la llamada completa y calcula 171
   estadísticos: cómo suena la voz (MFCC, espectro, armónicos) y cómo se
   comporta en la conversación (¿el llamante responde con la latencia
   perfectamente pareja de una máquina, o con la variabilidad natural de un
   humano?). Es el modelo "explicable": si dispara la alarma, podemos señalar
   exactamente qué feature la disparó.
2. **Una CNN pequeña entrenada desde cero** mira la llamada en ventanas de 4
   segundos sobre el espectrograma log-mel, buscando micro-artefactos locales
   que el GBM (al promediar toda la llamada) diluye.

**¿Por qué dos modelos y no uno solo más grande?** Porque medimos que fallan en
condiciones distintas (ver la tabla de robustez más abajo): el GBM se degrada
bajo canal telefónico real o ruido fuerte, mientras la CNN se mantiene estable.
Si sus errores estuvieran correlacionados, promediarlos no ayudaría — pero no lo
están, así que el ensamble es más robusto que cualquiera de los dos solos.

**¿Por qué una CNN propia de 154k parámetros y no un modelo pre-entrenado como
Wav2Vec2?** Por una restricción real del reto: **latencia y viabilidad de
despliegue**. Wav2Vec2 tarda 10–30 s por llamada en CPU (inviable en una línea
telefónica) y no se puede fine-tunear con las 282 llamadas de entrenamiento que
tenemos sin GPU. Nuestra CNN entrena en ~50 min en CPU y responde en 0.19 s —
la decisión correcta para los datos y las restricciones que teníamos, no un
atajo por falta de capacidad.

**¿Por qué no confiar ciegamente en el 100% de accuracy en validación?** Porque
lo investigamos y encontramos que una sola feature (`zcr_std`) ya predice casi
todo por sí sola — señal de que el dataset sintético tiene un atajo de
generación, no de que el problema esté resuelto. Por eso construimos una prueba
aparte (`eval_robustness.py`) que degrada el audio con canales que ningún
modelo vio en entrenamiento, para medir robustez real y no solo el número más
bonito. El detalle completo, con datos, está en las siguientes secciones.

## Arquitectura: dos modelos que miran cosas distintas

El sistema es un **ensamble de dos clasificadores con sesgos inductivos opuestos**.
No están puestos para inflar un número: miden señales diferentes y fallan ante
condiciones diferentes, y lo demostramos abajo con datos.

### Modelo 1 — GBM sobre 171 features interpretables (`src/features.py` + `scripts/train.py`)

`HistGradientBoosting` calibrado con isotónica, sobre estadísticos **globales** de
la llamada completa:

1. **Acústicas (canal 0).** MFCC + deltas, centroid, bandwidth, rolloff, flatness,
   ZCR, RMS, chroma y spectral contrast.
2. **Timing conversacional (canales 0 y 1).** Desde un VAD por energía por canal:
   latencia de respuesta del llamante tras cada turno del agente y, sobre todo, su
   **consistencia** (coeficiente de variación), más solapamientos y pausas.
   Una máquina responde con timing robóticamente parejo; un humano, no.

Su virtud es la interpretabilidad: podemos decirle al juez *qué* disparó la alarma.

### Modelo 2 — CNN por segmentos sobre log-mel (`src/neural.py` + `scripts/train_neural.py`)

Red convolucional de **154k parámetros entrenada desde cero**. Trocea el canal del
llamante en ventanas de **4 s con 50% de solape**, descarta las que no tienen voz,
y clasifica cada una por separado; el veredicto de la llamada es el promedio en
espacio logit de los segmentos.

Tres decisiones de diseño que sostienen el modelo:

- **CMVN por segmento** (normaliza media y varianza de cada banda mel en el tiempo).
  Destruye el nivel de piso de ruido y la coloración del canal, que es exactamente
  donde vive el atajo del dataset. Obliga a la red a aprender la *forma* del
  espectro, no su nivel.
- **Entrenamiento por segmento.** 282 llamadas de train se vuelven ~11.9k muestras.
  Nuestro límite real es la escasez de datos, no la capacidad del modelo.
- **Aumentación de canal** (`src/augment.py`: µ-law, ruido, ganancia) **+ SpecAugment**
  (enmascarado de bandas y tiempos) en cada época.

#### Por qué una CNN propia y no Wav2Vec2

No es una limitación, es la decisión correcta para este reto:

| | Wav2Vec2-base fine-tuneado | Nuestra CNN |
|---|---|---|
| Inferencia (llamada de 170 s, CPU) | 10–30 s | **0.19 s** |
| Entrenable con 282 llamadas sin GPU | no | **sí, ~50 min en CPU** |
| Parámetros | 95M | **154k** |
| Viable en telefonía real | requiere GPU por canal | **corre en el mismo host** |

El reto pide **baja latencia y viabilidad de despliegue**. Un modelo que tarda
30 s por llamada no sirve en una línea telefónica, por buena que sea su AUC.

## Validación honesta: por qué no presumimos el 100%

Con las features completas obtenemos AUC=1.0 / EER=0% en `val` (hablantes
disjuntos). **Eso es sospechoso, no un triunfo.** `zcr_std` **por sí sola** da
AUC=0.983: hay un atajo espectral en cómo se generaron los WAV sintéticos.

Medirse contra la propia aumentación sería un autogol estadístico. Así que
`scripts/eval_robustness.py` degrada `val` con canales que **ningún modelo vio jamás**
(nada de esto está en `src/augment.py`):

```
condicion                                GBM               CNN          ENSAMBLE
                                   AUC / EER         AUC / EER         AUC / EER
--------------------------------------------------------------------------------
limpio                         1.000 /  0.0%     1.000 /  0.0%     1.000 /  0.0%
pasabanda 300-3400Hz           0.990 /  9.9%     1.000 /  0.0%     1.000 /  0.0%
perdida ancho banda 4k         0.997 /  1.5%     1.000 /  0.0%     1.000 /  0.0%
reverberacion                  1.000 /  0.0%     1.000 /  0.0%     1.000 /  0.0%
ruido fuerte (5x)              0.960 /  7.0%     1.000 /  0.0%     0.994 /  7.0%
```

**Lectura:** el GBM se degrada bajo canal desconocido (EER 9.9% con pasabanda
telefónica real, 7.0% con ruido fuerte); la CNN se mantiene en 0% en las cuatro
condiciones. Esa evidencia fija el peso del ensamble en `W_NEURAL = 0.5`
(con 0.4 el GBM arrastraba el ensamble a 7% bajo ruido).

### Lo que NO podemos afirmar

Demostramos **robustez de canal**, no **robustez de motor**. Todos los audios
sintéticos del dataset provienen del mismo pipeline de generación, así que
ningún experimento local puede probar que aguantaremos un TTS no visto.
Nuestro seguro ante ese escenario son las **features de timing conversacional**,
que son independientes del engine: `resp_latency_mean` da AUC=0.922 y
`resp_latency_cv` AUC=0.860 por sí solas. Un motor de TTS nuevo sigue teniendo
turnos robóticamente parejos.

## Latencia

Medido end-to-end contra el endpoint, llamada de 170 s:

| Etapa | Tiempo |
|---|---|
| Extracción de 171 features | ~0.5 s |
| CNN (mel + 19 segmentos en batch) | ~0.19 s |
| **Total `/detect`** | **~1.0 s** |

`src/api/main.py` hace **warm-up al arranque**: librosa/numba pagan ~1.2 s de
inicialización perezosa en la primera llamada, y la pagamos nosotros para que la
primera petición del juez no la sufra.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (usar `source .venv/bin/activate` en Linux/Mac)
pip install -r requirements.txt
```

El dataset de audio (`altur-challenge-audio.zip`, ~640 MB) **no está en git**
por su tamaño — GitHub rechaza archivos >100 MB. Se comparte por fuera del repo:

- **Descarga:** https://drive.google.com/file/d/1cAiKkZ6Dj_Sr3rXOmrOKh-IKEHui86ED/view?usp=drive_link
- Descomprimirlo en la raíz del repo (crea `audio/`), junto a `manifest.csv`.

> Si actualizas el dataset, sube el nuevo zip al mismo link (o crea uno nuevo)
> y avisa al equipo — el archivo en sí nunca debe subirse a git.

## Entrenamiento

**1) GBM** (`scripts/train.py`) — genera `model.pkl` + `threshold.json`:

```bash
python scripts/train.py --data_dir . --augment
```

| Flag | Default | Descripción |
|---|---|---|
| `--data_dir` | *(requerido)* | Raíz con `audio/` y `manifest.csv` |
| `--turns_dir` | `None` | Carpeta con turnos de conversación, si aplica |
| `--out` | `model.pkl` | Ruta de salida del modelo entrenado |
| `--augment` | `False` | Activa aumentación de canal durante el entrenamiento |

**2) CNN** (`scripts/train_neural.py`) — genera `weights/neural_cnn.pt`:

```bash
python scripts/train_neural.py --data_dir . --epochs 25 --batch 128
```

| Flag | Default | Descripción |
|---|---|---|
| `--data_dir` | `.` | Raíz con `audio/` y `manifest.csv` |
| `--epochs` | `25` | Número de épocas |
| `--batch` | `128` | Tamaño de batch |
| `--out` | `weights/neural_cnn.pt` | Ruta de salida de los pesos |

**3) Verificar robustez** (val bajo canales no vistos):

```bash
python scripts/eval_robustness.py
```

## Servir

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# exponer publicamente para el juez:  ngrok http 8000
```

### Con Docker

Levanta la API (`/detect`, `/health`) y el dashboard Streamlit juntos, sin instalar
nada localmente más que Docker:

```bash
docker compose up --build
```

- API: http://localhost:8000
- Dashboard: http://localhost:8501

Los modelos (`model.pkl`, `weights/neural_cnn.pt`) ya están en la imagen porque
vienen commiteados en el repo — no hace falta el dataset de audio ni reentrenar
para levantar los contenedores. `audio/` está excluido del build vía
`.dockerignore` (pesa demasiado); para reentrenar dentro de Docker, móntalo como
volumen en vez de reconstruir la imagen:

```bash
docker compose run --rm -v "$(pwd)/audio:/app/audio" api python scripts/train.py --data_dir . --augment
```

## Endpoints

- `GET /health` — verifica que ambos modelos cargaron (`model_loaded`, `neural_loaded`).
- `POST /detect` — el del reto. Acepta el base64 en varios nombres de campo o crudo
  en el body; **nunca devuelve 500** al benchmark. Responde:
  ```json
  {"is_synthetic": true, "confidence": 0.9739,
   "desglose": {"acustica_timing": 1.0, "neuronal_cnn": 0.9479, "peso_neuronal": 0.5}}
  ```
- `POST /analyze` — para el dashboard: igual que `/detect` más la **línea de tiempo
  por segmento**, que muestra *en qué momento* de la llamada la voz se ve sintética.
  Se mantiene aparte para que `/detect` siga ligero.

Si los pesos neuronales no están presentes, el ensamble degrada limpiamente a
usar solo el GBM (`peso_neuronal: 0.0`) en vez de sesgar el promedio hacia 0.5.

## Estructura del proyecto

```
altur/
├── src/                    # Lógica core (features, modelo neuronal, aumentación, API)
│   ├── features.py         # extracción de 171 features (acústicas + timing)
│   ├── neural.py           # frontend log-mel, SpoofCNN, inferencia y agregación por segmentos
│   ├── augment.py          # degradaciones de canal para robustez
│   └── api/
│       └── main.py         # servidor FastAPI con el ensamble
├── scripts/                # entrenamiento y evaluación (uso puntual, no productivo)
│   ├── train.py            # entrena el GBM
│   ├── train_neural.py     # entrena la CNN
│   └── eval_robustness.py  # la prueba honesta: val bajo canales no vistos
├── tests/                  # scripts de humo contra la API corriendo
│   ├── test.py              # smoke test rápido (10 muestras de val)
│   └── test_full.py         # matriz de confusión completa sobre val
├── frontend.py              # dashboard Streamlit para la demo
├── model.pkl, threshold.json, neural_metrics.json   # artefactos del GBM
└── weights/neural_cnn.pt    # artefacto de la CNN
```

Todos los comandos (`scripts/*.py`, `uvicorn src.api.main:app`, `python tests/*.py`)
se ejecutan **desde la raíz del repo**.
