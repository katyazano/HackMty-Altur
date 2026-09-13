"""
main.py — Endpoint /detect para el reto Altur.

Endurecido contra los riesgos que te dejan en CERO el día del juez:
  - El documento fija el formato de RESPUESTA pero NO el nombre del campo del REQUEST.
    -> aceptamos varios nombres posibles Y base64 crudo en el body.
       (Aun así: CONFIRMA el esquema exacto con los ingenieros de Altur en sitio.)
  - Decodificación en memoria (sin archivos temporales) -> menor latencia.
  - Modelo cargado UNA vez al arranque.
  - Nunca revienta con 500 hacia el benchmark: si algo falla, responde JSON válido.

Arranque (desde la raíz del repo):
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    (host 0.0.0.0, NO 127.0.0.1, para que el juez alcance tu endpoint;
     y expón con: ngrok http 8000  -> usa esa URL pública)
"""

import base64
import binascii
import json
import os
import pickle
import sys
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from src.features import extract_all
from src.neural import get_neural_probability, load_model as load_neural

MODEL = None
FEATURE_NAMES = None
THRESHOLD = 0.5
NEURAL_OK = False

# Peso del modelo neuronal en el ensamble.
# Medido con eval_robustness.py sobre val degradado con canales NO vistos:
# la CNN mantiene EER 0% en las cuatro condiciones, el GBM sube a 9.9%
# (pasabanda 300-3400 Hz) y 7.0% (ruido fuerte). Con W=0.4 el GBM arrastraba
# el ensamble a 7% bajo ruido; 0.5 respeta esa evidencia sin apostar todo a
# un solo modelo ante un motor de TTS desconocido.
W_NEURAL = 0.5


def _warmup():
    """librosa/numba pagan ~1.2 s de lazy-init en la PRIMERA llamada. Lo
    pagamos aqui para que la primera peticion del juez no lo sufra."""
    import io
    import soundfile as sf
    sr = 8000
    stereo = np.zeros((sr * 5, 2), dtype=np.float32)
    stereo[:, 0] = np.random.RandomState(0).normal(0, 0.01, sr * 5)
    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV")
    data = buf.getvalue()
    try:
        extract_all(data)
        get_neural_probability(data)
    except Exception as e:
        print(f"[AVISO] warmup fallo: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, FEATURE_NAMES, THRESHOLD, NEURAL_OK
    model_path = os.path.join(REPO_ROOT, "model.pkl")
    threshold_path = os.path.join(REPO_ROOT, "threshold.json")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        MODEL = bundle["model"]
        FEATURE_NAMES = bundle["feature_names"]
        print(f"Modelo cargado ({len(FEATURE_NAMES)} features).")
    else:
        print("[AVISO] model.pkl no encontrado -> modo degradado (siempre responde JSON válido).")
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            content = f.read().lstrip("\ufeff")  # quita BOM si existe
            THRESHOLD = json.loads(content).get("threshold", 0.5)
        print(f"Umbral cargado: {THRESHOLD:.4f}")

    NEURAL_OK = load_neural() is not None
    print(f"Modelo neuronal: {'cargado' if NEURAL_OK else 'NO disponible (ensamble usa solo el GBM)'}")

    _warmup()
    print("Warmup completo. Listo.")
    yield


app = FastAPI(title="Altur Voice Anti-Spoofing", lifespan=lifespan)

# Campos donde podría venir el base64. Ajusta si Altur usa otro nombre.
_CANDIDATE_FIELDS = ["audio_base64", "audio", "wav_base64", "wav", "data", "clip", "file"]


async def _extract_b64(request: Request):
    """Saca el base64 del request sin importar cómo lo empaqueten los jueces."""
    ctype = request.headers.get("content-type", "")
    if "application/json" in ctype:
        body = await request.json()
        if isinstance(body, str):
            return body
        for k in _CANDIDATE_FIELDS:
            if isinstance(body, dict) and k in body and body[k]:
                return body[k]
        # último recurso: primer valor string largo del dict
        if isinstance(body, dict):
            for v in body.values():
                if isinstance(v, str) and len(v) > 100:
                    return v
        return None
    # body crudo (texto base64 o bytes)
    raw = await request.body()
    try:
        return raw.decode("utf-8").strip().strip('"')
    except Exception:
        return raw  # ya son bytes de audio


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None,
            "neural_loaded": NEURAL_OK, "threshold": THRESHOLD}


@app.post("/analyze")
async def analyze(request: Request):
    """Igual que /detect pero devuelve el detalle por segmento de la CNN, para
    que el dashboard muestre EN QUE MOMENTO la voz se ve sintetica.
    No lo usa el benchmark: /detect se mantiene ligero."""
    from neural import get_neural_timeline
    base = await detect(request)
    try:
        b64 = await _extract_b64(request)
        audio_bytes = bytes(b64) if isinstance(b64, (bytes, bytearray)) \
            else base64.b64decode(b64, validate=False)
        _, timeline = get_neural_timeline(audio_bytes)
        base["timeline"] = timeline
    except Exception as e:
        base["timeline_error"] = str(e)[:200]
    return base


@app.post("/detect")
async def detect(request: Request):
    try:
        b64 = await _extract_b64(request)
        if b64 is None:
            return {"is_synthetic": False, "confidence": 0.5, "error": "no_audio_field"}

        # Decodificar a bytes de audio
        if isinstance(b64, (bytes, bytearray)):
            audio_bytes = bytes(b64)
        else:
            try:
                audio_bytes = base64.b64decode(b64, validate=False)
            except (binascii.Error, ValueError):
                return {"is_synthetic": False, "confidence": 0.5, "error": "bad_base64"}

        # --- MODELO 1: Acústico y Timing (Tu modelo actual) ---
        vec, names = extract_all(audio_bytes)

        if MODEL is not None and FEATURE_NAMES is not None:
            index = {n: i for i, n in enumerate(names)}
            row = np.array([vec[index[n]] if n in index else 0.0
                            for n in FEATURE_NAMES], dtype=np.float32).reshape(1, -1)
            score_acustico = float(MODEL.predict_proba(row)[0, 1])
        else:
            score_acustico = 0.5

        # --- MODELO 2: CNN neuronal por segmentos sobre log-mel ---
        # Mira micro-artefactos locales en ventanas de 4 s, con CMVN por
        # segmento para no reaprender el atajo de piso de ruido del dataset.
        if NEURAL_OK:
            try:
                score_neural = get_neural_probability(audio_bytes)
            except Exception:
                score_neural = 0.5   # un fallo aqui no debe tumbar el veredicto
        else:
            score_neural = 0.5

        # --- ENSAMBLE FINAL (soft voting ponderado) ---
        # Si el neuronal no esta disponible su 0.5 sesgaria el promedio hacia
        # el centro, asi que en ese caso el GBM manda solo.
        if NEURAL_OK:
            score_final = (1 - W_NEURAL) * score_acustico + W_NEURAL * score_neural
        else:
            score_final = score_acustico

        # scripts/check_endpoint.py del juez reconstruye P(sintetico) asi:
        #     prob = confidence if is_synthetic else 1 - confidence
        # O sea: espera CONFIANZA EN EL VEREDICTO, no P(sintetico). Devolver
        # P(sintetico) directo hunde AUC a 0.505 y Brier a 0.515 aunque
        # aciertes 20/20, porque para cada humano el juez lee 1-0.01 = 0.99.
        is_syn = bool(score_final >= THRESHOLD)
        confidence = score_final if is_syn else 1.0 - score_final

        return {
            "is_synthetic": is_syn,
            "confidence": round(confidence, 4),
            "desglose": {
                "p_sintetico": round(score_final, 4),
                "acustica_timing": round(score_acustico, 4),
                "neuronal_cnn": round(score_neural, 4),
                "peso_neuronal": W_NEURAL if NEURAL_OK else 0.0,
            }
        }

    except Exception as e:
        # NUNCA devolver 500 al benchmark: mejor un veredicto neutral que un crash.
        return {"is_synthetic": False, "confidence": 0.5, "error": str(e)[:200]}