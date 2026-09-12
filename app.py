import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.config import settings
from src.api.schemas import DetectRequest, DetectResponse, HealthResponse
from src.api.audio_utils import decode_and_preprocess_audio
from src.api.inference_engine import InferenceEngine
from src.api.s3_audit import s3_audit_service

# Configure Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger("altur.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: Initializes the Triple Multi-Model Inference Engine
    during application startup and stores it in app.state.
    """
    logger.info(f"Starting {settings.app_name} (env={settings.environment})...")
    logger.info(f"Loading Multi-Model Ensemble weights & feature scalers into memory...")
    t0 = time.perf_counter()
    app.state.engine = InferenceEngine()
    load_time = time.perf_counter() - t0
    logger.info(f"Inference Engine warmed up successfully in {load_time:.3f}s on {app.state.engine.device}")
    yield
    logger.info(f"Shutting down {settings.app_name}...")


# Initialize FastAPI Application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production Voice Anti-Spoofing API for Bank Telephony Streams (8kHz Stereo Base64 WAV, Channel 0 = Caller)",
    lifespan=lifespan,
)

# Enable CORS for frontend & client integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
async def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
        "detect_endpoint": "POST /detect (Stereo 8kHz Base64 WAV, Channel 0 = Caller)",
        "calibrated_threshold": settings.calibrated_threshold,
    }


def get_inference_engine() -> InferenceEngine:
    """Helper to retrieve or lazily initialize the InferenceEngine."""
    if not hasattr(app.state, "engine") or app.state.engine is None:
        app.state.engine = InferenceEngine()
    return app.state.engine


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Service health and telemetry status endpoint."""
    engine = get_inference_engine()
    return HealthResponse(
        status="healthy",
        service=settings.app_name,
        version=settings.app_version,
        calibrated_threshold=settings.calibrated_threshold,
        model_ensemble="AASIST (GNN) + RawNet2 (Waveform) + Acoustic DSP + XGBoost",
        device=str(engine.device) if engine else "cpu",
    )


@app.post(
    "/detect",
    response_model=DetectResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect Synthetic AI Voices in Bank Call Audio",
    tags=["Anti-Spoofing"],
)
async def detect_synthetic_voice(
    request: DetectRequest,
    background_tasks: BackgroundTasks
):
    """
    **Voice Anti-Spoofing Detection Endpoint**:
    - **Input**: Stereo 8kHz Base64-encoded WAV audio clip.
      - **Channel 0**: Caller voice (evaluated for deepfake / synthetic speech).
      - **Channel 1**: Agent voice (separated / excluded).
    - **Returns**: Strict binary verdict on the caller:
      ```json
      {
        "is_synthetic": true,
        "confidence": 0.87
      }
      ```
    """
    call_id = request.call_id or f"call_{uuid.uuid4().hex[:12]}"

    # 1. Decode Base64, separate Channel 0, resample to 16kHz, and window audio
    try:
        caller_16k, raw_wav_bytes, orig_sr, channels = decode_and_preprocess_audio(
            base64_audio_str=request.audio,
            target_sr=settings.target_sample_rate
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except Exception as err:
        logger.error(f"Error preprocessing audio for {call_id}: {err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process input audio."
        )

    # 2. Schedule non-blocking S3 audit logging in background
    if settings.enable_s3_audit:
        background_tasks.add_task(s3_audit_service.upload_audio_async, raw_wav_bytes, call_id)

    # 3. Execute Multi-Model Anti-Spoofing Inference
    engine: InferenceEngine = get_inference_engine()
    try:
        is_synthetic, confidence = engine.predict(caller_16k)
    except Exception as inf_err:
        logger.error(f"Inference error for {call_id}: {inf_err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference model execution failed."
        )

    # 4. Return exact required contract
    return DetectResponse(
        is_synthetic=is_synthetic,
        confidence=confidence
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=(settings.environment == "development")
    )
