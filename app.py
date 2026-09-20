"""
app.py — Master Unified FastAPI Service for Altur Voice Anti-Spoofing AI.
Serves Engine 1 (Baseline CNN/GBM), Engine 2 (4-Pillar Multimodal SOTA), and Side-by-Side Dual Engine Comparison.
"""
import os
import sys
import time
import uuid
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, status, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Setup path and environment
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.config import settings
from src.api.schemas import (
    DetectRequest,
    DetectResponse,
    CompareResponse,
    HealthResponse
)
from src.audio_processor import AudioProcessor
from src.engines.unified_orchestrator import UnifiedOrchestrator
from src.api.s3_audit import s3_audit_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("altur.gateway")

WEB_INDEX = PROJECT_ROOT / "web" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes and warms up both Anti-Spoofing engines in memory.
    """
    logger.info(f"Initializing {settings.app_name}...")
    t0 = time.perf_counter()
    app.state.orchestrator = UnifiedOrchestrator()
    load_time = time.perf_counter() - t0
    logger.info(f"Dual-Engine Orchestrator initialized in {load_time:.3f}s.")
    yield
    logger.info("Shutting down Altur Anti-Spoofing Gateway...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Dual-Engine Enterprise Voice Anti-Spoofing Platform (Engine 1: Baseline CNN/GBM | Engine 2: 4-Pillar Multimodal SOTA)",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and React build assets
DIST_DIR = PROJECT_ROOT / "frontend" / "dist"

if (DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")


def get_orchestrator() -> UnifiedOrchestrator:
    if not hasattr(app.state, "orchestrator") or app.state.orchestrator is None:
        app.state.orchestrator = UnifiedOrchestrator()
    return app.state.orchestrator


WEB_INDEX = DIST_DIR / "index.html"


@app.get("/", tags=["UI"])
@app.get("/overview", tags=["UI"])
@app.get("/demo", tags=["UI"])
@app.get("/engine1", tags=["UI"])
@app.get("/engine2", tags=["UI"])
async def serve_ui():
    """Serves the Master Product Showcase & Overview (React SPA)."""
    if WEB_INDEX.exists():
        return FileResponse(str(WEB_INDEX))
    return {"service": settings.app_name, "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Service health and telemetry status endpoint."""
    orch = get_orchestrator()
    return HealthResponse(
        status="healthy",
        service=settings.app_name,
        version=settings.app_version,
        available_engines=[
            "Engine 1: Baseline Fast Acoustic + AudioCNN (8kHz)",
            "Engine 2: 4-Pillar Multimodal AASIST + RawNet2 + DSP + Whisper (16kHz)",
            "Dual-Engine Side-by-Side Comparator"
        ],
        device=str(orch.multimodal_engine.device),
        calibrated_threshold=settings.calibrated_threshold
    )


@app.get("/api/models/info", tags=["Catalog"])
async def get_models_info():
    """Returns comprehensive architectural documentation, dimensions, and benchmark summaries."""
    orch = get_orchestrator()
    return orch.get_catalog_info()


@app.get("/api/benchmarks", tags=["Catalog"])
async def get_benchmarks():
    """Returns comparative evaluation benchmarks for both models."""
    orch = get_orchestrator()
    info = orch.get_catalog_info()
    return {
        "benchmarks": info["benchmark_summary"],
        "evaluation_dataset": "HackMTY Bank Telephony Evaluation Corpus (Narrowband 8kHz Stereo)",
        "metrics_description": {
            "eer": "Equal Error Rate (Lower is better) - The rate where False Acceptance equals False Rejection.",
            "auc_roc": "Area Under ROC Curve (Higher is better) - Overall discriminatory capability.",
            "latency_ms": "Average end-to-end CPU inference time in milliseconds."
        }
    }


@app.post(
    "/detect",
    response_model=DetectResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect Synthetic AI Voices in Bank Call Audio (Standard Contract)",
    tags=["Anti-Spoofing"]
)
async def detect(
    request: DetectRequest,
    background_tasks: BackgroundTasks,
    engine: str = Query(default=None, description="Optional engine override: 'multimodal', 'baseline', 'dual'")
):
    """
    **Standard Anti-Spoofing Detection Endpoint**
    - Input: Base64-encoded WAV call audio. Channel 0 = Caller, Channel 1 = Bank Agent.
    - Returns strict standard challenge contract: `{"is_synthetic": bool, "confidence": float}`
    """
    call_id = request.call_id or f"call_{uuid.uuid4().hex[:12]}"
    try:
        audio_str = request.get_audio_str()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    selected_engine = engine or request.engine or "multimodal"
    orch = get_orchestrator()

    try:
        res = orch.predict_single(audio_str, engine_type=selected_engine)
    except Exception as e:
        logger.error(f"Detection failed for {call_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    if settings.enable_s3_audit:
        try:
            raw_bytes = AudioProcessor.decode_base64_to_bytes(audio_str)
            background_tasks.add_task(s3_audit_service.upload_audio_async, raw_bytes, call_id)
        except Exception:
            pass

    return DetectResponse(
        is_synthetic=res["is_synthetic"],
        confidence=res["confidence"]
    )


@app.post(
    "/detect/baseline",
    response_model=DetectResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect Synthetic Voices using Engine 1 (Fast Baseline CNN/GBM)",
    tags=["Anti-Spoofing"]
)
async def detect_baseline(request: DetectRequest):
    """
    Dedicated endpoint for Engine 1: Fast Acoustic DSP + Timing + SpoofCNN.
    Returns standard HackMTY evaluation contract: {"is_synthetic": bool, "confidence": float}
    """
    try:
        audio_str = request.get_audio_str()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    orch = get_orchestrator()
    try:
        res = orch.predict_single(audio_str, engine_type="baseline")
        return DetectResponse(
            is_synthetic=res["is_synthetic"],
            confidence=res["confidence"]
        )
    except Exception as e:
        logger.error(f"Baseline inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Baseline error: {str(e)}")


@app.post(
    "/detect/multimodal",
    response_model=DetectResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect Synthetic Voices using Engine 2 (4-Pillar Multimodal SOTA)",
    tags=["Anti-Spoofing"]
)
async def detect_multimodal(request: DetectRequest):
    """
    Dedicated endpoint for Engine 2: 4-Pillar AASIST + RawNet2 + DSP + Whisper + XGBoost.
    Returns standard HackMTY evaluation contract: {"is_synthetic": bool, "confidence": float}
    """
    try:
        audio_str = request.get_audio_str()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    orch = get_orchestrator()
    try:
        res = orch.predict_single(audio_str, engine_type="multimodal")
        return DetectResponse(
            is_synthetic=res["is_synthetic"],
            confidence=res["confidence"]
        )
    except Exception as e:
        logger.error(f"Multimodal inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Multimodal error: {str(e)}")


@app.post(
    "/api/detect/compare",
    summary="Side-by-Side Dual-Engine Comparison and Deep Attribution",
    tags=["Dual-Engine Analysis"]
)
async def detect_compare(request: DetectRequest):
    """
    Runs both Engine 1 and Engine 2 simultaneously, returning detailed probabilities,
    pillar attributions, consensus agreement, and timing breakdowns.
    """
    try:
        audio_str = request.get_audio_str()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    orch = get_orchestrator()
    try:
        result = orch.compare(audio_str)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Comparison execution error: {str(e)}")


@app.post(
    "/api/dialogue",
    summary="Transcribe 2-party conversational turns (Caller vs Agent)",
    tags=["UI Dialogue"]
)
async def transcribe_dialogue(request: DetectRequest):
    """
    Transcribes chronological dialogue turns between Channel 0 (Caller) and Channel 1 (Agent).
    """
    call_id = request.call_id or f"call_{uuid.uuid4().hex[:12]}"
    try:
        audio_str = request.get_audio_str()
        _, _, _, raw_wav_bytes = AudioProcessor.load_audio(audio_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio payload: {e}")

    try:
        from src.dialogue_transcriber import ConversationalTurnTranscriber
        transcriber = ConversationalTurnTranscriber(asr_model_size="tiny", device="cpu")
        result = transcriber.process_audio_bytes(raw_wav_bytes, call_id=call_id)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        return JSONResponse(content={
            "call_id": call_id,
            "total_turns": 0,
            "dialogue": [],
            "error": str(e)
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=(settings.environment == "development")
    )
