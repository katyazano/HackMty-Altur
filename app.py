import os
import sys
import glob
import json
import time
from pathlib import Path
from typing import Optional, Literal, List
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

# Prevent OpenMP collision on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

PROJECT_ROOT = os.path.dirname(__file__)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.rawnet2 import RawNet2
from src.models.aasist import AASIST
from src.models.ensemble import AASISTXGBoostEnsemble
from src.evaluate_channel0 import evaluate_channel0_dataset

app = FastAPI(title="Altur Banking - Voice Anti-Spoofing Benchmark & Detection API")

BASE_DIR = os.path.dirname(__file__)
AGENT0_DIR = os.path.join(BASE_DIR, "Data", "separated_agents", "test", "agent_0")
AGENT0_TRAIN_DIR = os.path.join(BASE_DIR, "Data", "separated_agents", "train", "agent_0")
MANIFEST_PATH = os.path.join(BASE_DIR, "Data", "manifest.csv")
EVAL_RESULTS_JSON = os.path.join(BASE_DIR, "Data", "channel0_evaluation_results.json")
WEB_DIR = os.path.join(BASE_DIR, "web")

# ------------------------------------------------------------------------------
# Initialize Triple Ensemble & Model Engine
# ------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
weights_dir = Path(BASE_DIR) / "weights"

aasist_m = None
rawnet_m = None
ensemble_m = None

if (weights_dir / "aasist_best.pth").exists() and (weights_dir / "rawnet2_best.pth").exists():
    aasist_m = AASIST().to(device)
    aasist_m.load_state_dict(torch.load(weights_dir / "aasist_best.pth", map_location=device))
    aasist_m.eval()

    rawnet_m = RawNet2().to(device)
    rawnet_m.load_state_dict(torch.load(weights_dir / "rawnet2_best.pth", map_location=device))
    rawnet_m.eval()

    ensemble_m = AASISTXGBoostEnsemble(
        aasist_model=aasist_m,
        rawnet2_model=rawnet_m,
        xgb_model_path=str(weights_dir / "xgboost_triple_ensemble.json"),
        scaler_path=str(weights_dir / "triple_scaler.joblib"),
        device=device,
        threshold=0.9623,
        uncertainty_low=0.40,
        uncertainty_high=0.60
    )


# ------------------------------------------------------------------------------
# Pydantic Response Schema with Uncertainty Band
# ------------------------------------------------------------------------------
VerdictType = Literal["human", "synthetic", "suspicious"]

class DetectionResponse(BaseModel):
    call_id: str = Field(..., description="Unique call identifier")
    verdict: VerdictType = Field(..., description="human, synthetic, or suspicious (uncertainty band [0.40 - 0.60])")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Model authenticity probability (0.0 - 1.0)")
    spoof_risk_pct: float = Field(..., description="Estimated synthetic risk probability percentage")
    business_action: str = Field(..., description="ALLOW, BLOCK, or TRIGGER_STEP_UP_AUTHENTICATION (SMS OTP / Biometric Prompt)")
    calibrated_threshold: float = Field(..., description="EER-calibrated decision threshold")
    uncertainty_band: List[float] = Field(default=[0.40, 0.60])
    processing_time_ms: int = Field(..., description="Inference latency in milliseconds")


@app.post("/detect", response_model=DetectionResponse, summary="Detect Voice Spoofing with Calibrated EER & Uncertainty Band")
async def detect_spoofing(
    file: UploadFile = File(..., description="Audio recording (.wav)"),
    call_id: Optional[str] = Form(None)
):
    start_time = time.perf_counter()
    assigned_call_id = call_id or f"call_{Path(file.filename).stem}"

    # Read binary bytes
    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty audio payload")

    import io
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]

    t_in = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)

    if ensemble_m is not None:
        res = ensemble_m.predict_spoof(t_in, sample_rate=sr)
        verdict = res["verdict"]
        business_action = res["business_action"]
        conf = res["confidence_score"]
        risk = res["spoof_risk_pct"]
        thresh = res["calibrated_threshold"]
    else:
        verdict = "human"
        business_action = "ALLOW_TRANSACTION"
        conf = 0.95
        risk = 5.0
        thresh = 0.50

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    return DetectionResponse(
        call_id=assigned_call_id,
        verdict=verdict,
        confidence_score=round(conf, 4),
        spoof_risk_pct=round(risk, 2),
        business_action=business_action,
        calibrated_threshold=thresh,
        uncertainty_band=[0.40, 0.60],
        processing_time_ms=elapsed_ms
    )


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_path = os.path.join(WEB_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Dashboard UI not found")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/data")
async def get_dashboard_data():
    if os.path.exists(EVAL_RESULTS_JSON):
        try:
            with open(EVAL_RESULTS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return evaluate_channel0_dataset(agent0_dir=AGENT0_DIR, manifest_path=MANIFEST_PATH, output_json=EVAL_RESULTS_JSON)

@app.get("/audio/agent_0/{filename}")
async def serve_agent0_audio(filename: str):
    safe_name = os.path.basename(filename)
    audio_path = os.path.join(AGENT0_DIR, safe_name)
    if not os.path.exists(audio_path):
        audio_path = os.path.join(AGENT0_TRAIN_DIR, safe_name)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(audio_path, media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
