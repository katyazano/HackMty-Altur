import os
from pathlib import Path
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseModel):
    app_name: str = "Altur Banking Voice Anti-Spoofing API"
    app_version: str = "1.0.0"
    environment: str = os.getenv("ENVIRONMENT", "development")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # Calibrated decision threshold (Equal Error Rate operating point)
    calibrated_threshold: float = float(os.getenv("CALIBRATED_THRESHOLD", "0.35"))

    # Audio specifications
    target_sample_rate: int = int(os.getenv("TARGET_SAMPLE_RATE", "16000"))
    telephony_sample_rate: int = int(os.getenv("TELEPHONY_SAMPLE_RATE", "8000"))
    max_audio_size_mb: int = int(os.getenv("MAX_AUDIO_SIZE_MB", "15"))

    # Model weight paths
    aasist_weights: str = os.getenv("AASIST_WEIGHTS", str(PROJECT_ROOT / "weights" / "aasist_best.pth"))
    rawnet2_weights: str = os.getenv("RAWNET2_WEIGHTS", str(PROJECT_ROOT / "weights" / "rawnet2_best.pth"))
    xgb_model_path: str = os.getenv("XGB_MODEL_PATH", str(PROJECT_ROOT / "weights" / "xgboost_triple_ensemble.json"))
    scaler_path: str = os.getenv("SCALER_PATH", str(PROJECT_ROOT / "weights" / "triple_scaler.joblib"))

    # AWS S3 Audit Configuration
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    s3_audit_bucket: str = os.getenv("S3_AUDIT_BUCKET", "antispoofing-audio-audit-altur")
    enable_s3_audit: bool = os.getenv("ENABLE_S3_AUDIT", "true").lower() in ("1", "true", "yes")


settings = Settings()
