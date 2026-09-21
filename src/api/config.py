"""
config.py — Unified Application Settings and Environment Variables.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    app_name: str = "Altur Voice AI Defense — Dual Engine Platform"
    app_version: str = "2.0.0"
    environment: str = os.getenv("ENV", "production")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # Audio Signal Processing Parameters
    target_sample_rate: int = 16000
    baseline_sample_rate: int = 8000

    # Decision Threshold
    calibrated_threshold: float = float(os.getenv("DECISION_THRESHOLD", "0.50"))

    # S3 Compliance Auditing (Optional)
    enable_s3_audit: bool = os.getenv("ENABLE_S3_AUDIT", "false").lower() == "true"
    s3_bucket_name: str = os.getenv("AWS_S3_BUCKET", "altur-telephony-audit-vault")
    s3_region: str = os.getenv("AWS_REGION", "us-east-1")


settings = Settings()
