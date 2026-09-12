from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class DetectRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    audio: str = Field(
        ...,
        alias="audio_base64",
        description="Base64-encoded stereo WAV audio clip (8kHz, Channel 0 = Caller, Channel 1 = Agent)."
    )
    call_id: Optional[str] = Field(
        default=None,
        description="Optional bank call identifier for tracking and S3 auditing."
    )


class DetectResponse(BaseModel):
    is_synthetic: bool = Field(
        ...,
        description="True if caller voice is synthetic (spoofed/deepfake), False if human."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability score that caller is synthetic [0.0 - 1.0]."
    )


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    calibrated_threshold: float
    model_ensemble: str
    device: str
