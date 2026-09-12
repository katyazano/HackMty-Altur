from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class DetectRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    audio: str = Field(
        ...,
        alias="audio_base64",
        description="Base64-encoded stereo WAV audio clip (8kHz, Channel 0 = Caller, Channel 1 = Agent)."
    )
    call_id: Optional[str] = Field(
        default=None,
        description="Optional bank call identifier for tracking and S3 auditing."
    )
    sample_rate: Optional[int] = Field(
        default=8000,
        description="Sample rate of the audio in Hz (default: 8000)."
    )
    channels: Optional[int] = Field(
        default=2,
        description="Number of audio channels (Channel 0 = Caller, Channel 1 = Agent)."
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
        description="Certainty in the returned verdict [0.0 - 1.0]."
    )


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    calibrated_threshold: float
    model_ensemble: str
    device: str
