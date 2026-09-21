"""
schemas.py — Pydantic Request & Response Data Contracts.
Strictly compatible with challenge evaluation specifications while supporting dual-engine comparison.
"""
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    """
    Standard Ingestion Request Schema.
    Accepts Base64-encoded WAV (8kHz / 16kHz stereo or mono).
    """
    audio: Optional[str] = Field(
        default=None,
        description="Base64-encoded WAV audio string. Channel 0 is Caller, Channel 1 is Agent."
    )
    audio_base64: Optional[str] = Field(
        default=None,
        description="Alternative field name for base64 audio payload."
    )
    wav_base64: Optional[str] = Field(
        default=None,
        description="Alternative field name for base64 audio payload."
    )
    data: Optional[str] = Field(
        default=None,
        description="Alternative field name for base64 audio payload."
    )
    call_id: Optional[str] = Field(
        default=None,
        description="Optional unique identifier for call session auditing."
    )
    engine: Optional[str] = Field(
        default="multimodal",
        description="Engine selector: 'multimodal' (SOTA 4-Pillar), 'baseline' (Fast CNN/GBM), or 'dual' (Ensemble)."
    )

    def get_audio_str(self) -> str:
        payload = self.audio or self.audio_base64 or self.wav_base64 or self.data
        if not payload:
            raise ValueError("No audio payload provided in request.")
        return payload


class DetectResponse(BaseModel):
    """
    Strict HackMTY Challenge Response Contract.
    """
    is_synthetic: bool = Field(
        ...,
        description="True if caller voice (Channel 0) is detected as AI-generated / deepfake / synthetic."
    )
    confidence: float = Field(
        ...,
        description="Confidence score in the prediction [0.0 - 1.0]."
    )


class ConsensusResult(BaseModel):
    models_agree: bool
    consensus_verdict: str
    agreement_status: str
    recommended_verdict: str
    probability_spread: float
    total_wall_latency_ms: float


class CompareResponse(BaseModel):
    consensus: ConsensusResult
    multimodal: Dict[str, Any]
    baseline: Dict[str, Any]
    audio_metadata: Dict[str, Any]
    speech_intervals: Optional[List[List[float]]] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    available_engines: List[str]
    device: str
    calibrated_threshold: float
