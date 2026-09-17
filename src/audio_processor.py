"""
audio_processor.py — Unified Audio Signal Processing Engine.
Standardizes audio ingestion for both Baseline (8kHz stereo) and Multimodal (16kHz isolated caller).
"""
import io
import base64
from math import gcd
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import resample_poly


class AudioProcessor:
    """
    Unified Audio Ingestion & Preprocessing Suite.
    """

    @staticmethod
    def decode_base64_to_bytes(base64_str: str) -> bytes:
        """Decodes raw base64 string or data-uri scheme."""
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        raw_bytes = base64.b64decode(base64_str.strip())
        if len(raw_bytes) == 0:
            raise ValueError("Decoded audio payload is empty.")
        return raw_bytes

    @staticmethod
    def load_audio(
        audio_input: Any
    ) -> Tuple[np.ndarray, int, int, bytes]:
        """
        Loads audio from bytes, file path, or base64.
        Returns:
            data (np.ndarray): Shape (samples, channels) or (samples,) float32
            sr (int): Sampling rate
            channels (int): Channel count (1 or 2)
            raw_bytes (bytes): Original audio bytes
        """
        if isinstance(audio_input, str):
            if audio_input.startswith("data:") or len(audio_input) > 260 or "/" not in audio_input[:20]:
                raw_bytes = AudioProcessor.decode_base64_to_bytes(audio_input)
                data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
            else:
                with open(audio_input, "rb") as f:
                    raw_bytes = f.read()
                data, sr = sf.read(audio_input, dtype="float32", always_2d=True)
        elif isinstance(audio_input, bytes):
            raw_bytes = audio_input
            data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

        channels = data.shape[1]
        return data, sr, channels, raw_bytes

    @staticmethod
    def extract_channels_8k(
        data: np.ndarray,
        orig_sr: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts Channel 0 (Caller) and Channel 1 (Agent) resampled to 8kHz.
        """
        if data.shape[1] == 1:
            caller = data[:, 0]
            agent = np.zeros_like(caller)
        else:
            caller = data[:, 0]
            agent = data[:, 1]

        if orig_sr != 8000:
            caller = librosa.resample(caller, orig_sr=orig_sr, target_sr=8000)
            agent = librosa.resample(agent, orig_sr=orig_sr, target_sr=8000)

        return caller.astype(np.float32), agent.astype(np.float32)

    @staticmethod
    def extract_caller_16k_clean(
        data: np.ndarray,
        orig_sr: int,
        top_db: float = 32.0,
        pad_ms: int = 180,
        min_silence_bridge_ms: int = 300
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Extracts Channel 0 (Caller) at 16kHz with dynamic energy VAD and silence bridging.
        """
        caller_raw = data[:, 0] if data.ndim == 2 else data
        caller_raw = caller_raw.astype(np.float32)

        if len(caller_raw) == 0 or np.max(np.abs(caller_raw)) < 1e-5:
            return np.zeros(16000, dtype=np.float32), []

        # Resample to 16kHz
        target_sr = 16000
        if orig_sr != target_sr:
            g = gcd(int(target_sr), int(orig_sr))
            caller_audio = resample_poly(caller_raw, target_sr // g, orig_sr // g).astype(np.float32)
        else:
            caller_audio = caller_raw

        total_samples = len(caller_audio)

        try:
            raw_intervals = librosa.effects.split(caller_audio, top_db=top_db, frame_length=1024, hop_length=256)
        except Exception:
            raw_intervals = np.array([[0, total_samples]])

        if len(raw_intervals) == 0:
            raw_intervals = np.array([[0, total_samples]])

        pad_samples = int(target_sr * (pad_ms / 1000.0))
        bridge_samples = int(target_sr * (min_silence_bridge_ms / 1000.0))

        merged = []
        for s, e in raw_intervals:
            sp = max(0, s - pad_samples)
            ep = min(total_samples, e + pad_samples)
            if not merged:
                merged.append([sp, ep])
            else:
                if sp <= (merged[-1][1] + bridge_samples):
                    merged[-1][1] = max(merged[-1][1], ep)
                else:
                    merged.append([sp, ep])

        intervals_s = [(round(s / target_sr, 3), round(e / target_sr, 3)) for s, e in merged]
        speech_segments = [caller_audio[s:e] for s, e in merged]
        active_speech = np.concatenate(speech_segments) if speech_segments else caller_audio

        return active_speech.astype(np.float32), intervals_s

    @staticmethod
    def get_audio_metadata(data: np.ndarray, sr: int) -> Dict[str, Any]:
        """Calculates audio telemetry and signal quality metrics."""
        duration_s = round(data.shape[0] / sr, 2)
        channels = data.shape[1] if data.ndim == 2 else 1
        peak_amp = round(float(np.max(np.abs(data))), 4)
        rms = round(float(np.sqrt(np.mean(data ** 2))), 4)
        return {
            "duration_s": duration_s,
            "sample_rate": sr,
            "channels": channels,
            "peak_amplitude": peak_amp,
            "rms_energy": rms,
            "is_stereo": channels >= 2
        }
