import io
import base64
import numpy as np
import soundfile as sf
from typing import Tuple
from src.dynamic_separator import dynamic_extract_caller_speech


def decode_and_preprocess_audio(
    base64_audio_str: str,
    target_sr: int = 16000,
    target_samples: int = None
) -> Tuple[np.ndarray, bytes, int, int]:
    """
    Decodes Base64-encoded audio, isolates Channel 0 (Caller),
    dynamically filters out silences/agent speaking turns using energy VAD,
    and returns 16kHz caller speech.

    Returns:
        caller_16k (np.ndarray): 1D float32 array @ 16kHz.
        raw_wav_bytes (bytes): Original audio bytes for S3 auditing.
        orig_sr (int): Original sample rate.
        num_channels (int): Number of audio channels.
    """
    # 1. Base64 Decode
    try:
        if "," in base64_audio_str:
            base64_audio_str = base64_audio_str.split(",", 1)[1]
        raw_wav_bytes = base64.b64decode(base64_audio_str)
    except Exception as e:
        raise ValueError(f"Invalid Base64 encoding: {e}")

    if len(raw_wav_bytes) == 0:
        raise ValueError("Decoded audio payload is empty (0 bytes).")

    # 2. Read audio via soundfile
    try:
        data, orig_sr = sf.read(io.BytesIO(raw_wav_bytes), dtype="float32")
    except Exception as e:
        raise ValueError(f"Unable to decode WAV audio: {e}")

    num_channels = data.shape[1] if data.ndim == 2 else 1

    # 3. Dynamic Channel 0 Speech Extraction with Energy VAD & Gap Bridging
    caller_16k, intervals = dynamic_extract_caller_speech(
        audio_data=data,
        orig_sr=orig_sr,
        target_sr=target_sr,
        target_samples=target_samples,
        top_db=32.0,
        pad_ms=180,
        min_silence_bridge_ms=300
    )

    return caller_16k.astype(np.float32), raw_wav_bytes, orig_sr, num_channels
