import io
import base64
import numpy as np
import soundfile as sf
import librosa
from typing import Tuple


def decode_and_preprocess_audio(
    base64_audio_str: str,
    target_sr: int = 16000,
    target_samples: int = 64000
) -> Tuple[np.ndarray, bytes, int, int]:
    """
    Decodes Base64-encoded audio, separates Channel 0 (Caller),
    and resamples to 16kHz for model inference.

    Returns:
        caller_16k (np.ndarray): 1D float32 array @ 16kHz (windowed to target_samples).
        raw_wav_bytes (bytes): Original audio bytes for S3 auditing.
        orig_sr (int): Original sample rate.
        num_channels (int): Number of audio channels.
    """
    # 1. Base64 Decode
    try:
        # Strip header if present (e.g., data:audio/wav;base64,...)
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

    # 3. Channel Separation (Channel 0 = Caller, Channel 1 = Agent)
    if data.ndim == 2:
        num_channels = data.shape[1]
        caller_audio = data[:, 0]  # Extract Channel 0 (Caller)
    else:
        num_channels = 1
        caller_audio = data

    # 4. Resample to target_sr (16000 Hz) if needed
    if orig_sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(target_sr), int(orig_sr))
        up = int(target_sr) // g
        down = int(orig_sr) // g
        caller_16k = resample_poly(caller_audio, up, down).astype(np.float32)
    else:
        caller_16k = caller_audio

    # 5. Fix length to target_samples (4.0s @ 16kHz) for low-latency inference
    n_samples = len(caller_16k)
    if n_samples > target_samples:
        start_idx = (n_samples - target_samples) // 2
        caller_16k = caller_16k[start_idx : start_idx + target_samples]
    elif n_samples < target_samples and n_samples > 0:
        # Repeat or zero-pad
        repeats = int(np.ceil(target_samples / n_samples))
        caller_16k = np.tile(caller_16k, repeats)[:target_samples]
    elif n_samples == 0:
        caller_16k = np.zeros(target_samples, dtype=np.float32)

    return caller_16k.astype(np.float32), raw_wav_bytes, orig_sr, num_channels
