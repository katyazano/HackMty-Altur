import numpy as np
import librosa
from scipy.signal import resample_poly
from math import gcd
from typing import Tuple, List, Optional


def dynamic_extract_caller_speech(
    audio_data: np.ndarray,
    orig_sr: int,
    target_sr: int = 16000,
    target_samples: Optional[int] = None,
    top_db: float = 32.0,
    frame_length: int = 1024,
    hop_length: int = 256,
    pad_ms: int = 180,
    min_silence_bridge_ms: int = 300
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Dynamically isolates Channel 0 (Caller) and extracts all active speech
    segments preserving full call length, word boundaries, and conversational turns.

    Args:
        audio_data (np.ndarray): Multi-channel or mono audio array.
        orig_sr (int): Sampling rate of input audio.
        target_sr (int): Target sampling rate for model inference (default: 16000).
        target_samples (Optional[int]): If provided, fixes length to target_samples (e.g., for batch tensors).
                                        If None, returns full concatenated caller speech.
        top_db (float): Silence threshold below peak dB (default: 32.0 dB).
        frame_length (int): Frame length for energy calculation.
        hop_length (int): Hop length for energy calculation.
        pad_ms (int): Safety padding in ms before and after each detected speech burst.
        min_silence_bridge_ms (int): Max silence gap in ms between phrases to bridge as continuous speech.

    Returns:
        speech_16k (np.ndarray): 1D float32 array @ target_sr (full length or target_samples).
        intervals_s (List[Tuple[float, float]]): Detected speech intervals in seconds [(start_s, end_s), ...].
    """
    # 1. Isolate Channel 0 (Caller)
    if audio_data.ndim == 2:
        caller_raw = audio_data[:, 0]
    else:
        caller_raw = audio_data

    caller_raw = caller_raw.astype(np.float32)

    # Empty / Silent signal guard
    if len(caller_raw) == 0 or np.max(np.abs(caller_raw)) < 1e-5:
        out_len = target_samples if target_samples is not None else int(target_sr * 1.0)
        return np.zeros(out_len, dtype=np.float32), []

    # 2. Resample to target_sr if necessary
    if orig_sr != target_sr:
        g = gcd(int(target_sr), int(orig_sr))
        up = int(target_sr) // g
        down = int(orig_sr) // g
        caller_audio = resample_poly(caller_raw, up, down).astype(np.float32)
        sr = target_sr
    else:
        caller_audio = caller_raw
        sr = orig_sr

    total_samples = len(caller_audio)

    # 3. Dynamic Energy VAD using non-silent intervals
    try:
        raw_intervals = librosa.effects.split(
            caller_audio,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )
    except Exception:
        raw_intervals = np.array([[0, total_samples]])

    # Fallback to relaxed threshold if nothing detected
    if len(raw_intervals) == 0:
        try:
            raw_intervals = librosa.effects.split(
                caller_audio,
                top_db=top_db + 8.0,
                frame_length=frame_length,
                hop_length=hop_length
            )
        except Exception:
            raw_intervals = np.array([[0, total_samples]])

    if len(raw_intervals) == 0:
        raw_intervals = np.array([[0, total_samples]])

    # 4. Apply Boundary Padding & Bridge Short Intra-Sentence Pauses
    pad_samples = int(sr * (pad_ms / 1000.0))
    bridge_samples = int(sr * (min_silence_bridge_ms / 1000.0))

    merged_intervals = []
    for start, end in raw_intervals:
        start_padded = max(0, start - pad_samples)
        end_padded = min(total_samples, end + pad_samples)

        if not merged_intervals:
            merged_intervals.append([start_padded, end_padded])
        else:
            prev_start, prev_end = merged_intervals[-1]
            # If the gap between previous speech end and current start is smaller than bridge_samples, merge!
            if start_padded <= (prev_end + bridge_samples):
                merged_intervals[-1][1] = max(prev_end, end_padded)
            else:
                merged_intervals.append([start_padded, end_padded])

    # Convert intervals to seconds for output
    intervals_s = [
        (round(s / sr, 3), round(e / sr, 3))
        for s, e in merged_intervals
    ]

    # 5. Extract & Concatenate active speech segments
    speech_segments = [caller_audio[s:e] for s, e in merged_intervals]
    if speech_segments:
        active_speech = np.concatenate(speech_segments)
    else:
        active_speech = caller_audio

    # 6. Length Formatting
    if target_samples is None:
        # Full length requested
        speech_16k = active_speech
    else:
        n_active = len(active_speech)
        if n_active >= target_samples:
            # Select most energetic window
            step = int(sr * 0.5)
            best_window = active_speech[:target_samples]
            best_energy = np.mean(best_window ** 2)

            for i in range(0, n_active - target_samples + 1, step):
                window = active_speech[i : i + target_samples]
                energy = np.mean(window ** 2)
                if energy > best_energy:
                    best_energy = energy
                    best_window = window

            speech_16k = best_window
        elif 0 < n_active < target_samples:
            repeats = int(np.ceil(target_samples / n_active))
            speech_16k = np.tile(active_speech, repeats)[:target_samples]
        else:
            speech_16k = np.zeros(target_samples, dtype=np.float32)

    return speech_16k.astype(np.float32), intervals_s
