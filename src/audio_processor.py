import os
import numpy as np
import librosa
import noisereduce as nr
from scipy.signal import butter, filtfilt

class AudioProcessor:
    """
    Audio Signal Processing (ASP) engine for Anti-Spoofing.
    Handles loading, audio standardization, spectral denoising, bandpass filtering, and VAD.
    """

    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr

    def load_and_normalize(self, file_path: str) -> tuple[np.ndarray, int]:
        """
        Loads audio file, converts to mono, resamples to target_sr, and normalizes amplitude.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Load audio and resample
        y, sr = librosa.load(file_path, sr=self.target_sr, mono=True)

        # Amplitude Normalization (Peak / RMS)
        max_amp = np.max(np.abs(y))
        if max_amp > 0:
            y = y / max_amp

        return y, self.target_sr

    def apply_denoising(self, y: np.ndarray, prop_decrease: float = 0.72) -> np.ndarray:
        """
        Applies Spectral Gating / Noise Reduction with balanced subtraction (prop_decrease=0.72).
        Removes background chatter while preserving soft speech formants.
        """
        try:
            reduced = nr.reduce_noise(y=y, sr=self.target_sr, prop_decrease=prop_decrease)
            return reduced
        except Exception:
            return y

    def apply_bandpass_filter(self, y: np.ndarray, lowcut: float = 90.0, highcut: float = 3900.0, order: int = 4) -> np.ndarray:
        """
        Applies Butterworth Bandpass filter (90Hz - 3900Hz) focusing on core human speech.
        """
        nyquist = 0.5 * self.target_sr
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered = filtfilt(b, a, y)
        return filtered

    def apply_vad(self, y: np.ndarray, top_db: int = 24) -> np.ndarray:
        """
        Voice Activity Detection (VAD) with balanced threshold (top_db=24).
        Suppresses background voice chatter while keeping spoken words intact.
        """
        non_silent_intervals = librosa.effects.split(y, top_db=top_db)
        if len(non_silent_intervals) == 0:
            return y

        processed = np.concatenate([y[start:end] for start, end in non_silent_intervals])
        return processed
