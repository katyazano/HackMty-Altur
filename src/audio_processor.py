import os
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from scipy.fftpack import dct
from scipy.signal import butter, filtfilt

class AudioProcessor:
    """
    Audio Signal Processing (ASP) engine for Voice Biometrics & Anti-Spoofing.
    Handles loading, audio standardization, spectral denoising, bandpass filtering, VAD, MFCC, and LFCC extraction.
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
        Strikes the perfect balance: suppresses background voice chatter while keeping spoken words intact.
        """
        non_silent_intervals = librosa.effects.split(y, top_db=top_db)
        if len(non_silent_intervals) == 0:
            return y

        processed = np.concatenate([y[start:end] for start, end in non_silent_intervals])
        return processed

    def extract_mfcc(self, y: np.ndarray, n_mfcc: int = 20) -> np.ndarray:
        """
        Extracts Mel-Frequency Cepstral Coefficients (MFCCs).
        Returns a compact 2x n_mfcc representation (mean + std over time).
        """
        mfcc = librosa.feature.mfcc(y=y, sr=self.target_sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        embedding = np.hstack((mfcc_mean, mfcc_std))
        return embedding

    def extract_lfcc(self, y: np.ndarray, n_lfcc: int = 20, n_fft: int = 512, hop_length: int = 160) -> np.ndarray:
        """
        Extracts Linear-Frequency Cepstral Coefficients (LFCCs).
        """
        stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        n_spec = stft.shape[0]

        num_filters = 40
        linear_filters = np.zeros((num_filters, n_spec))
        freqs = np.linspace(0, self.target_sr / 2, n_spec)
        filter_freqs = np.linspace(0, self.target_sr / 2, num_filters + 2)

        for i in range(num_filters):
            f_m_minus = filter_freqs[i]
            f_m = filter_freqs[i + 1]
            f_m_plus = filter_freqs[i + 2]

            for k in range(n_spec):
                if f_m_minus <= freqs[k] < f_m:
                    linear_filters[i, k] = (freqs[k] - f_m_minus) / (f_m - f_m_minus + 1e-8)
                elif f_m <= freqs[k] <= f_m_plus:
                    linear_filters[i, k] = (f_m_plus - freqs[k]) / (f_m_plus - f_m + 1e-8)

        energy = np.dot(linear_filters, stft)
        log_energy = np.log(energy + 1e-8)

        lfcc = dct(log_energy, type=2, axis=0, norm='ortho')[:n_lfcc]

        lfcc_mean = np.mean(lfcc, axis=1)
        lfcc_std = np.std(lfcc, axis=1)
        return np.hstack((lfcc_mean, lfcc_std))

    def compute_spectral_features(self, y: np.ndarray) -> dict:
        """
        Computes auxiliary signal metrics.
        """
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=self.target_sr)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
        
        stft = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=self.target_sr)
        high_freq_mask = freqs >= 4000
        
        total_energy = np.sum(stft**2) + 1e-8
        high_freq_energy = np.sum(stft[high_freq_mask, :]**2)
        hf_ratio = float(high_freq_energy / total_energy)

        return {
            "spectral_centroid_hz": round(centroid, 2),
            "zero_crossing_rate": round(zcr, 4),
            "high_freq_energy_ratio": round(hf_ratio, 4)
        }
