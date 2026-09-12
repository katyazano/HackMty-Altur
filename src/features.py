import os
import sys
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.aasist import AASIST


class AcousticFeatureExtractor:
    """
    Extracts high-dimensional acoustic, spectral, and prosodic DSP features
    specialized for speech anti-spoofing and vocoder artifact detection.
    """

    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 20):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.feature_names: List[str] = self._generate_feature_names()

    def _generate_feature_names(self) -> List[str]:
        names = []
        # MFCC stats
        for i in range(1, self.n_mfcc + 1):
            names.extend([f"mfcc_{i}_mean", f"mfcc_{i}_std"])
        # MFCC Delta stats
        for i in range(1, self.n_mfcc + 1):
            names.extend([f"mfcc_delta_{i}_mean", f"mfcc_delta_{i}_std"])
        # Spectral features
        names.extend(["spec_centroid_mean", "spec_centroid_std"])
        names.extend(["spec_bandwidth_mean", "spec_bandwidth_std"])
        names.extend(["spec_rolloff_mean", "spec_rolloff_std"])
        names.extend(["spec_flatness_mean", "spec_flatness_std"])
        for i in range(7):
            names.extend([f"spec_contrast_{i}_mean", f"spec_contrast_{i}_std"])
        # Temporal & Energy
        names.extend(["zcr_mean", "zcr_std"])
        names.extend(["rms_mean", "rms_std"])
        # Chroma
        for i in range(12):
            names.extend([f"chroma_{i}_mean", f"chroma_{i}_std"])
        # Prosody / Pitch
        names.extend(["pitch_f0_mean", "pitch_f0_std", "voiced_fraction"])
        # Telephony & Vocoder Cues
        names.extend(["hf_energy_ratio", "clipping_ratio", "crest_factor"])
        return names

    def extract_features(self, y: np.ndarray, sr: Optional[int] = None, max_samples: int = 64000) -> np.ndarray:
        """
        Computes 1D feature vector from audio waveform array.
        Uses standard 4-second (64k samples @ 16kHz) central slice for fast, robust extraction.
        """
        if sr is None:
            sr = self.sample_rate

        # Ensure mono float32
        if y.ndim > 1:
            y = y[:, 0]
        y = y.astype(np.float32)

        # Standard 4-second central representation (consistent with AASIST/RawNet2 input)
        nb_samples = len(y)
        if max_samples and nb_samples > max_samples:
            mid = nb_samples // 2
            half = max_samples // 2
            y = y[mid - half : mid + half]
        elif max_samples and nb_samples < max_samples:
            y = np.pad(y, (0, max_samples - nb_samples), mode="constant")

        # Handle silent inputs
        if len(y) < sr * 0.1 or np.max(np.abs(y)) < 1e-6:
            return np.zeros(len(self.feature_names), dtype=np.float32)

        features = []

        # 1. MFCCs & Deltas
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        for i in range(self.n_mfcc):
            features.extend([np.mean(mfcc[i]), np.std(mfcc[i])])
        for i in range(self.n_mfcc):
            features.extend([np.mean(mfcc_delta[i]), np.std(mfcc_delta[i])])

        # 2. Spectral Centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.extend([np.mean(cent), np.std(cent)])

        # 3. Spectral Bandwidth
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.extend([np.mean(bw), np.std(bw)])

        # 4. Spectral Roll-off
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.extend([np.mean(rolloff), np.std(rolloff)])

        # 5. Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=y)
        features.extend([np.mean(flatness), np.std(flatness)])

        # 6. Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for i in range(contrast.shape[0]):
            features.extend([np.mean(contrast[i]), np.std(contrast[i])])

        # 7. Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=y)
        features.extend([np.mean(zcr), np.std(zcr)])

        # 8. RMS Energy
        rms = librosa.feature.rms(y=y)
        features.extend([np.mean(rms), np.std(rms)])

        # 9. Chroma STFT
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for i in range(12):
            features.extend([np.mean(chroma[i]), np.std(chroma[i])])

        # 10. Pitch (F0) Estimation via Yin
        try:
            f0 = librosa.yin(y=y, fmin=60, fmax=350, sr=sr, frame_length=1024, hop_length=1024)
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 0:
                features.extend([np.mean(valid_f0), np.std(valid_f0), len(valid_f0) / len(f0)])
            else:
                features.extend([0.0, 0.0, 0.0])
        except Exception:
            features.extend([0.0, 0.0, 0.0])

        # 11. Telephony & Vocoder Cues
        # Energy above 3.4kHz (narrowband boundary)
        stft = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        hf_mask = freqs > 3400
        hf_energy = np.sum(stft[hf_mask, :])
        total_energy = np.sum(stft) + 1e-9
        hf_ratio = float(hf_energy / total_energy)
        features.append(hf_ratio)

        # Clipping Ratio
        clip_ratio = float(np.sum(np.abs(y) >= 0.98) / len(y))
        features.append(clip_ratio)

        # Crest Factor (Peak to RMS)
        mean_rms = np.mean(rms) + 1e-9
        crest = float(np.max(np.abs(y)) / mean_rms)
        features.append(crest)

        feat_arr = np.array(features, dtype=np.float32)
        # Replace any infs or nans with 0
        feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)
        return feat_arr


class AASISTEmbeddingExtractor:
    """
    Extracts deep 128-dim graph embeddings and logit outputs from a trained AASIST model.
    """

    def __init__(self, model: Optional[AASIST] = None, weights_path: Optional[str] = None, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = AASIST().to(self.device)
            if weights_path and os.path.exists(weights_path):
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def extract_from_waveform(self, waveform: torch.Tensor, target_samples: int = 64000) -> np.ndarray:
        """
        Runs forward pass on waveform tensor and returns:
        [latent_embedding (128-d), logits (2-d), softmax_probs (2-d)] -> total 132 dimensions.
        """
        self.model.eval()
        with torch.no_grad():
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            waveform = waveform.to(self.device)

            nb_samples = waveform.shape[-1]
            if nb_samples > target_samples:
                mid = nb_samples // 2
                waveform_in = waveform[:, mid - target_samples // 2 : mid + target_samples // 2]
            elif nb_samples < target_samples:
                waveform_in = F.pad(waveform, (0, target_samples - nb_samples))
            else:
                waveform_in = waveform

            feat, logits = self.model.extract_embeddings(waveform_in)
            probs = F.softmax(logits, dim=1)

            feat_np = feat.squeeze(0).cpu().numpy()
            logits_np = logits.squeeze(0).cpu().numpy()
            probs_np = probs.squeeze(0).cpu().numpy()

            combined = np.concatenate([feat_np, logits_np, probs_np], axis=0).astype(np.float32)
            return combined


class MultiModelFeaturePipeline:
    """
    Unified extractor combining AASIST spectro-temporal graph representations
    and domain-specific acoustic DSP features.
    """

    def __init__(self, aasist_extractor: AASISTEmbeddingExtractor, acoustic_extractor: Optional[AcousticFeatureExtractor] = None):
        self.aasist_extractor = aasist_extractor
        self.acoustic_extractor = acoustic_extractor or AcousticFeatureExtractor()

    def extract_combined_vector(self, waveform_tensor: torch.Tensor, sample_rate: int = 16000) -> np.ndarray:
        """
        Extracts concatenated feature vector for XGBoost meta-learner.
        """
        # 1. AASIST features (132-dim)
        aasist_feats = self.aasist_extractor.extract_from_waveform(waveform_tensor)

        # 2. Acoustic features (134-dim)
        waveform_np = waveform_tensor.detach().cpu().squeeze().numpy()
        acoustic_feats = self.acoustic_extractor.extract_features(waveform_np, sr=sample_rate)

        # 3. Concatenate
        combined = np.concatenate([aasist_feats, acoustic_feats], axis=0)
        return combined
