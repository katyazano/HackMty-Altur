"""
multimodal_engine.py — 4-Pillar Multimodal Anti-Spoofing Engine.
Integrates AASIST (GNN), RawNet2 (SincNet), 136-dim Acoustic DSP, and 24-dim Whisper NLP into XGBoost.
"""
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
import xgboost as xgb
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from src.models.aasist import AASIST
from src.models.rawnet2 import RawNet2
from src.features_multimodal import (
    AcousticFeatureExtractor,
    AASISTEmbeddingExtractor,
    RawNet2ScoreExtractor,
    WhisperMultimodalExtractor
)

logger = logging.getLogger("altur.multimodal")


class MultimodalEngine:
    """
    Engine 2: 4-Pillar Multimodal Deepfake Defense Engine (296 dimensions).
    1. AASIST: Graph Neural Network modeling spectral-temporal graph relationships (132 dims)
    2. RawNet2: SincNet raw waveform feature learning without STFT bias (4 dims)
    3. Acoustic DSP: High-resolution spectral, prosodic, MFCC, Wiener entropy (136 dims)
    4. Whisper NLP: Mexican Spanish filler & disfluency semantics + attention latents (24 dims)
    Meta-Learner: Calibrated Regularized XGBoost Quad Classifier.
    """

    def __init__(
        self,
        weights_dir: Optional[str] = None,
        calibrated_threshold: float = 0.50,
        device: Optional[str] = None
    ):
        self.repo_root = PROJECT_ROOT
        weights_path = Path(weights_dir) if weights_dir else (self.repo_root / "weights")
        
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.calibrated_threshold = calibrated_threshold

        # 1. Initialize AASIST
        self.aasist_model = AASIST().to(self.device)
        aasist_weights = weights_path / "aasist_best.pth"
        if aasist_weights.exists():
            self.aasist_model.load_state_dict(
                torch.load(str(aasist_weights), map_location=self.device)
            )
        self.aasist_model.eval()
        self.aasist_extractor = AASISTEmbeddingExtractor(
            model=self.aasist_model, device=str(self.device)
        )

        # 2. Initialize RawNet2
        self.rawnet2_model = RawNet2().to(self.device)
        rawnet_weights = weights_path / "rawnet2_best.pth"
        if rawnet_weights.exists():
            self.rawnet2_model.load_state_dict(
                torch.load(str(rawnet_weights), map_location=self.device)
            )
        self.rawnet2_model.eval()
        self.rawnet2_extractor = RawNet2ScoreExtractor(
            model=self.rawnet2_model, device=str(self.device)
        )

        # 3. Acoustic DSP Extractor
        self.acoustic_extractor = AcousticFeatureExtractor(sample_rate=16000)

        # 4. Whisper Multimodal Extractor
        self.whisper_extractor = WhisperMultimodalExtractor(model_size="tiny", device=str(self.device))

        # 5. Load XGBoost Classifier & Scaler
        self.xgb_model = xgb.XGBClassifier()
        self.scaler = None
        self.is_quad = False

        quad_model_path = weights_path / "xgboost_quad_ensemble.json"
        quad_scaler_path = weights_path / "quad_scaler.joblib"
        triple_model_path = weights_path / "xgboost_triple_ensemble.json"
        triple_scaler_path = weights_path / "triple_scaler.joblib"

        if quad_model_path.exists() and quad_scaler_path.exists():
            self.xgb_model.load_model(str(quad_model_path))
            self.scaler = joblib.load(str(quad_scaler_path))
            self.is_quad = True
        elif triple_model_path.exists():
            self.xgb_model.load_model(str(triple_model_path))
            if triple_scaler_path.exists():
                self.scaler = joblib.load(str(triple_scaler_path))

    def predict(
        self,
        caller_audio_16k: np.ndarray,
        return_transcript: bool = False
    ) -> Dict[str, Any]:
        """
        Runs 4-pillar multimodal feature extraction and XGBoost classification.
        """
        t0 = time.perf_counter()

        if len(caller_audio_16k) == 0:
            return {
                "engine": "multimodal",
                "name": "4-Pillar Multimodal (AASIST + RawNet2 + DSP + Whisper + XGBoost)",
                "is_synthetic": False,
                "confidence": 0.50,
                "probability_synthetic": 0.50,
                "threshold": self.calibrated_threshold,
                "latency_ms": 0.0,
                "pillar_breakdown": {},
                "turns": []
            }

        waveform_tensor = torch.tensor(caller_audio_16k, dtype=torch.float32)

        # Pillar 1: AASIST (132 dims)
        t_aasist_0 = time.perf_counter()
        aasist_feats = self.aasist_extractor.extract_from_waveform(waveform_tensor)
        aasist_time_ms = round((time.perf_counter() - t_aasist_0) * 1000, 1)

        # Pillar 2: RawNet2 (4 dims)
        t_raw_0 = time.perf_counter()
        rawnet_feats = self.rawnet2_extractor.extract_from_waveform(waveform_tensor)
        rawnet_time_ms = round((time.perf_counter() - t_raw_0) * 1000, 1)

        # Pillar 3: Acoustic DSP (136 dims)
        t_dsp_0 = time.perf_counter()
        acoustic_feats = self.acoustic_extractor.extract_features(caller_audio_16k, sr=16000)
        dsp_time_ms = round((time.perf_counter() - t_dsp_0) * 1000, 1)

        # Pillar 4: Whisper Multimodal (24 dims) + dialogue turns
        t_wh_0 = time.perf_counter()
        turns = []
        if self.is_quad:
            whisper_feats, turns = self.whisper_extractor.extract_features(
                caller_audio_16k, sr=16000, return_transcript=True
            )
            combined_feats = np.concatenate(
                [aasist_feats, rawnet_feats, acoustic_feats, whisper_feats], axis=0
            ).reshape(1, -1)
        else:
            combined_feats = np.concatenate(
                [aasist_feats, rawnet_feats, acoustic_feats], axis=0
            ).reshape(1, -1)
            whisper_feats = np.zeros(24, dtype=np.float32)
        whisper_time_ms = round((time.perf_counter() - t_wh_0) * 1000, 1)

        # Scale features
        if self.scaler is not None:
            scaled_feats = self.scaler.transform(combined_feats)
        else:
            scaled_feats = combined_feats

        # XGBoost Prediction
        probs = self.xgb_model.predict_proba(scaled_feats)[0]
        synthetic_prob = float(probs[1])

        # Calibrated decision
        is_synthetic = bool(synthetic_prob >= self.calibrated_threshold)
        certainty = synthetic_prob if is_synthetic else (1.0 - synthetic_prob)
        confidence = round(float(np.clip(certainty, 0.50, 1.0)), 4)
        total_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # Sub-pillar metrics
        aasist_prob = float(aasist_feats[131]) if len(aasist_feats) > 131 else synthetic_prob
        rawnet_prob = float(rawnet_feats[3]) if len(rawnet_feats) > 3 else synthetic_prob
        filler_density = float(whisper_feats[20]) if len(whisper_feats) > 20 else 0.0
        authenticity_score = float(whisper_feats[23]) if len(whisper_feats) > 23 else 0.50

        res = {
            "engine": "multimodal",
            "name": "4-Pillar Multimodal SOTA (AASIST + RawNet2 + DSP + Whisper + XGBoost)",
            "is_synthetic": is_synthetic,
            "confidence": confidence,
            "probability_synthetic": round(synthetic_prob, 4),
            "threshold": round(self.calibrated_threshold, 4),
            "latency_ms": total_latency_ms,
            "feature_dim": combined_feats.shape[1],
            "pillar_breakdown": {
                "aasist_gnn": {
                    "dims": 132,
                    "spoof_prob": round(aasist_prob, 4),
                    "latency_ms": aasist_time_ms,
                    "description": "Graph Attention Spectral-Temporal micro-inconsistencies"
                },
                "rawnet2_sinc": {
                    "dims": 4,
                    "spoof_prob": round(rawnet_prob, 4),
                    "latency_ms": rawnet_time_ms,
                    "description": "SincNet raw waveform filters avoiding STFT phase distortion"
                },
                "acoustic_dsp": {
                    "dims": 136,
                    "latency_ms": dsp_time_ms,
                    "description": "MFCCs, Wiener Entropy, Jitter, Shimmer, High-Freq Energy"
                },
                "whisper_nlp": {
                    "dims": 24,
                    "filler_density": round(filler_density, 4),
                    "authenticity_score": round(authenticity_score, 4),
                    "latency_ms": whisper_time_ms,
                    "description": "Encoder latents + Mexican Spanish colloquial filler pattern NLP"
                }
            }
        }
        if return_transcript:
            res["turns"] = turns or []
        return res
