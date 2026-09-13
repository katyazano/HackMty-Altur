import os
import sys
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union
import numpy as np
import torch
import xgboost as xgb
import joblib

# OpenMP runtime collision guard
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.aasist import AASIST
from src.models.rawnet2 import RawNet2
from src.features import (
    AcousticFeatureExtractor,
    AASISTEmbeddingExtractor,
    RawNet2ScoreExtractor,
    WhisperMultimodalExtractor
)
from src.api.config import settings

logger = logging.getLogger("altur.inference")


class InferenceEngine:
    """
    Production-grade 4-Pillar Multimodal Biometric Inference Engine.
    Combines AASIST (GNN), RawNet2 (Raw Waveform), Acoustic DSP, and Whisper Multimodal into XGBoost.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.calibrated_threshold = settings.calibrated_threshold

        # 1. Load AASIST
        self.aasist_model = AASIST().to(self.device)
        if os.path.exists(settings.aasist_weights):
            self.aasist_model.load_state_dict(
                torch.load(settings.aasist_weights, map_location=self.device)
            )
            logger.info(f"Loaded AASIST weights from {settings.aasist_weights}")
        else:
            logger.warning(f"AASIST weights not found at {settings.aasist_weights}")
        self.aasist_model.eval()
        self.aasist_extractor = AASISTEmbeddingExtractor(
            model=self.aasist_model, device=str(self.device)
        )

        # 2. Load RawNet2
        self.rawnet2_model = RawNet2().to(self.device)
        if os.path.exists(settings.rawnet2_weights):
            self.rawnet2_model.load_state_dict(
                torch.load(settings.rawnet2_weights, map_location=self.device)
            )
            logger.info(f"Loaded RawNet2 weights from {settings.rawnet2_weights}")
        else:
            logger.warning(f"RawNet2 weights not found at {settings.rawnet2_weights}")
        self.rawnet2_model.eval()
        self.rawnet2_extractor = RawNet2ScoreExtractor(
            model=self.rawnet2_model, device=str(self.device)
        )

        # 3. Acoustic Feature Extractor
        self.acoustic_extractor = AcousticFeatureExtractor(sample_rate=settings.target_sample_rate)

        # 4. Whisper Multimodal Extractor
        self.whisper_extractor = WhisperMultimodalExtractor(model_size="tiny", device=str(self.device))

        # 5. Load XGBoost Classifier & Scaler (Prefer 296-dim Quad; Fallback to 272-dim Triple)
        self.is_quad = False
        self.xgb_model = xgb.XGBClassifier()
        self.scaler = None

        if os.path.exists(settings.xgb_quad_model_path) and os.path.exists(settings.quad_scaler_path):
            self.xgb_model.load_model(settings.xgb_quad_model_path)
            self.scaler = joblib.load(settings.quad_scaler_path)
            self.is_quad = True
            logger.info(f"Loaded 4-Pillar 296-dim Quad XGBoost ensemble from {settings.xgb_quad_model_path}")
        elif os.path.exists(settings.xgb_model_path):
            self.xgb_model.load_model(settings.xgb_model_path)
            if os.path.exists(settings.scaler_path):
                self.scaler = joblib.load(settings.scaler_path)
            logger.info(f"Loaded 3-Model 272-dim Triple XGBoost ensemble from {settings.xgb_model_path}")

    def predict(
        self,
        caller_audio_16k: np.ndarray,
        return_transcript: bool = False
    ) -> Union[Tuple[bool, float], Tuple[bool, float, List[Dict[str, Any]]]]:
        """
        Executes end-to-end inference on 16kHz caller audio.

        Returns:
            is_synthetic (bool): True if probability >= calibrated threshold.
            confidence (float): Confidence in the returned verdict [0.0 - 1.0].
            turns (optional): Timestamped transcribed speech turns.
        """
        # 1. AASIST Latent Features (132-dim)
        waveform_tensor = torch.tensor(caller_audio_16k, dtype=torch.float32)
        aasist_feats = self.aasist_extractor.extract_from_waveform(waveform_tensor)

        # 2. RawNet2 Score Features (4-dim)
        rawnet_feats = self.rawnet2_extractor.extract_from_waveform(waveform_tensor)

        # 3. Acoustic DSP Features (136-dim)
        acoustic_feats = self.acoustic_extractor.extract_features(
            caller_audio_16k, sr=settings.target_sample_rate
        )

        turns = []
        if self.is_quad:
            # 4. Whisper Multimodal Features (24-dim) + Optional Transcript
            whisper_feats, turns = self.whisper_extractor.extract_features(
                caller_audio_16k, sr=settings.target_sample_rate, return_transcript=return_transcript
            )
            combined_feats = np.concatenate(
                [aasist_feats, rawnet_feats, acoustic_feats, whisper_feats], axis=0
            ).reshape(1, -1)
        else:
            # 272-dim fallback
            combined_feats = np.concatenate(
                [aasist_feats, rawnet_feats, acoustic_feats], axis=0
            ).reshape(1, -1)
            if return_transcript:
                _, turns = self.whisper_extractor.extract_features(
                    caller_audio_16k, sr=settings.target_sample_rate, return_transcript=True
                )

        # 5. Scale features
        if self.scaler is not None:
            combined_feats = self.scaler.transform(combined_feats)

        # 6. Predict Synthetic Probability (Class 1)
        if self.xgb_model is not None:
            probs = self.xgb_model.predict_proba(combined_feats)[0]
            synthetic_prob = float(probs[1])
        else:
            aasist_spoof = float(aasist_feats[131])
            rawnet_spoof = float(rawnet_feats[3])
            synthetic_prob = (aasist_spoof + rawnet_spoof) / 2.0

        # 7. Apply Calibrated Decision Threshold
        is_synthetic = bool(synthetic_prob >= self.calibrated_threshold)

        # 8. Compute confidence as certainty in the returned verdict
        if is_synthetic:
            certainty = synthetic_prob
        else:
            certainty = 1.0 - synthetic_prob

        confidence = round(float(np.clip(certainty, 0.50, 1.0)), 4)

        if return_transcript:
            return is_synthetic, confidence, turns
        return is_synthetic, confidence

