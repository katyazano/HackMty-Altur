import os
import sys
import logging
from pathlib import Path
from typing import Tuple
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
    RawNet2ScoreExtractor
)
from src.api.config import settings

logger = logging.getLogger("altur.inference")


class InferenceEngine:
    """
    Production-grade Triple Multi-Model Inference Engine.
    Combines AASIST (GNN), RawNet2 (Raw Waveform), and Acoustic DSP into XGBoost.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.calibrated_threshold = settings.calibrated_threshold

        # Ensure weights exist; if missing and dataset is present, auto-train
        if not os.path.exists(settings.xgb_model_path) and (PROJECT_ROOT / "Data" / "separated_agents").exists():
            logger.info("⚠️ Model weights not found in weights/. Auto-training Triple Multi-Model Ensemble on startup...")
            try:
                from src.xgboost_trainer import train_and_evaluate_triple_xgboost
                train_and_evaluate_triple_xgboost(epochs_base=15, n_splits=5, verbose=True)
                logger.info("✓ Auto-training completed successfully!")
            except Exception as train_err:
                logger.error(f"Failed to auto-train model weights: {train_err}", exc_info=True)

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

        # 4. Load XGBoost Classifier & Scaler
        self.xgb_model = xgb.XGBClassifier()
        if os.path.exists(settings.xgb_model_path):
            self.xgb_model.load_model(settings.xgb_model_path)
            logger.info(f"Loaded XGBoost ensemble from {settings.xgb_model_path}")
        else:
            logger.warning(f"XGBoost model not found at {settings.xgb_model_path}")

        self.scaler = None
        if os.path.exists(settings.scaler_path):
            self.scaler = joblib.load(settings.scaler_path)
            logger.info(f"Loaded feature scaler from {settings.scaler_path}")
        else:
            logger.warning(f"Feature scaler not found at {settings.scaler_path}")

    def predict(self, caller_audio_16k: np.ndarray) -> Tuple[bool, float]:
        """
        Executes end-to-end inference on 16kHz caller audio.

        Returns:
            is_synthetic (bool): True if probability >= calibrated threshold (0.9623).
            confidence (float): Estimated probability that voice is synthetic [0.0 - 1.0].
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

        # 4. Concatenate Triple Feature Vector (272-dim)
        combined_feats = np.concatenate(
            [aasist_feats, rawnet_feats, acoustic_feats], axis=0
        ).reshape(1, -1)

        # 5. Scale features
        if self.scaler is not None:
            combined_feats = self.scaler.transform(combined_feats)

        # 6. Predict Synthetic Probability (Class 1)
        if self.xgb_model is not None:
            probs = self.xgb_model.predict_proba(combined_feats)[0]
            synthetic_prob = float(probs[1])
        else:
            # Fallback if XGBoost is missing
            aasist_spoof = float(aasist_feats[131])
            rawnet_spoof = float(rawnet_feats[3])
            synthetic_prob = (aasist_spoof + rawnet_spoof) / 2.0

        # 7. Apply Calibrated EER Decision Threshold (0.9623)
        is_synthetic = bool(synthetic_prob >= self.calibrated_threshold)
        confidence = round(float(synthetic_prob), 4)

        return is_synthetic, confidence
