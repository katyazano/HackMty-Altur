import os
import sys

# Prevent OpenMP runtime collision between PyTorch and XGBoost on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
import joblib
import xgboost as xgb
from pathlib import Path
from typing import Dict, Any, Optional

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.aasist import AASIST
from src.features import AcousticFeatureExtractor, AASISTEmbeddingExtractor


class AASISTXGBoostEnsemble:
    """
    Multi-Model Stacked Ensemble Classifier:
    Combines AASIST Graph Neural Network representations with Acoustic & Spectral DSP
    features processed via an XGBoost Gradient Boosted Meta-Classifier.
    """

    def __init__(
        self,
        aasist_model: Optional[AASIST] = None,
        aasist_weights: Optional[str] = None,
        xgb_model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        device: Optional[str] = None,
        threshold: float = 0.50
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. Initialize AASIST
        if aasist_model is not None:
            self.aasist = aasist_model.to(self.device)
        else:
            self.aasist = AASIST().to(self.device)
            if aasist_weights and os.path.exists(aasist_weights):
                self.aasist.load_state_dict(torch.load(aasist_weights, map_location=self.device))
        self.aasist.eval()

        self.aasist_extractor = AASISTEmbeddingExtractor(model=self.aasist, device=str(self.device))
        self.acoustic_extractor = AcousticFeatureExtractor(sample_rate=16000)

        # 2. Initialize XGBoost Booster
        self.xgb_model = None
        if xgb_model_path and os.path.exists(xgb_model_path):
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(xgb_model_path)

        # 3. Scaler
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        self.threshold = threshold

    def predict_spoof(self, waveform_tensor: torch.Tensor, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Runs dual-model inference on input waveform and outputs ensemble classification.
        """
        # 1. Extract AASIST representations (132-dim)
        aasist_feats = self.aasist_extractor.extract_from_waveform(waveform_tensor)
        # Softmax probability from raw AASIST
        aasist_real_prob = float(aasist_feats[130])
        aasist_spoof_prob = float(aasist_feats[131])

        # 2. Extract Acoustic & Spectral features (134-dim)
        waveform_np = waveform_tensor.detach().cpu().squeeze().numpy()
        acoustic_feats = self.acoustic_extractor.extract_features(waveform_np, sr=sample_rate)

        # 3. Combine features (266-dim)
        combined_feats = np.concatenate([aasist_feats, acoustic_feats], axis=0).reshape(1, -1)

        # 4. Scale if scaler exists
        if self.scaler is not None:
            combined_feats = self.scaler.transform(combined_feats)

        # 5. XGBoost Inference (if model loaded) or weighted fallback
        if self.xgb_model is not None:
            # Predict probabilities: Class 0 = Human, Class 1 = Synthetic/Spoof
            probs = self.xgb_model.predict_proba(combined_feats)[0]
            real_prob = float(probs[0])
            spoof_prob = float(probs[1])
        else:
            # Fallback to AASIST baseline probabilities
            real_prob = aasist_real_prob
            spoof_prob = aasist_spoof_prob

        real_pct = round(real_prob * 100, 1)
        risk_pct = round(spoof_prob * 100, 1)
        is_real = real_prob >= self.threshold
        is_spoof = not is_real

        return {
            "model_name": "AASIST+XGBoost (Ensemble)",
            "is_real": is_real,
            "is_spoof": is_spoof,
            "real_score_pct": real_pct,
            "real_score_percentage": real_pct,
            "spoof_risk_pct": risk_pct,
            "confidence_pct": round(max(real_prob, spoof_prob) * 100, 2),
            "verdict": "AUTHENTIC HUMAN" if is_real else "SYNTHETIC / SPOOF",
            "aasist_real_prob": round(aasist_real_prob, 4),
            "xgboost_real_prob": round(real_prob, 4)
        }
