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
from typing import Dict, Any, Optional, Literal

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.aasist import AASIST
from src.models.rawnet2 import RawNet2
from src.features import AcousticFeatureExtractor, AASISTEmbeddingExtractor, RawNet2ScoreExtractor

VerdictType = Literal["human", "synthetic", "suspicious"]


class AASISTXGBoostEnsemble:
    """
    Triple Multi-Model Stacked Ensemble Classifier:
    Combines:
    1. AASIST Graph Neural Network (132-dim latent representations & logits)
    2. RawNet2 Raw-Waveform Architecture (4-dim logits & probabilities)
    3. Acoustic & Spectral DSP Feature Extractor (136-dim MFCCs, formants, prosody)
    4. Calibrated Decision Threshold with 0.40 - 0.60 Uncertainty Band.
    """

    def __init__(
        self,
        aasist_model: Optional[AASIST] = None,
        aasist_weights: Optional[str] = None,
        rawnet2_model: Optional[RawNet2] = None,
        rawnet2_weights: Optional[str] = None,
        xgb_model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        device: Optional[str] = None,
        threshold: float = 0.68,
        uncertainty_low: float = 0.40,
        uncertainty_high: float = 0.60
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

        # 2. Initialize RawNet2
        if rawnet2_model is not None:
            self.rawnet2 = rawnet2_model.to(self.device)
        else:
            self.rawnet2 = RawNet2().to(self.device)
            if rawnet2_weights and os.path.exists(rawnet2_weights):
                self.rawnet2.load_state_dict(torch.load(rawnet2_weights, map_location=self.device))
        self.rawnet2.eval()
        self.rawnet2_extractor = RawNet2ScoreExtractor(model=self.rawnet2, device=str(self.device))

        # 3. Acoustic Extractor
        self.acoustic_extractor = AcousticFeatureExtractor(sample_rate=16000)

        # 4. Initialize XGBoost Booster
        self.xgb_model = None
        if xgb_model_path and os.path.exists(xgb_model_path):
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(xgb_model_path)

        # 5. Scaler
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        # 6. Calibrated Threshold & Uncertainty Bounds
        self.threshold = threshold
        self.uncertainty_low = uncertainty_low
        self.uncertainty_high = uncertainty_high

    def predict_spoof(self, waveform_tensor: torch.Tensor, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Runs triple-model inference on input waveform and outputs ensemble classification
        with calibrated threshold and uncertainty band.
        """
        # 1. Extract AASIST representations (132-dim)
        aasist_feats = self.aasist_extractor.extract_from_waveform(waveform_tensor)
        aasist_real_prob = float(aasist_feats[130])
        aasist_spoof_prob = float(aasist_feats[131])

        # 2. Extract RawNet2 representations (4-dim)
        rawnet_feats = self.rawnet2_extractor.extract_from_waveform(waveform_tensor)
        rawnet_real_prob = float(rawnet_feats[2])
        rawnet_spoof_prob = float(rawnet_feats[3])

        # 3. Extract Acoustic DSP features (136-dim)
        waveform_np = waveform_tensor.detach().cpu().squeeze().numpy()
        acoustic_feats = self.acoustic_extractor.extract_features(waveform_np, sr=sample_rate)

        # 4. Check whether model expects 272 features (Triple) or 268 features (Dual)
        expected_features = 272
        if self.scaler is not None and hasattr(self.scaler, "n_features_in_"):
            expected_features = self.scaler.n_features_in_

        if expected_features == 272:
            combined_feats = np.concatenate([aasist_feats, rawnet_feats, acoustic_feats], axis=0).reshape(1, -1)
        else:
            combined_feats = np.concatenate([aasist_feats, acoustic_feats], axis=0).reshape(1, -1)

        # 5. Scale features
        if self.scaler is not None:
            combined_feats = self.scaler.transform(combined_feats)

        # 6. XGBoost Inference (Class 0 = Human, Class 1 = Synthetic/Spoof)
        if self.xgb_model is not None:
            probs = self.xgb_model.predict_proba(combined_feats)[0]
            real_prob = float(probs[0])
            spoof_prob = float(probs[1])
        else:
            real_prob = (aasist_real_prob + rawnet_real_prob) / 2.0
            spoof_prob = 1.0 - real_prob

        # 7. Uncertainty Band & Calibrated Threshold Decision Rule
        # confidence_score = probability of authenticity (real/human)
        confidence_score = real_prob

        if self.uncertainty_low <= spoof_prob <= self.uncertainty_high:
            verdict: VerdictType = "suspicious"
            business_action = "TRIGGER_STEP_UP_AUTHENTICATION (SMS OTP / Biometric Verification Required)"
            is_real = False
            is_spoof = False
        else:
            # Calibrated Decision Rule using EER-calibrated spoof threshold
            if spoof_prob >= self.threshold:
                verdict = "synthetic"
                business_action = "BLOCK_TRANSACTION (Synthetic AI Voice Detected)"
                is_real = False
                is_spoof = True
            else:
                verdict = "human"
                business_action = "ALLOW_TRANSACTION (Verified Human Voice)"
                is_real = True
                is_spoof = False

        real_pct = round(real_prob * 100, 1)
        risk_pct = round(spoof_prob * 100, 1)

        return {
            "model_name": "Triple Ensemble (AASIST + RawNet2 + XGBoost)",
            "verdict": verdict,
            "business_action": business_action,
            "confidence_score": round(confidence_score, 4),
            "spoof_probability": round(spoof_prob, 4),
            "calibrated_threshold": self.threshold,
            "uncertainty_band": [self.uncertainty_low, self.uncertainty_high],
            "is_real": is_real,
            "is_spoof": is_spoof,
            "real_score_pct": real_pct,
            "spoof_risk_pct": risk_pct,
            "model_breakdown": {
                "aasist_spoof_prob": round(aasist_spoof_prob, 4),
                "rawnet2_spoof_prob": round(rawnet_spoof_prob, 4),
                "xgboost_spoof_prob": round(spoof_prob, 4)
            }
        }
