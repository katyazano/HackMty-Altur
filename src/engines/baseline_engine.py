"""
baseline_engine.py — Baseline Anti-Spoofing Engine (Acoustic DSP + Timing + AudioCNN).
Combines HistGradientBoosting/LightGBM global features with micro-segment AudioCNN.
"""
import io
import os
import json
import time
import pickle
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from src.features_baseline import extract_from_arrays, load_stereo_from_bytes
from src.neural import SpoofCNN, get_neural_probability, load_model as load_neural

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class BaselineEngine:
    """
    Engine 1: Baseline Anti-Spoofing Engine.
    - Global acoustic features (MFCCs, spectral contrast, chroma, deltas)
    - Conversational timing features (response latencies, speech overlaps, turn durations)
    - 1D/2D AudioCNN micro-segment neural scoring
    - HistGradientBoosting meta-model
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        threshold_path: Optional[str] = None,
        w_neural: float = 0.50
    ):
        self.repo_root = PROJECT_ROOT
        self.model_path = model_path or str(self.repo_root / "model.pkl")
        self.weights_path = weights_path or str(self.repo_root / "weights" / "neural_cnn.pt")
        self.threshold_path = threshold_path or str(self.repo_root / "threshold.json")
        self.w_neural = w_neural

        self.model = None
        self.feature_names = []
        self.threshold = 0.50
        self.neural_loaded = False

        self._load_artifacts()

    def _load_artifacts(self):
        # 1. Load Scikit-Learn / LightGBM model bundle
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    bundle = pickle.load(f)
                self.model = bundle.get("model", bundle)
                self.feature_names = bundle.get("feature_names", [])
            except Exception as e:
                print(f"[BaselineEngine] Warning loading {self.model_path}: {e}")

        # 2. Load Decision Threshold
        if os.path.exists(self.threshold_path):
            try:
                with open(self.threshold_path, "r") as f:
                    data = json.loads(f.read().lstrip("\ufeff"))
                    self.threshold = float(data.get("threshold", 0.50))
            except Exception as e:
                print(f"[BaselineEngine] Warning loading threshold: {e}")

        # 3. Load AudioCNN
        try:
            neural_net = load_neural(self.weights_path)
            self.neural_loaded = neural_net is not None
        except Exception as e:
            print(f"[BaselineEngine] Warning loading AudioCNN: {e}")
            self.neural_loaded = False

    def predict(
        self,
        audio_bytes: bytes,
        caller: Optional[np.ndarray] = None,
        agent: Optional[np.ndarray] = None,
        sr: int = 8000
    ) -> Dict[str, Any]:
        """
        Executes baseline anti-spoofing inference.

        Returns:
            Dict containing:
              - is_synthetic (bool)
              - confidence (float)
              - probability_synthetic (float)
              - prob_gbm (float)
              - prob_neural (float)
              - threshold (float)
              - latency_ms (float)
              - feature_count (int)
        """
        t0 = time.perf_counter()

        if caller is None or agent is None:
            caller, agent, sr = load_stereo_from_bytes(audio_bytes)

        # 1. Acoustic & Timing Feature Extraction
        vec, names = extract_from_arrays(caller, agent, sr)
        x = vec.reshape(1, -1)

        # 2. GBM Prediction
        if self.model is not None:
            try:
                prob_gbm = float(self.model.predict_proba(x)[0, 1])
            except Exception:
                prob_gbm = 0.50
        else:
            prob_gbm = 0.50

        # 3. Neural CNN Prediction
        if self.neural_loaded:
            try:
                prob_neural = float(get_neural_probability(audio_bytes))
            except Exception:
                prob_neural = prob_gbm
        else:
            prob_neural = prob_gbm

        # 4. Ensemble Fusion
        if self.neural_loaded and self.model is not None:
            prob_final = (1.0 - self.w_neural) * prob_gbm + self.w_neural * prob_neural
        elif self.neural_loaded:
            prob_final = prob_neural
        else:
            prob_final = prob_gbm

        prob_final = float(np.clip(prob_final, 0.0, 1.0))
        is_synthetic = bool(prob_final >= self.threshold)

        # Confidence is distance from decision boundary normalized
        if is_synthetic:
            confidence = prob_final
        else:
            confidence = 1.0 - prob_final

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "engine": "baseline",
            "name": "Baseline Acoustic DSP + AudioCNN",
            "is_synthetic": is_synthetic,
            "confidence": round(confidence, 4),
            "probability_synthetic": round(prob_final, 4),
            "prob_gbm": round(prob_gbm, 4),
            "prob_neural": round(prob_neural, 4),
            "threshold": round(self.threshold, 4),
            "latency_ms": latency_ms,
            "feature_count": len(names),
        }
