import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb
import joblib

# Prevent OpenMP collisions
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.aasist import AASIST
from src.models.rawnet2 import RawNet2
from src.features import (
    AcousticFeatureExtractor,
    AASISTEmbeddingExtractor,
    RawNet2ScoreExtractor
)
from src.xgboost_trainer import extract_dataset_features_triple

# Calibrated EER operating point
CALIBRATED_EER_THRESHOLD = 0.9623


def evaluate_xgboost_binary(
    model_path: str = "weights/xgboost_triple_ensemble.json",
    scaler_path: str = "weights/triple_scaler.joblib",
    threshold: float = CALIBRATED_EER_THRESHOLD
) -> Dict[str, Any]:
    """
    Evaluates XGBoost model with continuous probability outputs,
    strict binary thresholding, and standardized binary JSON payloads.
    """
    model_file = PROJECT_ROOT / model_path
    scaler_file = PROJECT_ROOT / scaler_path

    if not model_file.exists() or not scaler_file.exists():
        raise FileNotFoundError(f"Missing model artifacts: {model_file} or {scaler_file}")

    # 1. Load trained model and feature scaler
    model = xgb.XGBClassifier()
    model.load_model(str(model_file))
    scaler = joblib.load(str(scaler_file))

    # 2. Extract features from held-out test split
    device = "cpu"
    aasist_weights = PROJECT_ROOT / "weights" / "aasist_best.pth"
    rawnet_weights = PROJECT_ROOT / "weights" / "rawnet2_best.pth"

    import torch
    aasist_m = AASIST().to(device)
    aasist_m.load_state_dict(torch.load(str(aasist_weights), map_location=device))
    aasist_m.eval()
    aasist_ext = AASISTEmbeddingExtractor(model=aasist_m, device=device)

    rawnet_m = RawNet2().to(device)
    rawnet_m.load_state_dict(torch.load(str(rawnet_weights), map_location=device))
    rawnet_m.eval()
    rawnet_ext = RawNet2ScoreExtractor(model=rawnet_m, device=device)

    ac_ext = AcousticFeatureExtractor(sample_rate=16000)

    test_dir = PROJECT_ROOT / "Data" / "separated_agents" / "test" / "agent_0"
    manifest_path = PROJECT_ROOT / "Data" / "manifest.csv"

    _, _, _, _, X_trip_te, y_true, _ = extract_dataset_features_triple(
        test_dir, manifest_path, aasist_ext, rawnet_ext, ac_ext, verbose=False
    )

    X_test_scaled = scaler.transform(X_trip_te)

    # -------------------------------------------------------------------------
    # Requirement 1: Probability Output Only (Positive Class: Synthetic = Class 1)
    # -------------------------------------------------------------------------
    y_probs: np.ndarray = model.predict_proba(X_test_scaled)[:, 1]

    # -------------------------------------------------------------------------
    # Requirement 2: Strict Binary Thresholding (Boolean Array)
    # -------------------------------------------------------------------------
    is_synthetic: np.ndarray = (y_probs >= threshold)  # dtype: bool (True=synthetic, False=human)
    y_true_bool: np.ndarray = (y_true == 1)

    # -------------------------------------------------------------------------
    # Requirement 3: Metric Recalculation
    # -------------------------------------------------------------------------
    cm = confusion_matrix(y_true_bool, is_synthetic, labels=[False, True])
    tn, fp, fn, tp = cm.ravel()
    accuracy = float(accuracy_score(y_true_bool, is_synthetic))
    f1 = float(f1_score(y_true_bool, is_synthetic, pos_label=True, zero_division=0))
    precision = float(precision_score(y_true_bool, is_synthetic, pos_label=True, zero_division=0))
    recall = float(recall_score(y_true_bool, is_synthetic, pos_label=True, zero_division=0))

    print("=" * 60)
    print(f"XGBOOST BINARY EVALUATION REPORT (Calibrated Threshold = {threshold})")
    print("=" * 60)
    print(f"Total Test Samples Evaluated : {len(y_true_bool)}")
    print(f"Calibrated Decision Threshold: {threshold:.4f}\n")
    print("CONFUSION MATRIX:")
    print(f"  [TN={tn:2d} (Human Correct)   | FP={fp:2d} (False Alarm)]")
    print(f"  [FN={fn:2d} (Missed Spoof)    | TP={tp:2d} (Spoof Blocked)]\n")
    print(f"Accuracy  : {accuracy * 100:.2f}% ({tp + tn}/{len(y_true_bool)})")
    print(f"F1-Score  : {f1 * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Requirement 4: Simulated JSON Payload Output
    # -------------------------------------------------------------------------
    simulated_payloads: List[Dict[str, Any]] = [
        {
            "is_synthetic": bool(pred),
            "confidence": round(float(prob), 4)
        }
        for pred, prob in zip(is_synthetic, y_probs)
    ]

    print("\nSAMPLE SIMULATED INFERENCE PAYLOADS (First 5 items):")
    print(json.dumps(simulated_payloads[:5], indent=2))

    return {
        "threshold": threshold,
        "metrics": {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": {
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn)
            }
        },
        "payload_schema": {
            "is_synthetic": "bool",
            "confidence": "float"
        },
        "sample_payloads": simulated_payloads[:5]
    }


if __name__ == "__main__":
    evaluate_xgboost_binary()
