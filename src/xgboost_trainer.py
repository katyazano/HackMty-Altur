import os
import sys

# Prevent OpenMP runtime collision between PyTorch and XGBoost on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import csv
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import soundfile as sf
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)
import xgboost as xgb
import torch

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.aasist import AASIST
from src.features import (
    AcousticFeatureExtractor,
    AASISTEmbeddingExtractor,
    MultiModelFeaturePipeline
)
from src.trainer import train_model


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Computes the Equal Error Rate (EER) and optimal operating threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    # Find the index where FPR and FNR are closest
    idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    threshold = thresholds[idx]
    return float(eer), float(threshold)


def extract_dataset_features(
    audio_dir: Path,
    manifest_path: Path,
    aasist_extractor: AASISTEmbeddingExtractor,
    acoustic_extractor: AcousticFeatureExtractor,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Extracts acoustic features, AASIST embeddings, and combined feature matrices
    for all audio files in the specified directory using ground-truth labels.
    """
    label_map = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = (row.get("anon_id") or row.get("id") or "").strip()
                label = (row.get("label") or "").strip().lower()
                # 0 = human (bonafide), 1 = synthetic (spoof)
                num_label = 0 if label == "human" else 1
                if cid:
                    label_map[cid] = num_label

    audio_files = sorted(list(audio_dir.glob("*.wav")))
    if not audio_files:
        raise FileNotFoundError(f"No .wav audio files found in {audio_dir}")

    if verbose:
        print(f"Extracting features from {len(audio_files)} files in {audio_dir.name}...", flush=True)

    X_acoustic_list = []
    X_aasist_list = []
    y_list = []
    call_ids = []

    for idx, wpath in enumerate(audio_files, 1):
        cid = wpath.stem.replace("_agent_0", "")
        if cid not in label_map:
            continue

        target_y = label_map[cid]

        try:
            data, sr = sf.read(str(wpath), dtype="float32")
            if data.ndim > 1:
                data = data[:, 0]

            # 1. Acoustic DSP features
            ac_feats = acoustic_extractor.extract_features(data, sr=sr)

            # 2. AASIST latent features
            waveform_t = torch.tensor(data, dtype=torch.float32)
            aa_feats = aasist_extractor.extract_from_waveform(waveform_t)

            X_acoustic_list.append(ac_feats)
            X_aasist_list.append(aa_feats)
            y_list.append(target_y)
            call_ids.append(cid)

            if verbose and (idx % 50 == 0 or idx == len(audio_files)):
                print(f"  [Progress] Processed {idx:3d}/{len(audio_files)} files ({idx/len(audio_files)*100:.1f}%)", flush=True)

        except Exception as err:
            print(f"  Warning: Skipped {wpath.name} due to error: {err}", flush=True)

    X_acoustic = np.array(X_acoustic_list, dtype=np.float32)
    X_aasist = np.array(X_aasist_list, dtype=np.float32)
    X_combined = np.concatenate([X_aasist, X_acoustic], axis=1)
    y = np.array(y_list, dtype=np.int64)

    # Feature names
    aasist_names = [f"aasist_emb_{i}" for i in range(128)] + ["aasist_logit_human", "aasist_logit_spoof", "aasist_prob_human", "aasist_prob_spoof"]
    combined_names = aasist_names + acoustic_extractor.feature_names

    if verbose:
        print(f"  ✓ Extracted {len(y)} samples. Combined feature vector dimension: {X_combined.shape[1]}")

    return X_acoustic, X_aasist, X_combined, y, combined_names


def train_and_evaluate_xgboost(
    epochs_aasist: int = 15,
    n_splits: int = 5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Main training and evaluation pipeline for AASIST + XGBoost multi-model architecture.
    """
    print("=" * 80)
    print("   TRAINING MULTI-MODEL ENSEMBLE: AASIST GRAPH GNN + XGBOOST META-LEARNER")
    print("=" * 80)

    weights_dir = PROJECT_ROOT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    aasist_weights = str(weights_dir / "aasist_best.pth")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Ensure AASIST base model is trained or load checkpoint
    if not os.path.exists(aasist_weights):
        print("\n>>> AASIST weights not found. Training base AASIST model first...", flush=True)
        aasist_model = AASIST()
        train_model(
            model=aasist_model,
            epochs=epochs_aasist,
            batch_size=16,
            lr=1e-4,
            save_path=aasist_weights,
            device=device,
            verbose=verbose
        )
    else:
        print(f"\n>>> Found existing AASIST checkpoint at: {aasist_weights}")

    # Initialize AASIST Extractor
    aasist_eval = AASIST().to(device)
    aasist_eval.load_state_dict(torch.load(aasist_weights, map_location=device))
    aasist_eval.eval()
    aasist_extractor = AASISTEmbeddingExtractor(model=aasist_eval, device=device)
    acoustic_extractor = AcousticFeatureExtractor(sample_rate=16000)

    # 2. Extract features from Train and Test sets
    train_dir = PROJECT_ROOT / "Data" / "separated_agents" / "train" / "agent_0"
    test_dir = PROJECT_ROOT / "Data" / "separated_agents" / "test" / "agent_0"
    manifest_path = PROJECT_ROOT / "Data" / "manifest.csv"

    print("\n>>> Extracting Multi-Model Training Features...", flush=True)
    X_ac_train, X_aa_train, X_comb_train, y_train, feat_names = extract_dataset_features(
        audio_dir=train_dir,
        manifest_path=manifest_path,
        aasist_extractor=aasist_extractor,
        acoustic_extractor=acoustic_extractor,
        verbose=verbose
    )

    print("\n>>> Extracting Multi-Model Test Features...", flush=True)
    X_ac_test, X_aa_test, X_comb_test, y_test, _ = extract_dataset_features(
        audio_dir=test_dir,
        manifest_path=manifest_path,
        aasist_extractor=aasist_extractor,
        acoustic_extractor=acoustic_extractor,
        verbose=verbose
    )

    # 3. Stratified 5-Fold Cross Validation on Training Split
    print(f"\n>>> Running Stratified {n_splits}-Fold Cross-Validation on Training Split...", flush=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores_combined = []
    cv_scores_acoustic = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_comb_train, y_train), 1):
        # A. Combined Ensemble
        scaler_fold = StandardScaler()
        X_tr_s = scaler_fold.fit_transform(X_comb_train[train_idx])
        X_va_s = scaler_fold.transform(X_comb_train[val_idx])

        clf_comb = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=42
        )
        clf_comb.fit(X_tr_s, y_train[train_idx])
        val_preds = clf_comb.predict(X_va_s)
        val_acc = accuracy_score(y_train[val_idx], val_preds)
        cv_scores_combined.append(val_acc)

        # B. Acoustic Only
        scaler_ac_fold = StandardScaler()
        X_tr_ac_s = scaler_ac_fold.fit_transform(X_ac_train[train_idx])
        X_va_ac_s = scaler_ac_fold.transform(X_ac_train[val_idx])

        clf_ac = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=42
        )
        clf_ac.fit(X_tr_ac_s, y_train[train_idx])
        val_ac_preds = clf_ac.predict(X_va_ac_s)
        cv_scores_acoustic.append(accuracy_score(y_train[val_idx], val_ac_preds))

    print(f"  * 5-Fold CV Accuracy (Acoustic XGBoost) : {np.mean(cv_scores_acoustic)*100:.2f}% ± {np.std(cv_scores_acoustic)*100:.2f}%")
    print(f"  * 5-Fold CV Accuracy (AASIST+XGBoost)   : {np.mean(cv_scores_combined)*100:.2f}% ± {np.std(cv_scores_combined)*100:.2f}%")

    # 4. Final Fit on Entire Training Data & Test Evaluation
    print("\n>>> Fitting Final Production Models on Full Training Set...", flush=True)

    # 4A. Train AASIST + XGBoost Ensemble
    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_comb_train)
    X_test_scaled = final_scaler.transform(X_comb_test)

    final_xgb_ensemble = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,
        random_state=42
    )
    t0 = time.perf_counter()
    final_xgb_ensemble.fit(X_train_scaled, y_train)
    train_time_s = time.perf_counter() - t0

    # Predictions
    t_inf_0 = time.perf_counter()
    y_test_probs = final_xgb_ensemble.predict_proba(X_test_scaled)[:, 1] # Probability of Synthetic (Class 1)
    inf_latency_per_sample_ms = ((time.perf_counter() - t_inf_0) / len(X_test_scaled)) * 1000
    y_test_preds = (y_test_probs >= 0.50).astype(int)

    # Metrics
    test_acc = accuracy_score(y_test, y_test_preds)
    test_prec = precision_score(y_test, y_test_preds, pos_label=1, zero_division=0)
    test_rec = recall_score(y_test, y_test_preds, pos_label=1, zero_division=0)
    test_f1 = f1_score(y_test, y_test_preds, pos_label=1, zero_division=0)
    test_auc = roc_auc_score(y_test, y_test_probs) if len(np.unique(y_test)) > 1 else 1.0
    test_eer, opt_thresh = compute_eer(y_test, y_test_probs)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_preds, labels=[0, 1]).ravel()

    # 4B. Train Standalone Acoustic XGBoost
    scaler_ac = StandardScaler()
    X_ac_tr_s = scaler_ac.fit_transform(X_ac_train)
    X_ac_te_s = scaler_ac.transform(X_ac_test)

    final_xgb_ac = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,
        random_state=42
    )
    final_xgb_ac.fit(X_ac_tr_s, y_train)
    y_ac_test_probs = final_xgb_ac.predict_proba(X_ac_te_s)[:, 1]
    y_ac_test_preds = (y_ac_test_probs >= 0.50).astype(int)
    ac_acc = accuracy_score(y_test, y_ac_test_preds)
    ac_f1 = f1_score(y_test, y_ac_test_preds, pos_label=1, zero_division=0)
    ac_auc = roc_auc_score(y_test, y_ac_test_probs) if len(np.unique(y_test)) > 1 else 1.0
    ac_eer, _ = compute_eer(y_test, y_ac_test_probs)

    # 5. Save Artifacts
    xgb_ensemble_path = str(weights_dir / "xgboost_aasist_ensemble.json")
    xgb_scaler_path = str(weights_dir / "xgboost_scaler.joblib")
    xgb_ac_path = str(weights_dir / "xgboost_acoustic_only.json")
    ac_scaler_path = str(weights_dir / "acoustic_scaler.joblib")

    final_xgb_ensemble.save_model(xgb_ensemble_path)
    joblib.dump(final_scaler, xgb_scaler_path)
    final_xgb_ac.save_model(xgb_ac_path)
    joblib.dump(scaler_ac, ac_scaler_path)

    # 6. Top Feature Importances (Gain)
    importances = final_xgb_ensemble.feature_importances_
    top_indices = np.argsort(importances)[::-1][:15]
    top_features = [{"feature": feat_names[i], "importance": float(importances[i])} for i in top_indices]

    report = {
        "model_name": "AASIST + XGBoost Ensemble",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_time_s": round(train_time_s, 3),
        "test_samples": int(len(y_test)),
        "metrics": {
            "accuracy": round(float(test_acc), 4),
            "precision": round(float(test_prec), 4),
            "recall": round(float(test_rec), 4),
            "f1_score": round(float(test_f1), 4),
            "roc_auc": round(float(test_auc), 4),
            "eer": round(float(test_eer), 4),
            "optimal_threshold": round(float(opt_thresh), 4),
            "inference_latency_ms": round(float(inf_latency_per_sample_ms), 2)
        },
        "confusion_matrix": {
            "true_positives_synthetic": int(tp),
            "false_positives_synthetic": int(fp),
            "true_negatives_human": int(tn),
            "false_negatives_human": int(fn)
        },
        "acoustic_only_metrics": {
            "accuracy": round(float(ac_acc), 4),
            "f1_score": round(float(ac_f1), 4),
            "roc_auc": round(float(ac_auc), 4),
            "eer": round(float(ac_eer), 4)
        },
        "top_features": top_features
    }

    report_path = PROJECT_ROOT / "Data" / "xgboost_training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print Final Summary
    print("\n" + "=" * 80)
    print("               MULTI-MODEL TEST BENCHMARK RESULTS")
    print("=" * 80)
    print(f"  * Ensemble Accuracy       : {test_acc * 100:.2f}%")
    print(f"  * Ensemble F1-Score       : {test_f1 * 100:.2f}%")
    print(f"  * Ensemble ROC-AUC        : {test_auc * 100:.2f}%")
    print(f"  * Equal Error Rate (EER)  : {test_eer * 100:.2f}%")
    print(f"  * XGB Latency per Sample  : {inf_latency_per_sample_ms:.2f} ms")
    print(f"  * Confusion Matrix        : TP={tp} (Spoof), TN={tn} (Human), FP={fp}, FN={fn}")
    print("\n  Top 5 Discriminative Features:")
    for rank, item in enumerate(top_features[:5], 1):
        print(f"    {rank}. {item['feature']:<30} (importance: {item['importance']:.4f})")
    print("=" * 80 + "\n", flush=True)

    return report


if __name__ == "__main__":
    train_and_evaluate_xgboost()
