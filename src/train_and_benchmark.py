import os
import sys

# Prevent OpenMP runtime collision between PyTorch and XGBoost on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import json
import csv
from pathlib import Path
import torch
import numpy as np
import soundfile as sf
import torch.nn.functional as F

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.rawnet2 import RawNet2
from src.models.aasist import AASIST
from src.models.ensemble import AASISTXGBoostEnsemble
from src.trainer import train_model
from src.xgboost_trainer import train_and_evaluate_xgboost
from src.features import AcousticFeatureExtractor, AASISTEmbeddingExtractor


def run_full_training_and_benchmark(epochs: int = 15, batch_size: int = 16, lr: float = 1e-4):
    print("=" * 80)
    print("   ALTUR ANTI-SPOOFING: MULTI-MODEL TRAINING & BENCHMARK SUITE")
    print("   (RawNet2 vs AASIST vs Acoustic-XGBoost vs AASIST+XGBoost Ensemble)")
    print("=" * 80)
    print(f"  * Target Deep Learning Epochs : {epochs}")
    print(f"  * Batch Size                  : {batch_size}")
    print(f"  * Learning Rate               : {lr}")
    print("=" * 80 + "\n", flush=True)

    weights_dir = PROJECT_ROOT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    rawnet_weights = str(weights_dir / "rawnet2_best.pth")
    aasist_weights = str(weights_dir / "aasist_best.pth")
    xgb_ensemble_weights = str(weights_dir / "xgboost_aasist_ensemble.json")
    xgb_scaler_path = str(weights_dir / "xgboost_scaler.joblib")
    xgb_ac_weights = str(weights_dir / "xgboost_acoustic_only.json")
    ac_scaler_path = str(weights_dir / "acoustic_scaler.joblib")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------
    # 1. Train RawNet2
    # -------------------------------------------------------------
    print(">>> [1/3] Training/Verifying RawNet2 Base Model...", flush=True)
    if not os.path.exists(rawnet_weights):
        rawnet_model = RawNet2()
        train_model(
            model=rawnet_model,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            save_path=rawnet_weights
        )
    else:
        print(f"  ✓ Found existing RawNet2 weights at {rawnet_weights}")

    # -------------------------------------------------------------
    # 2. Train AASIST Base Model
    # -------------------------------------------------------------
    print("\n>>> [2/3] Training/Verifying AASIST Graph GNN Model...", flush=True)
    if not os.path.exists(aasist_weights):
        aasist_model = AASIST()
        train_model(
            model=aasist_model,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            save_path=aasist_weights
        )
    else:
        print(f"  ✓ Found existing AASIST weights at {aasist_weights}")

    # -------------------------------------------------------------
    # 3. Train XGBoost Models (Acoustic-Only & AASIST+XGBoost Ensemble)
    # -------------------------------------------------------------
    print("\n>>> [3/3] Training & Optimizing XGBoost Classifiers...", flush=True)
    xgb_report = train_and_evaluate_xgboost(epochs_aasist=epochs, n_splits=5, verbose=True)

    # -------------------------------------------------------------
    # 4. Multi-Model Comprehensive Evaluation on Test Dataset
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("           EVALUATING ALL ARCHITECTURES ON TEST DATASET")
    print("=" * 80 + "\n", flush=True)

    # Load evaluated models
    rawnet_eval = RawNet2().to(device)
    rawnet_eval.load_state_dict(torch.load(rawnet_weights, map_location=device))
    rawnet_eval.eval()

    aasist_eval = AASIST().to(device)
    aasist_eval.load_state_dict(torch.load(aasist_weights, map_location=device))
    aasist_eval.eval()

    ensemble_eval = AASISTXGBoostEnsemble(
        aasist_model=aasist_eval,
        xgb_model_path=xgb_ensemble_weights,
        scaler_path=xgb_scaler_path,
        device=str(device)
    )

    import joblib
    import xgboost as xgb
    xgb_ac_model = xgb.XGBClassifier()
    xgb_ac_model.load_model(xgb_ac_weights)
    ac_scaler = joblib.load(ac_scaler_path)
    acoustic_extractor = AcousticFeatureExtractor(sample_rate=16000)

    # Discover test files and ground truths
    test_dir = PROJECT_ROOT / "Data" / "separated_agents" / "test" / "agent_0"
    manifest_path = PROJECT_ROOT / "Data" / "manifest.csv"

    gt_map = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = (row.get("anon_id") or row.get("id") or "").strip()
            if cid:
                gt_map[cid] = (row.get("label") or "").strip().lower()

    audio_files = sorted(list(test_dir.glob("*.wav")))
    final_records = []

    r_correct = 0
    a_correct = 0
    ac_xgb_correct = 0
    ens_correct = 0

    r_lats = []
    a_lats = []
    ac_xgb_lats = []
    ens_lats = []

    for wfile in audio_files:
        cid = wfile.stem.replace("_agent_0", "")
        gt = gt_map.get(cid, "unknown")
        is_human_gt = (gt == "human")
        is_synth_gt = (gt == "synthetic")

        data, sr = sf.read(str(wfile), dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]

        dur = round(len(data) / sr, 2)

        # 64,000 samples representation
        nb = len(data)
        if nb > 64000:
            mid = nb // 2
            data_fixed = data[mid - 32000 : mid + 32000]
        elif nb < 64000:
            data_fixed = np.pad(data, (0, 64000 - nb), mode="constant")
        else:
            data_fixed = data

        t_in = torch.tensor(data_fixed, dtype=torch.float32).unsqueeze(0).to(device)

        # 1. RawNet2
        t0 = time.perf_counter()
        with torch.no_grad():
            r_logits = rawnet_eval(t_in)
            r_probs = F.softmax(r_logits, dim=1)
            r_real_pct = round(r_probs[0, 0].item() * 100, 1)
            r_is_real = (r_real_pct >= 50.0)
        r_lat = (time.perf_counter() - t0) * 1000
        r_lats.append(r_lat)
        r_is_corr = (r_is_real == is_human_gt) if (is_human_gt or is_synth_gt) else None
        if r_is_corr:
            r_correct += 1

        # 2. AASIST
        t0 = time.perf_counter()
        with torch.no_grad():
            a_logits = aasist_eval(t_in)
            a_probs = F.softmax(a_logits, dim=1)
            a_real_pct = round(a_probs[0, 0].item() * 100, 1)
            a_is_real = (a_real_pct >= 50.0)
        a_lat = (time.perf_counter() - t0) * 1000
        a_lats.append(a_lat)
        a_is_corr = (a_is_real == is_human_gt) if (is_human_gt or is_synth_gt) else None
        if a_is_corr:
            a_correct += 1

        # 3. Standalone XGBoost (Acoustic DSP)
        t0 = time.perf_counter()
        ac_feat = acoustic_extractor.extract_features(data, sr=sr).reshape(1, -1)
        ac_feat_s = ac_scaler.transform(ac_feat)
        ac_xgb_prob_spoof = float(xgb_ac_model.predict_proba(ac_feat_s)[0, 1])
        ac_xgb_real_pct = round((1.0 - ac_xgb_prob_spoof) * 100, 1)
        ac_xgb_is_real = (ac_xgb_real_pct >= 50.0)
        ac_xgb_lat = (time.perf_counter() - t0) * 1000
        ac_xgb_lats.append(ac_xgb_lat)
        ac_xgb_is_corr = (ac_xgb_is_real == is_human_gt) if (is_human_gt or is_synth_gt) else None
        if ac_xgb_is_corr:
            ac_xgb_correct += 1

        # 4. Multi-Model Ensemble (AASIST + XGBoost)
        t0 = time.perf_counter()
        ens_res = ensemble_eval.predict_spoof(t_in, sample_rate=sr)
        ens_lat = (time.perf_counter() - t0) * 1000
        ens_lats.append(ens_lat)
        ens_is_real = ens_res["is_real"]
        ens_real_pct = ens_res["real_score_pct"]
        ens_is_corr = (ens_is_real == is_human_gt) if (is_human_gt or is_synth_gt) else None
        if ens_is_corr:
            ens_correct += 1

        final_records.append({
            "audio_id": cid,
            "filename": wfile.name,
            "duration_s": dur,
            "ground_truth": gt,
            "is_ground_truth_human": is_human_gt,
            "rawnet2": {"score_pct": r_real_pct, "prediction": "HUMAN" if r_is_real else "SYNTHETIC", "correct": r_is_corr, "latency_ms": round(r_lat, 2)},
            "aasist": {"score_pct": a_real_pct, "prediction": "HUMAN" if a_is_real else "SYNTHETIC", "correct": a_is_corr, "latency_ms": round(a_lat, 2)},
            "acoustic_xgboost": {"score_pct": ac_xgb_real_pct, "prediction": "HUMAN" if ac_xgb_is_real else "SYNTHETIC", "correct": ac_xgb_is_corr, "latency_ms": round(ac_xgb_lat, 2)},
            "ensemble_aasist_xgboost": {"score_pct": ens_real_pct, "prediction": "HUMAN" if ens_is_real else "SYNTHETIC", "correct": ens_is_corr, "latency_ms": round(ens_lat, 2)}
        })

    total_test = len(final_records)
    r_acc = round((r_correct / total_test) * 100, 2) if total_test > 0 else 0
    a_acc = round((a_correct / total_test) * 100, 2) if total_test > 0 else 0
    ac_xgb_acc = round((ac_xgb_correct / total_test) * 100, 2) if total_test > 0 else 0
    ens_acc = round((ens_correct / total_test) * 100, 2) if total_test > 0 else 0

    comparison_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_samples": total_test,
        "models": {
            "RawNet2": {
                "accuracy_pct": r_acc,
                "correct_count": r_correct,
                "mean_latency_ms": round(float(np.mean(r_lats)), 2)
            },
            "AASIST": {
                "accuracy_pct": a_acc,
                "correct_count": a_correct,
                "mean_latency_ms": round(float(np.mean(a_lats)), 2)
            },
            "Acoustic_XGBoost": {
                "accuracy_pct": ac_xgb_acc,
                "correct_count": ac_xgb_correct,
                "mean_latency_ms": round(float(np.mean(ac_xgb_lats)), 2)
            },
            "AASIST_XGBoost_Ensemble": {
                "accuracy_pct": ens_acc,
                "correct_count": ens_correct,
                "mean_latency_ms": round(float(np.mean(ens_lats)), 2)
            }
        },
        "records": final_records
    }

    # Save to JSON
    out_json = PROJECT_ROOT / "Data" / "multimodel_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)

    # Save to CSV
    out_csv = PROJECT_ROOT / "Data" / "multimodel_benchmark_results.csv"
    headers = [
        "audio_id", "ground_truth", "duration_s",
        "rawnet2_pred", "rawnet2_corr", "rawnet2_lat_ms",
        "aasist_pred", "aasist_corr", "aasist_lat_ms",
        "acoustic_xgb_pred", "acoustic_xgb_corr", "acoustic_xgb_lat_ms",
        "ensemble_pred", "ensemble_corr", "ensemble_lat_ms"
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in final_records:
            writer.writerow([
                r["audio_id"], r["ground_truth"], r["duration_s"],
                r["rawnet2"]["prediction"], r["rawnet2"]["correct"], r["rawnet2"]["latency_ms"],
                r["aasist"]["prediction"], r["aasist"]["correct"], r["aasist"]["latency_ms"],
                r["acoustic_xgboost"]["prediction"], r["acoustic_xgboost"]["correct"], r["acoustic_xgboost"]["latency_ms"],
                r["ensemble_aasist_xgboost"]["prediction"], r["ensemble_aasist_xgboost"]["correct"], r["ensemble_aasist_xgboost"]["latency_ms"]
            ])

    print("\n" + "=" * 85)
    print("                    MULTI-MODEL COMPREHENSIVE BENCHMARK TABLE")
    print("=" * 85)
    print(f"{'Model Architecture':<32} | {'Test Accuracy':<15} | {'Correct/Total':<15} | {'Mean Latency':<12}")
    print("-" * 85)
    print(f"{'RawNet2 (Baseline)':<32} | {r_acc:>12.2f}% | {r_correct:>6}/{total_test:<7} | {np.mean(r_lats):>8.2f} ms")
    print(f"{'AASIST-L (Graph GNN)':<32} | {a_acc:>12.2f}% | {a_correct:>6}/{total_test:<7} | {np.mean(a_lats):>8.2f} ms")
    print(f"{'XGBoost (Acoustic DSP only)':<32} | {ac_xgb_acc:>12.2f}% | {ac_xgb_correct:>6}/{total_test:<7} | {np.mean(ac_xgb_lats):>8.2f} ms")
    print(f"{'AASIST + XGBoost (Ensemble)':<32} | {ens_acc:>12.2f}% | {ens_correct:>6}/{total_test:<7} | {np.mean(ens_lats):>8.2f} ms")
    print("=" * 85)
    print(f"\n  ✓ Full JSON Benchmark saved to: {out_json}")
    print(f"  ✓ Full CSV Benchmark saved to : {out_csv}\n")

    return comparison_summary


if __name__ == "__main__":
    run_full_training_and_benchmark(epochs=15, batch_size=16, lr=1e-4)
