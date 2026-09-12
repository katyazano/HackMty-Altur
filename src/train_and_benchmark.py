import os
import sys
import time
import json
import csv
from pathlib import Path
import torch
import numpy as np
import soundfile as sf
import torch.nn.functional as F

# Prevent OpenMP collision
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.rawnet2 import RawNet2
from src.models.aasist import AASIST
from src.models.ensemble import AASISTXGBoostEnsemble
from src.trainer import train_model
from src.xgboost_trainer import train_and_evaluate_triple_xgboost
from src.features import AcousticFeatureExtractor, AASISTEmbeddingExtractor, RawNet2ScoreExtractor


def run_full_training_and_benchmark(epochs: int = 15, batch_size: int = 16, lr: float = 1e-4):
    print("=" * 85)
    print("   ALTUR ANTI-SPOOFING: TRIPLE MULTI-MODEL TRAINING & BENCHMARK SUITE")
    print("   (RawNet2 vs AASIST vs Acoustic-XGBoost vs Triple-Ensemble with Uncertainty)")
    print("=" * 85)

    weights_dir = PROJECT_ROOT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    rawnet_weights = str(weights_dir / "rawnet2_best.pth")
    aasist_weights = str(weights_dir / "aasist_best.pth")
    xgb_ensemble_weights = str(weights_dir / "xgboost_triple_ensemble.json")
    xgb_scaler_path = str(weights_dir / "triple_scaler.joblib")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Base models
    if not os.path.exists(rawnet_weights):
        train_model(model=RawNet2(), epochs=epochs, batch_size=batch_size, lr=lr, save_path=rawnet_weights)
    if not os.path.exists(aasist_weights):
        train_model(model=AASIST(), epochs=epochs, batch_size=batch_size, lr=lr, save_path=aasist_weights)

    # 2. Train Triple XGBoost with K-Fold and EER Calibration
    report = train_and_evaluate_triple_xgboost(epochs_base=epochs, n_splits=5, verbose=True)

    # 3. Load Evaluated Models
    rawnet_eval = RawNet2().to(device)
    rawnet_eval.load_state_dict(torch.load(rawnet_weights, map_location=device))
    rawnet_eval.eval()

    aasist_eval = AASIST().to(device)
    aasist_eval.load_state_dict(torch.load(aasist_weights, map_location=device))
    aasist_eval.eval()

    calibrated_thresh = report["threshold_calibration"]["optimal_eer_threshold"]

    ensemble_eval = AASISTXGBoostEnsemble(
        aasist_model=aasist_eval,
        rawnet2_model=rawnet_eval,
        xgb_model_path=xgb_ensemble_weights,
        scaler_path=xgb_scaler_path,
        device=str(device),
        threshold=calibrated_thresh,
        uncertainty_low=0.40,
        uncertainty_high=0.60
    )

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
    ens_correct = 0
    suspicious_count = 0

    r_lats, a_lats, ens_lats = [], [], []

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
        if (r_is_real == is_human_gt) if (is_human_gt or is_synth_gt) else False:
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
        if (a_is_real == is_human_gt) if (is_human_gt or is_synth_gt) else False:
            a_correct += 1

        # 3. Triple Ensemble with Uncertainty & Calibrated Threshold
        t0 = time.perf_counter()
        ens_res = ensemble_eval.predict_spoof(t_in, sample_rate=sr)
        ens_lat = (time.perf_counter() - t0) * 1000
        ens_lats.append(ens_lat)

        verdict = ens_res["verdict"]
        if verdict == "suspicious":
            suspicious_count += 1
            ens_is_corr = None
        elif verdict == "human":
            ens_is_corr = is_human_gt
            if is_human_gt:
                ens_correct += 1
        else: # synthetic
            ens_is_corr = is_synth_gt
            if is_synth_gt:
                ens_correct += 1

        final_records.append({
            "audio_id": cid,
            "filename": wfile.name,
            "duration_s": dur,
            "ground_truth": gt,
            "rawnet2": {"score_pct": r_real_pct, "prediction": "HUMAN" if r_is_real else "SYNTHETIC", "latency_ms": round(r_lat, 2)},
            "aasist": {"score_pct": a_real_pct, "prediction": "HUMAN" if a_is_real else "SYNTHETIC", "latency_ms": round(a_lat, 2)},
            "triple_ensemble": {
                "verdict": verdict,
                "confidence_score": ens_res["confidence_score"],
                "business_action": ens_res["business_action"],
                "is_correct": ens_is_corr,
                "latency_ms": round(ens_lat, 2)
            }
        })

    total_test = len(final_records)
    r_acc = round((r_correct / total_test) * 100, 2)
    a_acc = round((a_correct / total_test) * 100, 2)
    decisive_total = total_test - suspicious_count
    ens_acc = round((ens_correct / decisive_total) * 100, 2) if decisive_total > 0 else 0.0

    comparison_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_samples": total_test,
        "calibrated_threshold": calibrated_thresh,
        "uncertainty_band": [0.40, 0.60],
        "suspicious_count": suspicious_count,
        "models": {
            "RawNet2": {"accuracy_pct": r_acc, "correct_count": r_correct, "mean_latency_ms": round(float(np.mean(r_lats)), 2)},
            "AASIST": {"accuracy_pct": a_acc, "correct_count": a_correct, "mean_latency_ms": round(float(np.mean(a_lats)), 2)},
            "Triple_Ensemble": {"accuracy_pct": ens_acc, "correct_count": ens_correct, "mean_latency_ms": round(float(np.mean(ens_lats)), 2)}
        },
        "records": final_records
    }

    out_json = PROJECT_ROOT / "Data" / "multimodel_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)

    print("\n" + "=" * 85)
    print("               FINAL TRIPLE MULTI-MODEL BENCHMARK TABLE")
    print("=" * 85)
    print(f"{'Model Architecture':<35} | {'Test Accuracy':<15} | {'Correct/Total':<15} | {'Mean Latency':<12}")
    print("-" * 85)
    print(f"{'RawNet2 (Baseline)':<35} | {r_acc:>12.2f}% | {r_correct:>6}/{total_test:<7} | {np.mean(r_lats):>8.2f} ms")
    print(f"{'AASIST-L (Graph GNN)':<35} | {a_acc:>12.2f}% | {a_correct:>6}/{total_test:<7} | {np.mean(a_lats):>8.2f} ms")
    print(f"{'Triple Ensemble (AASIST+RawNet2+XGB)':<35} | {ens_acc:>12.2f}% | {ens_correct:>6}/{decisive_total:<7} | {np.mean(ens_lats):>8.2f} ms")
    print("=" * 85 + "\n")

    return comparison_summary


if __name__ == "__main__":
    run_full_training_and_benchmark(epochs=15, batch_size=16, lr=1e-4)
