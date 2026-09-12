import os
import sys
import csv
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def benchmark_model(
    model: nn.Module,
    test_dir: Optional[str] = None,
    manifest_path: Optional[str] = None,
    device: Optional[str] = None,
    save_csv: Optional[str] = None,
    save_json: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Universal benchmarking function to evaluate any PyTorch model on test audio data.
    
    Args:
        model: Any PyTorch nn.Module (e.g. RawNet2, AASIST, or any custom model)
        test_dir: Directory containing test Channel 0 audio files (.wav)
        manifest_path: Path to manifest.csv containing ground truth labels
        device: Device string ('cuda', 'cpu', or None for auto-detect)
        save_csv: Optional path to export CSV results
        save_json: Optional path to export JSON results
        verbose: Whether to print summary table and progress
        
    Returns:
        Dictionary containing overall accuracy, metrics, latency, and per-sample predictions.
    """
    # 1. Resolve paths
    if test_dir is None:
        test_dir = str(PROJECT_ROOT / "Data" / "separated_agents" / "test" / "agent_0")
    if manifest_path is None:
        manifest_path = str(PROJECT_ROOT / "Data" / "manifest.csv")

    test_path = Path(test_dir)
    if not test_path.exists():
        raise FileNotFoundError(f"Test audio directory not found: {test_path}")

    # 2. Setup Device & Model
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    model = model.to(target_device)
    model.eval()

    model_name = model.__class__.__name__

    # 3. Load Ground Truth Labels from Manifest
    label_map = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = (row.get("anon_id") or row.get("id") or "").strip()
                label = (row.get("label") or "").strip().lower()
                if cid:
                    label_map[cid] = label

    # 4. Discover audio files
    audio_files = sorted(list(test_path.glob("*.wav")))
    if not audio_files:
        raise FileNotFoundError(f"No .wav files found in {test_path}")

    if verbose:
        print("=" * 80)
        print(f"       UNIVERSAL BENCHMARK EVALUATOR: {model_name.upper()}")
        print("=" * 80)
        print(f"  * Audio Source    : {test_path} ({len(audio_files)} recordings)")
        print(f"  * Manifest Source : {manifest_path}")
        print(f"  * Device          : {target_device}")
        print("=" * 80 + "\n")

    results = []
    correct_count = 0
    total_evaluated = 0
    total_latency_ms = 0.0

    # Confusion matrix counters: [TP, FP, TN, FN] (Positive = Human/Bonafide, Negative = Synthetic/Spoof)
    tp = fp = tn = fn = 0

    t_start = time.time()

    for idx, audio_file in enumerate(audio_files, 1):
        file_stem = audio_file.stem
        call_id = file_stem.replace("_agent_0", "")

        ground_truth = label_map.get(call_id, "unknown")
        is_human_gt = (ground_truth == "human")
        is_synthetic_gt = (ground_truth == "synthetic")

        try:
            data, sr = sf.read(str(audio_file), dtype="float32")
            if data.ndim > 1:
                data = data[:, 0]

            duration = round(len(data) / sr, 2)

            # Standard 64,000 samples (4.0s @ 16kHz) windowing
            nb_samples = len(data)
            target_samples = 64000
            if nb_samples > target_samples:
                mid = nb_samples // 2
                half = target_samples // 2
                data = data[mid - half : mid + half]
            elif nb_samples < target_samples:
                data = np.pad(data, (0, target_samples - nb_samples), mode="constant")

            tensor_in = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(target_device)

            # Inference
            t0 = time.perf_counter()
            with torch.no_grad():
                # Check if model has custom predict_spoof or standard forward
                if hasattr(model, "predict_spoof") and callable(getattr(model, "predict_spoof")):
                    res = model.predict_spoof(tensor_in)
                    real_score = res.get("real_score_pct", 50.0)
                    is_real_pred = res.get("is_real", real_score >= 50.0)
                else:
                    logits = model(tensor_in)
                    probs = F.softmax(logits, dim=1)
                    # Class 0: Human, Class 1: Synthetic
                    real_prob = probs[0, 0].item()
                    real_score = round(real_prob * 100, 1)
                    is_real_pred = (real_score >= 50.0)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            total_latency_ms += latency_ms

            pred_label = "human" if is_real_pred else "synthetic"
            is_correct = (is_real_pred == is_human_gt) if (is_human_gt or is_synthetic_gt) else None

            if is_correct is True:
                correct_count += 1

            if is_human_gt and is_real_pred:
                tp += 1
            elif is_synthetic_gt and is_real_pred:
                fp += 1
            elif is_synthetic_gt and not is_real_pred:
                tn += 1
            elif is_human_gt and not is_real_pred:
                fn += 1

            total_evaluated += 1

            record = {
                "audio_id": call_id,
                "filename": audio_file.name,
                "duration_s": duration,
                "ground_truth": ground_truth,
                "prediction": pred_label,
                "score_pct": real_score,
                "is_correct": is_correct,
                "latency_ms": latency_ms
            }
            results.append(record)

            if verbose and (idx % 25 == 0 or idx == len(audio_files)):
                print(f"  [{idx:2d}/{len(audio_files)}] ID: {call_id} | GT: {ground_truth.upper():9s} | Pred: {pred_label.upper():9s} ({real_score:5.1f}%) | Latency: {latency_ms}ms")

        except Exception as e:
            print(f"Error evaluating {audio_file.name}: {e}")

    total_time_s = round(time.time() - t_start, 2)
    accuracy_pct = round((correct_count / total_evaluated) * 100, 2) if total_evaluated > 0 else 0.0
    avg_latency_ms = round(total_latency_ms / total_evaluated, 2) if total_evaluated > 0 else 0.0

    # Precision, Recall, F1 for Human class
    precision_human = round((tp / (tp + fp)) * 100, 2) if (tp + fp) > 0 else 0.0
    recall_human = round((tp / (tp + fn)) * 100, 2) if (tp + fn) > 0 else 0.0
    f1_human = round((2 * precision_human * recall_human) / (precision_human + recall_human), 2) if (precision_human + recall_human) > 0 else 0.0

    benchmark_summary = {
        "model_name": model_name,
        "total_samples": total_evaluated,
        "accuracy_pct": accuracy_pct,
        "correct_count": correct_count,
        "incorrect_count": total_evaluated - correct_count,
        "avg_latency_ms": avg_latency_ms,
        "total_evaluation_time_s": total_time_s,
        "confusion_matrix": {
            "true_human (TP)": tp,
            "false_human (FP)": fp,
            "true_synthetic (TN)": tn,
            "false_synthetic (FN)": fn
        },
        "metrics_human": {
            "precision_pct": precision_human,
            "recall_pct": recall_human,
            "f1_score": f1_human
        },
        "results": results
    }

    # Save to JSON if requested
    if save_json:
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(benchmark_summary, f, indent=2)

    # Save to CSV if requested
    if save_csv:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        headers = ["audio_id", "ground_truth", "prediction", "score_pct", "is_correct", "duration_s", "latency_ms"]
        with open(save_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for r in results:
                writer.writerow([r["audio_id"], r["ground_truth"], r["prediction"], r["score_pct"], r["is_correct"], r["duration_s"], r["latency_ms"]])

    if verbose:
        print("\n" + "=" * 80)
        print(f"             BENCHMARK REPORT: {model_name.upper()}")
        print("=" * 80)
        print(f"  * Total Evaluated : {total_evaluated} test recordings in {total_time_s}s")
        print(f"  * Overall Accuracy: {accuracy_pct}% ({correct_count}/{total_evaluated} correct)")
        print(f"  * Avg Latency     : {avg_latency_ms} ms / sample")
        print(f"  * Human Precision : {precision_human}%")
        print(f"  * Human Recall    : {recall_human}%")
        print(f"  * Human F1-Score  : {f1_human}")
        print("=" * 80 + "\n")

    return benchmark_summary


if __name__ == "__main__":
    import argparse
    from src.models.rawnet2 import RawNet2
    from src.models.aasist import AASIST

    parser = argparse.ArgumentParser(description="Universal Model Benchmark Runner")
    parser.add_argument("--model", type=str, default="rawnet2", choices=["rawnet2", "aasist"], help="Model to benchmark")
    parser.add_argument("--weights", type=str, default=None, help="Optional path to custom trained weights .pth")
    parser.add_argument("--save_csv", type=str, default=None, help="Optional path to save CSV")
    parser.add_argument("--save_json", type=str, default=None, help="Optional path to save JSON")
    args = parser.parse_args()

    if args.model.lower() == "rawnet2":
        net = RawNet2()
    elif args.model.lower() == "aasist":
        net = AASIST()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    if args.weights and os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}...")
        net.load_state_dict(torch.load(args.weights, map_location="cpu"))

    benchmark_model(
        model=net,
        save_csv=args.save_csv,
        save_json=args.save_json
    )
