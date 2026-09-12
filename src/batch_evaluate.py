import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import soundfile as sf
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.inference_engine import InferenceEngine
from src.dynamic_separator import dynamic_extract_caller_speech


def batch_evaluate_directory(
    audio_dir: Path,
    manifest_path: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Runs dynamic VAD separation + multi-model inference across a directory of raw stereo/mono WAV files.
    """
    audio_files = sorted(list(audio_dir.glob("*.wav")) + list(audio_dir.glob("**/*.wav")))
    if not audio_files:
        print(f"Error: No .wav files found in {audio_dir}")
        return {}

    # Load Ground Truth Labels if manifest exists
    labels_map = {}
    if manifest_path and manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = (row.get("anon_id") or row.get("id") or "").strip()
                label = (row.get("label") or "").strip().lower()
                num_label = 0 if label == "human" else 1
                if cid:
                    labels_map[cid] = num_label

    print("=" * 80)
    print("      ALTUR BATCH AUDIO EVALUATION - DYNAMIC CHANNEL SEPARATION & INFERENCE")
    print("=" * 80)
    print(f"  * Audio Directory  : {audio_dir}")
    print(f"  * Total Files      : {len(audio_files)}")
    print(f"  * Manifest Labels  : {len(labels_map)} labels loaded" if labels_map else "  * Manifest Labels  : None (Unsupervised Testing)")
    print("=" * 80 + "\n")

    engine = InferenceEngine()
    if threshold is not None:
        engine.calibrated_threshold = threshold

    results = []
    y_true = []
    y_pred = []
    y_scores = []
    latencies = []

    for idx, apath in enumerate(audio_files, 1):
        cid = apath.stem
        t0 = time.perf_counter()

        try:
            # 1. Load raw audio
            data, orig_sr = sf.read(str(apath), dtype="float32")

            # 2. Dynamic Caller Speech Extraction
            caller_16k, intervals = dynamic_extract_caller_speech(
                audio_data=data,
                orig_sr=orig_sr,
                target_sr=16000,
                target_samples=None,
                top_db=32.0,
                pad_ms=180,
                min_silence_bridge_ms=300
            )

            # 3. Model Inference
            is_synthetic, confidence = engine.predict(caller_16k)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            latencies.append(elapsed_ms)

            gt_label = labels_map.get(cid, None)
            verdict_str = "SYNTHETIC" if is_synthetic else "HUMAN"

            res_entry = {
                "call_id": cid,
                "file_path": str(apath),
                "duration_s": round(len(data) / orig_sr, 2),
                "caller_speech_s": round(len(caller_16k) / 16000, 2),
                "speech_turns": len(intervals),
                "is_synthetic": is_synthetic,
                "confidence": confidence,
                "verdict": verdict_str,
                "latency_ms": round(elapsed_ms, 2)
            }

            if gt_label is not None:
                gt_str = "SYNTHETIC" if gt_label == 1 else "HUMAN"
                is_correct = (1 if is_synthetic else 0) == gt_label
                res_entry["ground_truth"] = gt_str
                res_entry["correct"] = is_correct
                y_true.append(gt_label)
                y_pred.append(1 if is_synthetic else 0)
                y_scores.append(confidence)

            results.append(res_entry)

            status_icon = "✓" if (gt_label is None or res_entry.get("correct")) else "✗"
            gt_tag = f" | GT: {res_entry.get('ground_truth')}" if gt_label is not None else ""
            print(f"  [{idx:>3}/{len(audio_files)}] {status_icon} {cid} -> {verdict_str} (Conf: {confidence:.4f}, Latency: {elapsed_ms:.1f}ms{gt_tag})")

        except Exception as e:
            print(f"  [{idx:>3}/{len(audio_files)}] ✗ Error processing {cid}: {e}")

    # Export to CSV if output path specified
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"\n✓ Saved batch evaluation results to: {output_csv}")

    # Print Summary Metrics
    print("\n" + "=" * 80)
    print("                              EVALUATION SUMMARY")
    print("=" * 80)
    print(f"  * Total Evaluated Files : {len(results)}")
    print(f"  * Mean Inference Latency: {np.mean(latencies):.2f} ms")
    print(f"  * Median Latency        : {np.median(latencies):.2f} ms")

    if y_true and len(y_true) == len(y_pred):
        acc = accuracy_score(y_true, y_pred) * 100
        prec = precision_score(y_true, y_pred, zero_division=0) * 100
        rec = recall_score(y_true, y_pred, zero_division=0) * 100
        f1 = f1_score(y_true, y_pred, zero_division=0) * 100
        cm = confusion_matrix(y_true, y_pred)

        print(f"  * Accuracy              : {acc:.2f}%")
        print(f"  * Precision             : {prec:.2f}%")
        print(f"  * Recall                : {rec:.2f}%")
        print(f"  * F1-Score              : {f1:.2f}%")
        print(f"  * Confusion Matrix      : TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    print("=" * 80 + "\n")

    return {
        "total_files": len(results),
        "mean_latency_ms": float(np.mean(latencies)),
        "accuracy": float(accuracy_score(y_true, y_pred) * 100) if y_true else None,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate a directory of audio calls with dynamic channel separation.")
    parser.add_argument("--audio_dir", type=str, default="Data/audio/test", help="Directory with WAV files")
    parser.add_argument("--manifest", type=str, default="Data/test_manifest.csv", help="Path to manifest CSV with ground-truth labels")
    parser.add_argument("--output_csv", type=str, default="Data/batch_evaluation_results.csv", help="Output CSV path")
    parser.add_argument("--threshold", type=float, default=None, help="Custom decision threshold (e.g. 0.9623)")

    args = parser.parse_args()
    batch_evaluate_directory(
        audio_dir=Path(args.audio_dir),
        manifest_path=Path(args.manifest) if args.manifest else None,
        output_csv=Path(args.output_csv) if args.output_csv else None,
        threshold=args.threshold
    )


if __name__ == "__main__":
    main()
