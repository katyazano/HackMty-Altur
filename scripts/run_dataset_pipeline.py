import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm

# Resolve project root
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.unified_analyzer import UnifiedDialogueAnalyzer


def evaluate_dataset(manifest_csv: str, asr_model_size: str = "tiny", max_samples: int = None, output_prefix: str = "full_dataset"):
    print("\n" + "=" * 95)
    print(f"🚀 RUNNING UNIFIED PIPELINE ACROSS DATASET: {manifest_csv}")
    print("=" * 95)

    df = pd.read_csv(manifest_csv)
    if max_samples is not None and max_samples > 0:
        df = df.head(max_samples)

    total_calls = len(df)
    print(f"Found {total_calls} calls to process.\n")

    # Output directories
    out_dir = Path("Data/dataset_transcripts")
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path("Data/dataset_temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Initialize master analyzer once
    analyzer = UnifiedDialogueAnalyzer(asr_model_size=asr_model_size, device="cpu")

    results = []
    t_start_total = time.perf_counter()

    for idx, row in df.iterrows():
        call_id = str(row["id"])
        true_label = str(row.get("label", "unknown")).lower()
        is_true_synthetic = (true_label == "synthetic")

        # Find audio file
        audio_path = f"Data/audio/test/{call_id}.wav"
        if not os.path.exists(audio_path):
            audio_path = f"Data/audio/train/{call_id}.wav"
        if not os.path.exists(audio_path):
            print(f"⚠️ Audio file not found for {call_id}, skipping...")
            continue

        # Find turns metadata
        turns_path = f"Data/turns/test/{call_id}.json"
        if not os.path.exists(turns_path):
            turns_path = f"Data/turns/train/{call_id}.json"
        if not os.path.exists(turns_path):
            turns_path = None

        # Load stereo audio & split
        data, sr = sf.read(audio_path, dtype="float32")
        ch0_caller = data[:, 0]
        ch1_agent = data[:, 1]

        caller_path = str(temp_dir / f"{call_id}_caller.wav")
        agent_path = str(temp_dir / f"{call_id}_agent.wav")
        sf.write(caller_path, ch0_caller, sr)
        sf.write(agent_path, ch1_agent, sr)

        call_json_out = str(out_dir / f"{call_id}_transcript.json")

        t_call_start = time.perf_counter()
        report = analyzer.process_call(
            caller_audio=caller_path,
            agent_audio=agent_path,
            turns_metadata=turns_path,
            output_json_path=call_json_out
        )
        t_call_elapsed = time.perf_counter() - t_call_start

        pred_synthetic = report["anti_spoofing_verdict"]["is_synthetic"]
        conf = report["anti_spoofing_verdict"]["confidence"]
        dur = report["call_summary"]["caller_duration_sec"]
        rtf = report["call_summary"]["real_time_factor"]
        turns_cnt = report["call_summary"]["total_dialogue_turns"]
        fillers = report["semantic_nlp_profile"]["detected_human_fillers"]
        filler_dens = report["semantic_nlp_profile"]["filler_density"]
        authen_score = report["semantic_nlp_profile"]["semantic_authenticity_score"]

        correct = (pred_synthetic == is_true_synthetic) if true_label != "unknown" else None

        results.append({
            "call_id": call_id,
            "true_label": true_label,
            "pred_synthetic": pred_synthetic,
            "confidence": conf,
            "correct": correct,
            "duration_sec": dur,
            "proc_time_sec": round(t_call_elapsed, 2),
            "rtf": rtf,
            "total_turns": turns_cnt,
            "filler_count": len(fillers),
            "filler_density": filler_dens,
            "detected_fillers": ", ".join(fillers),
            "semantic_authenticity_score": authen_score,
            "json_path": call_json_out
        })

        tag = "✓" if correct else "✗"
        print(f"[{idx+1}/{total_calls}] {call_id}: True={true_label:<9} | Pred={'SYNTHETIC' if pred_synthetic else 'HUMAN':<9} | Conf={conf*100:5.1f}% | RTF={rtf:.3f}x | {tag}")

    total_elapsed = time.perf_counter() - t_start_total
    res_df = pd.DataFrame(results)

    # Save summary CSV
    csv_out = f"Data/{output_prefix}_evaluation_results.csv"
    res_df.to_csv(csv_out, index=False)

    # Compute Metrics
    evaluated_df = res_df[res_df["true_label"].isin(["synthetic", "human"])]
    metrics = {}
    if not evaluated_df.empty:
        total_eval = len(evaluated_df)
        acc = (evaluated_df["correct"].sum() / total_eval) * 100.0

        synth_df = evaluated_df[evaluated_df["true_label"] == "synthetic"]
        synth_recall = (synth_df["pred_synthetic"].sum() / max(1, len(synth_df))) * 100.0

        human_df = evaluated_df[evaluated_df["true_label"] == "human"]
        human_spec = ((~human_df["pred_synthetic"]).sum() / max(1, len(human_df))) * 100.0

        balanced_acc = (synth_recall + human_spec) / 2.0
        avg_rtf = evaluated_df["rtf"].mean()
        avg_conf = evaluated_df["confidence"].mean() * 100.0

        metrics = {
            "total_calls_evaluated": total_eval,
            "accuracy_percent": round(acc, 2),
            "balanced_accuracy_percent": round(balanced_acc, 2),
            "synthetic_detection_rate_percent": round(synth_recall, 2),
            "human_bona_fide_pass_rate_percent": round(human_spec, 2),
            "average_confidence_percent": round(avg_conf, 2),
            "average_rtf": round(avg_rtf, 4),
            "total_processing_time_sec": round(total_elapsed, 2)
        }

        json_out = f"Data/{output_prefix}_evaluation_summary.json"
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    print("\n" + "=" * 95)
    print("📊 DATASET EVALUATION BENCHMARK SUMMARY")
    print("=" * 95)
    if metrics:
        print(f"• Total Calls Evaluated:        {metrics['total_calls_evaluated']}")
        print(f"• Balanced Accuracy:            {metrics['balanced_accuracy_percent']}%")
        print(f"• Overall Accuracy:             {metrics['accuracy_percent']}%")
        print(f"• Synthetic (Spoof) Detection:  {metrics['synthetic_detection_rate_percent']}%")
        print(f"• Human (Bona Fide) Pass Rate:  {metrics['human_bona_fide_pass_rate_percent']}%")
        print(f"• Average Model Confidence:     {metrics['average_confidence_percent']}%")
        print(f"• Average Real-Time Factor:     {metrics['average_rtf']}x (~{1/max(0.001, metrics['average_rtf']):.1f}x faster than real-time)")
        print(f"• Total Elapsed Time:           {metrics['total_processing_time_sec']}s")
    print(f"• CSV Report Saved:             {csv_out}")
    print(f"• Transcripts Saved In:         {out_dir}/")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Unified Pipeline across dataset")
    parser.add_argument("--manifest", type=str, default="Data/test_manifest.csv", help="Path to manifest CSV")
    parser.add_argument("--model-size", type=str, default="tiny", help="Whisper model size")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--prefix", type=str, default="full_dataset", help="Output file prefix")
    args = parser.parse_args()

    evaluate_dataset(
        manifest_csv=args.manifest,
        asr_model_size=args.model_size,
        max_samples=args.max_samples,
        output_prefix=args.prefix
    )
