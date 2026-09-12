import os
import sys
import csv
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
import soundfile as sf
import torch
import numpy as np

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.rawnet2 import RawNet2
from src.models.aasist import AASIST

def load_manifest_ground_truth(manifest_path: str) -> Dict[str, dict]:
    gt_map = {}
    if not os.path.exists(manifest_path):
        return gt_map

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("anon_id") or row.get("id")
            if cid:
                cid = cid.strip()
                gt_map[cid] = {
                    "id": cid,
                    "label": row.get("label", "unknown").strip().lower(),
                    "split": row.get("split", "train").strip().lower(),
                    "duration_s": row.get("duration_s") or row.get("duration")
                }
    return gt_map

def evaluate_channel0_dataset(
    agent0_dir: Optional[str] = None,
    manifest_path: Optional[str] = None,
    output_json: Optional[str] = None,
    output_csv: Optional[str] = None,
    max_workers: int = 8
) -> List[dict]:
    if agent0_dir is None:
        agent0_dir = str(PROJECT_ROOT / "Data" / "separated_agents" / "train" / "agent_0")
    if manifest_path is None:
        manifest_path = str(PROJECT_ROOT / "Data" / "manifest.csv")
    if output_json is None:
        output_json = str(PROJECT_ROOT / "Data" / "channel0_evaluation_results.json")
    if output_csv is None:
        output_csv = str(PROJECT_ROOT / "Data" / "channel0_evaluation_results.csv")

    agent0_path = Path(agent0_dir)
    if not agent0_path.exists():
        print(f"Error: Agent 0 directory does not exist: {agent0_path}", flush=True)
        sys.exit(1)

    print("=" * 80, flush=True)
    print("      CHANNEL 0 (AGENT 0) VOICE ANTI-SPOOFING BENCHMARK EVALUATION", flush=True)
    print("=" * 80, flush=True)
    print(f"  * Audio Directory : {agent0_path}", flush=True)
    print(f"  * Manifest Source : {manifest_path}", flush=True)
    print("=" * 80 + "\n", flush=True)

    gt_map = load_manifest_ground_truth(manifest_path)
    audio_files = sorted(list(agent0_path.glob("*.wav")))
    if not audio_files:
        print(f"Error: No .wav files found in {agent0_path}", flush=True)
        sys.exit(1)

    print(f"Discovered {len(audio_files)} Channel 0 audio files. Starting parallel evaluation...\n", flush=True)

    # Thread-local models
    models = {
        "rawnet": RawNet2(),
        "aasist": AASIST()
    }
    models["rawnet"].eval()
    models["aasist"].eval()

    results = []
    processed = 0
    total = len(audio_files)
    t_start = time.time()

    def process_single_audio(audio_file: Path) -> Optional[dict]:
        file_stem = audio_file.stem
        call_id = file_stem.replace("_agent_0", "")

        meta = gt_map.get(call_id, {})
        raw_gt = meta.get("label", "unknown")
        is_human_gt = (raw_gt == "human")
        is_synthetic_gt = (raw_gt == "synthetic")

        try:
            data, sr = sf.read(str(audio_file), dtype="float32")
            if data.ndim > 1:
                data = data[:, 0]
            
            dur = round(len(data) / sr, 2)
            tensor_in = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

            t0 = time.perf_counter()
            r_res = models["rawnet"].predict_spoof(tensor_in)
            r_lat = round((time.perf_counter() - t0) * 1000, 2)

            t0 = time.perf_counter()
            a_res = models["aasist"].predict_spoof(tensor_in)
            a_lat = round((time.perf_counter() - t0) * 1000, 2)

            r_is_real = r_res["is_real"]
            a_is_real = a_res["is_real"]
            r_score = r_res["real_score_pct"]
            a_score = a_res["real_score_pct"]

            r_is_correct = (r_is_real == is_human_gt) if (is_human_gt or is_synthetic_gt) else None
            a_is_correct = (a_is_real == is_human_gt) if (is_human_gt or is_synthetic_gt) else None

            return {
                "audio_id": call_id,
                "filename": audio_file.name,
                "audio_url": f"/audio/agent_0/{audio_file.name}",
                "duration_s": dur,
                "ground_truth": raw_gt,
                "is_ground_truth_human": is_human_gt,
                "rawnet2_score": r_score,
                "rawnet2_prediction": "HUMAN" if r_is_real else "SYNTHETIC",
                "rawnet2_is_real": r_is_real,
                "rawnet2_is_correct": r_is_correct,
                "rawnet2_latency_ms": r_lat,
                "aasistl_score": a_score,
                "aasistl_prediction": "HUMAN" if a_is_real else "SYNTHETIC",
                "aasistl_is_real": a_is_real,
                "aasistl_is_correct": a_is_correct,
                "aasistl_latency_ms": a_lat
            }
        except Exception as e:
            print(f"Error on {audio_file.name}: {e}", flush=True)
            return None

    # Run multi-threaded evaluation
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(process_single_audio, audio_files):
            if res:
                results.append(res)
                processed += 1
                if processed % 30 == 0 or processed == total:
                    print(f"  [{processed:3d}/{total}] Evaluated {res['audio_id']} | GT: {res['ground_truth'].upper():9s} | RawNet2: {res['rawnet2_score']:5.1f}% | AASIST: {res['aasistl_score']:5.1f}%", flush=True)

    elapsed = round(time.time() - t_start, 2)
    rawnet_correct = sum(1 for r in results if r["rawnet2_is_correct"] is True)
    aasist_correct = sum(1 for r in results if r["aasistl_is_correct"] is True)

    # Sort results by audio_id
    results.sort(key=lambda x: x["audio_id"])

    # Save JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": len(results),
            "evaluation_time_s": elapsed,
            "rawnet2_accuracy_pct": round((rawnet_correct / len(results) * 100), 2) if results else 0,
            "aasistl_accuracy_pct": round((aasist_correct / len(results) * 100), 2) if results else 0,
            "results": results
        }, f, indent=2)

    # Save CSV
    headers = [
        "audio_id", "ground_truth", "duration_s",
        "rawnet2_score", "rawnet2_prediction", "rawnet2_is_correct", "rawnet2_latency_ms",
        "aasistl_score", "aasistl_prediction", "aasistl_is_correct", "aasistl_latency_ms"
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in results:
            writer.writerow([
                r["audio_id"], r["ground_truth"], r["duration_s"],
                r["rawnet2_score"], r["rawnet2_prediction"], r["rawnet2_is_correct"], r["rawnet2_latency_ms"],
                r["aasistl_score"], r["aasistl_prediction"], r["aasistl_is_correct"], r["aasistl_latency_ms"]
            ])

    print("\n" + "=" * 80, flush=True)
    print("                    EVALUATION BENCHMARK COMPLETE", flush=True)
    print("=" * 80, flush=True)
    print(f"  * Total Evaluated : {len(results)} recordings in {elapsed}s", flush=True)
    print(f"  * RawNet2 Accuracy: {round((rawnet_correct / len(results) * 100), 2) if results else 0}% ({rawnet_correct}/{len(results)})", flush=True)
    print(f"  * AASIST Accuracy : {round((aasist_correct / len(results) * 100), 2) if results else 0}% ({aasist_correct}/{len(results)})", flush=True)
    print(f"  * JSON Results Saved: {output_json}", flush=True)
    print(f"  * CSV Results Saved : {output_csv}", flush=True)
    print("=" * 80 + "\n", flush=True)

    return results

if __name__ == "__main__":
    evaluate_channel0_dataset()
