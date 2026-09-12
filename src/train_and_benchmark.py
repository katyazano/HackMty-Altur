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

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.rawnet2 import RawNet2
from src.models.aasist import AASIST
from src.trainer import train_model
from src.evaluator import benchmark_model

def run_full_training_and_benchmark(epochs: int = 15, batch_size: int = 16, lr: float = 1e-4):
    print("=" * 80)
    print("      ALTUR ANTI-SPOOFING: END-TO-END TRAINING & TEST BENCHMARK")
    print("=" * 80)
    print(f"  * Target Epochs  : {epochs}")
    print(f"  * Batch Size     : {batch_size}")
    print(f"  * Learning Rate  : {lr}")
    print("=" * 80 + "\n", flush=True)

    weights_dir = PROJECT_ROOT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    rawnet_weights = str(weights_dir / "rawnet2_best.pth")
    aasist_weights = str(weights_dir / "aasist_best.pth")

    # -------------------------------------------------------------
    # 1. Train RawNet2
    # -------------------------------------------------------------
    print(">>> [1/2] Training RawNet2 Architecture...", flush=True)
    rawnet_model = RawNet2()
    rawnet_res = train_model(
        model=rawnet_model,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        save_path=rawnet_weights
    )

    # -------------------------------------------------------------
    # 2. Train AASIST
    # -------------------------------------------------------------
    print("\n>>> [2/2] Training AASIST Architecture...", flush=True)
    aasist_model = AASIST()
    aasist_res = train_model(
        model=aasist_model,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        save_path=aasist_weights
    )

    # -------------------------------------------------------------
    # 3. Load Best Checkpoints & Run Test Benchmark
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("           EVALUATING TRAINED MODELS ON TEST DATASET")
    print("=" * 80 + "\n", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading best RawNet2 weights from {rawnet_weights}...")
    rawnet_eval = RawNet2().to(device)
    rawnet_eval.load_state_dict(torch.load(rawnet_weights, map_location=device))
    rawnet_eval.eval()

    print(f"Loading best AASIST weights from {aasist_weights}...")
    aasist_eval = AASIST().to(device)
    aasist_eval.load_state_dict(torch.load(aasist_weights, map_location=device))
    aasist_eval.eval()

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
            data = data[mid - 32000 : mid + 32000]
        elif nb < 64000:
            data = np.pad(data, (0, 64000 - nb), mode="constant")

        t_in = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)

        # RawNet2 inference
        t0 = time.perf_counter()
        with torch.no_grad():
            r_logits = rawnet_eval(t_in)
            r_probs = F.softmax(r_logits, dim=1)
            r_real_pct = round(r_probs[0, 0].item() * 100, 1)
            r_is_real = (r_real_pct >= 50.0)
        r_lat = round((time.perf_counter() - t0) * 1000, 2)
        r_is_corr = (r_is_real == is_human_gt) if (is_human_gt or is_synth_gt) else None
        if r_is_corr:
            r_correct += 1

        # AASIST inference
        t0 = time.perf_counter()
        with torch.no_grad():
            a_logits = aasist_eval(t_in)
            a_probs = F.softmax(a_logits, dim=1)
            a_real_pct = round(a_probs[0, 0].item() * 100, 1)
            a_is_real = (a_real_pct >= 50.0)
        a_lat = round((time.perf_counter() - t0) * 1000, 2)
        a_is_corr = (a_is_real == is_human_gt) if (is_human_gt or is_synth_gt) else None
        if a_is_corr:
            a_correct += 1

        final_records.append({
            "audio_id": cid,
            "filename": wfile.name,
            "audio_url": f"/audio/agent_0/{wfile.name}",
            "duration_s": dur,
            "ground_truth": gt,
            "is_ground_truth_human": is_human_gt,
            "rawnet2_score": r_real_pct,
            "rawnet2_prediction": "HUMAN" if r_is_real else "SYNTHETIC",
            "rawnet2_is_real": r_is_real,
            "rawnet2_is_correct": r_is_corr,
            "rawnet2_latency_ms": r_lat,
            "aasistl_score": a_real_pct,
            "aasistl_prediction": "HUMAN" if a_is_real else "SYNTHETIC",
            "aasistl_is_real": a_is_real,
            "aasistl_is_correct": a_is_corr,
            "aasistl_latency_ms": a_lat
        })

    # Save to JSON
    out_json = PROJECT_ROOT / "Data" / "channel0_evaluation_results.json"
    total_test = len(final_records)
    r_acc = round((r_correct / total_test) * 100, 2) if total_test > 0 else 0
    a_acc = round((a_correct / total_test) * 100, 2) if total_test > 0 else 0

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": total_test,
            "rawnet2_accuracy_pct": r_acc,
            "aasistl_accuracy_pct": a_acc,
            "results": final_records
        }, f, indent=2)

    # Save to CSV
    out_csv = PROJECT_ROOT / "Data" / "channel0_evaluation_results.csv"
    headers = [
        "audio_id", "ground_truth", "duration_s",
        "rawnet2_score", "rawnet2_prediction", "rawnet2_is_correct", "rawnet2_latency_ms",
        "aasistl_score", "aasistl_prediction", "aasistl_is_correct", "aasistl_latency_ms"
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in final_records:
            writer.writerow([
                r["audio_id"], r["ground_truth"], r["duration_s"],
                r["rawnet2_score"], r["rawnet2_prediction"], r["rawnet2_is_correct"], r["rawnet2_latency_ms"],
                r["aasistl_score"], r["aasistl_prediction"], r["aasistl_is_correct"], r["aasistl_latency_ms"]
            ])

    print("\n" + "=" * 80)
    print("                    FINAL BENCHMARK COMPARISON")
    print("=" * 80)
    print(f"  * Total Test Samples   : {total_test}")
    print(f"  * RawNet2 Test Accuracy: {r_acc}% ({r_correct}/{total_test})")
    print(f"  * AASIST Test Accuracy : {a_acc}% ({a_correct}/{total_test})")
    print(f"  * Results JSON Saved   : {out_json}")
    print(f"  * Results CSV Saved    : {out_csv}")
    print("=" * 80 + "\n", flush=True)

if __name__ == "__main__":
    run_full_training_and_benchmark(epochs=15, batch_size=16, lr=1e-4)
