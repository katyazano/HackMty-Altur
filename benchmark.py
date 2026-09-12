import os
import sys
import time
import glob
import numpy as np
import soundfile as sf
import torch

# Configure UTF-8 encoding for Windows console compatibility
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Add workspace to sys.path
sys.path.insert(0, os.path.dirname(__file__))

from src.audio_processor import AudioProcessor
from src.models.rawnet2 import RawNet2
from src.models.aasist import AASIST

RAW_DIR = os.path.join(os.path.dirname(__file__), "demo_real_audios")

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_benchmark():
    print("=" * 80)
    print("      ALTUR BANKING - ANTI-SPOOFING MODEL BENCHMARK (RawNet2 vs AASIST)")
    print("=" * 80)

    # 1. Initialize Engines
    print("\n[1/3] Initializing Deep Learning Anti-Spoofing Models...")
    processor = AudioProcessor(target_sr=16000)

    rawnet2_model = RawNet2()
    rawnet2_model.eval()
    rawnet2_params = count_parameters(rawnet2_model)

    aasist_model = AASIST()
    aasist_model.eval()
    aasist_params = count_parameters(aasist_model)

    print(f"  • Model 1: RawNet2 (SincNet + ResNet-GRU)        -> {rawnet2_params:,} parameters")
    print(f"  • Model 2: AASIST-L (Spectro-Temporal GAT)      -> {aasist_params:,} parameters")

    # 2. Gather Test Audio Files
    print("\n[2/3] Loading Test Audio Corpus...")
    valid_exts = ('.ogg', '.wav', '.mp3', '.flac')
    available_tests = []
    
    if os.path.exists(RAW_DIR):
        for fname in sorted(os.listdir(RAW_DIR)):
            if fname.lower().endswith(valid_exts):
                available_tests.append((fname, os.path.join(RAW_DIR, fname)))

    print(f"  ✓ {len(available_tests)} audio files ready for benchmark.")

    # 3. Benchmark Execution
    print("\n[3/3] Running Side-by-Side Inference Benchmark...")
    print("-" * 80)
    header = f"{'Audio Sample':<35} | {'RawNet2':<20} | {'AASIST-L':<20}"
    print(header)
    print("-" * 80)

    results = []

    for label, audio_path in available_tests:
        y_raw, sr = processor.load_and_normalize(audio_path)
        y_clean = processor.apply_vad(processor.apply_bandpass_filter(processor.apply_denoising(y_raw)))

        tensor_in = torch.tensor(y_clean, dtype=torch.float32).unsqueeze(0)

        # A. Benchmark RawNet2
        t0 = time.perf_counter()
        raw_res = rawnet2_model.predict_spoof(tensor_in)
        raw_latency = (time.perf_counter() - t0) * 1000

        # B. Benchmark AASIST-L
        t0 = time.perf_counter()
        aasist_res = aasist_model.predict_spoof(tensor_in)
        aasist_latency = (time.perf_counter() - t0) * 1000

        raw_str = f"{'REAL' if raw_res.get('is_real', not raw_res.get('is_spoof', False)) else 'SPOOF'} ({raw_res.get('real_score_pct', 0)}%, {raw_latency:.1f}ms)"
        aasist_str = f"{'REAL' if aasist_res.get('is_real', not aasist_res.get('is_spoof', False)) else 'SPOOF'} ({aasist_res.get('real_score_pct', 0)}%, {aasist_latency:.1f}ms)"

        print(f"{label:<35} | {raw_str:<20} | {aasist_str:<20}")

        results.append({
            "label": label,
            "rawnet2": {**raw_res, "latency_ms": round(raw_latency, 2)},
            "aasist": {**aasist_res, "latency_ms": round(aasist_latency, 2)}
        })

    print("-" * 80)
    print("\n" + "=" * 80)
    print("                           SUMMARY & COMPARISON")
    print("=" * 80)
    print("  • AASIST-L: Graph attention network for spectro-temporal artifact detection.")
    print("  • RawNet2:  SincNet + Residual GRU network directly from raw waveforms.")
    print("=" * 80 + "\n")

    return results

if __name__ == "__main__":
    run_benchmark()
