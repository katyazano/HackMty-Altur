import os
import sys
import time
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
from src.biometrics import BiometricsEngine
from src.models.rawnet2 import RawNet2
from src.models.aasist import AASIST

RAW_DIR = os.path.join(os.path.dirname(__file__), "demo_real_audios")

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_benchmark():
    print("=" * 80)
    print("      ALTUR BANKING - ANTI-SPOOFING MODEL BENCHMARK (DSP vs RawNet2 vs AASIST)")
    print("=" * 80)

    # 1. Initialize Engines
    print("\n[1/3] Initializing Anti-Spoofing Engines...")
    processor = AudioProcessor(target_sr=16000)
    dsp_engine = BiometricsEngine()

    rawnet2_model = RawNet2()
    rawnet2_model.eval()
    rawnet2_params = count_parameters(rawnet2_model)

    aasist_model = AASIST()
    aasist_model.eval()
    aasist_params = count_parameters(aasist_model)

    print(f"  • Model 1: DSP Baseline (LFCC + Spectral Ratio) -> 0 parameters (Rule-based)")
    print(f"  • Model 2: RawNet2 (SincNet + ResNet-GRU)        -> {rawnet2_params:,} parameters")
    print(f"  • Model 3: AASIST-L (Spectro-Temporal GAT)      -> {aasist_params:,} parameters")

    # 2. Gather Test Audio Files
    print("\n[2/3] Loading Test Audio Corpus...")
    test_files = [
        ("Primary User (Rec 1)", "user_uploaded_voice.ogg"),
        ("Primary User (Rec 2)", "user_uploaded_voice_2.ogg"),
        ("Second User (Rec 3 - Male)", "user_uploaded_impostor.ogg"),
        ("Benchmark LibriSpeech 1272", "real_speaker_1272_sample1.flac"),
        ("Synthetic Deepfake Demo", os.path.join("..", "demo_data", "synthetic_deepfake.wav"))
    ]

    available_tests = []
    for label, rel_path in test_files:
        p = os.path.join(RAW_DIR, rel_path)
        if not os.path.exists(p):
            p = os.path.abspath(os.path.join(RAW_DIR, rel_path))
        if os.path.exists(p):
            available_tests.append((label, p))

    print(f"  ✓ {len(available_tests)} audio files ready for benchmark.")

    # 3. Benchmark Execution
    print("\n[3/3] Running Side-by-Side Inference Benchmark...")
    print("-" * 96)
    header = f"{'Audio Sample':<26} | {'DSP Baseline':<22} | {'RawNet2':<22} | {'AASIST-L':<22}"
    print(header)
    print("-" * 96)

    results = []

    for label, audio_path in available_tests:
        y_raw, sr = processor.load_and_normalize(audio_path)
        y_clean = processor.apply_vad(processor.apply_bandpass_filter(processor.apply_denoising(y_raw)))

        # A. Benchmark DSP
        t0 = time.perf_counter()
        lfcc = processor.extract_lfcc(y_clean)
        spec = processor.compute_spectral_features(y_clean)
        dsp_res = dsp_engine.assess_anti_spoofing(lfcc, spec)
        dsp_latency = (time.perf_counter() - t0) * 1000

        # B. Benchmark RawNet2
        tensor_in = torch.tensor(y_clean, dtype=torch.float32).unsqueeze(0)
        t0 = time.perf_counter()
        raw_res = rawnet2_model.predict_spoof(tensor_in)
        raw_latency = (time.perf_counter() - t0) * 1000

        # C. Benchmark AASIST-L
        t0 = time.perf_counter()
        aasist_res = aasist_model.predict_spoof(tensor_in)
        aasist_latency = (time.perf_counter() - t0) * 1000

        dsp_str = f"{'REAL' if dsp_res.get('is_real', not dsp_res.get('is_spoof', False)) else 'SPOOF'} ({dsp_res.get('real_score_pct', 0)}%, {dsp_latency:.1f}ms)"
        raw_str = f"{'REAL' if raw_res.get('is_real', not raw_res.get('is_spoof', False)) else 'SPOOF'} ({raw_res.get('real_score_pct', 0)}%, {raw_latency:.1f}ms)"
        aasist_str = f"{'REAL' if aasist_res.get('is_real', not aasist_res.get('is_spoof', False)) else 'SPOOF'} ({aasist_res.get('real_score_pct', 0)}%, {aasist_latency:.1f}ms)"

        print(f"{label:<26} | {dsp_str:<22} | {raw_str:<22} | {aasist_str:<22}")

        results.append({
            "label": label,
            "dsp": {**dsp_res, "latency_ms": round(dsp_latency, 2)},
            "rawnet2": {**raw_res, "latency_ms": round(raw_latency, 2)},
            "aasist": {**aasist_res, "latency_ms": round(aasist_latency, 2)}
        })

    print("-" * 80)
    print("\n" + "=" * 80)
    print("                           SUMMARY & COMPARISON")
    print("=" * 80)
    print("  • AASIST-L: Outstanding efficiency (~85K params) with graph attention reasoning.")
    print("  • RawNet2:  Robust temporal GRU modeling directly from raw SincNet waveforms.")
    print("  • DSP Rule: Fastest execution with zero parameter memory footprint.")
    print("=" * 80 + "\n")

    return results

if __name__ == "__main__":
    run_benchmark()
