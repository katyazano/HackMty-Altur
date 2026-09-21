"""
benchmark_comparison.py — Runs synthetic & test evaluations on both Baseline and Multimodal engines.
"""
import io
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engines.unified_orchestrator import UnifiedOrchestrator


def generate_synthetic_stereo_wav(duration_s=4.0, sr=8000, is_synthetic=True):
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    if is_synthetic:
        # Flat harmonic robotic tone without pitch micro-jitter
        caller = 0.3 * np.sin(2 * np.pi * 240 * t) + 0.15 * np.sin(2 * np.pi * 480 * t)
    else:
        # Modulated natural speech-like waveform
        f0 = 140 + 15 * np.sin(2 * np.pi * 1.2 * t)
        caller = 0.3 * np.sin(2 * np.pi * f0 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 2.5 * t))

    agent = 0.2 * np.sin(2 * np.pi * 180 * t)
    stereo = np.stack([caller, agent], axis=1).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV")
    return buf.getvalue()


def run_benchmark():
    print("=" * 70)
    print(" ALTUR DUAL-ENGINE BIOMETRIC BENCHMARK VERIFICATION")
    print("=" * 70)

    orchestrator = UnifiedOrchestrator()
    print("✔ Engines Loaded Successfully:")
    print("   - Engine 1: Baseline Acoustic DSP + AudioCNN")
    print("   - Engine 2: 4-Pillar Multimodal (AASIST + RawNet2 + DSP + Whisper + XGBoost)\n")

    # Generate synthetic call
    synth_wav = generate_synthetic_stereo_wav(duration_s=4.0, is_synthetic=True)
    real_wav = generate_synthetic_stereo_wav(duration_s=4.0, is_synthetic=False)

    print("▶ Evaluating Test Audio 1: Synthetic Carrier Buzz...")
    comp_synth = orchestrator.compare(synth_wav)
    print(f"   Baseline:   is_synthetic={comp_synth['baseline']['is_synthetic']}, P(synth)={comp_synth['baseline']['probability_synthetic']}, Latency={comp_synth['baseline']['latency_ms']}ms")
    print(f"   Multimodal: is_synthetic={comp_synth['multimodal']['is_synthetic']}, P(synth)={comp_synth['multimodal']['probability_synthetic']}, Latency={comp_synth['multimodal']['latency_ms']}ms")
    print(f"   Consensus:  {comp_synth['consensus']['agreement_status']} -> {comp_synth['consensus']['recommended_verdict']}\n")

    print("▶ Evaluating Test Audio 2: Modulated Human-Style Voice...")
    comp_real = orchestrator.compare(real_wav)
    print(f"   Baseline:   is_synthetic={comp_real['baseline']['is_synthetic']}, P(synth)={comp_real['baseline']['probability_synthetic']}, Latency={comp_real['baseline']['latency_ms']}ms")
    print(f"   Multimodal: is_synthetic={comp_real['multimodal']['is_synthetic']}, P(synth)={comp_real['multimodal']['probability_synthetic']}, Latency={comp_real['multimodal']['latency_ms']}ms")
    print(f"   Consensus:  {comp_real['consensus']['agreement_status']} -> {comp_real['consensus']['recommended_verdict']}\n")

    print("=" * 70)
    print("✔ Benchmark verification complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
