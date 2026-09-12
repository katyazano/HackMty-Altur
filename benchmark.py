import os
import sys
import time
import glob
import numpy as np
import soundfile as sf
import torch

# Configure UTF-8 encoding for Windows/Mac console compatibility
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
from src.models.ensemble import AASISTXGBoostEnsemble
from src.features import AcousticFeatureExtractor
import joblib
import xgboost as xgb

PROJECT_ROOT = os.path.dirname(__file__)
RAW_DIR = os.path.join(PROJECT_ROOT, "Data", "audio", "test")
if not os.path.exists(RAW_DIR):
    RAW_DIR = os.path.join(PROJECT_ROOT, "Data", "separated_agents", "test", "agent_0")

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_benchmark():
    print("=" * 105)
    print("      ALTUR BANKING - MULTI-MODEL ANTI-SPOOFING BENCHMARK")
    print("      (RawNet2 vs AASIST vs Acoustic-XGBoost vs AASIST+XGBoost Ensemble)")
    print("=" * 105)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AudioProcessor(target_sr=16000)

    # 1. Initialize Models
    print("\n[1/3] Initializing Anti-Spoofing Model Architecture Engines...")

    # A. RawNet2
    rawnet2_weights = os.path.join(PROJECT_ROOT, "weights", "rawnet2_best.pth")
    rawnet2_model = RawNet2().to(device)
    if os.path.exists(rawnet2_weights):
        rawnet2_model.load_state_dict(torch.load(rawnet2_weights, map_location=device))
    rawnet2_model.eval()

    # B. AASIST
    aasist_weights = os.path.join(PROJECT_ROOT, "weights", "aasist_best.pth")
    aasist_model = AASIST().to(device)
    if os.path.exists(aasist_weights):
        aasist_model.load_state_dict(torch.load(aasist_weights, map_location=device))
    aasist_model.eval()

    # C. AASIST + XGBoost Ensemble
    xgb_ensemble_weights = os.path.join(PROJECT_ROOT, "weights", "xgboost_aasist_ensemble.json")
    xgb_scaler_path = os.path.join(PROJECT_ROOT, "weights", "xgboost_scaler.joblib")
    ensemble_model = AASISTXGBoostEnsemble(
        aasist_model=aasist_model,
        xgb_model_path=xgb_ensemble_weights if os.path.exists(xgb_ensemble_weights) else None,
        scaler_path=xgb_scaler_path if os.path.exists(xgb_scaler_path) else None,
        device=str(device)
    )

    # D. Acoustic-Only XGBoost
    xgb_ac_weights = os.path.join(PROJECT_ROOT, "weights", "xgboost_acoustic_only.json")
    ac_scaler_path = os.path.join(PROJECT_ROOT, "weights", "acoustic_scaler.joblib")
    acoustic_extractor = AcousticFeatureExtractor(sample_rate=16000)
    xgb_ac_model = None
    ac_scaler = None
    if os.path.exists(xgb_ac_weights) and os.path.exists(ac_scaler_path):
        xgb_ac_model = xgb.XGBClassifier()
        xgb_ac_model.load_model(xgb_ac_weights)
        ac_scaler = joblib.load(ac_scaler_path)

    print(f"  • Model 1: RawNet2 (SincNet + ResNet-GRU)          -> {count_parameters(rawnet2_model):,} parameters")
    print(f"  • Model 2: AASIST-L (Spectro-Temporal GAT)        -> {count_parameters(aasist_model):,} parameters")
    print(f"  • Model 3: XGBoost (Acoustic DSP Features)        -> 134 acoustic/spectral features")
    print(f"  • Model 4: AASIST + XGBoost (Multi-Model Ensemble)-> 266 combined latent+DSP features")

    # 2. Gather Test Audio Files
    print("\n[2/3] Loading Test Audio Corpus...")
    valid_exts = ('.ogg', '.wav', '.mp3', '.flac')
    available_tests = []
    
    if os.path.exists(RAW_DIR):
        for fname in sorted(os.listdir(RAW_DIR))[:10]: # Top 10 for quick CLI demo
            if fname.lower().endswith(valid_exts):
                available_tests.append((fname, os.path.join(RAW_DIR, fname)))

    print(f"  ✓ {len(available_tests)} sample audio files ready for side-by-side benchmark.")

    # 3. Benchmark Execution
    print("\n[3/3] Running Side-by-Side Inference Benchmark...")
    print("-" * 105)
    header = f"{'Audio Sample':<28} | {'RawNet2':<16} | {'AASIST-L':<16} | {'Acoustic-XGB':<16} | {'AASIST+XGB (Ensemble)':<20}"
    print(header)
    print("-" * 105)

    results = []

    for label, audio_path in available_tests:
        y_raw, sr = processor.load_and_normalize(audio_path)
        y_clean = processor.apply_vad(processor.apply_bandpass_filter(processor.apply_denoising(y_raw)))
        tensor_in = torch.tensor(y_clean, dtype=torch.float32).unsqueeze(0)

        # 1. RawNet2
        t0 = time.perf_counter()
        raw_res = rawnet2_model.predict_spoof(tensor_in)
        raw_lat = (time.perf_counter() - t0) * 1000

        # 2. AASIST
        t0 = time.perf_counter()
        aasist_res = aasist_model.predict_spoof(tensor_in)
        aasist_lat = (time.perf_counter() - t0) * 1000

        # 3. Acoustic XGBoost
        ac_str = "N/A"
        if xgb_ac_model is not None and ac_scaler is not None:
            t0 = time.perf_counter()
            ac_f = acoustic_extractor.extract_features(y_clean, sr=sr).reshape(1, -1)
            ac_f_s = ac_scaler.transform(ac_f)
            p_spoof = float(xgb_ac_model.predict_proba(ac_f_s)[0, 1])
            ac_real_pct = round((1.0 - p_spoof) * 100, 1)
            ac_lat = (time.perf_counter() - t0) * 1000
            ac_str = f"{'REAL' if ac_real_pct >= 50 else 'SPOOF'} ({ac_real_pct}%)"

        # 4. AASIST + XGBoost Ensemble
        t0 = time.perf_counter()
        ens_res = ensemble_model.predict_spoof(tensor_in, sample_rate=sr)
        ens_lat = (time.perf_counter() - t0) * 1000
        ens_str = f"{'REAL' if ens_res['is_real'] else 'SPOOF'} ({ens_res['real_score_pct']}%, {ens_lat:.1f}ms)"

        raw_str = f"{'REAL' if raw_res.get('is_real') else 'SPOOF'} ({raw_res.get('real_score_pct', 0)}%)"
        aasist_str = f"{'REAL' if aasist_res.get('is_real') else 'SPOOF'} ({aasist_res.get('real_score_pct', 0)}%)"

        print(f"{label[:28]:<28} | {raw_str:<16} | {aasist_str:<16} | {ac_str:<16} | {ens_str:<20}")

        results.append({
            "label": label,
            "rawnet2": {**raw_res, "latency_ms": round(raw_lat, 2)},
            "aasist": {**aasist_res, "latency_ms": round(aasist_lat, 2)},
            "ensemble": {**ens_res, "latency_ms": round(ens_lat, 2)}
        })

    print("-" * 105)
    print("\n" + "=" * 105)
    print("                           SUMMARY & COMPARISON")
    print("=" * 105)
    print("  • AASIST-L               : Graph attention network for spectro-temporal artifact detection.")
    print("  • RawNet2                : SincNet + Residual GRU network directly from raw waveforms.")
    print("  • AASIST+XGBoost Ensemble: Stacked meta-learner combining graph embeddings & acoustic DSP.")
    print("=" * 105 + "\n")

    return results

if __name__ == "__main__":
    run_benchmark()
