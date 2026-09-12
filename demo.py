import os
import sys
import numpy as np
import soundfile as sf

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

REAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "demo_real_audios")
PROCESSED_DIR = os.path.join(REAL_DATA_DIR, "processed_output")

def get_audio_files():
    """
    Returns user uploaded voice recordings and benchmark samples.
    """
    user_voice_1 = os.path.join(REAL_DATA_DIR, "user_uploaded_voice.ogg")
    user_voice_2 = os.path.join(REAL_DATA_DIR, "user_uploaded_voice_2.ogg")
    user_voice_3 = os.path.join(REAL_DATA_DIR, "user_uploaded_impostor.ogg")
    spk1_enrolled = os.path.join(REAL_DATA_DIR, "real_speaker_1272_sample1.flac")

    files = {
        "primary_user_rec1": user_voice_1,
        "primary_user_rec2": user_voice_2,
        "second_user_rec3": user_voice_3,
        "benchmark_speaker_1272": spk1_enrolled,
    }

    for name, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required audio file: {path}")

    return files

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    print("=" * 75)
    print("  ALTUR BANKING CHALLENGE - BALANCED VAD THRESHOLD (SWEET SPOT top_db=24)")
    print("=" * 75)

    print("\n[1/4] Loading All User Uploaded Audio Files...")
    audio_files = get_audio_files()
    for name, path in audio_files.items():
        print(f"  * RAW AUDIO: {name.upper()} -> {os.path.basename(path)}")

    processor = AudioProcessor(target_sr=16000)
    engine = BiometricsEngine(speaker_similarity_threshold=0.985)

    features = {}
    print("\n[2/4] Executing Audio Signal Processing (ASP) Pipeline...")
    print("  -> Step A: 16kHz Resampling & Mono Peak Normalization")
    print("  -> Step B: Balanced Spectral Denoising (prop_decrease=0.72)")
    print("  -> Step C: Speech Bandpass Filter (90Hz - 3900Hz)")
    print("  -> Step D: Balanced VAD Threshold (top_db=24 - Sweet Spot)")

    for key, raw_path in audio_files.items():
        # Load raw audio
        y_raw, sr = processor.load_and_normalize(raw_path)
        
        # Apply Spectral Denoising (balanced 72% reduction)
        y_denoised = processor.apply_denoising(y_raw, prop_decrease=0.72)
        
        # Apply Vocal Bandpass Filter (90Hz - 3900Hz)
        y_filtered = processor.apply_bandpass_filter(y_denoised, lowcut=90.0, highcut=3900.0)
        
        # Apply Voice Activity Detection with balanced threshold (top_db=24)
        y_final = processor.apply_vad(y_filtered, top_db=24)

        # Export processed audio
        processed_path = os.path.join(PROCESSED_DIR, f"processed_{key}_denoised.wav")
        sf.write(processed_path, y_final, sr)

        # Extract features
        mfcc = processor.extract_mfcc(y_final)
        lfcc = processor.extract_lfcc(y_final)
        spec = processor.compute_spectral_features(y_final)

        raw_duration = round(len(y_raw) / sr, 2)
        final_duration = round(len(y_final) / sr, 2)

        features[key] = {
            "y": y_final,
            "mfcc": mfcc,
            "lfcc": lfcc,
            "spectral": spec,
            "raw_path": raw_path,
            "processed_path": processed_path,
            "raw_duration": raw_duration,
            "final_duration": final_duration
        }

        print(f"\n  [OK] Sample: {key.upper()}")
        print(f"       Raw Audio Duration       : {raw_duration}s")
        print(f"       Cleaned Audio Duration   : {final_duration}s (Balanced VAD top_db=24)")
        print(f"       Exported Audio File      : {os.path.basename(processed_path)}")
        print(f"       Audio Metrics            : Centroid={spec['spectral_centroid_hz']}Hz | HF-Energy={spec['high_freq_energy_ratio']}")

    print("\n" + "=" * 75)
    print(" [3/4] SPEAKER VERIFICATION & AUTHENTICATION TEST")
    print("=" * 75)
    
    enrolled_mfcc = features["primary_user_rec1"]["mfcc"]

    # Test A: Primary User Rec 1 vs Primary User Rec 2 (Same Speaker)
    res_same = engine.verify_speaker(enrolled_mfcc, features["primary_user_rec2"]["mfcc"])
    status_same = "[AUTHORIZED - SAME USER MATCH]" if res_same["is_match"] else "[DENIED]"
    
    print("\n  TEST A: PRIMARY USER REC 1 vs PRIMARY USER REC 2 (Same Speaker Attempt)")
    print(f"  Similarity Score : {res_same['similarity_score']} (Match Confidence: {res_same['confidence_percentage']}%)")
    print(f"  Threshold Used   : {res_same['threshold_used']}")
    print(f"  Verification     : {status_same}")

    # Test B: Primary User Rec 1 vs Second User Rec 3 (Different Speaker)
    res_diff = engine.verify_speaker(enrolled_mfcc, features["second_user_rec3"]["mfcc"])
    status_diff = "[AUTHORIZED - MATCH]" if res_diff["is_match"] else "[DENIED - DIFFERENT SPEAKER IMPOSTOR]"
    
    print("\n  TEST B: PRIMARY USER REC 1 vs SECOND USER REC 3 (Impostor Attempt)")
    print(f"  Similarity Score : {res_diff['similarity_score']} (Match Confidence: {res_diff['confidence_percentage']}%)")
    print(f"  Threshold Used   : {res_diff['threshold_used']}")
    print(f"  Verification     : {status_diff}")

    print("\n" + "=" * 75)
    print(" [4/4] ANTI-SPOOFING ASSESSMENT")
    print("=" * 75)

    for key, data in features.items():
        spoof_res = engine.assess_anti_spoofing(data["lfcc"], data["spectral"])
        badge = "[REJECT - SYNTHETIC]" if spoof_res["is_spoof"] else "[PASS - AUTHENTIC]"
        print(f"\n  Audio Sample : {key.upper()}")
        print(f"  Verdict      : {spoof_res['verdict']}")
        print(f"  Risk Score   : {spoof_res['risk_score_percentage']}%")
        print(f"  Security Flag: {badge}")

    print("\n" + "=" * 75)
    print(" RE-PROCESSED AUDIO EXPORTED WITH BALANCED VAD (top_db=24)")
    print("=" * 75)

if __name__ == "__main__":
    main()
