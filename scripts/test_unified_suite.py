import os
import sys
import glob
import time
import json
from pathlib import Path
import numpy as np
import soundfile as sf

# Resolve project root
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.unified_analyzer import UnifiedDialogueAnalyzer


def run_test_suite():
    print("\n" + "=" * 90)
    print("🚀 STARTING COMPREHENSIVE TEST SUITE: Unified Dialogue Transcriber & 3-Model Biometric Engine")
    print("=" * 90)

    # 1. Initialize Analyzer
    t0 = time.perf_counter()
    analyzer = UnifiedDialogueAnalyzer(asr_model_size="tiny", device="cpu")
    init_time = round(time.perf_counter() - t0, 3)
    print(f"✓ Model Initialized in {init_time}s\n")

    test_calls = glob.glob("Data/audio/test/*.wav")[:3]
    if not test_calls:
        test_calls = glob.glob("Data/audio/train/*.wav")[:3]

    out_dir = Path("Data/test_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path("Data/test_temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for idx, audio_p in enumerate(test_calls, 1):
        call_id = Path(audio_p).stem
        turns_p = f"Data/turns/test/{call_id}.json"
        if not os.path.exists(turns_p):
            turns_p = f"Data/turns/train/{call_id}.json"

        print(f"\n--- [TEST {idx}/{len(test_calls)}] Evaluating Call: {call_id} ---")

        # Load stereo audio
        data, sr = sf.read(audio_p, dtype="float32")
        ch0_caller = data[:, 0]
        ch1_agent = data[:, 1]

        # Save separated tracks for testing
        caller_path = str(temp_dir / f"{call_id}_caller.wav")
        agent_path = str(temp_dir / f"{call_id}_agent.wav")
        sf.write(caller_path, ch0_caller, sr)
        sf.write(agent_path, ch1_agent, sr)

        # TEST CASE A: Synchronized Timeline with Metadata
        print(f"▶ Case A (Synchronized Timeline + Metadata):")
        report_a_path = str(out_dir / f"{call_id}_report_synchronized.json")
        res_a = analyzer.process_call(
            caller_audio=caller_path,
            agent_audio=agent_path,
            turns_metadata=turns_p if os.path.exists(turns_p) else None,
            output_json_path=report_a_path
        )

        verdict_str = "⚠️ SYNTHETIC" if res_a["anti_spoofing_verdict"]["is_synthetic"] else "✅ AUTHENTIC"
        print(f"  • Verdict: {verdict_str} (Confidence: {res_a['anti_spoofing_verdict']['confidence']*100:.1f}%)")
        print(f"  • RTF:     {res_a['call_summary']['real_time_factor']}x ({res_a['call_summary']['processing_time_sec']}s for {res_a['call_summary']['caller_duration_sec']}s audio)")
        print(f"  • Turns:   {res_a['call_summary']['total_dialogue_turns']} turns extracted")
        print(f"  • Fillers: {res_a['semantic_nlp_profile']['detected_human_fillers']} (Density: {res_a['semantic_nlp_profile']['filler_density']*100:.1f}%)")

        # TEST CASE B: Stripped/Concatenated Audio (No Metadata / No Silence)
        # Create concatenated active speech only
        active_caller = ch0_caller[np.abs(ch0_caller) > 0.01]
        active_agent = ch1_agent[np.abs(ch1_agent) > 0.01]
        if len(active_caller) < int(sr * 1.0):
            active_caller = ch0_caller
        if len(active_agent) < int(sr * 1.0):
            active_agent = ch1_agent

        stripped_caller_p = str(temp_dir / f"{call_id}_caller_stripped.wav")
        stripped_agent_p = str(temp_dir / f"{call_id}_agent_stripped.wav")
        sf.write(stripped_caller_p, active_caller, sr)
        sf.write(stripped_agent_p, active_agent, sr)

        print(f"\n▶ Case B (Stripped Speech / Silence Removed):")
        report_b_path = str(out_dir / f"{call_id}_report_stripped.json")
        res_b = analyzer.process_call(
            caller_audio=stripped_caller_p,
            agent_audio=stripped_agent_p,
            turns_metadata=None,
            output_json_path=report_b_path
        )
        print(f"  • Timeline Sync: {res_b['conversational_dynamics']['timeline_synchronized']}")
        print(f"  • Dynamics Note: {res_b['conversational_dynamics']['note']}")
        print(f"  • Verdict:       {'⚠️ SYNTHETIC' if res_b['anti_spoofing_verdict']['is_synthetic'] else '✅ AUTHENTIC'} (Confidence: {res_b['anti_spoofing_verdict']['confidence']*100:.1f}%)")

        results.append({
            "call_id": call_id,
            "duration_s": res_a["call_summary"]["caller_duration_sec"],
            "proc_time_s": res_a["call_summary"]["processing_time_sec"],
            "rtf": res_a["call_summary"]["real_time_factor"],
            "is_synthetic": res_a["anti_spoofing_verdict"]["is_synthetic"],
            "confidence": res_a["anti_spoofing_verdict"]["confidence"],
            "fillers": res_a["semantic_nlp_profile"]["detected_human_fillers"],
            "turns": res_a["call_summary"]["total_dialogue_turns"]
        })

    print("\n" + "=" * 90)
    print("📊 BATCH TEST SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Call ID':<22} | {'Dur (s)':<8} | {'Proc (s)':<9} | {'RTF':<7} | {'Verdict':<12} | {'Conf':<7} | {'Turns':<6}")
    print("-" * 90)
    for r in results:
        v_tag = "SYNTHETIC" if r["is_synthetic"] else "AUTHENTIC"
        print(f"{r['call_id']:<22} | {r['duration_s']:<8.1f} | {r['proc_time_s']:<9.2f} | {r['rtf']:<7.3f} | {v_tag:<12} | {r['confidence']*100:<6.1f}% | {r['turns']:<6}")
    print("=" * 90)
    print(f"✓ All reports saved to {out_dir}/\n")


if __name__ == "__main__":
    run_test_suite()
