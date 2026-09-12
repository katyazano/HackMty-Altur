import os
import sys
import glob
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import numpy as np
import soundfile as sf
import librosa

# Resolve project root directory
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def separate_call_agents(
    call_id: str,
    audio_path: str,
    turns_path: Optional[str],
    output_dir: str,
    mode: str = "concat",
    target_sr: Optional[int] = None,
    padding_ms: int = 50
) -> dict:
    """
    Separates a two-party call audio into two distinct audio files for Agent 0 and Agent 1.
    """
    # 1. Load Audio
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim == 1:
        # Mono fallback: duplicate track
        data = np.stack([data, data], axis=-1)
    
    total_samples = len(data)
    total_duration = total_samples / sr

    # 2. Load Turns Metadata
    turns = []
    if turns_path and os.path.exists(turns_path):
        try:
            with open(turns_path, "r", encoding="utf-8") as f:
                turns_meta = json.load(f)
                turns = turns_meta.get("turns", [])
        except Exception as e:
            print(f"Warning: Failed to load turns from {turns_path}: {e}")

    # Output paths
    agent_0_dir = os.path.join(output_dir, "agent_0")
    agent_1_dir = os.path.join(output_dir, "agent_1")
    os.makedirs(agent_0_dir, exist_ok=True)
    os.makedirs(agent_1_dir, exist_ok=True)

    out_path_0 = os.path.join(agent_0_dir, f"{call_id}_agent_0.wav")
    out_path_1 = os.path.join(agent_1_dir, f"{call_id}_agent_1.wav")

    # If no turns available, fallback to raw stereo channel split
    if not turns:
        audio_0 = data[:, 0]
        audio_1 = data[:, 1]
    elif mode == "concat":
        # Extract and concatenate all speech turns for each agent
        padding_samples = int(sr * (padding_ms / 1000.0))
        silence_pad = np.zeros(padding_samples, dtype=np.float32)

        segments_0 = []
        segments_1 = []

        for turn in turns:
            ch = turn.get("channel", 0)
            start_s = max(0.0, float(turn.get("start", 0.0)))
            end_s = min(total_duration, float(turn.get("end", total_duration)))
            
            start_idx = int(start_s * sr)
            end_idx = int(end_s * sr)

            if end_idx <= start_idx:
                continue

            if ch == 0:
                seg = data[start_idx:end_idx, 0]
                if len(segments_0) > 0 and padding_samples > 0:
                    segments_0.append(silence_pad)
                segments_0.append(seg)
            elif ch == 1:
                seg = data[start_idx:end_idx, 1]
                if len(segments_1) > 0 and padding_samples > 0:
                    segments_1.append(silence_pad)
                segments_1.append(seg)

        audio_0 = np.concatenate(segments_0) if segments_0 else np.zeros(int(sr * 0.1), dtype=np.float32)
        audio_1 = np.concatenate(segments_1) if segments_1 else np.zeros(int(sr * 0.1), dtype=np.float32)

    elif mode == "timeline":
        # Keep full timeline length, zeroing out inactive turns
        audio_0 = np.zeros(total_samples, dtype=np.float32)
        audio_1 = np.zeros(total_samples, dtype=np.float32)

        for turn in turns:
            ch = turn.get("channel", 0)
            start_idx = max(0, int(float(turn.get("start", 0.0)) * sr))
            end_idx = min(total_samples, int(float(turn.get("end", total_duration)) * sr))

            if end_idx <= start_idx:
                continue

            if ch == 0:
                audio_0[start_idx:end_idx] = data[start_idx:end_idx, 0]
            elif ch == 1:
                audio_1[start_idx:end_idx] = data[start_idx:end_idx, 1]

    elif mode == "full_channel":
        audio_0 = data[:, 0]
        audio_1 = data[:, 1]
    else:
        raise ValueError(f"Unknown separation mode: {mode}")

    # Optional Resampling (e.g. 8kHz -> 16kHz for neural anti-spoofing models)
    out_sr = sr
    if target_sr and target_sr != sr:
        if len(audio_0) > 0:
            audio_0 = librosa.resample(audio_0, orig_sr=sr, target_sr=target_sr)
        if len(audio_1) > 0:
            audio_1 = librosa.resample(audio_1, orig_sr=sr, target_sr=target_sr)
        out_sr = target_sr

    # Write output wav files
    sf.write(out_path_0, audio_0, out_sr)
    sf.write(out_path_1, audio_1, out_sr)

    dur_0 = len(audio_0) / out_sr
    dur_1 = len(audio_1) / out_sr
    turns_0 = sum(1 for t in turns if t.get("channel") == 0)
    turns_1 = sum(1 for t in turns if t.get("channel") == 1)

    return {
        "call_id": call_id,
        "original_duration_s": round(total_duration, 2),
        "agent_0_path": out_path_0,
        "agent_0_duration_s": round(dur_0, 2),
        "agent_0_turns": turns_0,
        "agent_1_path": out_path_1,
        "agent_1_duration_s": round(dur_1, 2),
        "agent_1_turns": turns_1,
        "sample_rate": out_sr,
        "mode": mode
    }

def main():
    default_audio = os.path.join(PROJECT_ROOT, "Data", "audio")
    default_turns = os.path.join(PROJECT_ROOT, "Data", "turns")
    default_output = os.path.join(PROJECT_ROOT, "Data", "separated_agents")

    parser = argparse.ArgumentParser(
        description="Separate 2-party call recordings into Agent 0 and Agent 1 audio tracks using Data/turns metadata."
    )
    parser.add_argument("--audio_dir", type=str, default=default_audio, help="Directory containing call audio files (.wav)")
    parser.add_argument("--turns_dir", type=str, default=default_turns, help="Directory containing turn metadata (.json)")
    parser.add_argument("--output_dir", type=str, default=default_output, help="Directory to save separated audio tracks")
    parser.add_argument("--mode", type=str, default="concat", choices=["concat", "timeline", "full_channel"],
                        help="Separation mode: 'concat' (speech segments merged), 'timeline' (timeline preserved with silence), 'full_channel' (raw channels)")
    parser.add_argument("--target_sr", type=int, default=16000, help="Target sample rate (e.g. 16000 for RawNet2/AASIST, 8000 for native)")
    parser.add_argument("--padding_ms", type=int, default=50, help="Silence padding in ms between concatenated turns in concat mode")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker threads")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of calls to process (for quick testing)")

    args = parser.parse_args()

    audio_dir = os.path.abspath(args.audio_dir)
    turns_dir = os.path.abspath(args.turns_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory not found at {audio_dir}")
        sys.exit(1)

    # Discover audio files (supports flat or nested train/test folders)
    audio_files = sorted(list(set(glob.glob(os.path.join(audio_dir, "*.wav")) + glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True))))
    if not audio_files:
        print(f"Error: No .wav files found in {audio_dir}")
        sys.exit(1)

    if args.limit:
        audio_files = audio_files[:args.limit]

    print("=" * 80)
    print("      ALTUR CALL AUDIO SEPARATION - AGENT 0 & AGENT 1 EXTRACTION")
    print("=" * 80)
    print(f"  * Audio Directory : {audio_dir}")
    print(f"  * Turns Directory : {turns_dir}")
    print(f"  * Output Directory: {output_dir}")
    print(f"  * Separation Mode : {args.mode}")
    print(f"  * Target Rate     : {args.target_sr} Hz")
    print(f"  * Files to Process: {len(audio_files)}")
    print("=" * 80 + "\n")

    os.makedirs(output_dir, exist_ok=True)

    # Pre-index turns metadata (supports flat or nested train/test folders)
    turns_map = {}
    if os.path.exists(turns_dir):
        all_turns = glob.glob(os.path.join(turns_dir, "*.json")) + glob.glob(os.path.join(turns_dir, "**", "*.json"), recursive=True)
        for tfile in all_turns:
            cid = os.path.splitext(os.path.basename(tfile))[0]
            turns_map[cid] = tfile

    tasks = []
    for audio_path in audio_files:
        call_id = os.path.splitext(os.path.basename(audio_path))[0]
        turns_path = turns_map.get(call_id, None)
        tasks.append((call_id, audio_path, turns_path))

    results = []
    processed_count = 0

    def process_task(task_args):
        cid, apath, tpath = task_args
        return separate_call_agents(
            call_id=cid,
            audio_path=apath,
            turns_path=tpath,
            output_dir=output_dir,
            mode=args.mode,
            target_sr=args.target_sr,
            padding_ms=args.padding_ms
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for res in executor.map(process_task, tasks):
            results.append(res)
            processed_count += 1
            if processed_count % 25 == 0 or processed_count == len(tasks):
                print(f"  [{processed_count:>3}/{len(tasks)}] Processed: {res['call_id']} -> Agent 0: {res['agent_0_duration_s']}s ({res['agent_0_turns']} turns), Agent 1: {res['agent_1_duration_s']}s ({res['agent_1_turns']} turns)")

    # Write Manifest Summary
    manifest_csv = os.path.join(output_dir, "manifest_separated.csv")
    with open(manifest_csv, "w", encoding="utf-8") as f:
        f.write("call_id,original_duration_s,agent_0_path,agent_0_duration_s,agent_0_turns,agent_1_path,agent_1_duration_s,agent_1_turns,sample_rate,mode\n")
        for r in results:
            f.write(f"{r['call_id']},{r['original_duration_s']},{r['agent_0_path']},{r['agent_0_duration_s']},{r['agent_0_turns']},{r['agent_1_path']},{r['agent_1_duration_s']},{r['agent_1_turns']},{r['sample_rate']},{r['mode']}\n")

    print("\n" + "=" * 80)
    print("                              SEPARATION COMPLETE")
    print("=" * 80)
    print(f"  * Processed Calls : {len(results)}")
    print(f"  * Agent 0 Audios  : {os.path.join(output_dir, 'agent_0')}")
    print(f"  * Agent 1 Audios  : {os.path.join(output_dir, 'agent_1')}")
    print(f"  * Summary Manifest: {manifest_csv}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
