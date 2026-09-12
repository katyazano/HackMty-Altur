import os
import sys
import argparse
from pathlib import Path

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.split_manifest import split_manifest
from src.split_audio_files import separate_audio_splits
from src.split_turns_files import separate_turns_splits
from src.separate_agents import main as separate_agents_main

def run_full_dataset_setup():
    print("=" * 80)
    print("           ALTUR VOICE ANTI-SPOOFING DATASET AUTOMATED SETUP")
    print("=" * 80 + "\n", flush=True)

    data_dir = PROJECT_ROOT / "Data"
    manifest_csv = data_dir / "manifest.csv"
    train_manifest = data_dir / "train_manifest.csv"
    test_manifest = data_dir / "test_manifest.csv"

    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        sys.exit(1)

    # Step 1: Split manifest into train_manifest.csv and test_manifest.csv
    print(">>> [Step 1/4] Splitting Data/manifest.csv into train and test tables...", flush=True)
    split_manifest(manifest_path=str(manifest_csv), output_dir=str(data_dir))

    # Step 2: Separate audio files into Data/audio/train/ and Data/audio/test/
    print("\n>>> [Step 2/4] Organizing audio files into Data/audio/train and Data/audio/test...", flush=True)
    separate_audio_splits(
        audio_dir=str(data_dir / "audio"),
        test_manifest=str(test_manifest),
        train_manifest=str(train_manifest),
        action="move"
    )

    # Step 3: Separate turns metadata into Data/turns/train/ and Data/turns/test/
    print("\n>>> [Step 3/4] Organizing turns JSON files into Data/turns/train and Data/turns/test...", flush=True)
    separate_turns_splits(
        turns_dir=str(data_dir / "turns"),
        test_manifest=str(test_manifest),
        train_manifest=str(train_manifest),
        action="move"
    )

    # Step 4: Extract Agent 0 and Agent 1 audio tracks for train and test sets
    print("\n>>> [Step 4/4] Slicing dual-channel calls into Agent 0 and Agent 1 audio tracks...", flush=True)
    
    # Train set channel separation
    print("\n--- Processing Train Set Audios ---", flush=True)
    sys.argv = [
        "separate_agents.py",
        "--audio_dir", str(data_dir / "audio" / "train"),
        "--turns_dir", str(data_dir / "turns" / "train"),
        "--output_dir", str(data_dir / "separated_agents" / "train"),
        "--workers", "8"
    ]
    try:
        separate_agents_main()
    except SystemExit:
        pass

    # Test set channel separation
    print("\n--- Processing Test Set Audios ---", flush=True)
    sys.argv = [
        "separate_agents.py",
        "--audio_dir", str(data_dir / "audio" / "test"),
        "--turns_dir", str(data_dir / "turns" / "test"),
        "--output_dir", str(data_dir / "separated_agents" / "test"),
        "--workers", "8"
    ]
    try:
        separate_agents_main()
    except SystemExit:
        pass

    print("\n" + "=" * 80)
    print("                    DATASET SETUP COMPLETE!")
    print("=" * 80)
    print(f"  * Audio Folders     : {data_dir / 'audio' / 'train'} & {data_dir / 'audio' / 'test'}")
    print(f"  * Turns Folders     : {data_dir / 'turns' / 'train'} & {data_dir / 'turns' / 'test'}")
    print(f"  * Separated Channel 0: {data_dir / 'separated_agents' / 'test' / 'agent_0'}")
    print("=" * 80 + "\n", flush=True)

if __name__ == "__main__":
    run_full_dataset_setup()
