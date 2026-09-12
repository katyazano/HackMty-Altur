import os
import sys
import csv
import shutil
import argparse
from pathlib import Path

# Project root directory detection
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def separate_audio_splits(
    audio_dir: str = None,
    test_manifest: str = None,
    train_manifest: str = None,
    action: str = "move"
):
    """
    Separates audios in Data/audio into train and test folders based on manifest IDs.
    """
    if audio_dir is None:
        audio_dir = str(PROJECT_ROOT / "Data" / "audio")
    if test_manifest is None:
        test_manifest = str(PROJECT_ROOT / "Data" / "test_manifest.csv")
    if train_manifest is None:
        train_manifest = str(PROJECT_ROOT / "Data" / "train_manifest.csv")

    audio_path = Path(audio_dir)
    test_manifest_path = Path(test_manifest)
    train_manifest_path = Path(train_manifest)

    if not audio_path.exists():
        print(f"Error: Audio directory does not exist: {audio_path}")
        sys.exit(1)

    if not test_manifest_path.exists():
        print(f"Error: Test manifest not found: {test_manifest_path}")
        sys.exit(1)

    # 1. Read Test Manifest IDs
    test_ids = set()
    with open(test_manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val_id = row.get("id") or row.get("anon_id")
            if val_id:
                test_ids.add(val_id.strip())

    print(f"Loaded {len(test_ids)} test audio IDs from {test_manifest_path.name}")

    # 2. Read Train Manifest IDs (if exists, for validation)
    train_ids = set()
    if train_manifest_path.exists():
        with open(train_manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val_id = row.get("id") or row.get("anon_id")
                if val_id:
                    train_ids.add(val_id.strip())
        print(f"Loaded {len(train_ids)} train audio IDs from {train_manifest_path.name}")

    # Target folders inside audio_dir
    train_dir = audio_path / "train"
    test_dir = audio_path / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # 3. Scan all loose WAV files in audio_dir
    loose_wavs = [f for f in audio_path.iterdir() if f.is_file() and f.suffix.lower() == ".wav"]

    if not loose_wavs:
        # Check if already organized
        existing_train = list(train_dir.glob("*.wav"))
        existing_test = list(test_dir.glob("*.wav"))
        if existing_train or existing_test:
            print(f"\nAudio files are already separated:")
            print(f"  • Train audio: {len(existing_train)} files in {train_dir}")
            print(f"  • Test audio : {len(existing_test)} files in {test_dir}")
            return
        else:
            print(f"Warning: No .wav files found directly in {audio_path}")
            return

    print(f"\nFound {len(loose_wavs)} WAV files to organize. Processing with action='{action}'...\n")

    moved_test = 0
    moved_train = 0
    unmatched = 0

    for wav_file in loose_wavs:
        file_stem = wav_file.stem  # e.g., 'call_0294f969f98b'

        if file_stem in test_ids:
            target_path = test_dir / wav_file.name
            if action == "move":
                shutil.move(str(wav_file), str(target_path))
            else:
                shutil.copy2(str(wav_file), str(target_path))
            moved_test += 1
        else:
            # Belongs to train set
            target_path = train_dir / wav_file.name
            if action == "move":
                shutil.move(str(wav_file), str(target_path))
            else:
                shutil.copy2(str(wav_file), str(target_path))
            moved_train += 1

    print("=" * 70)
    print("             AUDIO DATASET SPLIT COMPLETE")
    print("=" * 70)
    print(f"  • Action Performed: {action.upper()}")
    print(f"  • Test Audios     : {moved_test} files -> {test_dir}")
    print(f"  • Train Audios    : {moved_train} files -> {train_dir}")
    print(f"  • Total Processed : {moved_test + moved_train} files")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Separate audio files into train and test directories.")
    parser.add_argument("--audio_dir", type=str, default=None, help="Directory containing audio files")
    parser.add_argument("--test_manifest", type=str, default=None, help="Path to test_manifest.csv")
    parser.add_argument("--train_manifest", type=str, default=None, help="Path to train_manifest.csv")
    parser.add_argument("--action", type=str, default="move", choices=["move", "copy"], help="File operation: 'move' or 'copy'")
    args = parser.parse_args()

    separate_audio_splits(
        audio_dir=args.audio_dir,
        test_manifest=args.test_manifest,
        train_manifest=args.train_manifest,
        action=args.action
    )
