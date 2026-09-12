import os
import sys
import csv
import argparse
from pathlib import Path

# Project root directory detection
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def split_manifest(
    manifest_path: str = None,
    output_dir: str = None
):
    """
    Reads Data/manifest.csv and separates it into train_manifest.csv and test_manifest.csv
    with standard columns: id, label, split, duration.
    """
    if manifest_path is None:
        manifest_path = str(PROJECT_ROOT / "Data" / "manifest.csv")
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "Data")

    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file not found at {manifest_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    train_out = os.path.join(output_dir, "train_manifest.csv")
    test_out = os.path.join(output_dir, "test_manifest.csv")

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    train_rows = [r for r in reader if r.get("split") == "train"]
    test_rows = [r for r in reader if r.get("split") in ("val", "test")]

    headers = ["id", "label", "split", "duration"]

    with open(train_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in train_rows:
            writer.writerow([r.get("anon_id"), r.get("label"), r.get("split"), r.get("duration_s")])

    with open(test_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in test_rows:
            writer.writerow([r.get("anon_id"), r.get("label"), r.get("split"), r.get("duration_s")])

    print("=" * 70)
    print("             MANIFEST DATASET SPLIT COMPLETE")
    print("=" * 70)
    print(f"  • Source Manifest  : {manifest_path} ({len(reader)} records)")
    print(f"  • Training Manifest: {train_out} ({len(train_rows)} records)")
    print(f"  • Testing Manifest : {test_out} ({len(test_rows)} records)")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split manifest.csv into train and test tables.")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest.csv")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save split CSVs")
    args = parser.parse_args()
    split_manifest(args.manifest, args.output_dir)
