import os
import sys
import csv
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class AntiSpoofingAudioDataset(Dataset):
    """
    Universal PyTorch Dataset for voice anti-spoofing audio files.
    Reads Channel 0 audio files and pairs them with manifest ground truth labels.
    """
    def __init__(
        self,
        audio_dir: str,
        manifest_path: str,
        target_sr: int = 16000,
        sample_length: int = 32000,  # 2.0 seconds @ 16kHz for fast CPU training
        split_filter: Optional[str] = None
    ):
        self.audio_dir = Path(audio_dir)
        self.target_sr = target_sr
        self.sample_length = sample_length
        self.samples = []

        # 1. Load Manifest
        label_map = {}
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = (row.get("anon_id") or row.get("id") or "").strip()
                    label = (row.get("label") or "").strip().lower()
                    split = (row.get("split") or "").strip().lower()
                    if split_filter and split != split_filter.lower():
                        continue
                    # 0 = human (bonafide), 1 = synthetic (spoof)
                    num_label = 0 if label == "human" else 1
                    label_map[cid] = num_label

        # 2. Discover and pre-load audio files into RAM (total size < 75 MB)
        wav_files = sorted(list(self.audio_dir.glob("*.wav")))
        for wfile in wav_files:
            cid = wfile.stem.replace("_agent_0", "")
            target_label = label_map.get(cid, 1)

            try:
                data, sr = sf.read(str(wfile), dtype="float32")
                if data.ndim > 1:
                    data = data[:, 0]

                nb_samples = len(data)
                if nb_samples > self.sample_length:
                    mid = nb_samples // 2
                    half = self.sample_length // 2
                    data = data[mid - half : mid + half]
                elif nb_samples < self.sample_length:
                    pad = self.sample_length - nb_samples
                    data = np.pad(data, (0, pad), mode="constant")

                waveform = torch.tensor(data, dtype=torch.float32)
                self.samples.append((waveform, target_label, cid))
            except Exception as e:
                print(f"Warning: Failed to load {wfile.name}: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        waveform, label, cid = self.samples[idx]
        return waveform, label, cid


def train_model(
    model: nn.Module,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    save_path: Optional[str] = None,
    train_dir: Optional[str] = None,
    test_dir: Optional[str] = None,
    manifest_path: Optional[str] = None,
    device: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Universal training function for any anti-spoofing PyTorch model.
    
    Args:
        model: Any PyTorch nn.Module (e.g. RawNet2, AASIST, or any custom model)
        epochs: Number of training epochs
        batch_size: Batch size for training and validation
        lr: Learning rate for Adam optimizer
        weight_decay: L2 regularization factor
        save_path: Path to save the best model weights (.pth)
        train_dir: Directory containing training Channel 0 audios
        test_dir: Directory containing test Channel 0 audios
        manifest_path: Path to manifest.csv
        device: Device string ('cuda', 'cpu', or None for auto-detect)
        verbose: Whether to print training progress per epoch
        
    Returns:
        Dictionary containing best accuracy, best epoch, training history, and model.
    """
    # 1. Resolve paths
    if train_dir is None:
        train_dir = str(PROJECT_ROOT / "Data" / "separated_agents" / "train" / "agent_0")
    if test_dir is None:
        test_dir = str(PROJECT_ROOT / "Data" / "separated_agents" / "test" / "agent_0")
    if manifest_path is None:
        manifest_path = str(PROJECT_ROOT / "Data" / "manifest.csv")

    model_name = model.__class__.__name__
    if save_path is None:
        weights_dir = PROJECT_ROOT / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(weights_dir / f"{model_name.lower()}_best.pth")
    else:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # 2. Setup Device
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    model = model.to(target_device)

    if verbose:
        print("=" * 80)
        print(f"      UNIVERSAL ANTI-SPOOFING MODEL TRAINER: {model_name.upper()}")
        print("=" * 80)
        print(f"  * Device          : {target_device}")
        print(f"  * Epochs          : {epochs}")
        print(f"  * Batch Size      : {batch_size}")
        print(f"  * Learning Rate   : {lr}")
        print(f"  * Save Checkpoint : {save_path}")
        print(f"  * Train Directory : {train_dir}")
        print(f"  * Test Directory  : {test_dir}")
        print("=" * 80 + "\n")

    # 3. Create Datasets and DataLoaders
    train_dataset = AntiSpoofingAudioDataset(audio_dir=train_dir, manifest_path=manifest_path)
    test_dataset = AntiSpoofingAudioDataset(audio_dir=test_dir, manifest_path=manifest_path)

    if len(train_dataset) == 0:
        raise ValueError(f"No training audio files found in {train_dir}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    if verbose:
        print(f"Loaded {len(train_dataset)} training samples and {len(test_dataset)} test validation samples.\n")

    # 4. Setup Criterion, Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_val_acc = 0.0
    best_epoch = 0

    t_start = time.time()

    # 5. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for b_idx, (waveforms, labels, _) in enumerate(train_loader, 1):
            waveforms = waveforms.to(target_device)
            labels = labels.to(target_device)

            optimizer.zero_grad()
            logits = model(waveforms)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            train_loss += batch_loss * len(labels)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

            if verbose and (b_idx % 3 == 0 or b_idx == len(train_loader)):
                print(f"  [Epoch {epoch:02d}/{epochs:02d} | Batch {b_idx:2d}/{len(train_loader)}] Step Loss: {batch_loss:.4f} | Running Train Acc: {(train_correct/train_total)*100:.1f}%", flush=True)

        scheduler.step()

        epoch_train_loss = train_loss / train_total if train_total > 0 else 0.0
        epoch_train_acc = (train_correct / train_total) * 100 if train_total > 0 else 0.0

        # 6. Validation Loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for waveforms, labels, _ in test_loader:
                waveforms = waveforms.to(target_device)
                labels = labels.to(target_device)

                logits = model(waveforms)
                loss = criterion(logits, labels)

                val_loss += loss.item() * len(labels)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        epoch_val_loss = val_loss / val_total if val_total > 0 else 0.0
        epoch_val_acc = (val_correct / val_total) * 100 if val_total > 0 else 0.0

        is_best = epoch_val_acc > best_val_acc
        if is_best:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)

        epoch_stat = {
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc, 2),
            "val_loss": round(epoch_val_loss, 4),
            "val_acc": round(epoch_val_acc, 2),
            "is_best": is_best
        }
        history.append(epoch_stat)

        if verbose:
            best_tag = " -> [BEST SAVED]" if is_best else ""
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:5.1f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:5.1f}%{best_tag}", flush=True)

    total_time = round(time.time() - t_start, 2)

    if verbose:
        print("\n" + "=" * 80)
        print("                         TRAINING COMPLETE")
        print("=" * 80)
        print(f"  * Total Time     : {total_time}s")
        print(f"  * Best Epoch     : {best_epoch}")
        print(f"  * Best Val Acc   : {best_val_acc:.2f}%")
        print(f"  * Checkpoint Path: {save_path}")
        print("=" * 80 + "\n")

    return {
        "model_name": model_name,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "save_path": save_path,
        "history": history,
        "model": model
    }


if __name__ == "__main__":
    import argparse
    from src.models.rawnet2 import RawNet2
    from src.models.aasist import AASIST

    parser = argparse.ArgumentParser(description="Universal Voice Anti-Spoofing Model Trainer")
    parser.add_argument("--model", type=str, default="rawnet2", choices=["rawnet2", "aasist"], help="Model architecture to train")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save best weights")
    args = parser.parse_args()

    if args.model.lower() == "rawnet2":
        net = RawNet2()
    elif args.model.lower() == "aasist":
        net = AASIST()
    else:
        raise ValueError(f"Unknown model architecture: {args.model}")

    train_model(
        model=net,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path
    )
