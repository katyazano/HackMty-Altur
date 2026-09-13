"""
train_neural.py — Entrena la CNN por segmentos de neural.py.

    python scripts/train_neural.py --data_dir .

Reporta AUC/EER a nivel LLAMADA sobre val limpio y sobre val degradado
(codec G.711 + ruido). El segundo numero es el que importa: es el proxy de
"engine/canal no visto" del criterio de robustez.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.augment import augment_stereo, mulaw_roundtrip
from src.neural import (N_MELS, SEG_FRAMES, SpoofCNN, aggregate, logmel,
                        segment_starts, segments_from_mel, voiced_frames)

SEGS_PER_CALL = 16
N_AUG = 2
CACHE = "neural_cache.npz"


def compute_eer(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2.0)


def load_caller(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data.T[0], sr


def call_segments(caller, rng=None, max_segs=SEGS_PER_CALL):
    mel = logmel(caller)
    starts = segment_starts(mel, voiced_frames(caller), max_segments=max_segs, rng=rng)
    return segments_from_mel(mel, starts)


def build(data_dir, augment=True):
    if os.path.exists(CACHE):
        print(f"   (usando cache {CACHE})")
        d = np.load(CACHE, allow_pickle=True)
        return d["X"], d["y"], d["call"], d["split"]

    df = pd.read_csv(os.path.join(data_dir, "manifest.csv"))
    df["path"] = df["anon_id"].apply(lambda a: os.path.join(data_dir, "audio", a + ".wav"))
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
    print(f"   {len(df)} llamadas -> segmentando...")

    X, y, call, split = [], [], [], []
    for i, row in df.iterrows():
        caller, _ = load_caller(row["path"])
        label = 1 if row["label"] == "synthetic" else 0
        rng = np.random.default_rng(i)

        versions = [caller]
        # solo aumentamos train: val debe quedar limpio para medir honesto
        if augment and row["split"] == "train":
            for _ in range(N_AUG):
                c, _a = augment_stereo(caller, caller, rng)
                versions.append(c)

        for v in versions:
            segs = call_segments(v, rng)
            if len(segs) == 0:
                continue
            X.append(segs.astype(np.float16))
            y.extend([label] * len(segs))
            call.extend([row["anon_id"]] * len(segs))
            split.extend([row["split"]] * len(segs))
        if (i + 1) % 50 == 0:
            print(f"     {i+1}/{len(df)}")

    X = np.concatenate(X)
    y = np.array(y, dtype=np.int64)
    call = np.array(call)
    split = np.array(split)
    np.savez(CACHE, X=X, y=y, call=call, split=split)
    return X, y, call, split


def spec_augment(batch, rng):
    """Enmascara bandas y tiempos al azar. Tras CMVN, 0 = la media."""
    b = batch.clone()
    n = b.shape[0]
    for i in range(n):
        if rng.random() < 0.5:
            f = rng.integers(1, 9)
            f0 = rng.integers(0, N_MELS - f)
            b[i, :, f0:f0 + f, :] = 0.0
        if rng.random() < 0.5:
            t = rng.integers(1, 25)
            t0 = rng.integers(0, SEG_FRAMES - t)
            b[i, :, :, t0:t0 + t] = 0.0
    return b


@torch.no_grad()
def eval_calls(net, X, y, call, idx):
    """AUC/EER a nivel llamada agregando los segmentos de cada una."""
    net.eval()
    segs = torch.from_numpy(X[idx].astype(np.float32))
    probs = torch.sigmoid(net(segs)).numpy()
    calls, labels, scores = call[idx], y[idx], []
    uniq = list(dict.fromkeys(calls))
    for c in uniq:
        scores.append(aggregate(probs[calls == c]))
    ylab = np.array([labels[calls == c][0] for c in uniq])
    scores = np.array(scores)
    return roc_auc_score(ylab, scores), compute_eer(ylab, scores), len(uniq)


def degrade_val(data_dir):
    """Val pasado por codec G.711 + ruido: proxy de canal no visto."""
    df = pd.read_csv(os.path.join(data_dir, "manifest.csv"))
    df["path"] = df["anon_id"].apply(lambda a: os.path.join(data_dir, "audio", a + ".wav"))
    df = df[(df["split"] == "val") & df["path"].apply(os.path.exists)].reset_index(drop=True)
    X, y, call = [], [], []
    rng = np.random.default_rng(1234)
    for _, row in df.iterrows():
        caller, _ = load_caller(row["path"])
        c = mulaw_roundtrip(caller) + rng.normal(0, 0.002, len(caller)).astype(np.float32)
        segs = call_segments(c.astype(np.float32))
        if len(segs) == 0:
            continue
        X.append(segs)
        lab = 1 if row["label"] == "synthetic" else 0
        y.extend([lab] * len(segs))
        call.extend([row["anon_id"]] * len(segs))
    return np.concatenate(X), np.array(y), np.array(call)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", default="weights/neural_cnn.pt")
    args = ap.parse_args()

    torch.manual_seed(42)
    torch.set_num_threads(os.cpu_count() or 4)

    print("1. Construyendo segmentos...")
    X, y, call, split = build(args.data_dir)
    tr = np.flatnonzero(split == "train")
    va = np.flatnonzero(split == "val")
    print(f"   {len(X)} segmentos | train={len(tr)} val={len(va)}")

    net = SpoofCNN()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"   CNN con {n_params/1000:.0f}k parametros")

    pos = float(y[tr].sum())
    pos_weight = torch.tensor([(len(tr) - pos) / max(pos, 1)])
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    rng = np.random.default_rng(0)
    Xtr = X[tr]                       # se queda en float16; casteamos por lote
    ytr = torch.from_numpy(y[tr].astype(np.float32))

    print("2. Entrenando...")
    best_auc, best_state = -1.0, None
    for ep in range(args.epochs):
        net.train()
        perm = np.random.permutation(len(Xtr))
        total = 0.0
        for i in range(0, len(perm), args.batch):
            sel = perm[i:i + args.batch]
            xb = spec_augment(torch.from_numpy(Xtr[sel].astype(np.float32)), rng)
            opt.zero_grad()
            loss = crit(net(xb), ytr[sel])
            loss.backward()
            opt.step()
            total += loss.item() * len(sel)
        sched.step()
        auc, eer, ncalls = eval_calls(net, X, y, call, va)
        flag = ""
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            flag = "  <- mejor"
        print(f"   ep{ep+1:02d} loss={total/len(perm):.4f}  "
              f"VAL({ncalls} llamadas) AUC={auc:.4f} EER={eer*100:.1f}%{flag}")

    net.load_state_dict(best_state)

    print("3. Robustez: val degradado (G.711 + ruido)...")
    Xd, yd, cd = degrade_val(args.data_dir)
    auc_d, eer_d, n_d = eval_calls(net, Xd, yd, cd, np.arange(len(Xd)))
    print(f"   >> VAL degradado ({n_d} llamadas) AUC={auc_d:.4f} EER={eer_d*100:.1f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(net.state_dict(), args.out)
    with open("neural_metrics.json", "w") as f:
        json.dump({"val_auc": best_auc, "val_auc_degraded": auc_d,
                   "val_eer_degraded": eer_d, "params": n_params}, f, indent=2)
    print(f"4. Guardado {args.out}")


if __name__ == "__main__":
    main()
