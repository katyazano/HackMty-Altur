"""
neural.py — Modelo neuronal complementario para el reto Altur.

Segunda opinion frente al HistGradientBoosting de features.py. Mientras el GBM
mira estadisticos GLOBALES de la llamada entera, esta CNN mira micro-artefactos
LOCALES dentro de ventanas de 4 s y vota por segmento. Errores decorrelacionados
-> el ensamble aporta de verdad.

Tres decisiones que sostienen el diseno:

  1. CMVN por segmento (media/varianza por banda mel). Destruye el nivel de piso
     de ruido y la coloracion del canal, que es justo donde vive el atajo
     'zcr_std' (AUC 0.983 el solo) del dataset. Obliga a la red a aprender la
     FORMA del espectro, no su nivel.

  2. Entrenamiento por segmento. 282 llamadas de train se vuelven ~11k muestras.
     Nuestro limite real es la escasez de datos, no la capacidad del modelo.

  3. CNN chica desde cero (~250k params) en vez de Wav2Vec2. En CPU sin CUDA,
     Wav2Vec2 sobre 170 s de audio cuesta 10-30 s por llamada y no hay tiempo
     de GPU para fine-tunearlo con 282 ejemplos. Esta red corre en decenas de ms.
"""
import io
import os

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

TARGET_SR = 8000
N_MELS = 64
N_FFT = 512
HOP = 160                      # 20 ms por frame
SEG_FRAMES = 200               # 4 s por segmento
SEG_HOP_FRAMES = 100           # 2 s de salto -> 50% de solape
MAX_SEGMENTS = 24              # cota dura de latencia en inferencia
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(REPO_ROOT, "weights", "neural_cnn.pt")


# ---------------------------------------------------------------------------
# Frontend: waveform -> log-mel
# ---------------------------------------------------------------------------
def logmel(y, sr=TARGET_SR):
    """Log-mel de la senal completa. Un solo STFT."""
    m = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0)
    return np.log(m + 1e-10).astype(np.float32)


def voiced_frames(y, sr=TARGET_SR):
    """Indices de frames con voz, para no entrenar/inferir sobre silencio."""
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    if rms.max() <= 0:
        return np.array([], dtype=int)
    thr = max(np.median(rms) * 1.2, rms.max() * 0.05)
    return np.flatnonzero(rms > thr)


def segment_starts(mel, voiced_idx, max_segments=None, rng=None):
    """Inicios de ventana que caen sobre region con voz."""
    n = mel.shape[1]
    if n < SEG_FRAMES:
        return []
    cands = [s for s in range(0, n - SEG_FRAMES + 1, SEG_HOP_FRAMES)]
    if len(voiced_idx):
        voiced_set = np.zeros(n, dtype=bool)
        voiced_set[voiced_idx] = True
        # exige que al menos 40% del segmento tenga voz
        keep = [s for s in cands if voiced_set[s:s + SEG_FRAMES].mean() >= 0.4]
        if keep:
            cands = keep
    if max_segments and len(cands) > max_segments:
        if rng is not None:
            cands = sorted(rng.choice(cands, max_segments, replace=False))
        else:
            # muestreo uniforme a lo largo de la llamada
            idx = np.linspace(0, len(cands) - 1, max_segments).round().astype(int)
            cands = [cands[i] for i in idx]
    return cands


def cmvn(seg):
    """Normaliza cada banda mel en el tiempo. Mata el nivel de canal."""
    mu = seg.mean(axis=1, keepdims=True)
    sd = seg.std(axis=1, keepdims=True) + 1e-5
    return (seg - mu) / sd


def segments_from_mel(mel, starts):
    if not starts:
        return np.zeros((0, 1, N_MELS, SEG_FRAMES), dtype=np.float32)
    out = np.stack([cmvn(mel[:, s:s + SEG_FRAMES]) for s in starts])
    return out[:, None, :, :].astype(np.float32)


# ---------------------------------------------------------------------------
# Arquitectura
# ---------------------------------------------------------------------------
class SpoofCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(1, 16), block(16, 32), block(32, 64), block(64, 64))
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(128, 64),
            nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        h = self.features(x)                     # (B, 64, F', T')
        h = h.mean(axis=2)                       # colapsa frecuencia -> (B, 64, T')
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)   # (B, 128)
        return self.head(pooled).squeeze(1)      # logits


# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------
_MODEL = None


def load_model(path=WEIGHTS_PATH):
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(path):
            return None
        net = SpoofCNN()
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        _MODEL = net
    return _MODEL


def _caller_channel(audio_bytes):
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    caller = data.T[0]
    if sr != TARGET_SR:
        caller = librosa.resample(caller, orig_sr=sr, target_sr=TARGET_SR)
    return caller


@torch.no_grad()
def score_segments(caller, net):
    """Devuelve las probabilidades por segmento (array, puede ir vacio)."""
    mel = logmel(caller)
    starts = segment_starts(mel, voiced_frames(caller), max_segments=MAX_SEGMENTS)
    segs = segments_from_mel(mel, starts)
    if len(segs) == 0:
        return np.array([]), []
    logits = net(torch.from_numpy(segs))
    return torch.sigmoid(logits).numpy(), [s * HOP / TARGET_SR for s in starts]


def aggregate(probs):
    """Un humano real no tiene segmentos sinteticos; un clon los tiene casi
    todos. Promediar en espacio logit es mas estable que promediar probas."""
    if len(probs) == 0:
        return 0.5
    p = np.clip(probs, 1e-6, 1 - 1e-6)
    return float(1.0 / (1.0 + np.exp(-np.mean(np.log(p / (1 - p))))))


def get_neural_probability(audio_bytes):
    """Probabilidad de que el llamante sea sintetico. 0.5 si no hay pesos."""
    net = load_model()
    if net is None:
        return 0.5
    probs, _ = score_segments(_caller_channel(audio_bytes), net)
    return aggregate(probs)


def get_neural_timeline(audio_bytes):
    """Como get_neural_probability pero con el detalle por segmento,
    para que el dashboard muestre DONDE la voz se ve sintetica."""
    net = load_model()
    if net is None:
        return 0.5, []
    probs, times = score_segments(_caller_channel(audio_bytes), net)
    return aggregate(probs), [
        {"t": round(t, 1), "p": round(float(p), 4)} for t, p in zip(times, probs)]
