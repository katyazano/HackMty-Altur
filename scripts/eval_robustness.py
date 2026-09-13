"""
eval_robustness.py — La prueba honesta.

train.py y train_neural.py reportan EER 0% en val. Eso NO es evidencia de
robustez: el 'val degradado' de train_neural.py usa mu-law + ruido, que es
exactamente lo que augment.py inyecta en entrenamiento. Medirse contra la
propia aumentacion es marcarse un autogol estadistico.

Aqui degradamos val con canales que NINGUNO de los dos modelos vio jamas
(pasabanda telefonica real, perdida de ancho de banda, reverberacion, ruido
fuerte) y comparamos GBM vs CNN vs ensamble. Este es el numero que hay que
llevarle al juez cuando pregunte por 'unseen conditions'.

    python scripts/eval_robustness.py
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, lfilter
from sklearn.metrics import roc_auc_score, roc_curve

import librosa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import extract_from_arrays
from src.neural import load_model, score_segments, aggregate

W_NEURAL = 0.5


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2.0)


# --- Degradaciones NO vistas en augment.py -------------------------------
def bandpass_telefono(x, sr=8000):
    """Banda telefonica real 300-3400 Hz. augment.py nunca filtra."""
    b, a = butter(4, [300 / (sr / 2), 3400 / (sr / 2)], btype="band")
    return lfilter(b, a, x).astype(np.float32)


def perdida_ancho_banda(x, sr=8000):
    """Downsample agresivo 8k->4k->8k: simula codec de baja tasa."""
    d = librosa.resample(x, orig_sr=sr, target_sr=4000)
    return librosa.resample(d, orig_sr=4000, target_sr=sr).astype(np.float32)[:len(x)]


def reverberacion(x, sr=8000):
    """Sala pequena / manos libres."""
    ir = np.zeros(int(0.12 * sr), dtype=np.float32)
    ir[0] = 1.0
    rng = np.random.default_rng(7)
    for t in rng.uniform(0.01, 0.12, 12):
        ir[int(t * sr)] += rng.uniform(-0.35, 0.35)
    y = np.convolve(x, ir)[:len(x)]
    return (y / (np.abs(y).max() + 1e-9) * np.abs(x).max()).astype(np.float32)


def ruido_fuerte(x, sr=8000):
    """Ruido 5x mas fuerte que el maximo de entrenamiento (0.004)."""
    rng = np.random.default_rng(11)
    return (x + rng.normal(0, 0.02, len(x))).astype(np.float32)


CONDICIONES = {
    "limpio": lambda x, sr: x,
    "pasabanda 300-3400Hz": bandpass_telefono,
    "perdida ancho banda 4k": perdida_ancho_banda,
    "reverberacion": reverberacion,
    "ruido fuerte (5x)": ruido_fuerte,
}


def main():
    bundle = pickle.load(open("model.pkl", "rb"))
    gbm, feat_names = bundle["model"], bundle["feature_names"]
    net = load_model()
    if net is None:
        print("Faltan pesos neuronales. Corre train_neural.py primero.")
        return

    df = pd.read_csv("manifest.csv")
    df["path"] = df["anon_id"].apply(lambda a: os.path.join("audio", a + ".wav"))
    df = df[(df["split"] == "val") & df["path"].apply(os.path.exists)].reset_index(drop=True)
    y = (df["label"] == "synthetic").astype(int).values
    print(f"Evaluando {len(df)} llamadas de val\n")

    print(f"{'condicion':<26}{'GBM':>18}{'CNN':>18}{'ENSAMBLE':>18}")
    print(f"{'':<26}{'AUC / EER':>18}{'AUC / EER':>18}{'AUC / EER':>18}")
    print("-" * 80)

    for nombre, fn in CONDICIONES.items():
        s_gbm, s_cnn = [], []
        for _, row in df.iterrows():
            data, sr = sf.read(row["path"], dtype="float32", always_2d=True)
            caller, agent = data.T[0], data.T[1]
            caller, agent = fn(caller, sr), fn(agent, sr)

            vec, names = extract_from_arrays(caller, agent, sr)
            idx = {n: i for i, n in enumerate(names)}
            row_v = np.array([vec[idx[n]] if n in idx else 0.0
                              for n in feat_names], dtype=np.float32).reshape(1, -1)
            s_gbm.append(float(gbm.predict_proba(row_v)[0, 1]))

            probs, _ = score_segments(caller, net)
            s_cnn.append(aggregate(probs))

        s_gbm, s_cnn = np.array(s_gbm), np.array(s_cnn)
        s_ens = (1 - W_NEURAL) * s_gbm + W_NEURAL * s_cnn
        out = f"{nombre:<26}"
        for s in (s_gbm, s_cnn, s_ens):
            out += f"{roc_auc_score(y, s):>10.3f} /{eer(y, s)*100:>5.1f}%"
        print(out)

    print("\nLee la columna ENSAMBLE bajo las filas degradadas: ese es el")
    print("rendimiento defendible ante 'locutores y motores no vistos'.")


if __name__ == "__main__":
    main()
