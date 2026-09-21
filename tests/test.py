import os
import base64
import pandas as pd
import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

URL = "http://127.0.0.1:8000/detect"
manifest_path = os.path.join(REPO_ROOT, "Data", "manifest.csv")
df = pd.read_csv(manifest_path)

# Toma 5 humanos y 5 sintéticos del val
val = df[df["split"] == "val"]
muestra = pd.concat([
    val[val["label"] == "human"].head(5),
    val[val["label"] == "synthetic"].head(5)
])

ok, total = 0, 0
print(f"{'archivo':<20} {'real':<12} {'predicho':<12} {'confidence':<12} {'resultado'}")
print("-" * 70)

audio_dirs = [
    os.path.join(REPO_ROOT, "Data", "benchmark_audios"),
    os.path.join(REPO_ROOT, "audio"),
    "/home/kontgo/Downloads/altur-challenge-audio/audio"
]

for _, row in muestra.iterrows():
    path = None
    for ad in audio_dirs:
        candidate = os.path.join(ad, row["anon_id"] + ".wav")
        if os.path.exists(candidate):
            path = candidate
            break
    if not path:
        continue

    b64 = base64.b64encode(open(path, "rb").read()).decode()
    resp = requests.post(URL, json={"audio_base64": b64}).json()

    pred = "synthetic" if resp["is_synthetic"] else "human"
    conf = resp["confidence"]
    acierto = pred == row["label"]
    ok += acierto
    total += 1

    emoji = "✅" if acierto else "❌"
    print(f"{row['anon_id'][:18]:<20} {row['label']:<12} {pred:<12} {conf:<12} {emoji}")

print("-" * 70)
print(f"Resultado: {ok}/{total} correctos")