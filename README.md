# 🎙️ Altur Voice Anti-Spoofing & Deepfake Detection Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![HackMTY](https://img.shields.io/badge/HackMTY-Altur%20Challenge-blueviolet)](https://hackmty.com/)

An end-to-end **AI Voice Anti-Spoofing & Deepfake Detection Platform** developed for the **Altur Banking Challenge at HackMTY**.

The platform ingests 2-party call recordings, extracts Channel 0 (Agent 0 - evaluation target) using turn metadata, and benchmarks deep learning models (`RawNet2` and `AASIST-L`) against ground-truth human vs. synthetic speech labels.

---

## ⚡ 1-Command Dataset Setup

If you have raw data in `Data/` (with loose `.wav` recordings, `manifest.csv`, and `turns/*.json`), organize and slice the full dataset in one command:

```bash
python src/setup_dataset.py
```

### What `setup_dataset.py` does automatically:
1. **Splits Manifest**: Generates `train_manifest.csv` (282 calls) and `test_manifest.csv` (71 calls).
2. **Organizes Audios & Turns**: Moves loose recordings and JSON files into `Data/audio/train/`, `Data/audio/test/`, `Data/turns/train/`, and `Data/turns/test/`.
3. **Channel Separation**: Slices stereo calls into isolated Agent 0 audio tracks inside `Data/separated_agents/train/agent_0/` and `Data/separated_agents/test/agent_0/`.

---

## 📦 Large File Management (Git LFS Instructions)

To track `.wav` audio files and `.pth` model checkpoints in Git without bloating repository size:

### 1. Install & Initialize Git LFS
```bash
# Run once on your system
git lfs install
```

### 2. Track Audio Files & Weights
```bash
git lfs track "*.wav"
git lfs track "*.pth"
git add .gitattributes
git commit -m "Track audio and model checkpoints with Git LFS"
```

### 3. Cloning or Pulling with Git LFS
When team members clone or pull the repository:
```bash
git clone https://github.com/katyazano/HackMty-Altur.git
git lfs pull
```
*(Git LFS will automatically place all audio files into their exact target folders).*

---

## 🚀 Running with Docker

### 1. Build and Run
```bash
docker compose up --build -d
```

### 2. Open the Interactive Benchmark Dashboard
Open your browser to:
```
http://localhost:8000
```

---

## 🧠 Training & Benchmarking Models

### Train Any Model in 1 Line of Code ([`src/trainer.py`](file:///c:/Users/kathe/Desktop/HackMTY/src/trainer.py))
```python
from src.trainer import train_model
from src.models.aasist import AASIST

# Train AASIST model with in-memory dataset caching
model = AASIST()
results = train_model(model=model, epochs=15, batch_size=16)
```

**Terminal CLI Training**:
```bash
python src/trainer.py --model aasist --epochs 15
python src/trainer.py --model rawnet2 --epochs 15
```

### Benchmark Any Model on Test Dataset ([`src/evaluator.py`](file:///c:/Users/kathe/Desktop/HackMTY/src/evaluator.py))
```bash
python src/evaluator.py --model aasist --weights weights/aasist_best.pth
```

---

## 📁 Repository Structure

```
HackMTY/
├── app.py                     # FastAPI server & benchmark API
├── Dockerfile                 # Multi-stage Python 3.12 slim container
├── docker-compose.yml         # Container orchestration with Data volume mount
├── requirements.txt           # Python dependencies
├── src/
│   ├── setup_dataset.py       # Master 1-command dataset setup script
│   ├── separate_agents.py     # Agent turn extraction (Channel 0 / Channel 1)
│   ├── split_audio_files.py   # Train / Test audio set organizer
│   ├── split_turns_files.py   # Train / Test turns metadata organizer
│   ├── split_manifest.py      # Train / Test manifest splitter
│   ├── evaluate_channel0.py   # Test dataset benchmark pipeline
│   ├── trainer.py             # Universal PyTorch model trainer with RAM caching
│   ├── evaluator.py           # Universal PyTorch model benchmark evaluator
│   ├── audio_processor.py     # ASP pipeline (Denoising, Bandpass, VAD)
│   └── models/
│       ├── sincnet.py         # Learnable SincNet raw filterbank frontend
│       ├── rawnet2.py         # RawNet2 raw waveform deepfake detector
│       └── aasist.py          # AASIST-L Graph Attention Network
└── web/
    └── index.html             # Interactive Channel 0 benchmark web interface
```
