# 🎙️ Altur Voice Anti-Spoofing & Deepfake Detection Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![HackMTY](https://img.shields.io/badge/HackMTY-Altur%20Challenge-blueviolet)](https://hackmty.com/)

An end-to-end **AI Voice Anti-Spoofing & Deepfake Detection Platform** developed for the **Altur Banking Challenge at HackMTY**.

The platform ingests 2-party call recordings, extracts Channel 0 (Agent 0 - evaluation target) using turn metadata, and benchmarks deep learning models (`RawNet2` and `AASIST-L`) against ground-truth human vs. synthetic speech labels.

---

## 📋 Step-by-Step Quickstart & Setup Guide

### 1. Clone the Repository & Install Dependencies
```bash
git clone https://github.com/katyazano/HackMty-Altur.git
cd HackMty-Altur
pip install -r requirements.txt
```

### 2. Place Your Raw Data
Place your raw audio files and metadata inside the `Data/` folder:
```
Data/
├── manifest.csv
├── turns/          # Loose turn JSON files (*.json)
└── audio/          # Loose audio recordings (*.wav)
```

### 3. Run Automated 1-Command Dataset Setup
Organize the manifests, audio splits, turn metadata, and Channel 0/1 tracks in one command:
```bash
python src/setup_dataset.py
```

**What `setup_dataset.py` does automatically**:
* **Splits Manifest**: Generates `train_manifest.csv` (282 calls) and `test_manifest.csv` (71 calls).
* **Organizes Audios & Turns**: Sorts recordings and JSON files into `Data/audio/train/`, `Data/audio/test/`, `Data/turns/train/`, and `Data/turns/test/`.
* **Channel Separation**: Slices stereo calls into isolated Agent 0 audio tracks inside `Data/separated_agents/train/agent_0/` and `Data/separated_agents/test/agent_0/`.

---

## 🧠 Model Training & Benchmarking

### 1. Train Models
Train models on the 282 Channel 0 training audio files:
```bash
python src/trainer.py --model aasist --epochs 15
python src/trainer.py --model rawnet2 --epochs 15
```

### 2. Benchmark Trained Models on Test Set
Evaluate trained models against the 71 test set Channel 0 audio files:
```bash
python src/evaluator.py --model aasist --weights weights/aasist_best.pth
python src/evaluator.py --model rawnet2 --weights weights/rawnet2_best.pth
```

---

## 🚀 Running the Interactive Web Dashboard with Docker

### 1. Build and Launch Container
```bash
docker compose up --build -d
```

### 2. Open the Dashboard
Open your browser to:
```
http://localhost:8000
```
*(Displays real-time test predictions, ground-truth badges, accuracy metrics, and in-browser Channel 0 audio playback).*

---

## 📦 Optional: Tracking Heavy Audio Files & Weights with Git LFS

If you wish to store `.wav` audio files or `.pth` model checkpoints in Git:

```bash
# 1. Initialize Git LFS once
git lfs install

# 2. Track large binary formats
git lfs track "*.wav"
git lfs track "*.pth"

# 3. Pull LFS files on new clones
git lfs pull
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
