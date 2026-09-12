# 🎙️ Altur Voice Biometrics & Anti-Spoofing Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![HackMTY](https://img.shields.io/badge/HackMTY-Altur%20Challenge-blueviolet)](https://hackmty.com/)

An end-to-end **Voice Biometrics (Speaker Verification)** and **Deepfake / AI Voice Anti-Spoofing Detection** platform developed for the **Altur Banking Challenge at HackMTY**.

The system features real-time audio signal processing (ASP), multi-model anti-spoofing benchmarking (DSP Baseline vs. RawNet2 vs. AASIST-L), and an interactive web dashboard with live hyperparameter tuning and ground-truth validation.

---

## 🚀 Docker Quickstart

The entire project is containerized with Docker and Docker Compose. All dependencies (Python 3.12, PyTorch CPU, FFmpeg, libsndfile) and runtime directories are handled automatically inside the container.

### 1. Build and Run

```bash
docker compose up --build
```

### 2. Open the Dashboard

Open your browser and navigate to:
```
http://localhost:8000
```

### 3. Stop the Container

```bash
docker compose down
```

---

## 🔬 Anti-Spoofing Models Benchmark

| Model | Architecture | Parameters | CPU Latency | Key Strengths |
| :--- | :--- | :--- | :--- | :--- |
| **DSP Baseline** | Rule-based Signal Processing | 0 (Rule-based) | < 5 ms | Ultra-fast, catches basic spectral anomalies. |
| **RawNet2** | Raw Waveform (SincNet + ResNet-GRU) | ~10.3M | ~1,200 ms | Time-domain analysis without spectrogram quantization loss. |
| **AASIST-L** | Spectro-Temporal Graph Attention | **~47.6K** | **~80 ms** | **Lightweight & fast SOTA**, captures subtle synthetic vocoder artifacts. |

---

## 🖥️ Web Dashboard Features

- **Live ASP Controls:** Sliders for VAD Threshold (`top_db`), Spectral Noise Reduction (`prop_decrease`), Bandpass Filtering (`lowcut`/`highcut`), and Speaker Similarity Threshold.
- **Dual Audio Player:** Compare Original Raw Audio (🎧) vs. Cleaned Output (🔊) in real-time.
- **1-Click ASVspoof Benchmark:** Load verified real human and synthetic deepfake samples with ground-truth accuracy indicators (`✓ Correct` / `✗ Miss`).
- **Audio Management:** Upload custom recordings, delete individual tracks, or clear the entire library.

---

## 📁 Repository Structure

```
HackMTY/
├── app.py                 # FastAPI backend & web server
├── benchmark.py           # CLI side-by-side benchmark runner
├── Dockerfile             # Multi-stage Python 3.12 slim container
├── docker-compose.yml     # Orchestration with live volume mounts
├── requirements.txt       # Python dependencies
├── src/
│   ├── audio_processor.py # ASP pipeline (Denoising, Bandpass, VAD, MFCC/LFCC)
│   ├── biometrics.py      # Voice biometrics & DSP anti-spoofing engine
│   └── models/
│       ├── sincnet.py     # Learnable SincNet raw filterbanks
│       ├── rawnet2.py     # RawNet2 deepfake detector
│       └── aasist.py      # AASIST-L Graph Attention Network
└── web/
    └── index.html         # Interactive web interface
```

---

## 🔌 API Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the web control center |
| `POST` | `/api/process` | Re-evaluates all audios with updated ASP parameters |
| `POST` | `/api/upload` | Uploads a new audio sample |
| `POST` | `/api/load-sample-pack` | Ingests the benchmark audio suite |
| `DELETE` | `/api/audios/{key}` | Deletes a single audio recording |
| `DELETE` | `/api/audios-all` | Removes all audio recordings |
| `GET` | `/audio/raw/{file}` | Streams the original untouched audio |
| `GET` | `/audio/processed/{file}` | Streams the cleaned & filtered audio |
