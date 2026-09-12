# 🎙️ Altur Banking: Voice Anti-Spoofing & Deepfake AI Defense Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%2FGPU-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-eb4034?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS S3](https://img.shields.io/badge/AWS-S3%20Audit-232F3E?logo=amazon-aws&logoColor=white)](https://aws.amazon.com/s3/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4-38B2AC?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![HackMTY](https://img.shields.io/badge/HackMTY-Altur%20Challenge-blueviolet)](https://hackmty.com/)

A production-grade, multi-model AI voice biometric anti-spoofing engine developed for the **Altur Banking Challenge at HackMTY**.

The system analyzes inbound 2-party bank telephony audio (8kHz narrowband stereo / 16kHz resampled), separates **Channel 0 (Caller)** from Channel 1 (Agent), and executes stacked multi-model inference combining Graph Neural Networks (**AASIST**), Raw Waveform Convolutions (**RawNet2**), and Spectral DSP features into a regularized **XGBoost** meta-classifier with **Equal Error Rate (EER) threshold calibration** and non-blocking AWS S3 compliance auditing.

---

## 🏛️ Multi-Model Stacked Architecture (272 Dimensions)

```
                       Inbound Call Audio (.wav)
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
       Channel 0: Caller (Target)      Channel 1: Bank Agent (Excluded)
                  │
        ┌─────────┼────────────────────────┐
        ▼         ▼                        ▼
    ┌────────┐ ┌────────┐ ┌─────────────────────────────────┐
    │ AASIST │ │RawNet2 │ │ Acoustic DSP Feature Extractor  │
    │  (GNN) │ │(SincNet│ │ (MFCCs, Spectral Centroid,      │
    │ 132-dim│ │ 4-dim) │ │  Roll-off, Chroma, Prosody)     │
    │        │ │        │ │            136-dim              │
    └───┬────┘ └───┬────┘ └────────────────┬────────────────┘
        │          │                       │
        └──────────┼───────────────────────┘
                   ▼
       Concatenated Feature Vector (272-dim)
                   │
                   ▼
      StandardScaler Feature Normalization
                   │
                   ▼
       Regularized XGBoost Meta-Learner
                   │
                   ▼
     Continuous Probability: P(synthetic)
                   │
                   ▼
   Calibrated Decision Threshold (τ* = 0.9623)
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
 🟢 Human Verified     🔴 Synthetic AI Spoof
(is_synthetic: false)  (is_synthetic: true)
```

### 🧠 Core Performance & Calibration Benchmarks
* **5-Fold Full Corpus Cross-Validation**: **`96.88% ± 1.41%`** (Anti-overfitting verified across 353 recordings).
* **Calibrated Equal Error Rate (EER)**: **`8.47%`** achieved at optimal threshold **$\tau^* = 0.9623$** (minimizing $|FPR - FNR|$).
* **False Alarms**: Reduced by **`50%`** compared to arbitrary 0.50 decision boundaries.
* **Inference Latency**: **`~73ms`** per request.

---

## 🛠️ Technology Stack

| Category | Technologies / Libraries |
| :--- | :--- |
| **Deep Learning & ML** | PyTorch, AASIST (Graph Attention Networks), RawNet2 (SincNet + ResNet-GRU), XGBoost, Scikit-Learn |
| **Audio DSP** | Librosa, SoundFile, SciPy (Polyphase Resampling), Noisereduce |
| **Backend API** | FastAPI, Uvicorn, Pydantic v2, Python 3.12 |
| **Cloud & Auditing** | AWS S3 (`boto3` async audit logging), Terraform IaC (`infra/`) |
| **Data Analysis** | Jupyter Notebook, Pandas, NumPy, Matplotlib, Seaborn |
| **Frontend UI** | HTML5, TailwindCSS, WebAudio API, Canvas Waveform Visualizer, FontAwesome |
| **DevOps** | Docker, Docker Compose, Git LFS |

---

## 📋 Quickstart & Local Environment (`.venv`)

### 1. Clone & System Dependencies
* **macOS**: `brew install libsndfile ffmpeg libomp`
* **Ubuntu / Debian**: `sudo apt-get update && sudo apt-get install -y libsndfile1 ffmpeg libomp-dev curl`

### 2. Setup Virtual Environment
```bash
git clone https://github.com/katyazano/HackMty-Altur.git
cd HackMty-Altur

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Model Training & Benchmarking (Local `.venv`)
If you are starting from scratch or retraining:
```bash
# Optional 1-command dataset prep (manifest split, audio sorting, Channel 0 isolation)
python src/setup_dataset.py

# 1-Command Full Training Suite (Trains AASIST + RawNet2 + Triple XGBoost with 5-Fold CV & EER Calibration)
python src/train_and_benchmark.py
```
> Trained checkpoints (`aasist_best.pth`, `rawnet2_best.pth`, `xgboost_triple_ensemble.json`, `triple_scaler.joblib`) are automatically exported to `weights/`.

### 4. Start Local FastAPI Server & Web UI
```bash
PORT=8080 python app.py
```
* **Interactive UI**: [http://localhost:8080/](http://localhost:8080/)
* **Swagger API Docs**: [http://localhost:8080/docs](http://localhost:8080/docs)
* **Health Check**: [http://localhost:8080/health](http://localhost:8080/health)

---

## 🐳 Docker Containerization & Training

The Docker setup supports both **standalone model training** and **production API serving** with volume mounts so weights persist directly to your host machine:

### 1. Train Models Inside Docker (1-Command)
To train all base models (AASIST, RawNet2) and the calibrated XGBoost ensemble inside an isolated Docker container:
```bash
docker compose run --rm trainer
```
*(All trained model weights and benchmark reports are saved directly to `./weights/` and `./Data/` on your host machine).*

### 2. Run the Full API & Web UI Inside Docker
```bash
# Build and launch API container in background
docker compose up --build -d

# Inspect live container logs
docker compose logs -f altur-voice-api

# Stop container
docker compose down
```
> **Self-Healing Bootstrap**: If `docker compose up` is executed when `weights/` is empty, the container **automatically detects missing weights and triggers training upon boot** before serving traffic.

---

## 📊 Interactive Jupyter Notebook Analysis

The repository includes a comprehensive, interactive Jupyter Notebook ([`multimodel_analysis.ipynb`](multimodel_analysis.ipynb)) containing ROC curves, EER threshold calibration plots, confusion matrix heatmaps, 5-fold cross-validation reports, feature importance charts, and inline audio players (`IPython.display.Audio`).

### Launch Jupyter Notebook:
```bash
# Launch interactive browser session
.venv/bin/jupyter notebook multimodel_analysis.ipynb
```
*(Or launch JupyterLab via `.venv/bin/jupyter lab multimodel_analysis.ipynb`)*

### Headless Automated Execution:
To re-run and render all cells from the command line:
```bash
jupyter nbconvert --to notebook --execute multimodel_analysis.ipynb --output multimodel_analysis_executed.ipynb
```

---

## 📡 API Endpoint Reference

### `POST /detect`
Receives an 8kHz Base64-encoded stereo WAV clip, isolates **Channel 0 (Caller)**, resamples to 16kHz, and evaluates voice authenticity against the calibrated threshold ($\tau^* = 0.9623$).

#### Request Body (`application/json`):
```json
{
  "audio": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
  "call_id": "call_0847d7417bb1"
}
```

#### Response (`application/json`):
```json
{
  "is_synthetic": true,
  "confidence": 0.9863
}
```

---

## 🧪 Automated Testing

Run the automated test suite covering endpoint health, stereo 8kHz decoding, real audio inference, and validation edge cases:

```bash
# Run unit & integration tests
.venv/bin/python tests/test_api.py
```

---

## 📁 Repository Structure

```
HackMty-Altur/
├── app.py                         # FastAPI server with lifespan, Web UI routes & /detect
├── Dockerfile                     # Production multi-stage container (PyTorch CPU + DSP)
├── docker-compose.yml             # Container orchestration with volume mounts
├── requirements.txt               # Pinned Python dependencies
├── .env.example                   # Environment configuration template
├── multimodel_analysis.ipynb      # Interactive Jupyter Notebook analysis suite
├── tests/
│   └── test_api.py                # Automated unit and integration test suite
├── web/
│   └── index.html                 # Interactive WebAudio UI with Canvas waveform & mic recording
├── weights/                       # Pre-trained model checkpoints & scalers
│   ├── aasist_best.pth            # AASIST Graph Attention Network weights
│   ├── rawnet2_best.pth           # RawNet2 SincNet weights
│   ├── xgboost_triple_ensemble.json # Stacked 272-dim XGBoost meta-learner
│   └── triple_scaler.joblib       # Fitted StandardScaler for 272 features
├── src/
│   ├── api/
│   │   ├── config.py              # Pydantic Settings & environment variables
│   │   ├── schemas.py             # Strict request & response contracts
│   │   ├── audio_utils.py         # Base64 stereo decoding, Channel 0 isolation, resampling
│   │   ├── inference_engine.py    # Multi-model inference pipeline & threshold evaluator
│   │   └── s3_audit.py            # Non-blocking async AWS S3 compliance auditing
│   ├── features.py                # AASIST, RawNet2 & Acoustic DSP feature extractors
│   ├── evaluate_xgboost.py        # Standalone binary evaluation & metric recalculator
│   ├── train_and_benchmark.py     # 4-Model evaluation & comparative benchmark suite
│   ├── xgboost_trainer.py         # K-Fold CV trainer and EER threshold calibrator
│   ├── setup_dataset.py           # Automated 1-command audio dataset organizer
│   └── models/
│       ├── sincnet.py             # Learnable SincNet raw filterbank frontend
│       ├── rawnet2.py             # RawNet2 waveform architecture
│       ├── aasist.py              # AASIST Graph Neural Network
│       └── ensemble.py            # Stacked ensemble abstractions
└── Data/                          # Manifests, benchmark reports & metadata
    ├── manifest.csv
    ├── train_manifest.csv
    ├── test_manifest.csv
    ├── xgboost_training_report.json
    └── multimodel_benchmark_results.json
```

---

## 🔒 Security & S3 Compliance Auditing

In banking environments, regulatory compliance requires persistent audit trails for all customer voice interactions:
* When `ENABLE_S3_AUDIT=true`, the API asynchronously streams original raw audio payloads to **AWS S3** (`raw-recordings/8khz/YYYY/MM/DD/{call_id}.wav`).
* Uploads run inside a non-blocking background thread pool without adding latency to the client response.
* If AWS credentials are not configured locally, the service automatically falls back gracefully.

---

## 🏆 HackMTY 2026 - Altur Banking Challenge
Built with ❤️ for Altur Banking.
