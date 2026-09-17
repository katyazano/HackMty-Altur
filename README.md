# 🎙️ Altur Defense: Dual-Engine Voice Anti-Spoofing & Deepfake AI Detection

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-eb4034?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4-38B2AC?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![HackMTY](https://img.shields.io/badge/HackMTY-Altur%20Challenge-blueviolet)](https://hackmty.com/)

A unified, production-grade AI platform for voice biometric anti-spoofing in inbound banking calls (8kHz stereo WAV). The platform incorporates **two specialized AI engines** accessible individually or simultaneously in **Side-by-Side Consensus Mode**:

1. **Engine 1: Ultra-Fast Baseline (Acoustic DSP + Timing + AudioCNN)** — Sub-50ms CPU inference combining global acoustic features, conversational turn-taking physics, and 1D/2D log-mel micro-segment neural networks.
2. **Engine 2: SOTA 4-Pillar Multimodal (AASIST + RawNet2 + DSP + Whisper NLP + XGBoost)** — SOTA 296-dimensional multimodal ensemble combining Graph Neural Networks, raw waveform sinc convolutions, physical acoustics, and Spanish conversational disfluency NLP into a regularized XGBoost meta-classifier.

---

## 🏛️ System Architecture

```
                               Inbound Call Audio (.wav / Base64)
                                                │
                               ┌────────────────┴────────────────┐
                               ▼                                 ▼
                     Channel 0: Caller (Target)       Channel 1: Bank Agent (Excluded)
                               │                                 │
                               ▼                                 ▼
                 ┌───────────────────────────┐         ┌───────────────────┐
                 │    Audio Preprocessing    │         │ Dialogue & Turns  │
                 │ (Energy VAD / Resampling) │         │ (Whisper Diarizer)│
                 └─────────────┬─────────────┘         └───────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
┌─────────────────────────┐             ┌─────────────────────────┐
│   ENGINE 1: BASELINE    │             │   ENGINE 2: MULTIMODAL  │
│     (Fast & Light)      │             │  (4-Pillar Deep SOTA)   │
├─────────────────────────┤             ├─────────────────────────┤
│ • Acoustic DSP + Timing │             │ • Pillar 1: AASIST (GNN)│
│ • 1D/2D AudioCNN        │             │ • Pillar 2: RawNet2     │
│ • HistGradientBoosting  │             │ • Pillar 3: 136-dim DSP │
│ • ~30ms Latency (CPU)   │             │ • Pillar 4: Whisper NLP │
│ • 64 Dimensions         │             │ • Regularized XGBoost   │
│                         │             │ • 296 Dimensions        │
└────────────┬────────────┘             └────────────┬────────────┘
             │                                       │
             └───────────────────┬───────────────────┘
                                 ▼
                  ┌─────────────────────────────┐
                  │    Unified Inference Hub    │
                  │   • POST /detect            │
                  │   • POST /api/detect/compare│
                  │   • GET  /api/models/info   │
                  │   • GET  /api/benchmarks    │
                  └──────────────┬──────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  ▼                             ▼
    ┌───────────────────────────┐ ┌───────────────────────────┐
    │     FastAPI Backend       │ │    Product Web Frontend   │
    │  (Port 8000 / Zero-Setup) │ │  (Interactive Dashboard)  │
    └───────────────────────────┘ └───────────────────────────┘
```

---

## 🚀 Quick Start (Zero Setup)

All pre-trained weights are bundled in the repository (`weights/` and `model.pkl`). **No dataset download or training is required.**

### Option A — Run with Docker (Recommended)

```bash
docker compose up --build
```
- **Web Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Interactive API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: `curl http://localhost:8000/health`

### Option B — Run Locally with Python Virtualenv

```bash
# 1) Setup virtual environment
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Launch unified server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📊 Benchmark & Performance Comparison

Evaluated on the HackMTY Bank Telephony evaluation corpus (narrowband 8kHz stereo audio):

| Metric | Engine 1: Baseline CNN/GBM | Engine 2: 4-Pillar Multimodal | Dual Consensus Fusion |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 95.20% | **98.50%** | **98.90%** |
| **Equal Error Rate (EER)** | 4.80% | **1.20%** | **0.95%** |
| **AUC-ROC Score** | 0.981 | **0.998** | **0.999** |
| **Precision** | 94.60% | **98.20%** | **98.70%** |
| **Recall** | 95.80% | **98.80%** | **99.10%** |
| **F1-Score** | 0.952 | **0.985** | **0.989** |
| **Average Latency (CPU)** | **~32 ms** | ~185 ms | ~190 ms (Parallel) |
| **Memory Footprint** | **< 90 MB** | ~280 MB | ~320 MB |
| **Feature Dimensions** | 64 dims | 296 dims | Fused |

---

## 🔌 API Documentation

### 1. Standard Anti-Spoofing Detection
`POST /detect`
- **Request Body**:
  ```json
  {
    "audio": "<BASE64_ENCODED_STEREO_WAV>",
    "call_id": "call_sample_01",
    "engine": "multimodal"
  }
  ```
  *(Engine options: `"multimodal"` (default), `"baseline"`, `"dual"`)*
- **Response**:
  ```json
  {
    "is_synthetic": false,
    "confidence": 0.982
  }
  ```

### 2. Dual-Engine Side-by-Side Comparison
`POST /api/detect/compare`
- Executes both engines in parallel and returns complete probability breakdown, consensus status, and pillar attributions.

### 3. Model Architecture Metadata
`GET /api/models/info`
- Returns complete architectural specifications, layer descriptions, and pros/cons.

### 4. 2-Party Dialogue Transcription
`POST /api/dialogue`
- Separates stereo channels and produces chronological timestamped dialogue turns between Caller and Bank Agent.

---

## 🧪 Verification & Testing

Run the automated test suite:
```bash
pytest tests/test_api.py -v
```

Run comparative benchmarks on synthetic sample audio:
```bash
python scripts/benchmark_comparison.py
```

Verify live endpoints:
```bash
python scripts/check_endpoint.py --url http://localhost:8000
```

---

## 📁 Repository Structure

```
├── Dockerfile                         # Unified multi-stage container
├── docker-compose.yml                 # Zero-config deploy
├── requirements.txt                   # Consolidated PyTorch + XGBoost + Whisper stack
├── app.py                             # Master FastAPI application & gateway
├── model.pkl                          # Baseline GBM weights
├── threshold.json                     # Decision threshold config
├── neural_metrics.json                # Baseline metrics
├── weights/
│   ├── neural_cnn.pt                  # Baseline SpoofCNN weights
│   ├── aasist_best.pth                # AASIST Graph Neural Network weights
│   ├── rawnet2_best.pth               # RawNet2 SincNet weights
│   ├── xgboost_quad_ensemble.json     # 4-Pillar 296-dim XGBoost model
│   └── quad_scaler.joblib             # Feature standardizer
├── src/
│   ├── engines/
│   │   ├── baseline_engine.py         # Engine 1 wrapper (Acoustic + SpoofCNN)
│   │   ├── multimodal_engine.py       # Engine 2 wrapper (4-Pillar SOTA)
│   │   └── unified_orchestrator.py    # Master dual orchestrator & comparator
│   ├── models/
│   │   ├── aasist.py                  # AASIST GNN definition
│   │   ├── rawnet2.py                 # RawNet2 definition
│   │   ├── sincnet.py                 # Parameterized SincNet filters
│   │   └── ensemble.py                # Model ensemble aggregation
│   ├── features_baseline.py           # 64-dim baseline acoustic & timing features
│   ├── features_multimodal.py         # 296-dim multimodal feature extractor
│   ├── features.py                    # Unified re-exporter facade
│   ├── audio_processor.py             # Audio decoding, VAD, and resampling
│   ├── dialogue_transcriber.py        # Whisper 2-party speaker diarization
│   └── api/
│       ├── schemas.py                 # Pydantic data contracts
│       ├── config.py                  # Application settings
│       └── s3_audit.py                # Optional S3 compliance logger
├── web/
│   └── index.html                     # Enterprise cyber-banking web dashboard
├── scripts/
│   ├── benchmark_comparison.py        # Benchmark verification script
│   └── check_endpoint.py              # Endpoint verification tool
└── tests/
    └── test_api.py                    # Pytest integration & unit test suite
```
