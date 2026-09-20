# mergeConflict • Dual-Engine Voice Anti-Spoofing & Deepfake Detection Platform

Master backend service engineered for the **Altur HackMTY Voice Anti-Spoofing AI Challenge**.

Features high-performance dual anti-spoofing architectures:
1. **Engine 1: Ultra-Fast Baseline (~30ms CPU)**: 64-dimensional spectral acoustic physics + lightweight 1D AudioCNN + HistGradientBoosting.
2. **Engine 2: 4-Pillar Multimodal SOTA (296 dimensions)**: Graph Attention Neural Network (AASIST), RawNet2 SincNet waveforms, 136-dim DSP acoustic physics, and Whisper Spanish NLP timing.

---

## 🚀 Quick Start (Single-Command Docker)

Start the entire dual-engine API with Docker Compose:
```bash
docker compose up --build
```
The API is live at **`http://localhost:8000`**.
Interactive OpenAPI / Swagger docs are available at **`http://localhost:8000/docs`**.

---

## 📡 API Endpoints

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/detect/baseline` | `POST` | Engine 1 frontline low-latency biometrics (~30ms) |
| `/detect/multimodal` | `POST` | Engine 2 deep 4-pillar representation detection |
| `/compare` | `POST` | Side-by-side execution of both engines with agreement metrics |
| `/health` | `GET` | Service status, device, and warm model telemetry |
| `/docs` | `GET` | Interactive Swagger API documentation |

### Contract Response Schema
```json
{
  "is_synthetic": false,
  "confidence": 0.982
}
```

---

## 🧪 Local Python Setup & Testing

1. **Create and activate environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run tests**:
   ```bash
   pytest tests/test_api.py -v
   ```

3. **Start local uvicorn server**:
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```
