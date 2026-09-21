# merge-conflict • Dual-Engine Voice Anti-Spoofing & Deepfake Detection

High-performance, industrial web application engineered for the **Altur HackMTY Voice Anti-Spoofing Challenge**. 

Provides real-time audio testing workbenches, live backend health monitoring, and in-depth documentation for both detection engines.

---

## ⚡ Key Highlights & Architecture

- **Engine 1: Ultra-Fast Baseline (64 dimensions)**:
  - Frontline voice biometrics optimized for telephony and call center gateways (~700ms CPU).
  - Uses ITU-T P.56 normalization, 25ms Hamming STFT, 64-dimensional acoustic DSP physics, and a HistGradientBoosting decision forest.
- **Engine 2: 4-Pillar Multimodal SOTA (296 dimensions)**:
  - Deep representation learning (~1300ms) for high-stakes verification.
  - Combines **AASIST** (Graph Neural Network), **RawNet2** (Parametric SincNet waveforms), **136-dim Deep DSP**, and **Whisper NLP** conversational token analysis fused via an **XGBoost Quad** meta-learner.
- **Live Health Monitoring & Dashboard Redirect**:
  - Automatically polls backend status (`http://localhost:8000`) with visual feedback (pulsing orange dot for online, gray for offline).
  - Direct redirect to the live analytics dashboard (`http://localhost:8000/dashboard`).
- **Interactive Workbench**:
  - Drag-and-drop `.wav` audio upload, live millisecond latency calculation, and formatted JSON response terminal.
- **API Integration Guides**:
  - Copy-paste integration code in Python, cURL, and Node.js for both engines.

---

## 🚀 Quick Start with Docker (Port 8001)

Run the frontend container with Docker Compose:

```bash
docker compose up --build
```

Access the web interface at **`http://localhost:8001`**.

---

## 💻 Local Development (Vite + React + TypeScript)

### 1. Prerequisites
- Node.js 18+
- npm 9+

### 2. Install Dependencies
```bash
npm install
```

### 3. Environment Configuration
By default, the frontend connects to the backend at `http://localhost:8000`. You can override this using `.env`:
```bash
cp .env.example .env
```

### 4. Run Development Server
```bash
npm run dev
```
Navigate to **`http://localhost:8001`** in your browser.

### 5. Build for Production
```bash
npm run build
```

---

## 🔌 API Endpoints Reference

| Engine | Endpoint | Runtime | Dimensions | Target Spoofs |
| :--- | :--- | :--- | :--- | :--- |
| **Engine 1** | `POST /detect/baseline` | ~700 ms (CPU) | 64 | Replay, Vocoder artifacts |
| **Engine 2** | `POST /detect/multimodal` | ~1300 ms (CPU) | 296 | Zero-shot clones, ElevenLabs, XTTS-v2, HiFi-GAN |
| **Dashboard** | `GET /dashboard` | — | — | Live telemetry & metrics |

### Expected Response Schema:
```json
{
  "is_synthetic": false,
  "confidence": 0.982
}
```

---

## 👥 Contributors (Team `merge-conflict`)

- [pontro](https://github.com/pontro)
- [Chewbaccas](https://github.com/Chewbaccas)
- [katyazano](https://github.com/katyazano)
- [cris-hernandezz](https://github.com/cris-hernandezz)

Built for **HackMTY 2026** // Altur Voice Anti-Spoofing Challenge.

