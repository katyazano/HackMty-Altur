# mergeConflict • Frontend Web Interface

High-performance modular React + TypeScript + Vite web interface for the **Altur HackMTY Voice Anti-Spoofing & Deepfake Detection Challenge**.

## Features
- **Dual Engine Workbenches**: Live testing for Engine 1 (Real-Time Baseline) and Engine 2 (4-Pillar Multimodal SOTA).
- **Interactive Audio Ingestion**: Drag-and-drop / `.wav` upload sending requests to backend at `http://localhost:8000`.
- **Terminal View**: Color-coded JSON syntax highlighter with dimmed placeholder state and live latency display.
- **Copy Endpoint Strips**: One-click clipboard copy for API endpoints.
- **Documentation Pages**: In-depth architecture breakdown and benchmarks for both engines.

---

## 🚀 Quick Start (Single-Command Docker on Port 8001)

Run with Docker Compose:
```bash
docker compose up --build
```
Open **`http://localhost:8001`** in your browser.

---

## 💻 Local Development

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Configure backend URL** (defaults to `http://localhost:8000`):
   ```bash
   cp .env.example .env
   ```

3. **Start Vite Dev Server** (Port 8001):
   ```bash
   npm run dev
   ```
   Open **`http://localhost:8001`**.

4. **Production Build**:
   ```bash
   npm run build
   ```
