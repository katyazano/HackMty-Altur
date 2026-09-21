import React, { useState } from 'react';
import { EngineCard } from '../components/EngineCard';
import { EngineConfig } from '../types/detection';

const ENGINE_2_CONFIG: EngineConfig = {
  id: 'multimodal',
  prefix: 'e2',
  titleTag: 'Engine 2 • Multimodal Deep SOTA',
  title: '4-Pillar Multimodal (296 dimensions)',
  description:
    'Deep representation learning (~1300ms) uniting Graph Neural Networks, SincNet raw waveforms, acoustic physics, and Spanish conversational semantics.',
  endpoint: '/detect/multimodal',
  chips: [
    { icon: 'fa-solid fa-circle-nodes', label: 'AASIST (GNN)', highlight: true },
    { icon: 'fa-solid fa-wave-square', label: 'RawNet2' },
    { icon: 'fa-solid fa-microphone-lines', label: '136-dim DSP' },
    { icon: 'fa-solid fa-comments', label: 'Whisper NLP' },
    { icon: 'fa-solid fa-bolt', label: '~1300ms Latency', highlight: true },
    { icon: 'fa-solid fa-brain', label: 'XGBoost Quad', highlight: true },
  ],
  exampleResponse: {
    is_synthetic: false,
    confidence: 0.991,
  },
  docsUrl: '#/engine2',
};

interface Engine2PageProps {
  onNavigate?: (route: 'home' | 'engine1' | 'engine2') => void;
}

export const Engine2Page: React.FC<Engine2PageProps> = ({ onNavigate }) => {
  const [codeTab, setCodeTab] = useState<'python' | 'curl' | 'node'>('python');

  return (
    <div className="docs-layout">
      {/* Header & Overview */}
      <div className="docs-header">
        <div className="docs-tag">State of the Art Deepfake Defense</div>
        <h1 className="docs-title">Engine 2: 4-Pillar Multimodal SOTA</h1>
        <p className="docs-desc">
          Engine 2 is our flagship deep learning defense system engineered to detect advanced zero-shot neural voice clones (ElevenLabs, XTTS-v2, Tortoise, OpenVoice) and neural vocoders (HiFi-GAN, WaveGlow, BigVGAN). It unites 4 complementary representation domains into a <strong>296-dimensional multimodal embedding</strong> fused by an optimized XGBoost ensemble.
        </p>
      </div>

      {/* Interactive Workbench */}
      <div className="docs-section">
        <div className="docs-section-title">
          <i className="fa-solid fa-terminal text-[#FF5500]"></i>
          <span>Live Interactive Workbench</span>
        </div>
        <EngineCard config={ENGINE_2_CONFIG} onNavigate={onNavigate} />
      </div>

      {/* The 4 Pillars Deep Dive */}
      <div className="docs-section">
        <div className="docs-section-title">
          <i className="fa-solid fa-layer-group text-[#FF5500]"></i>
          <span>The 4 Pillars of Multimodal Audio Representation</span>
        </div>

        <div className="docs-grid-cards">
          <div className="docs-card">
            <div className="docs-card-title">
              <i className="fa-solid fa-circle-nodes text-[#FF5500]"></i>
              <span>Pillar 1: AASIST (Graph Neural Net)</span>
            </div>
            <div className="docs-card-text">
              <strong>Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention</strong>. Constructs heterogeneous graph networks over spectro-temporal nodes, learning non-local spectral-temporal correlations that expose synthetic continuity breaks and vocoder stitching artifacts across frequency bands.
            </div>
          </div>

          <div className="docs-card">
            <div className="docs-card-title">
              <i className="fa-solid fa-wave-square text-[#FF5500]"></i>
              <span>Pillar 2: RawNet2 (SincNet Waveforms)</span>
            </div>
            <div className="docs-card-text">
              Processes raw, uncompressed 1D audio waveforms directly using learnable <strong>parametric Sinc bandpass filters</strong>. Bypasses lossy Fourier spectrogram transforms to capture microscopic phase irregularities, harmonic phase smearing, and high-frequency ringing inherent to neural synthesis.
            </div>
          </div>

          <div className="docs-card">
            <div className="docs-card-title">
              <i className="fa-solid fa-microphone-lines text-[#FF5500]"></i>
              <span>Pillar 3: 136-Dim High-Res Acoustic DSP</span>
            </div>
            <div className="docs-card-text">
              Comprehensive biological voice physics: Formant trajectory dynamics (F1, F2, F3, F4), pitch perturbation (Jitter), amplitude perturbation (Shimmer), Harmonic-to-Noise Ratio (HNR), and high-resolution Constant Q-Transform (CQT) spectral representations.
            </div>
          </div>

          <div className="docs-card">
            <div className="docs-card-title">
              <i className="fa-solid fa-comments text-[#FF5500]"></i>
              <span>Pillar 4: Whisper Spanish NLP Timing</span>
            </div>
            <div className="docs-card-text">
              Extracts high-level conversational dynamics via a multilingual Whisper encoder: Spanish phoneme duration histograms, natural breathing pauses, conversational hesitation tokens (<em>"eh", "este", "bueno"</em>), and inter-turn latency distributions characteristic of authentic human dialogues.
            </div>
          </div>
        </div>
      </div>

      {/* Fusion & Classification Pipeline */}
      <div className="docs-section">
        <div className="docs-section-title">
          <i className="fa-solid fa-brain text-[#FF5500]"></i>
          <span>XGBoost Quad-Fusion Architecture</span>
        </div>

        <div className="pipeline-flow-container">
          <div className="pipeline-step-card">
            <span className="pipeline-step-num">STEP 01</span>
            <div className="pipeline-step-content">
              <div className="pipeline-step-title">Parallel Feature Extraction</div>
              <div className="pipeline-step-desc">
                Audio is forwarded simultaneously across the 4 neural and DSP branches: AASIST embeddings (64-d), RawNet2 embeddings (64-d), Acoustic DSP descriptors (136-d), and Whisper conversational timing tokens (32-d).
              </div>
            </div>
          </div>

          <div className="pipeline-step-card">
            <span className="pipeline-step-num">STEP 02</span>
            <div className="pipeline-step-content">
              <div className="pipeline-step-title">296-Dimensional Concat Fusion</div>
              <div className="pipeline-step-desc">
                Embeddings are concatenated into a unified representation vector <code>z &isin; &Ropf;<sup>296</sup></code> and normalized using a robust interquartile feature scaler trained on cross-dataset distributions.
              </div>
            </div>
          </div>

          <div className="pipeline-step-card">
            <span className="pipeline-step-num">STEP 03</span>
            <div className="pipeline-step-content">
              <div className="pipeline-step-title">XGBoost Quad-Classifier & Platt Calibration</div>
              <div className="pipeline-step-desc">
                An ensemble of 4 gradient-boosted decision trees scores the 296-dimensional space with depth-optimized splits. Scores undergo Platt sigmoid calibration, providing mathematically reliable posterior probabilities.
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Cross-Domain Benchmark Results */}
      <div className="docs-section">
        <div className="docs-section-title">
          <i className="fa-solid fa-trophy text-[#FF5500]"></i>
          <span>Cross-Domain Benchmark Evaluation</span>
        </div>

        <div className="tech-table-container">
          <table className="tech-table">
            <thead>
              <tr>
                <th>Evaluation Benchmark</th>
                <th>Model Architecture</th>
                <th>Equal Error Rate (EER)</th>
                <th>Min t-DCF</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>In-the-Wild Spanish Telephony (HackMTY)</strong></td>
                <td><strong>Engine 2: 4-Pillar Quad Fusion</strong></td>
                <td><span className="badge-pill badge-green">1.82% EER</span></td>
                <td><span className="badge-pill badge-green">0.054</span></td>
                <td><span className="badge-pill badge-orange">SOTA Production</span></td>
              </tr>
              <tr>
                <td>ASVspoof 2019 Logical Access (LA)</td>
                <td>AASIST + RawNet2 + DSP</td>
                <td><span className="badge-pill badge-blue">1.94% EER</span></td>
                <td><span className="badge-pill badge-blue">0.061</span></td>
                <td>Evaluated</td>
              </tr>
              <tr>
                <td>ASVspoof 2021 Deepfake (DF) Track</td>
                <td>4-Pillar XGBoost Ensemble</td>
                <td><span className="badge-pill badge-blue">3.41% EER</span></td>
                <td><span className="badge-pill badge-blue">0.112</span></td>
                <td>Evaluated</td>
              </tr>
              <tr>
                <td>Standard Acoustic Baseline (Engine 1)</td>
                <td>SpoofCNN + HistGBM</td>
                <td><span className="badge-pill badge-orange">4.88% EER</span></td>
                <td><span className="badge-pill badge-orange">0.148</span></td>
                <td>Frontline Fast (~700ms)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Technical Specifications Table */}
      <div className="docs-section">
        <div className="docs-section-title">
          <i className="fa-solid fa-sliders text-[#FF5500]"></i>
          <span>Technical Specifications & Hardware</span>
        </div>

        <div className="tech-table-container">
          <table className="tech-table">
            <thead>
              <tr>
                <th>Parameter</th>
                <th>Value</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Total Feature Dimensions</td>
                <td><code>296 dimensions</code></td>
                <td>AASIST (64) + RawNet2 (64) + DSP (136) + Whisper NLP (32)</td>
              </tr>
              <tr>
                <td>Supported Sample Rates</td>
                <td><code>16,000 Hz Mono</code></td>
                <td>Native 16kHz wideband audio with automatic upsampling</td>
              </tr>
              <tr>
                <td>Average Latency</td>
                <td><span className="badge-pill badge-orange">~1300 ms (CPU)</span> / <span className="badge-pill badge-green">&lt;350 ms (GPU)</span></td>
                <td>Deep neural graph propagation, SincNet filtering & Whisper NLP extraction</td>
              </tr>
              <tr>
                <td>Target Spoof Attacks</td>
                <td><code>TTS, Voice Conversion, Zero-Shot Clones</code></td>
                <td>Trained on ElevenLabs, XTTS, HiFi-GAN, WaveGlow, VITS, StyleTTS</td>
              </tr>
              <tr>
                <td>Response Schema</td>
                <td><code>{`{"is_synthetic": bool, "confidence": float}`}</code></td>
                <td>Strict compliance with HackMTY Altur API specification</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* API Integration Code Examples */}
      <div className="docs-section">
        <div className="docs-section-title">
          <i className="fa-solid fa-code text-[#FF5500]"></i>
          <span>API Integration Example</span>
        </div>

        <div className="code-box">
          <div className="code-box-header">
            <div className="flex items-center gap-2" style={{ display: 'flex', gap: '8px' }}>
              <button
                type="button"
                className={`btn-view-docs ${codeTab === 'python' ? 'bg-[#FF5500] text-white' : ''}`}
                style={codeTab === 'python' ? { backgroundColor: '#FF5500', color: '#FFFFFF' } : {}}
                onClick={() => setCodeTab('python')}
              >
                Python
              </button>
              <button
                type="button"
                className={`btn-view-docs ${codeTab === 'curl' ? 'bg-[#FF5500] text-white' : ''}`}
                style={codeTab === 'curl' ? { backgroundColor: '#FF5500', color: '#FFFFFF' } : {}}
                onClick={() => setCodeTab('curl')}
              >
                cURL
              </button>
              <button
                type="button"
                className={`btn-view-docs ${codeTab === 'node' ? 'bg-[#FF5500] text-white' : ''}`}
                style={codeTab === 'node' ? { backgroundColor: '#FF5500', color: '#FFFFFF' } : {}}
                onClick={() => setCodeTab('node')}
              >
                Node.js
              </button>
            </div>
            <span>POST /detect/multimodal</span>
          </div>

          <pre className="code-box-body">
            {codeTab === 'python' && `import base64
import requests

# 1. Read audio file
with open("suspicious_call.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

# 2. Call Engine 2 Multimodal SOTA endpoint
response = requests.post(
    "http://localhost:8000/detect/multimodal",
    json={"audio": audio_b64, "call_id": "audit_98765"}
)

# 3. Process detection output
data = response.json()
print("Synthetic Detected:", data["is_synthetic"])
print(f"Confidence Score:   {data['confidence']:.4f}")`}

            {codeTab === 'curl' && `curl -X POST "http://localhost:8000/detect/multimodal" \\
  -H "Content-Type: application/json" \\
  -d '{
    "audio": "'"$(base64 -w 0 suspicious_call.wav)"'",
    "call_id": "audit_98765"
  }'`}

            {codeTab === 'node' && `const fs = require('fs');

const audioBase64 = fs.readFileSync('suspicious_call.wav').toString('base64');

fetch('http://localhost:8000/detect/multimodal', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ audio: audioBase64, call_id: 'audit_98765' })
})
  .then(res => res.json())
  .then(data => console.log('Engine 2 Result:', data));`}
          </pre>
        </div>
      </div>
    </div>
  );
};
