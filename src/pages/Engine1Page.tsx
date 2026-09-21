import React, { useState } from 'react';
import { EngineCard } from '../components/EngineCard';
import { EngineConfig } from '../types/detection';

const ENGINE_1_CONFIG: EngineConfig = {
  id: 'baseline',
  prefix: 'e1',
  titleTag: 'Engine 1 • Real-Time Baseline',
  title: 'Ultra-Fast Baseline (64 dimensions)',
  description:
    'Ultra-low latency (~30ms CPU) frontline voice biometrics combining acoustic DSP physics with micro-segment convolutional scoring.',
  endpoint: '/detect/baseline',
  chips: [
    { icon: 'fa-solid fa-microchip', label: 'SpoofCNN (AudioCNN)', highlight: true },
    { icon: 'fa-solid fa-tree', label: 'HistGradientBoosting' },
    { icon: 'fa-solid fa-bolt', label: '~32ms Latency', highlight: true },
  ],
  exampleResponse: {
    is_synthetic: false,
    confidence: 0.982,
  },
  docsUrl: '#/engine1',
};

interface Engine1PageProps {
  onNavigate?: (route: 'home' | 'engine1' | 'engine2') => void;
}

export const Engine1Page: React.FC<Engine1PageProps> = ({ onNavigate }) => {
  const [codeTab, setCodeTab] = useState<'python' | 'curl' | 'node'>('python');

  return (
    <div className="docs-layout">
      {/* Header & Overview */}
      <div className="docs-header">
        <div className="docs-tag">Frontline Telephony Voice Biometrics</div>
        <h1 className="docs-title">Engine 1: Ultra-Fast Baseline</h1>
        <p className="docs-desc">
          Engine 1 is engineered specifically for frontline interactive telephony systems (IVRs, call centers, VoIP gateways) where every millisecond counts. It delivers deterministic, calibrated anti-spoofing verdicts within a strict <strong>&lt;50ms response budget</strong> on standard CPU instances without GPU dependencies.
        </p>
      </div>

      {/* Interactive Workbench */}
      <div className="docs-section">
        <div className="docs-section-title">
          <i className="fa-solid fa-terminal text-[#FF5500]"></i>
          <span>Live Interactive Workbench</span>
        </div>
        <EngineCard config={ENGINE_1_CONFIG} onNavigate={onNavigate} />
      </div>

      {/* Architecture & Pipeline Stages */}
      <div className="docs-section">
        <div className="docs-section-title">
          <i className="fa-solid fa-diagram-project text-[#FF5500]"></i>
          <span>4-Stage Low-Latency Processing Pipeline</span>
        </div>

        <div className="pipeline-flow-container">
          <div className="pipeline-step-card">
            <span className="pipeline-step-num">STAGE 01</span>
            <div className="pipeline-step-content">
              <div className="pipeline-step-title">
                <i className="fa-solid fa-wave-square text-[#FF5500]"></i>
                <span>Audio Ingestion & ITU-T Normalization</span>
              </div>
              <div className="pipeline-step-desc">
                Ingests raw PCM WAV audio (8kHz or 16kHz). Automatically applies ITU-T P.56 active speech level normalization, trims silent margins (-40dB energy threshold), and segments the caller audio stream using 25ms Hamming analysis windows with 10ms frame stride.
              </div>
            </div>
          </div>

          <div className="pipeline-step-card">
            <span className="pipeline-step-num">STAGE 02</span>
            <div className="pipeline-step-content">
              <div className="pipeline-step-title">
                <i className="fa-solid fa-chart-simple text-[#FF5500]"></i>
                <span>64-Dimensional Acoustic Physics DSP Extractor</span>
              </div>
              <div className="pipeline-step-desc">
                Computes 13 Mel-Frequency Cepstral Coefficients (MFCCs) alongside their velocity ($\Delta$) and acceleration ($\Delta\Delta$) temporal derivatives. Augments spectral representations with Spectral Centroid, Spectral Rolloff (85% energy point), Zero-Crossing Rate (ZCR), and Sub-band Spectral Flux.
              </div>
            </div>
          </div>

          <div className="pipeline-step-card">
            <span className="pipeline-step-num">STAGE 03</span>
            <div className="pipeline-step-content">
              <div className="pipeline-step-title">
                <i className="fa-solid fa-microchip text-[#FF5500]"></i>
                <span>Dual-Branch Model Architecture</span>
              </div>
              <div className="pipeline-step-desc">
                <strong>Branch A (SpoofCNN)</strong>: A 4-layer 1D Convolutional Neural Network processing raw micro-segments to detect high-frequency vocoder phase discrepancies.<br />
                <strong>Branch B (HistGradientBoosting)</strong>: Fast histogram-binned gradient boosted decision trees analyzing statistical distribution of acoustic features.
              </div>
            </div>
          </div>

          <div className="pipeline-step-card">
            <span className="pipeline-step-num">STAGE 04</span>
            <div className="pipeline-step-content">
              <div className="pipeline-step-title">
                <i className="fa-solid fa-scale-balanced text-[#FF5500]"></i>
                <span>Ensemble Fusion & Calibrated Confidence</span>
              </div>
              <div className="pipeline-step-desc">
                Fuses both branch outputs with a weighted ensemble and Platt scaling calibrator, emitting the standardized HackMTY payload <code>{`{"is_synthetic": bool, "confidence": float}`}</code> in ~32ms total CPU runtime.
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* DSP & Physics Details */}
      <div className="docs-section">
        <div className="docs-section-title">
          <i className="fa-solid fa-atom text-[#FF5500]"></i>
          <span>Acoustic DSP Features Extracted</span>
        </div>
        <div className="docs-grid-cards">
          <div className="docs-card">
            <div className="docs-card-title">13 MFCCs + Velocity ($\Delta$) + Acceleration ($\Delta\Delta$)</div>
            <div className="docs-card-text">
              Captures human vocal tract resonances. Synthesizers often struggle with natural formant transitions between phonemes, creating abrupt spectral jumps visible in $\Delta\Delta$ coefficients.
            </div>
          </div>
          <div className="docs-card">
            <div className="docs-card-title">Spectral Centroid & Rolloff (85%)</div>
            <div className="docs-card-text">
              Measures the "brightness" and high-frequency cutoff of the audio spectrum. Neural vocoders frequently leave unnatural high-frequency energy tails or excessive high-band attenuation.
            </div>
          </div>
          <div className="docs-card">
            <div className="docs-card-title">Sub-Band Spectral Flux</div>
            <div className="docs-card-text">
              Tracks the rate of change in spectral power across frequency bands. Human speech exhibits organic harmonic variation, whereas TTS models exhibit synthetic stiffness across steady-state vowels.
            </div>
          </div>
          <div className="docs-card">
            <div className="docs-card-title">Zero-Crossing Rate (ZCR) Dynamic Range</div>
            <div className="docs-card-text">
              Analyzes the frequency of signal sign changes. Exposes noise gate anomalies and micro-glitches common in real-time neural vocoder generation.
            </div>
          </div>
        </div>
      </div>

      {/* Technical Specifications Table */}
      <div className="docs-section">
        <div className="docs-section-title">
          <i className="fa-solid fa-sliders text-[#FF5500]"></i>
          <span>Technical Specifications & Hyperparameters</span>
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
                <td>Feature Dimensions</td>
                <td><code>64 floats</code></td>
                <td>Compact acoustic physics feature representation</td>
              </tr>
              <tr>
                <td>Sample Rate Ingestion</td>
                <td><code>8,000 Hz / 16,000 Hz</code></td>
                <td>Supports standard G.711 telephony and wideband audio</td>
              </tr>
              <tr>
                <td>Average Latency</td>
                <td><span className="badge-pill badge-green"><i className="fa-solid fa-bolt"></i> ~32 ms (CPU)</span></td>
                <td>Measured on standard 2-vCPU virtual machine</td>
              </tr>
              <tr>
                <td>Memory Footprint</td>
                <td><span className="badge-pill badge-blue">&lt; 45 MB RAM</span></td>
                <td>Lightweight footprint suitable for high-density edge deployments</td>
              </tr>
              <tr>
                <td>Inference Device</td>
                <td><code>CPU / AVX2</code></td>
                <td>Runs fully on CPU with zero GPU requirement</td>
              </tr>
              <tr>
                <td>Decision Threshold</td>
                <td><code>0.50 (Calibrated)</code></td>
                <td>Platt sigmoid calibration to probability scale [0.0, 1.0]</td>
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
            <span>POST /detect/baseline</span>
          </div>

          <pre className="code-box-body">
            {codeTab === 'python' && `import base64
import requests

# 1. Encode local .wav file to base64
with open("call_audio.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

# 2. Call Engine 1 baseline endpoint
response = requests.post(
    "http://localhost:8000/detect/baseline",
    json={"audio": audio_b64, "call_id": "call_12345"}
)

# 3. Standard HackMTY contract result
data = response.json()
print(f"Is Synthetic: {data['is_synthetic']}")
print(f"Confidence:   {data['confidence']:.4f}")`}

            {codeTab === 'curl' && `curl -X POST "http://localhost:8000/detect/baseline" \\
  -H "Content-Type: application/json" \\
  -d '{
    "audio": "'"$(base64 -w 0 call_audio.wav)"'",
    "call_id": "call_12345"
  }'`}

            {codeTab === 'node' && `const fs = require('fs');

const audioBase64 = fs.readFileSync('call_audio.wav').toString('base64');

fetch('http://localhost:8000/detect/baseline', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ audio: audioBase64, call_id: 'call_12345' })
})
  .then(res => res.json())
  .then(data => console.log('Result:', data));`}
          </pre>
        </div>
      </div>
    </div>
  );
};
