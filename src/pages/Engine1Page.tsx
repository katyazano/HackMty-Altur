import React from 'react';
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
  return (
    <div className="docs-layout">
      {/* Header */}
      <div className="docs-header">
        <div className="docs-tag">Frontline Telephony Biometrics</div>
        <h1 className="docs-title">Engine 1: Ultra-Fast Baseline</h1>
        <p className="docs-desc">
          Engineered for strict real-time telephony SLAs (&lt;50ms response budget).
          Combines 64-dimensional acoustic spectral physics with lightweight 1D convolutional feature extraction and tree boosting.
        </p>
      </div>

      {/* Interactive Workbench */}
      <div className="docs-section">
        <div className="docs-section-title">
          <i className="fa-solid fa-terminal text-[#FF5500]"></i>
          <span>Live Inference Workbench</span>
        </div>
        <EngineCard config={ENGINE_1_CONFIG} onNavigate={onNavigate} />
      </div>

      {/* Technical Architecture */}
      <div className="docs-section">
        <div className="docs-section-title">
          <i className="fa-solid fa-microchip text-[#FF5500]"></i>
          <span>Pipeline Architecture</span>
        </div>
        <div className="docs-grid-cards">
          <div className="docs-card">
            <div className="docs-card-title">1. Audio Standardization</div>
            <div className="docs-card-text">
              Direct ingestion of raw PCM WAV audio, resampled to 16kHz mono with ITU-T P.56 active speech level normalization and 25ms Hamming window framing.
            </div>
          </div>
          <div className="docs-card">
            <div className="docs-card-title">2. 64-Dim DSP Extractor</div>
            <div className="docs-card-text">
              Computes MFCCs (13 coefficients), spectral centroid, spectral rolloff, zero-crossing rate, sub-band spectral flux, and short-term energy dynamics.
            </div>
          </div>
          <div className="docs-card">
            <div className="docs-card-title">3. SpoofCNN + HistGBM Ensemble</div>
            <div className="docs-card-text">
              A 4-layer 1D CNN processes raw micro-frames while a HistGradientBoostingClassifier scores tabular acoustic descriptors with calibrated sigmoid outputs.
            </div>
          </div>
        </div>
      </div>

      {/* Latency & Hardware Metrics */}
      <div className="docs-section">
        <div className="docs-section-title">
          <i className="fa-solid fa-gauge-high text-[#FF5500]"></i>
          <span>Performance Benchmarks</span>
        </div>
        <div className="docs-grid-cards">
          <div className="docs-card">
            <div className="docs-card-title">Latency (CPU)</div>
            <div className="docs-card-text">
              <strong>~32ms average</strong> on standard 2-core cloud virtual machines without requiring specialized GPU accelerators.
            </div>
          </div>
          <div className="docs-card">
            <div className="docs-card-title">Memory Footprint</div>
            <div className="docs-card-text">
              Under <strong>45 MB RAM</strong> in memory warm state, enabling high concurrency and edge deployment.
            </div>
          </div>
          <div className="docs-card">
            <div className="docs-card-title">API Compatibility</div>
            <div className="docs-card-text">
              Fully compliant with the HackMTY Altur Contract Schema: <code>{`{"is_synthetic": bool, "confidence": float}`}</code>.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
