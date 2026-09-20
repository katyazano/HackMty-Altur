import React from 'react';
import { EngineCard } from '../components/EngineCard';
import { EngineConfig } from '../types/detection';

const ENGINE_2_CONFIG: EngineConfig = {
  id: 'multimodal',
  prefix: 'e2',
  titleTag: 'Engine 2 • Multimodal Deep SOTA',
  title: '4-Pillar Multimodal (296 dimensions)',
  description:
    'Deep representation learning uniting Graph Neural Networks, SincNet raw waveforms, acoustic physics, and Spanish conversational semantics.',
  endpoint: '/detect/multimodal',
  chips: [
    { icon: 'fa-solid fa-circle-nodes', label: 'AASIST (GNN)', highlight: true },
    { icon: 'fa-solid fa-wave-square', label: 'RawNet2' },
    { icon: 'fa-solid fa-microphone-lines', label: '136-dim DSP' },
    { icon: 'fa-solid fa-comments', label: 'Whisper NLP' },
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
  return (
    <div className="docs-layout">
      {/* Header */}
      <div className="docs-header">
        <div className="docs-tag">State of the Art Deepfake Defense</div>
        <h1 className="docs-title">Engine 2: 4-Pillar Multimodal SOTA</h1>
        <p className="docs-desc">
          State-of-the-art defense against zero-shot voice clones, neural vocoders (HiFi-GAN, WaveGlow), and Spanish-language synthetic conversational attacks.
        </p>
      </div>

      {/* Interactive Workbench */}
      <div className="docs-section">
        <div className="docs-section-title">
          <i className="fa-solid fa-terminal text-[#FF5500]"></i>
          <span>Live Inference Workbench</span>
        </div>
        <EngineCard config={ENGINE_2_CONFIG} onNavigate={onNavigate} />
      </div>

      {/* 4 Pillars Deep Dive */}
      <div className="docs-section">
        <div className="docs-section-title">
          <i className="fa-solid fa-layer-group text-[#FF5500]"></i>
          <span>The 4 Pillars of Multimodal Representation</span>
        </div>
        <div className="docs-grid-cards">
          <div className="docs-card">
            <div className="docs-card-title">Pillar 1: AASIST Graph Neural Net</div>
            <div className="docs-card-text">
              Heterogeneous Graph Attention Network modeling spectral-temporal correlations across sub-band spectro-temporal graphs to expose synthetic artifacts.
            </div>
          </div>
          <div className="docs-card">
            <div className="docs-card-title">Pillar 2: RawNet2 SincNet</div>
            <div className="docs-card-text">
              Learnable parametric bandpass Sinc filters processing raw uncompressed waveforms directly, catching phase irregularities introduced by neural vocoders.
            </div>
          </div>
          <div className="docs-card">
            <div className="docs-card-title">Pillar 3: 136-Dim Acoustic DSP</div>
            <div className="docs-card-text">
              Formant trajectory tracking (F1-F4), Jitter, Shimmer, Harmonic-to-Noise Ratio (HNR), and high-resolution Constant Q-Transform (CQT) spectral phase.
            </div>
          </div>
          <div className="docs-card">
            <div className="docs-card-title">Pillar 4: Whisper Spanish NLP</div>
            <div className="docs-card-text">
              Multilingual Whisper encoder extracting Spanish phoneme timing, pause distributions, and conversational hesitation patterns that synthetic models fail to reproduce.
            </div>
          </div>
        </div>
      </div>

      {/* Fusion & Classification */}
      <div className="docs-section">
        <div className="docs-section-title">
          <i className="fa-solid fa-brain text-[#FF5500]"></i>
          <span>XGBoost Quad Fusion</span>
        </div>
        <div className="docs-grid-cards">
          <div className="docs-card">
            <div className="docs-card-title">296-Dimensional Embeddings</div>
            <div className="docs-card-text">
              Concat-fused feature vectors projected through an optimized XGBoost Classifier trained on cross-domain anti-spoofing benchmarks (ASVspoof 2019/2021 & In-the-Wild).
            </div>
          </div>
          <div className="docs-card">
            <div className="docs-card-title">Calibration & Audit</div>
            <div className="docs-card-text">
              Platt scaling calibration ensuring mathematically sound probability scores, with optional asynchronous audit telemetry logging.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
