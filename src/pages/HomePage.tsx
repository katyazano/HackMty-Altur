import React from 'react';
import { HeroBanner } from '../components/HeroBanner';
import { TeamRail } from '../components/TeamRail';
import { EngineCard } from '../components/EngineCard';
import { EngineConfig } from '../types/detection';

const ENGINE_1_CONFIG: EngineConfig = {
  id: 'baseline',
  prefix: 'e1',
  titleTag: 'Engine 1 • Real-Time Baseline',
  title: 'Ultra-Fast Baseline (64 dimensions)',
  description:
    'Frontline voice biometrics (~700ms CPU) combining acoustic DSP physics with micro-segment convolutional scoring.',
  endpoint: '/detect/baseline',
  chips: [
    { icon: 'fa-solid fa-microchip', label: 'SpoofCNN (AudioCNN)', highlight: true },
    { icon: 'fa-solid fa-tree', label: 'HistGradientBoosting' },
    { icon: 'fa-solid fa-bolt', label: '~700ms Latency', highlight: true },
  ],
  exampleResponse: {
    is_synthetic: false,
    confidence: 0.982,
  },
  docsUrl: '#/engine1',
};

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

interface HomePageProps {
  onNavigate?: (route: 'home' | 'engine1' | 'engine2') => void;
}

export const HomePage: React.FC<HomePageProps> = ({ onNavigate }) => {
  return (
    <div className="main-content-layout">
      {/* Top Row: Hero Stage + Team Ribbon + Team Member Monograms Rail */}
      <div className="hero-and-rail-grid">
        <HeroBanner />
        <TeamRail />
      </div>

      {/* Dual Workbench Comparison: Engine 1 & Engine 2 Split Section */}
      <div className="hero-split-section">
        <EngineCard
          config={ENGINE_1_CONFIG}
          onNavigate={onNavigate}
          className="split-col-left"
        />
        <EngineCard
          config={ENGINE_2_CONFIG}
          onNavigate={onNavigate}
          className="split-col-right"
        />
      </div>
    </div>
  );
};
