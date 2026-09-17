"""
Anti-Spoofing Detection Engines Package
- BaselineEngine: Fast Acoustic & Conversational Timing + AudioCNN (Lightweight, ~30ms)
- MultimodalEngine: 4-Pillar Deep SOTA (AASIST GNN + RawNet2 + 136-dim DSP + Whisper NLP + XGBoost)
- UnifiedOrchestrator: Dual Engine Orchestration & Comparison Gateway
"""
from src.engines.baseline_engine import BaselineEngine
from src.engines.multimodal_engine import MultimodalEngine
from src.engines.unified_orchestrator import UnifiedOrchestrator

__all__ = ["BaselineEngine", "MultimodalEngine", "UnifiedOrchestrator"]
