"""
features.py — Unified Feature Extraction Module for Altur Voice Anti-Spoofing.
Re-exports both Baseline feature extraction routines and 4-Pillar Multimodal extractors.
"""
# 1. Baseline Extraction Routines
from src.features_baseline import (
    load_stereo_from_bytes,
    energy_vad,
    acoustic_features as baseline_acoustic_features,
    timing_features,
    timing_features_from_turns,
    extract_from_arrays,
    extract_all,
    TARGET_SR,
    N_MFCC,
    FRAME_MS,
    HOP_MS,
    MAX_RESPONSE_GAP,
    MERGE_GAP,
    MIN_SEG
)

# 2. Multimodal 4-Pillar Extractors
from src.features_multimodal import (
    AcousticFeatureExtractor,
    AASISTEmbeddingExtractor,
    RawNet2ScoreExtractor,
    TripleEnsembleFeaturePipeline,
    WhisperMultimodalExtractor,
    QuadEnsembleFeaturePipeline
)

__all__ = [
    # Baseline
    "load_stereo_from_bytes",
    "energy_vad",
    "baseline_acoustic_features",
    "timing_features",
    "timing_features_from_turns",
    "extract_from_arrays",
    "extract_all",
    "TARGET_SR",
    "N_MFCC",
    "FRAME_MS",
    "HOP_MS",
    "MAX_RESPONSE_GAP",
    "MERGE_GAP",
    "MIN_SEG",
    # Multimodal
    "AcousticFeatureExtractor",
    "AASISTEmbeddingExtractor",
    "RawNet2ScoreExtractor",
    "TripleEnsembleFeaturePipeline",
    "WhisperMultimodalExtractor",
    "QuadEnsembleFeaturePipeline",
]
