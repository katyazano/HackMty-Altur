import numpy as np

class BiometricsEngine:
    """
    Engine for Speaker Verification (Voice Biometrics) and Anti-Spoofing / Deepfake Risk Assessment.
    """

    def __init__(self, speaker_similarity_threshold: float = 0.70):
        self.speaker_similarity_threshold = speaker_similarity_threshold

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calculates cosine similarity between two feature vectors.
        """
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        similarity = dot_product / (norm_v1 * norm_v2)
        return float(np.clip(similarity, -1.0, 1.0))

    def verify_speaker(self, enrolled_mfcc: np.ndarray, candidate_mfcc: np.ndarray) -> dict:
        """
        Compares candidate speaker's MFCC embedding against enrolled speaker's template.
        """
        similarity = self.cosine_similarity(enrolled_mfcc, candidate_mfcc)
        # Convert [-1, 1] cosine range to percentage [0%, 100%]
        match_percentage = round(float((similarity + 1) / 2 * 100), 2)
        is_match = similarity >= self.speaker_similarity_threshold

        return {
            "is_match": bool(is_match),
            "similarity_score": round(similarity, 4),
            "confidence_percentage": match_percentage,
            "threshold_used": self.speaker_similarity_threshold
        }

    def assess_anti_spoofing(self, lfcc_features: np.ndarray, spectral_features: dict) -> dict:
        """
        Assesses whether the audio is a real human voice vs. a synthetic / AI-generated voice or replay attack.
        Uses LFCC variance anomalies and high-frequency spectral ratios.
        """
        # Synthetic TTS voices often have unusually low variance in higher LFCC cepstral bins
        lfcc_std_sum = float(np.sum(lfcc_features[len(lfcc_features)//2:]))
        hf_energy = spectral_features.get("high_freq_energy_ratio", 0.0)

        # Baseline heuristic calculation for mini demo
        risk_points = 0.0

        # Heuristic 1: Extremely smooth/flat high-frequency LFCC variance (vocoder artifact)
        if lfcc_std_sum < 0.5:
            risk_points += 40.0
        elif lfcc_std_sum < 1.0:
            risk_points += 20.0

        # Heuristic 2: Abnormally low or suppressed high-frequency energy ratio (over-compressed TTS audio)
        if hf_energy < 0.005 or hf_energy > 0.35:
            risk_points += 35.0

        # Heuristic 3: Spectral Centroid sharpness anomaly
        centroid = spectral_features.get("spectral_centroid_hz", 1500)
        if centroid < 1000 or centroid > 3800:
            risk_points += 25.0

        risk_score = min(100.0, max(0.0, risk_points))
        is_spoof = risk_score >= 50.0

        verdict = "SYNTHETIC / AI VOICE DETECTED" if is_spoof else "AUTHENTIC HUMAN VOICE"

        return {
            "verdict": verdict,
            "is_spoof": is_spoof,
            "risk_score_percentage": round(risk_score, 1),
            "lfcc_variance_metric": round(lfcc_std_sum, 4),
            "high_freq_energy_ratio": hf_energy
        }
