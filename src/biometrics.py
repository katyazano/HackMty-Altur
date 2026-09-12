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
        Uses spectral rolloff, spectral flatness, LFCC subband variance, and centroid distributions.
        """
        lfcc_std_sum = float(np.sum(lfcc_features[len(lfcc_features)//2:]))
        hf_energy = spectral_features.get("high_freq_energy_ratio", 0.0)
        centroid = spectral_features.get("spectral_centroid_hz", 1500)
        rolloff = spectral_features.get("spectral_rolloff_hz", 1500)
        flatness = spectral_features.get("spectral_flatness", 0.0001)

        risk_points = 0.0

        # Heuristic 1: High Spectral Rolloff (High-frequency synthetic vocoder excitation)
        if rolloff > 3000:
            risk_points += 45.0
        elif rolloff > 1500:
            risk_points += 30.0
        elif rolloff > 900:
            risk_points += 15.0

        # Heuristic 2: Elevated Spectral Flatness (Wiener entropy from vocoder/noise smearing)
        if flatness > 0.0003:
            risk_points += 40.0
        elif flatness > 0.00004:
            risk_points += 25.0

        # Heuristic 3: Spectral Centroid anomaly (Vocoder phase buzz)
        if centroid > 1800:
            risk_points += 25.0
        elif centroid > 900:
            risk_points += 15.0

        # Heuristic 4: Extremely smooth/flat high-frequency LFCC variance
        if lfcc_std_sum < 0.2:
            risk_points += 20.0

        risk_score = min(95.0, max(5.0, risk_points))
        real_score = round(100.0 - risk_score, 1)
        is_spoof = real_score < 50.0

        verdict = "SYNTHETIC / AI VOICE DETECTED" if is_spoof else "AUTHENTIC HUMAN VOICE"

        return {
            "verdict": verdict,
            "is_spoof": is_spoof,
            "real_score_pct": real_score,
            "real_score_percentage": real_score,
            "risk_score_percentage": round(risk_score, 1),
            "lfcc_variance_metric": round(lfcc_std_sum, 4),
            "spectral_flatness": flatness,
            "spectral_rolloff_hz": rolloff,
            "high_freq_energy_ratio": hf_energy
        }
