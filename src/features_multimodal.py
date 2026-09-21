import os
import sys
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.aasist import AASIST
from src.models.rawnet2 import RawNet2


class AcousticFeatureExtractor:
    """
    Extracts high-dimensional acoustic, spectral, and prosodic DSP features
    specialized for speech anti-spoofing and vocoder artifact detection.
    """

    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 20):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.feature_names: List[str] = self._generate_feature_names()

    def _generate_feature_names(self) -> List[str]:
        names = []
        # MFCC stats
        for i in range(1, self.n_mfcc + 1):
            names.extend([f"mfcc_{i}_mean", f"mfcc_{i}_std"])
        # MFCC Delta stats
        for i in range(1, self.n_mfcc + 1):
            names.extend([f"mfcc_delta_{i}_mean", f"mfcc_delta_{i}_std"])
        # Spectral features
        names.extend(["spec_centroid_mean", "spec_centroid_std"])
        names.extend(["spec_bandwidth_mean", "spec_bandwidth_std"])
        names.extend(["spec_rolloff_mean", "spec_rolloff_std"])
        names.extend(["spec_flatness_mean", "spec_flatness_std"])
        for i in range(7):
            names.extend([f"spec_contrast_{i}_mean", f"spec_contrast_{i}_std"])
        # Temporal & Energy
        names.extend(["zcr_mean", "zcr_std"])
        names.extend(["rms_mean", "rms_std"])
        # Chroma
        for i in range(12):
            names.extend([f"chroma_{i}_mean", f"chroma_{i}_std"])
        # Prosody / Pitch
        names.extend(["pitch_f0_mean", "pitch_f0_std", "voiced_fraction"])
        # Telephony & Vocoder Cues
        names.extend(["hf_energy_ratio", "clipping_ratio", "crest_factor"])
        return names

    def extract_features(self, y: np.ndarray, sr: Optional[int] = None, max_samples: int = 64000) -> np.ndarray:
        """
        Computes 1D feature vector from audio waveform array.
        Uses standard 4-second (64k samples @ 16kHz) central slice for fast, robust extraction.
        """
        if sr is None:
            sr = self.sample_rate

        # Ensure mono float32
        if y.ndim > 1:
            y = y[:, 0]
        y = y.astype(np.float32)

        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate).astype(np.float32)
            sr = self.sample_rate


        # Standard 4-second central representation (consistent with AASIST/RawNet2 input)
        nb_samples = len(y)
        if max_samples and nb_samples > max_samples:
            mid = nb_samples // 2
            half = max_samples // 2
            y = y[mid - half : mid + half]
        elif max_samples and nb_samples < max_samples:
            y = np.pad(y, (0, max_samples - nb_samples), mode="constant")

        # Handle silent inputs
        if len(y) < sr * 0.1 or np.max(np.abs(y)) < 1e-6:
            return np.zeros(len(self.feature_names), dtype=np.float32)

        features = []

        # 1. MFCCs & Deltas
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        for i in range(self.n_mfcc):
            features.extend([np.mean(mfcc[i]), np.std(mfcc[i])])
        for i in range(self.n_mfcc):
            features.extend([np.mean(mfcc_delta[i]), np.std(mfcc_delta[i])])

        # 2. Spectral Centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.extend([np.mean(cent), np.std(cent)])

        # 3. Spectral Bandwidth
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.extend([np.mean(bw), np.std(bw)])

        # 4. Spectral Roll-off
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.extend([np.mean(rolloff), np.std(rolloff)])

        # 5. Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=y)
        features.extend([np.mean(flatness), np.std(flatness)])

        # 6. Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for i in range(contrast.shape[0]):
            features.extend([np.mean(contrast[i]), np.std(contrast[i])])

        # 7. Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=y)
        features.extend([np.mean(zcr), np.std(zcr)])

        # 8. RMS Energy
        rms = librosa.feature.rms(y=y)
        features.extend([np.mean(rms), np.std(rms)])

        # 9. Chroma STFT
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for i in range(12):
            features.extend([np.mean(chroma[i]), np.std(chroma[i])])

        # 10. Pitch (F0) Estimation via Yin
        try:
            f0 = librosa.yin(y=y, fmin=60, fmax=350, sr=sr, frame_length=1024, hop_length=1024)
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 0:
                features.extend([np.mean(valid_f0), np.std(valid_f0), len(valid_f0) / len(f0)])
            else:
                features.extend([0.0, 0.0, 0.0])
        except Exception:
            features.extend([0.0, 0.0, 0.0])

        # 11. Telephony & Vocoder Cues
        # Energy above 3.4kHz (narrowband boundary)
        stft = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        hf_mask = freqs > 3400
        hf_energy = np.sum(stft[hf_mask, :])
        total_energy = np.sum(stft) + 1e-9
        hf_ratio = float(hf_energy / total_energy)
        features.append(hf_ratio)

        # Clipping Ratio
        clip_ratio = float(np.sum(np.abs(y) >= 0.98) / len(y))
        features.append(clip_ratio)

        # Crest Factor (Peak to RMS)
        mean_rms = np.mean(rms) + 1e-9
        crest = float(np.max(np.abs(y)) / mean_rms)
        features.append(crest)

        feat_arr = np.array(features, dtype=np.float32)
        # Replace any infs or nans with 0
        feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)
        return feat_arr


class AASISTEmbeddingExtractor:
    """
    Extracts deep 128-dim graph embeddings and logit outputs from a trained AASIST model.
    """

    def __init__(self, model: Optional[AASIST] = None, weights_path: Optional[str] = None, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = AASIST().to(self.device)
            if weights_path and os.path.exists(weights_path):
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def extract_from_waveform(self, waveform: torch.Tensor, target_samples: int = 64000) -> np.ndarray:
        """
        Runs forward pass on waveform tensor and returns:
        [latent_embedding (128-d), logits (2-d), softmax_probs (2-d)] -> total 132 dimensions.
        """
        self.model.eval()
        with torch.no_grad():
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            waveform = waveform.to(self.device)

            nb_samples = waveform.shape[-1]
            if nb_samples > target_samples:
                mid = nb_samples // 2
                waveform_in = waveform[:, mid - target_samples // 2 : mid + target_samples // 2]
            elif nb_samples < target_samples:
                waveform_in = F.pad(waveform, (0, target_samples - nb_samples))
            else:
                waveform_in = waveform

            feat, logits = self.model.extract_embeddings(waveform_in)
            probs = F.softmax(logits, dim=1)

            feat_np = feat.squeeze(0).cpu().numpy()
            logits_np = logits.squeeze(0).cpu().numpy()
            probs_np = probs.squeeze(0).cpu().numpy()

            combined = np.concatenate([feat_np, logits_np, probs_np], axis=0).astype(np.float32)
            return combined


class RawNet2ScoreExtractor:
    """
    Extracts raw waveform predictions and logits from the RawNet2 architecture.
    """

    def __init__(self, model: Optional[RawNet2] = None, weights_path: Optional[str] = None, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = RawNet2().to(self.device)
            if weights_path and os.path.exists(weights_path):
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def extract_from_waveform(self, waveform: torch.Tensor, target_samples: int = 64000) -> np.ndarray:
        """
        Runs forward pass on waveform tensor and returns:
        [rawnet2_logits (2-d), rawnet2_prob_human (1-d), rawnet2_prob_synthetic (1-d)] -> total 4 dimensions.
        """
        self.model.eval()
        with torch.no_grad():
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            waveform = waveform.to(self.device)

            nb_samples = waveform.shape[-1]
            if nb_samples > target_samples:
                mid = nb_samples // 2
                waveform_in = waveform[:, mid - target_samples // 2 : mid + target_samples // 2]
            elif nb_samples < target_samples:
                waveform_in = F.pad(waveform, (0, target_samples - nb_samples))
            else:
                waveform_in = waveform

            logits = self.model(waveform_in)
            probs = F.softmax(logits, dim=1)

            logits_np = logits.squeeze(0).cpu().numpy()
            probs_np = probs.squeeze(0).cpu().numpy()

            return np.concatenate([logits_np, probs_np], axis=0).astype(np.float32)


class TripleEnsembleFeaturePipeline:
    """
    Unified triple-extractor combining:
    1. AASIST spectro-temporal graph representations (132-dim)
    2. RawNet2 raw-waveform logits & probabilities (4-dim)
    3. Acoustic DSP & Spectral features (136-dim)
    Total combined dimension = 272 features.
    """

    def __init__(
        self,
        aasist_extractor: AASISTEmbeddingExtractor,
        rawnet2_extractor: RawNet2ScoreExtractor,
        acoustic_extractor: Optional[AcousticFeatureExtractor] = None
    ):
        self.aasist_extractor = aasist_extractor
        self.rawnet2_extractor = rawnet2_extractor
        self.acoustic_extractor = acoustic_extractor or AcousticFeatureExtractor()

    def extract_combined_vector(self, waveform_tensor: torch.Tensor, sample_rate: int = 16000) -> np.ndarray:
        # 1. AASIST (132-dim)
        aasist_feats = self.aasist_extractor.extract_from_waveform(waveform_tensor)
        # 2. RawNet2 (4-dim)
        rawnet_feats = self.rawnet2_extractor.extract_from_waveform(waveform_tensor)
        # 3. Acoustic DSP (136-dim)
        waveform_np = waveform_tensor.detach().cpu().squeeze().numpy()
        acoustic_feats = self.acoustic_extractor.extract_features(waveform_np, sr=sample_rate)

        return np.concatenate([aasist_feats, rawnet_feats, acoustic_feats], axis=0)


class WhisperMultimodalExtractor:
    """
    Multimodal ASR Extractor:
    1. Encoder Transformer Latents: 16 dimensions (pooled attention embeddings)
    2. Decoder Token Uncertainty: 4 dimensions (avg_logprob, min_logprob, compression_ratio, no_speech_prob)
    3. Semantic & Disfluency NLP: 4 dimensions (filler_density, speech_rate_wps, lexical_diversity_ttr, authenticity_score)
    Total: 24 dimensions + transcribed turns for UI.
    """

    HUMAN_COLLOQUIAL_FILLERS = [
        r"\beste\b", r"\beste\.\.\.", r"\beh\b", r"\bem\b", r"\bah\b",
        r"\bbueno\b", r"\bo sea\b", r"\bajá\b", r"\bpues\b", r"\bmira\b",
        r"\ba ver\b", r"\bfíjate\b", r"\bverdad\b", r"\bósea\b", r"\bahorita\b"
    ]

    ROBOTIC_LLM_PATTERNS = [
        r"\bentiendo perfectamente\b", r"\bcon gusto le asisto\b",
        r"\ben qué más puedo servirle\b", r"\bcomo modelo de lenguaje\b",
        r"\bsoy un asistente virtual\b", r"\bgracias por comunicarse\b"
    ]

    def __init__(self, model_size: str = "tiny", device: str = "cpu"):
        self.device = device
        self.model_size = model_size
        self.whisper_pt_model = None
        self.faster_whisper_model = None
        self._init_models()

    def _init_models(self):
        # 1. PyTorch Whisper for Encoder Latents
        try:
            import whisper
            self.whisper_pt_model = whisper.load_model(self.model_size, device=self.device)
            self.whisper_pt_model.eval()
        except Exception as e:
            pass

        # 2. Faster-Whisper for high-speed transcription & token stats
        try:
            from faster_whisper import WhisperModel
            self.faster_whisper_model = WhisperModel(
                self.model_size, device=self.device, compute_type="int8", cpu_threads=4
            )
        except Exception as e:
            pass

    def extract_features(
        self,
        audio_16k: np.ndarray,
        sr: int = 16000,
        return_transcript: bool = False
    ) -> Tuple[np.ndarray, Optional[List[Dict[str, Any]]]]:
        """
        Extracts 24-dimensional multimodal Whisper feature vector + optional transcript.
        """
        if len(audio_16k) == 0:
            zeros = np.zeros(24, dtype=np.float32)
            return (zeros, []) if return_transcript else (zeros, None)

        # ----------------------------------------------------
        # 1. Encoder Latents (16 dims)
        # ----------------------------------------------------
        latents_16 = np.zeros(16, dtype=np.float32)
        if self.whisper_pt_model is not None:
            try:
                import whisper
                audio_tensor = torch.from_numpy(audio_16k.astype(np.float32))
                mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio_tensor))
                with torch.no_grad():
                    enc_out = self.whisper_pt_model.encoder(mel.unsqueeze(0).to(self.device))
                    # enc_out shape: [1, 1500, hidden_dim]
                    mean_p = torch.mean(enc_out, dim=1).squeeze().cpu().numpy()
                    std_p = torch.std(enc_out, dim=1).squeeze().cpu().numpy()

                    hidden_dim = len(mean_p)
                    stride = max(1, hidden_dim // 8)
                    latents_16 = np.concatenate([mean_p[::stride][:8], std_p[::stride][:8]]).astype(np.float32)
            except Exception:
                pass

        # ----------------------------------------------------
        # 2. Decoder & Transcription Pass
        # ----------------------------------------------------
        turns = []
        avg_logprobs = []
        compression_ratios = []
        no_speech_probs = []

        full_text_list = []

        if self.faster_whisper_model is not None:
            try:
                segments_iter, _ = self.faster_whisper_model.transcribe(
                    audio_16k,
                    language="es",
                    initial_prompt="Llamada bancaria de atención en México. Cuenta, saldo, tarjeta, NIP, este, o sea...",
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=250),
                    condition_on_previous_text=False
                )
                for seg in segments_iter:
                    t_text = seg.text.strip()
                    if t_text:
                        turns.append({
                            "start_time": round(float(seg.start), 2),
                            "end_time": round(float(seg.end), 2),
                            "text": t_text,
                            "avg_logprob": round(float(seg.avg_logprob), 3) if hasattr(seg, "avg_logprob") else 0.0,
                            "no_speech_prob": round(float(seg.no_speech_prob), 3) if hasattr(seg, "no_speech_prob") else 0.0,
                            "compression_ratio": round(float(seg.compression_ratio), 3) if hasattr(seg, "compression_ratio") else 1.0
                        })
                        full_text_list.append(t_text)
                        if hasattr(seg, "avg_logprob"):
                            avg_logprobs.append(float(seg.avg_logprob))
                        if hasattr(seg, "compression_ratio"):
                            compression_ratios.append(float(seg.compression_ratio))
                        if hasattr(seg, "no_speech_prob"):
                            no_speech_probs.append(float(seg.no_speech_prob))
            except Exception:
                pass

        # ----------------------------------------------------
        # 3. Decoder Token Stats (4 dims)
        # ----------------------------------------------------
        mean_logprob = float(np.mean(avg_logprobs)) if avg_logprobs else -1.5
        min_logprob = float(np.min(avg_logprobs)) if avg_logprobs else -3.0
        mean_comp = float(np.mean(compression_ratios)) if compression_ratios else 1.0
        mean_no_speech = float(np.mean(no_speech_probs)) if no_speech_probs else 0.1

        decoder_feats_4 = np.array([mean_logprob, min_logprob, mean_comp, mean_no_speech], dtype=np.float32)

        # ----------------------------------------------------
        # 4. Semantic & Disfluency Metrics (4 dims)
        # ----------------------------------------------------
        import re
        full_text = " ".join(full_text_list).lower().strip()
        words = re.findall(r"\b\w+\b", full_text)
        total_words = len(words)
        duration_sec = max(0.5, len(audio_16k) / sr)

        if total_words == 0:
            filler_density = 0.0
            speech_rate_wps = 0.0
            ttr = 0.0
            authenticity_score = 0.50
        else:
            filler_count = 0
            for pat in self.HUMAN_COLLOQUIAL_FILLERS:
                filler_count += len(re.findall(pat, full_text))

            robotic_count = 0
            for pat in self.ROBOTIC_LLM_PATTERNS:
                robotic_count += len(re.findall(pat, full_text))

            filler_density = filler_count / total_words
            speech_rate_wps = total_words / duration_sec
            ttr = len(set(words)) / total_words
            authenticity_score = 0.50 + (filler_density * 2.5) - (robotic_count * 0.25)
            authenticity_score = float(np.clip(authenticity_score, 0.05, 0.98))

        semantic_feats_4 = np.array([
            filler_density, speech_rate_wps, ttr, authenticity_score
        ], dtype=np.float32)

        whisper_24 = np.concatenate([latents_16, decoder_feats_4, semantic_feats_4], axis=0).astype(np.float32)
        whisper_24 = np.nan_to_num(whisper_24)

        if return_transcript:
            return whisper_24, turns
        return whisper_24, None


class QuadEnsembleFeaturePipeline:
    """
    Master 4-Pillar Multimodal Pipeline:
    1. AASIST GNN (132-dim)
    2. RawNet2 SincNet (4-dim)
    3. Acoustic DSP (136-dim)
    4. Whisper Multimodal (24-dim: 16 Latents + 4 Decoder + 4 Semantics)
    Total combined dimensions = 296 features.
    """

    def __init__(
        self,
        aasist_extractor: AASISTEmbeddingExtractor,
        rawnet2_extractor: RawNet2ScoreExtractor,
        acoustic_extractor: Optional[AcousticFeatureExtractor] = None,
        whisper_extractor: Optional[WhisperMultimodalExtractor] = None
    ):
        self.aasist_extractor = aasist_extractor
        self.rawnet2_extractor = rawnet2_extractor
        self.acoustic_extractor = acoustic_extractor or AcousticFeatureExtractor()
        self.whisper_extractor = whisper_extractor or WhisperMultimodalExtractor()

    def extract_combined_vector(
        self,
        waveform_tensor: torch.Tensor,
        sample_rate: int = 16000,
        return_transcript: bool = False
    ) -> Tuple[np.ndarray, Optional[List[Dict[str, Any]]]]:
        # 1. AASIST (132-dim)
        aasist_feats = self.aasist_extractor.extract_from_waveform(waveform_tensor)
        # 2. RawNet2 (4-dim)
        rawnet_feats = self.rawnet2_extractor.extract_from_waveform(waveform_tensor)
        # 3. Acoustic DSP (136-dim)
        waveform_np = waveform_tensor.detach().cpu().squeeze().numpy()
        acoustic_feats = self.acoustic_extractor.extract_features(waveform_np, sr=sample_rate)
        # 4. Whisper Multimodal (24-dim)
        whisper_feats, turns = self.whisper_extractor.extract_features(
            waveform_np, sr=sample_rate, return_transcript=return_transcript
        )

        quad_vector = np.concatenate([aasist_feats, rawnet_feats, acoustic_feats, whisper_feats], axis=0)
        return (quad_vector, turns) if return_transcript else (quad_vector, None)

