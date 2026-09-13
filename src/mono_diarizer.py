import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import soundfile as sf
import librosa
from sklearn.cluster import AgglomerativeClustering, KMeans

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FastDialogueDiarizer:
    """
    Ultra-Fast Diarizer and Transcriber for raw mono/mixed 2-person conversations.
    - Uses faster-whisper INT8 with greedy search (beam_size=1) and VAD filtering.
    - Fast vectorized MFCC/Spectral voiceprint extraction.
    """

    def __init__(self, model_size: str = "tiny", device: str = "cpu", cpu_threads: int = 8):
        self.device = device
        self.model_size = model_size
        self.cpu_threads = cpu_threads
        self._model = None
        self._init_model()

    def _init_model(self):
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="int8",
                cpu_threads=self.cpu_threads
            )
            print(f"✓ Loaded faster-whisper ({self.model_size}, INT8, {self.cpu_threads} threads)")
        except Exception as e:
            print(f"⚠️ Error loading faster-whisper: {e}")

    def extract_fast_voiceprint(self, audio_seg: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Lightweight, vector-optimized 28-dimensional acoustic vocal embedding."""
        if len(audio_seg) < int(sr * 0.15):
            return np.zeros(28, dtype=np.float32)

        # Fast MFCCs (13 coefficients mean + std) -> 26 dims
        mfccs = librosa.feature.mfcc(y=audio_seg, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        # Fast Spectral Centroid (mean + std) -> 2 dims
        cent = librosa.feature.spectral_centroid(y=audio_seg, sr=sr, n_fft=512, hop_length=256)
        cent_mean = np.mean(cent)
        cent_std = np.std(cent)

        feat = np.concatenate([mfcc_mean, mfcc_std, [cent_mean, cent_std]]).astype(np.float32)
        return np.nan_to_num(feat)

    def process_conversation(self, audio_path: str) -> Dict[str, Any]:
        t_start = time.perf_counter()

        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim == 2:
            data = np.mean(data, axis=1)

        if sr != 16000:
            audio_16k = librosa.resample(data, orig_sr=sr, target_sr=16000).astype(np.float32)
            sr = 16000
        else:
            audio_16k = data.astype(np.float32)

        total_duration = len(audio_16k) / sr

        print(f"\n================================================================================")
        print(f"🎙️ FAST TRANSCRIPTION & DIARIZATION: {Path(audio_path).stem} ({total_duration:.1f}s)")
        print(f"================================================================================")

        t_asr_start = time.perf_counter()

        # 1. Fast Greedy Transcription with Silero VAD filtering
        segments_raw, info = self._model.transcribe(
            audio_16k,
            language="es",
            beam_size=1,            # Greedy decoding = maximum speed
            best_of=1,
            temperature=0.0,
            vad_filter=True,        # Skip silent parts
            vad_parameters=dict(min_silence_duration_ms=300),
            condition_on_previous_text=False
        )

        segments = []
        embeddings = []

        for seg in segments_raw:
            text = seg.text.strip()
            if not text:
                continue

            start_idx = int(seg.start * sr)
            end_idx = int(seg.end * sr)
            seg_audio = audio_16k[start_idx:end_idx]

            emb = self.extract_fast_voiceprint(seg_audio, sr=sr)

            segments.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "duration": round(seg.end - seg.start, 2),
                "text": text
            })
            embeddings.append(emb)

        t_asr_elapsed = time.perf_counter() - t_asr_start

        if not segments:
            print("No speech detected.")
            return {}

        # 2. Cluster voiceprints into 2 Speakers
        if len(segments) >= 2:
            X = np.array(embeddings)
            norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
            X_norm = X / norm

            try:
                clustering = AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average")
                labels = clustering.fit_predict(X_norm)
            except Exception:
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=5)
                labels = kmeans.fit_predict(X_norm)
        else:
            labels = [0] * len(segments)

        # 3. Assemble Output Dialogue
        dialogue = []
        spk0_texts = []
        spk1_texts = []

        for seg, spk_id in zip(segments, labels):
            spk_label = int(spk_id)
            if spk_label == 0:
                spk0_texts.append(seg["text"])
                prefix = "👤 [Speaker 1]"
            else:
                spk1_texts.append(seg["text"])
                prefix = "🗣️ [Speaker 2]"

            print(f"[{seg['start']:05.1f}s - {seg['end']:05.1f}s] {prefix}: {seg['text']}")

            dialogue.append({
                "start": seg["start"],
                "end": seg["end"],
                "duration": seg["duration"],
                "speaker": f"Speaker {spk_label + 1}",
                "text": seg["text"]
            })

        t_total_elapsed = time.perf_counter() - t_start

        print(f"--------------------------------------------------------------------------------")
        print(f"⚡ Audio Duration: {total_duration:.1f}s | Processing Time: {t_total_elapsed:.2f}s (ASR: {t_asr_elapsed:.2f}s)")
        print(f"🚀 Real-Time Factor (RTF): {t_total_elapsed / total_duration:.3f}x ({(total_duration / t_total_elapsed):.1f}x faster than real-time)")
        print(f"================================================================================\n")

        return {
            "call_id": Path(audio_path).stem,
            "duration_s": round(total_duration, 2),
            "total_turns": len(dialogue),
            "processing_time_s": round(t_total_elapsed, 2),
            "speaker_1_transcript": " ".join(spk0_texts).strip(),
            "speaker_2_transcript": " ".join(spk1_texts).strip(),
            "dialogue": dialogue
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--model", type=str, default="tiny")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    diarizer = FastDialogueDiarizer(model_size=args.model)
    res = diarizer.process_conversation(args.audio)

    if args.out and res:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved results to: {args.out}")


if __name__ == "__main__":
    main()
