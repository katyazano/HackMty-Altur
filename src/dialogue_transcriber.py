import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dynamic_separator import dynamic_extract_caller_speech


class ConversationalTurnTranscriber:
    """
    Processes 2-party stereo call audios, detects who is talking (Channel 0 = Caller vs Channel 1 = Agent),
    slices conversational turns chronologically, and transcribes each turn using ASR (Whisper / faster-whisper / HuggingFace).
    """

    def __init__(self, asr_model_size: str = "tiny", device: str = "cpu"):
        self.device = device
        self.asr_model_size = asr_model_size
        self._asr_engine = None
        self._asr_type = None

        self._init_asr_engine()

    def _init_asr_engine(self):
        """Initializes the fastest available ASR backend."""
        # 1. Try faster-whisper (CTranslate2 INT8)
        try:
            from faster_whisper import WhisperModel
            self._asr_engine = WhisperModel(self.asr_model_size, device=self.device, compute_type="int8", cpu_threads=4)
            self._asr_type = "faster_whisper"
            print(f"✓ Loaded faster-whisper ({self.asr_model_size}) on {self.device}")
            return
        except ImportError:
            pass

        # 2. Try standard OpenAI Whisper
        try:
            import whisper
            self._asr_engine = whisper.load_model(self.asr_model_size, device=self.device)
            self._asr_type = "openai_whisper"
            print(f"✓ Loaded openai-whisper ({self.asr_model_size}) on {self.device}")
            return
        except ImportError:
            pass

        # 3. Try HuggingFace Transformers pipeline
        try:
            from transformers import pipeline
            self._asr_engine = pipeline(
                "automatic-speech-recognition",
                model=f"openai/whisper-{self.asr_model_size}",
                device=self.device
            )
            self._asr_type = "transformers"
            print(f"✓ Loaded HuggingFace transformers pipeline (whisper-{self.asr_model_size})")
            return
        except ImportError:
            pass

        print("⚠️ No ASR engine installed (faster-whisper, whisper, or transformers). Install with: pip install faster-whisper")
        self._asr_type = None

    def transcribe_audio_segment(self, audio_16k: np.ndarray) -> str:
        """Transcribes a single mono 16kHz float32 audio segment."""
        if self._asr_engine is None or len(audio_16k) == 0:
            return ""

        # Skip near-silent audio
        if np.max(np.abs(audio_16k)) < 1e-4:
            return ""

        try:
            if self._asr_type == "faster_whisper":
                segments, _ = self._asr_engine.transcribe(
                    audio_16k,
                    language="es",
                    beam_size=1,
                    temperature=0.0,
                    without_timestamps=True
                )
                return " ".join([s.text for s in segments]).strip()

            elif self._asr_type == "openai_whisper":
                res = self._asr_engine.transcribe(
                    audio_16k,
                    language="es",
                    fp16=False,
                    beam_size=1,
                    temperature=0.0,
                    without_timestamps=True
                )
                return res.get("text", "").strip() if isinstance(res, dict) else ""

            elif self._asr_type == "transformers":
                res = self._asr_engine({"raw": audio_16k, "sampling_rate": 16000})
                return res.get("text", "").strip() if isinstance(res, dict) else ""

        except Exception as e:
            return f"[Transcription error: {e}]"

        return ""

    def extract_turns_from_audio(
        self,
        audio_data: np.ndarray,
        sr: int,
        turns_meta_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extracts chronological turns.
        If turns.json is provided, uses ground-truth timestamps.
        Otherwise, dynamically computes VAD intervals for Channel 0 (Caller) and Channel 1 (Agent).
        """
        total_duration = len(audio_data) / sr
        turns = []

        # 1. If metadata JSON exists, use annotated turns
        if turns_meta_path and os.path.exists(turns_meta_path):
            try:
                with open(turns_meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    raw_turns = meta.get("turns", [])
                    for t in raw_turns:
                        ch = int(t.get("channel", 0))
                        start_s = max(0.0, float(t.get("start", 0.0)))
                        end_s = min(total_duration, float(t.get("end", total_duration)))
                        speaker_label = "Caller (Channel 0)" if ch == 0 else "Bank Agent (Channel 1)"
                        turns.append({
                            "channel": ch,
                            "speaker": speaker_label,
                            "start": round(start_s, 2),
                            "end": round(end_s, 2),
                            "duration": round(end_s - start_s, 2)
                        })
                # Sort chronologically
                turns.sort(key=lambda x: x["start"])
                return turns
            except Exception as e:
                print(f"Warning: Failed to parse turns JSON: {e}")

        # 2. Autonomous Dynamic Extraction (Zero metadata fallback)
        import librosa
        if audio_data.ndim == 2:
            ch0 = audio_data[:, 0]
            ch1 = audio_data[:, 1]
        else:
            ch0 = audio_data
            ch1 = np.zeros_like(audio_data)

        # Dynamic VAD on Channel 0 (Caller)
        _, intervals_0 = dynamic_extract_caller_speech(ch0, orig_sr=sr, target_sr=16000, target_samples=None)
        for s, e in intervals_0:
            turns.append({
                "channel": 0,
                "speaker": "Caller (Channel 0)",
                "start": s,
                "end": e,
                "duration": round(e - s, 2)
            })

        # Dynamic VAD on Channel 1 (Agent)
        _, intervals_1 = dynamic_extract_caller_speech(ch1, orig_sr=sr, target_sr=16000, target_samples=None)
        for s, e in intervals_1:
            turns.append({
                "channel": 1,
                "speaker": "Bank Agent (Channel 1)",
                "start": s,
                "end": e,
                "duration": round(e - s, 2)
            })

        turns.sort(key=lambda x: x["start"])
        return turns

    def process_call_audio(
        self,
        audio_path: str,
        turns_meta_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        End-to-end function: reads audio, segments into turns with speaker ID,
        and transcribes each turn.
        """
        data, sr = sf.read(audio_path, dtype="float32")
        total_duration = len(data) / sr

        # Resample full audio to 16kHz for ASR
        if sr != 16000:
            import librosa
            if data.ndim == 2:
                ch0_16k = librosa.resample(data[:, 0], orig_sr=sr, target_sr=16000)
                ch1_16k = librosa.resample(data[:, 1], orig_sr=sr, target_sr=16000)
                data_16k = np.stack([ch0_16k, ch1_16k], axis=-1)
            else:
                data_16k = librosa.resample(data, orig_sr=sr, target_sr=16000)
            work_sr = 16000
        else:
            data_16k = data
            work_sr = sr

        # 1. Identify Speaker Turns
        turns = self.extract_turns_from_audio(data, sr=sr, turns_meta_path=turns_meta_path)

        # 2. Slice and Transcribe each turn
        transcript_dialogue = []
        caller_full_text = []
        agent_full_text = []

        print(f"\n🎙️ Transcribing Dialogue for: {Path(audio_path).stem} ({total_duration:.1f}s)")
        print("-" * 80)

        for idx, t in enumerate(turns, 1):
            ch = t["channel"]
            start_idx = int(t["start"] * work_sr)
            end_idx = int(t["end"] * work_sr)

            if end_idx <= start_idx:
                continue

            if data_16k.ndim == 2:
                segment_audio = data_16k[start_idx:end_idx, ch]
            else:
                segment_audio = data_16k[start_idx:end_idx]

            # Transcribe
            text = self.transcribe_audio_segment(segment_audio) if self._asr_engine else "[ASR not installed]"
            t["text"] = text

            if ch == 0:
                caller_full_text.append(text)
                prefix = "👤 [CALLER]"
            else:
                agent_full_text.append(text)
                prefix = "🤖 [AGENT] "

            print(f"[{t['start']:05.1f}s - {t['end']:05.1f}s] {prefix}: {text}")
            transcript_dialogue.append(t)

        print("-" * 80)

        return {
            "call_id": Path(audio_path).stem,
            "total_duration_s": round(total_duration, 2),
            "total_turns": len(transcript_dialogue),
            "dialogue": transcript_dialogue,
            "caller_transcript": " ".join(caller_full_text).strip(),
            "agent_transcript": " ".join(agent_full_text).strip()
        }


def main():
    parser = argparse.ArgumentParser(description="Transcribe 2-party bank conversations with speaker diarization.")
    parser.add_argument("--audio", type=str, required=True, help="Path to stereo WAV call audio file")
    parser.add_argument("--turns", type=str, default=None, help="Optional path to turns.json file")
    parser.add_argument("--model", type=str, default="tiny", choices=["tiny", "base", "small"], help="ASR model size")
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path")

    args = parser.parse_args()

    transcriber = ConversationalTurnTranscriber(asr_model_size=args.model)
    result = transcriber.process_call_audio(args.audio, turns_meta_path=args.turns)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved dialogue transcript to: {args.out}")


if __name__ == "__main__":
    main()
