import os
import sys
import re
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import soundfile as sf
import librosa

# OpenMP runtime collision guard
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.inference_engine import InferenceEngine
from src.api.config import settings

logger = logging.getLogger("altur.unified_analyzer")


class SpanishSemanticFeatures:
    """
    Extracts semantic conversational markers and disfluency patterns from transcribed text
    to differentiate natural human conversational speech from synthetic/LLM voice bots.
    """

    HUMAN_COLLOQUIAL_FILLERS = [
        r"\beste\b", r"\beste\.\.\.", r"\beh\b", r"\bem\b", r"\bah\b",
        r"\bbueno\b", r"\bo sea\b", r"\bajá\b", r"\bpues\b", r"\bmira\b",
        r"\ba ver\b", r"\bfíjate\b", r"\bverdad\b", r"\bósea\b", r"\bahorita\b",
        r"\bórale\b", r"\bno manches\b", r"\bqué onda\b", r"\bándale\b",
        r"\bsí, este\b", r"\beh\.\.\.", r"\bumm\b", r"\bhum\b"
    ]

    ROBOTIC_LLM_PATTERNS = [
        r"\bentiendo perfectamente\b",
        r"\bcon gusto le asisto\b",
        r"\ben qué más puedo servirle\b",
        r"\bcomo modelo de lenguaje\b",
        r"\bsoy un asistente virtual\b",
        r"\bgracias por comunicarse\b",
        r"\bpor supuesto, le ayudo con eso de inmediato\b",
        r"\bestoy aquí para ayudarle\b",
        r"\blamento el inconveniente\b"
    ]

    @classmethod
    def analyze(cls, text: str) -> Dict[str, Any]:
        """Analyzes text for authentic conversational fillers vs robotic patterns."""
        clean_text = text.lower().strip()
        words = re.findall(r"\b\w+\b", clean_text)
        total_words = len(words)

        if total_words == 0:
            return {
                "total_words": 0,
                "filler_count": 0,
                "filler_density": 0.0,
                "robotic_phrase_count": 0,
                "lexical_diversity_ttr": 0.0,
                "detected_fillers": [],
                "semantic_authenticity_score": 0.50
            }

        detected_fillers = []
        filler_count = 0
        for pattern in cls.HUMAN_COLLOQUIAL_FILLERS:
            matches = re.findall(pattern, clean_text)
            if matches:
                filler_count += len(matches)
                detected_fillers.extend(matches)

        robotic_phrase_count = 0
        for pattern in cls.ROBOTIC_LLM_PATTERNS:
            matches = re.findall(pattern, clean_text)
            if matches:
                robotic_phrase_count += len(matches)

        unique_words = len(set(words))
        ttr = round(unique_words / total_words, 4)
        filler_density = round(filler_count / total_words, 4)

        # Authenticity scoring:
        # High filler density + informal phrasing = high human score
        # High robotic phrasing + zero fillers = low human score / synthetic indicator
        authenticity_score = 0.50 + (filler_density * 2.5) - (robotic_phrase_count * 0.25)
        authenticity_score = round(float(np.clip(authenticity_score, 0.05, 0.98)), 4)

        return {
            "total_words": total_words,
            "unique_words": unique_words,
            "filler_count": filler_count,
            "filler_density": filler_density,
            "robotic_phrase_count": robotic_phrase_count,
            "lexical_diversity_ttr": ttr,
            "detected_fillers": list(set(detected_fillers)),
            "semantic_authenticity_score": authenticity_score
        }


class UnifiedDialogueAnalyzer:
    """
    Master Unified Pipeline:
    1. Dual-stream ASR (Spanish faster-whisper INT8).
    2. Conversational Dynamics (Interleaved turns, Latency Δt, Overlap).
    3. Semantic Spanish Analysis (Authentic fillers vs robotic patterns).
    4. 272-dim 3-Model Biometric Anti-Spoofing (AASIST + RawNet2 + DSP -> XGBoost).
    5. UI-Ready JSON Export.
    """

    MEXICAN_BANKING_PROMPT = (
        "Llamada telefónica de atención a clientes y banca en México. "
        "Palabras clave: tarjeta, NIP, cuenta, CLABE, transferencia, saldo, "
        "SPEI, RFC, homoclave, cargo no reconocido, ahorita, este, o sea, bueno."
    )

    def __init__(
        self,
        asr_model_size: str = "tiny",
        device: str = "cpu",
        cpu_threads: int = 4
    ):
        self.device = device
        self.asr_model_size = asr_model_size
        self.cpu_threads = cpu_threads

        # 1. Initialize Biometric Multi-Model Engine (AASIST + RawNet2 + DSP + XGBoost)
        logger.info("Initializing 3-Model Biometric Anti-Spoofing Engine...")
        self.biometric_engine = InferenceEngine()

        # 2. Initialize Whisper ASR Engine
        self.whisper_model = None
        self._init_asr()

    def _init_asr(self):
        try:
            from faster_whisper import WhisperModel
            self.whisper_model = WhisperModel(
                self.asr_model_size,
                device=self.device,
                compute_type="int8",
                cpu_threads=self.cpu_threads
            )
            logger.info(f"✓ Loaded faster-whisper ({self.asr_model_size}, INT8) on {self.device}")
        except Exception as e:
            logger.warning(f"Could not load faster-whisper: {e}. Falling back to standard whisper if available.")
            try:
                import whisper
                self.whisper_model = whisper.load_model(self.asr_model_size, device=self.device)
            except Exception as e2:
                logger.error(f"Failed to load any Whisper ASR model: {e2}")

    def load_audio(self, audio_source: Union[str, np.ndarray], target_sr: int = 16000) -> Tuple[np.ndarray, float]:
        """Loads and normalizes an audio track to 16kHz float32 mono."""
        if isinstance(audio_source, str):
            data, sr = sf.read(audio_source, dtype="float32")
        else:
            data = audio_source
            sr = target_sr

        if data.ndim == 2:
            data = np.mean(data, axis=1)

        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr).astype(np.float32)

        duration = len(data) / target_sr
        return data, duration

    def transcribe_stream(
        self,
        audio_16k: np.ndarray,
        speaker_name: str,
        channel_id: int
    ) -> List[Dict[str, Any]]:
        """Transcribes an audio stream and extracts timestamped utterance segments."""
        if self.whisper_model is None or len(audio_16k) == 0:
            return []

        # Skip near-silent audio
        if np.max(np.abs(audio_16k)) < 1e-4:
            return []

        turns = []
        try:
            if hasattr(self.whisper_model, "transcribe"):
                segments_iter, _ = self.whisper_model.transcribe(
                    audio_16k,
                    language="es",
                    initial_prompt=self.MEXICAN_BANKING_PROMPT,
                    beam_size=1,             # Greedy search for maximum throughput
                    best_of=1,
                    temperature=0.0,
                    vad_filter=True,         # Silero VAD to segment speech
                    vad_parameters=dict(min_silence_duration_ms=250),
                    condition_on_previous_text=False
                )

                for seg in segments_iter:
                    text = seg.text.strip()
                    if text:
                        turns.append({
                            "speaker": speaker_name,
                            "channel": channel_id,
                            "start_time": round(float(seg.start), 2),
                            "end_time": round(float(seg.end), 2),
                            "duration": round(float(seg.end - seg.start), 2),
                            "text": text,
                            "avg_logprob": round(float(seg.avg_logprob), 3) if hasattr(seg, "avg_logprob") else 0.0
                        })
        except Exception as e:
            logger.error(f"Error during transcription of {speaker_name}: {e}")

        return turns

    def compute_conversational_dynamics(
        self,
        caller_turns: List[Dict[str, Any]],
        agent_turns: List[Dict[str, Any]],
        is_synchronized_timeline: bool
    ) -> Dict[str, Any]:
        """Computes dialogue interleaving, response latency, and overlap dynamics."""
        # 1. Chronological Merge
        all_turns = caller_turns + agent_turns
        all_turns.sort(key=lambda x: x["start_time"])

        if not is_synchronized_timeline or not caller_turns or not agent_turns:
            return {
                "timeline_synchronized": False,
                "interleaved_dialogue": all_turns,
                "total_turns": len(all_turns),
                "caller_turn_count": len(caller_turns),
                "agent_turn_count": len(agent_turns),
                "turn_taking_latency_ms": None,
                "overlap_percentage": None,
                "cross_talk_duration_sec": 0.0,
                "note": "Audio files were processed as independent continuous speech streams."
            }

        # 2. Compute Latency (Δt) between Agent question and Caller response
        latencies = []
        for i in range(len(all_turns) - 1):
            curr_turn = all_turns[i]
            next_turn = all_turns[i + 1]

            if curr_turn["speaker"] == "Agent" and next_turn["speaker"] == "Caller":
                delta_t = next_turn["start_time"] - curr_turn["end_time"]
                latencies.append(delta_t)

        # 3. Compute Overlapping Speech (Cross-Talk)
        total_overlap_sec = 0.0
        for c_turn in caller_turns:
            for a_turn in agent_turns:
                start_overlap = max(c_turn["start_time"], a_turn["start_time"])
                end_overlap = min(c_turn["end_time"], a_turn["end_time"])
                if end_overlap > start_overlap:
                    total_overlap_sec += (end_overlap - start_overlap)

        mean_latency = float(np.mean(latencies)) if latencies else 0.0
        median_latency = float(np.median(latencies)) if latencies else 0.0

        return {
            "timeline_synchronized": True,
            "interleaved_dialogue": all_turns,
            "total_turns": len(all_turns),
            "caller_turn_count": len(caller_turns),
            "agent_turn_count": len(agent_turns),
            "turn_taking_latency_ms": {
                "mean_ms": round(mean_latency * 1000.0, 1),
                "median_ms": round(median_latency * 1000.0, 1),
                "samples_evaluated": len(latencies)
            },
            "cross_talk_duration_sec": round(total_overlap_sec, 2),
            "note": "Turn-taking latency and cross-talk evaluated on synchronized timeline."
        }

    def process_call(
        self,
        caller_audio: Union[str, np.ndarray],
        agent_audio: Optional[Union[str, np.ndarray]] = None,
        turns_metadata: Optional[Union[str, Dict[str, Any]]] = None,
        output_json_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end multi-model analysis:
        1. Biometric 3-model anti-spoofing on caller audio.
        2. Fast dual-channel Spanish ASR.
        3. Conversational dynamics & Mexican Spanish semantic profiling.
        4. Structured JSON export for future UI.
        """
        t_start = time.perf_counter()

        # 1. Load Caller Audio
        caller_16k, caller_dur = self.load_audio(caller_audio, target_sr=16000)

        # 2. Load Agent Audio (if provided)
        agent_16k, agent_dur = None, 0.0
        if agent_audio is not None:
            agent_16k, agent_dur = self.load_audio(agent_audio, target_sr=16000)

        # 3. Check if we have timeline metadata
        is_synchronized = False
        parsed_turns_meta = None
        if turns_metadata:
            if isinstance(turns_metadata, str) and os.path.exists(turns_metadata):
                with open(turns_metadata, "r", encoding="utf-8") as f:
                    parsed_turns_meta = json.load(f)
                    is_synchronized = True
            elif isinstance(turns_metadata, dict):
                parsed_turns_meta = turns_metadata
                is_synchronized = True

        # 4. Run 4-Pillar Multimodal Anti-Spoofing on Isolated Caller Audio (Single Unified Pass)
        logger.info("Executing 4-Pillar Multimodal Biometric Anti-Spoofing on caller stream...")
        is_synthetic, confidence, raw_caller_turns = self.biometric_engine.predict(caller_16k, return_transcript=True)

        caller_turns = []
        for t in (raw_caller_turns or []):
            caller_turns.append({
                "speaker": "Caller",
                "channel": 0,
                "start_time": t.get("start_time", 0.0),
                "end_time": t.get("end_time", 0.0),
                "duration": round(float(t.get("end_time", 0.0) - t.get("start_time", 0.0)), 2),
                "text": t.get("text", ""),
                "avg_logprob": t.get("avg_logprob", 0.0),
                "no_speech_prob": t.get("no_speech_prob", 0.0),
                "compression_ratio": t.get("compression_ratio", 1.0)
            })

        # 5. Run Spanish ASR on Agent (if provided)
        agent_turns = []
        if agent_16k is not None:
            logger.info("Transcribing agent speech...")
            agent_turns = self.transcribe_stream(agent_16k, speaker_name="Agent", channel_id=1)

        # 6. Extract Semantic NLP Metrics from Caller Speech
        caller_full_text = " ".join([t["text"] for t in caller_turns])
        semantic_profile = SpanishSemanticFeatures.analyze(caller_full_text)


        # 7. Conversational Dynamics & Dialogue Interleaving
        dynamics = self.compute_conversational_dynamics(
            caller_turns=caller_turns,
            agent_turns=agent_turns,
            is_synchronized_timeline=is_synchronized
        )

        total_exec_time = round(time.perf_counter() - t_start, 3)

        # 8. Construct Unified UI Report
        report = {
            "call_summary": {
                "caller_duration_sec": round(caller_dur, 2),
                "agent_duration_sec": round(agent_dur, 2),
                "processing_time_sec": total_exec_time,
                "real_time_factor": round(total_exec_time / max(0.1, caller_dur + agent_dur), 3),
                "total_dialogue_turns": len(dynamics["interleaved_dialogue"])
            },
            "anti_spoofing_verdict": {
                "is_synthetic": is_synthetic,
                "confidence": confidence,
                "decision_threshold": self.biometric_engine.calibrated_threshold,
                "risk_level": "CRITICAL" if is_synthetic and confidence > 0.80 else ("SUSPICIOUS" if is_synthetic else "AUTHENTIC")
            },
            "semantic_nlp_profile": {
                "detected_language": "es-MX",
                "total_words": semantic_profile["total_words"],
                "filler_density": semantic_profile["filler_density"],
                "detected_human_fillers": semantic_profile["detected_fillers"],
                "robotic_phrases_found": semantic_profile["robotic_phrase_count"],
                "lexical_diversity_ttr": semantic_profile["lexical_diversity_ttr"],
                "semantic_authenticity_score": semantic_profile["semantic_authenticity_score"]
            },
            "conversational_dynamics": {
                "timeline_synchronized": dynamics["timeline_synchronized"],
                "turn_taking_latency": dynamics.get("turn_taking_latency_ms"),
                "cross_talk_duration_sec": dynamics.get("cross_talk_duration_sec"),
                "note": dynamics.get("note")
            },
            "transcript": dynamics["interleaved_dialogue"]
        }

        # 9. Optional JSON Save
        if output_json_path:
            out_p = Path(output_json_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Saved unified dialogue report to {output_json_path}")

        return report


def main():
    parser = argparse.ArgumentParser(description="Unified Dialogue Transcriber & 3-Model Biometric Engine")
    parser.add_argument("--caller", type=str, required=True, help="Path to caller audio (Channel 0)")
    parser.add_argument("--agent", type=str, default=None, help="Path to agent audio (Channel 1, optional)")
    parser.add_argument("--turns", type=str, default=None, help="Path to turns.json metadata (optional)")
    parser.add_argument("--model-size", type=str, default="tiny", help="Whisper model size (tiny, base, small, medium)")
    parser.add_argument("--output", type=str, default="Data/unified_call_report.json", help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    analyzer = UnifiedDialogueAnalyzer(asr_model_size=args.model_size)
    report = analyzer.process_call(
        caller_audio=args.caller,
        agent_audio=args.agent,
        turns_metadata=args.turns,
        output_json_path=args.output
    )

    print("\n" + "=" * 80)
    print("📊 UNIFIED MULTI-MODEL CALL REPORT")
    print("=" * 80)
    print(f"• Verdict:         {'⚠️ SYNTHETIC / DEEPFAKE' if report['anti_spoofing_verdict']['is_synthetic'] else '✅ AUTHENTIC HUMAN'}")
    print(f"• Confidence:      {report['anti_spoofing_verdict']['confidence'] * 100:.1f}%")
    print(f"• Risk Level:      {report['anti_spoofing_verdict']['risk_level']}")
    print(f"• Semantic Fillers: {report['semantic_nlp_profile']['filler_density'] * 100:.1f}% ({', '.join(report['semantic_nlp_profile']['detected_human_fillers']) or 'None'})")
    print(f"• Total Turns:     {report['call_summary']['total_dialogue_turns']}")
    print(f"• Saved JSON:      {args.output}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
