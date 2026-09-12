import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import base64
import numpy as np
import soundfile as sf
import unittest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def generate_synthetic_stereo_wav_base64(duration_s: float = 2.0, sr: int = 8000) -> str:
    """Generates an in-memory 8kHz stereo WAV base64 string for testing."""
    n_samples = int(duration_s * sr)
    caller = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, duration_s, n_samples))).astype(np.float32)
    agent = (0.3 * np.sin(2 * np.pi * 880 * np.linspace(0, duration_s, n_samples))).astype(np.float32)
    stereo_data = np.stack([caller, agent], axis=-1)

    buf = io.BytesIO()
    sf.write(buf, stereo_data, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


class TestAntiSpoofingAPI(unittest.TestCase):
    def test_root_endpoint(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_api_info_endpoint(self):
        response = client.get("/api/info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("service", data)
        self.assertIn("docs", data)
        self.assertEqual(data["calibrated_threshold"], 0.9623)

    def test_health_endpoint(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["calibrated_threshold"], 0.9623)
        self.assertIn("model_ensemble", data)

    def test_detect_endpoint_stereo_8khz(self):
        b64_audio = generate_synthetic_stereo_wav_base64()
        payload = {
            "audio": b64_audio,
            "call_id": "test_call_stereo_001"
        }
        response = client.post("/detect", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Strict 2-key contract verification
        self.assertIn("is_synthetic", data)
        self.assertIn("confidence", data)
        self.assertEqual(set(data.keys()), {"is_synthetic", "confidence"})
        self.assertIsInstance(data["is_synthetic"], bool)
        self.assertIsInstance(data["confidence"], (float, int))
        self.assertTrue(0.0 <= data["confidence"] <= 1.0)

    def test_detect_real_dataset_audio(self):
        test_wav = PROJECT_ROOT / "Data" / "separated_agents" / "test" / "agent_0" / "call_0847d7417bb1_agent_0.wav"
        if test_wav.exists():
            with open(test_wav, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
            payload = {
                "audio": b64_data,
                "call_id": "call_0847d7417bb1"
            }
            response = client.post("/detect", json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(set(data.keys()), {"is_synthetic", "confidence"})
            self.assertIsInstance(data["is_synthetic"], bool)
            self.assertIsInstance(data["confidence"], (float, int))
            print(f"\nReal audio inference result: is_synthetic={data['is_synthetic']}, confidence={data['confidence']}")

    def test_detect_endpoint_invalid_base64(self):
        payload = {
            "audio": "not_valid_base64!!!",
            "call_id": "test_call_invalid"
        }
        response = client.post("/detect", json=payload)
        self.assertEqual(response.status_code, 400)
        detail = response.json().get("detail", "")
        self.assertTrue("Invalid Base64" in detail or "Unable to decode" in detail)

    def test_detect_endpoint_empty_payload(self):
        payload = {
            "audio": "",
            "call_id": "test_call_empty"
        }
        response = client.post("/detect", json=payload)
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
