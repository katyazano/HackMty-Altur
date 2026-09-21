import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import io
import base64
import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def generate_test_base64_audio(duration_s=2.0, sr=8000):
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    caller = 0.3 * np.sin(2 * np.pi * 220 * t)
    agent = 0.2 * np.sin(2 * np.pi * 150 * t)
    stereo = np.stack([caller, agent], axis=1).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "available_engines" in data


def test_models_info_endpoint(client):
    response = client.get("/api/models/info")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) >= 2
    assert "benchmark_summary" in data


def test_benchmarks_endpoint(client):
    response = client.get("/api/benchmarks")
    assert response.status_code == 200
    data = response.json()
    assert "benchmarks" in data
    assert "multimodal" in data["benchmarks"]
    assert "baseline" in data["benchmarks"]


def test_detect_standard_contract(client):
    audio_b64 = generate_test_base64_audio()
    response = client.post("/detect", json={"audio": audio_b64, "call_id": "test_call_01"})
    assert response.status_code == 200
    data = response.json()
    assert "is_synthetic" in data
    assert isinstance(data["is_synthetic"], bool)
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0


def test_detect_engine_selection(client):
    audio_b64 = generate_test_base64_audio()
    # Baseline
    res_base = client.post("/detect?engine=baseline", json={"audio": audio_b64})
    assert res_base.status_code == 200
    assert "is_synthetic" in res_base.json()

    # Multimodal
    res_multi = client.post("/detect?engine=multimodal", json={"audio": audio_b64})
    assert res_multi.status_code == 200
    assert "is_synthetic" in res_multi.json()


def test_compare_endpoint(client):
    audio_b64 = generate_test_base64_audio()
    response = client.post("/api/detect/compare", json={"audio": audio_b64, "call_id": "test_compare_01"})
    assert response.status_code == 200
    data = response.json()
    assert "consensus" in data
    assert "multimodal" in data
    assert "baseline" in data
    assert "audio_metadata" in data
    assert data["audio_metadata"]["is_stereo"] is True
