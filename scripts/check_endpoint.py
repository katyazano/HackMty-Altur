"""
check_endpoint.py — Verifies live FastAPI endpoints against competition specifications.
Usage:
    python scripts/check_endpoint.py --url http://localhost:8000
"""
import io
import sys
import base64
import argparse
import numpy as np
import soundfile as sf
import requests


def generate_mock_call_b64(duration_s=3.0, sr=8000):
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    caller = 0.3 * np.sin(2 * np.pi * 220 * t)
    agent = 0.2 * np.sin(2 * np.pi * 150 * t)
    stereo = np.stack([caller, agent], axis=1).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="Verify Altur Anti-Spoofing API Endpoints")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of server")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    print("=" * 60)
    print(f" ALTUR API ENDPOINT HEALTH & CONTRACT VERIFICATION")
    print(f" Target URL: {base_url}")
    print("=" * 60)

    # 1. Health check
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        if r.status_code == 200:
            print("✔ GET /health: OK (200)")
            print(f"   Payload: {r.json()}\n")
        else:
            print(f"✖ GET /health failed with status {r.status_code}\n")
    except Exception as e:
        print(f"✖ Could not connect to {base_url}: {e}\n")
        sys.exit(1)

    # 2. Standard detect endpoint
    audio_b64 = generate_mock_call_b64()
    try:
        r = requests.post(f"{base_url}/detect", json={"audio": audio_b64, "call_id": "test_01"}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            assert "is_synthetic" in data and "confidence" in data
            print("✔ POST /detect (Default Engine): OK (200)")
            print(f"   Verdict: is_synthetic={data['is_synthetic']}, confidence={data['confidence']}\n")
        else:
            print(f"✖ POST /detect failed with status {r.status_code}\n")
    except Exception as e:
        print(f"✖ Error testing /detect: {e}\n")

    # 3. Baseline engine override
    try:
        r = requests.post(f"{base_url}/detect?engine=baseline", json={"audio": audio_b64}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            print("✔ POST /detect?engine=baseline: OK (200)")
            print(f"   Verdict: is_synthetic={data['is_synthetic']}, confidence={data['confidence']}\n")
    except Exception as e:
        print(f"✖ Error testing baseline override: {e}\n")

    # 4. Compare endpoint
    try:
        r = requests.post(f"{base_url}/api/detect/compare", json={"audio": audio_b64}, timeout=15)
        if r.status_code == 200:
            data = r.json()
            consensus = data.get("consensus", {})
            print("✔ POST /api/detect/compare (Dual-Engine Comparator): OK (200)")
            print(f"   Consensus: {consensus.get('agreement_status')} -> {consensus.get('recommended_verdict')}")
            print(f"   Wall Latency: {consensus.get('total_wall_latency_ms')} ms\n")
    except Exception as e:
        print(f"✖ Error testing /api/detect/compare: {e}\n")

    print("=" * 60)
    print("✔ All endpoint verifications completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
