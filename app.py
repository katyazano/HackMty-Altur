import os
import sys
import glob
import json
import time
from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Add workspace to sys.path
PROJECT_ROOT = os.path.dirname(__file__)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.rawnet2 import RawNet2
from src.models.aasist import AASIST
from src.evaluate_channel0 import evaluate_channel0_dataset

app = FastAPI(title="Altur Banking - Voice Anti-Spoofing Benchmark")

BASE_DIR = os.path.dirname(__file__)
AGENT0_DIR = os.path.join(BASE_DIR, "Data", "separated_agents", "train", "agent_0")
MANIFEST_PATH = os.path.join(BASE_DIR, "Data", "manifest.csv")
EVAL_RESULTS_JSON = os.path.join(BASE_DIR, "Data", "channel0_evaluation_results.json")
WEB_DIR = os.path.join(BASE_DIR, "web")

def get_channel0_benchmark_data():
    """
    Returns the benchmark evaluation results for Channel 0 (Agent 0) audios.
    If cached JSON exists, loads it; otherwise runs the evaluation.
    """
    if os.path.exists(EVAL_RESULTS_JSON):
        try:
            with open(EVAL_RESULTS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except Exception as e:
            print(f"Error loading {EVAL_RESULTS_JSON}: {e}")

    # Fallback to computing on the fly
    results = evaluate_channel0_dataset(
        agent0_dir=AGENT0_DIR,
        manifest_path=MANIFEST_PATH,
        output_json=EVAL_RESULTS_JSON
    )
    with open(EVAL_RESULTS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_path = os.path.join(WEB_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Dashboard UI not found")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/data")
async def get_dashboard_data():
    """
    Endpoint returning Channel 0 anti-spoofing evaluations for RawNet2 and AASIST-L.
    """
    return get_channel0_benchmark_data()

@app.post("/api/re-evaluate")
async def re_evaluate_benchmark():
    """
    Triggers re-evaluation of all Channel 0 audios.
    """
    results = evaluate_channel0_dataset(
        agent0_dir=AGENT0_DIR,
        manifest_path=MANIFEST_PATH,
        output_json=EVAL_RESULTS_JSON
    )
    return get_channel0_benchmark_data()

@app.get("/audio/agent_0/{filename}")
async def serve_agent0_audio(filename: str):
    """
    Streams the separated Channel 0 audio file for in-browser playback.
    """
    safe_name = os.path.basename(filename)
    audio_path = os.path.join(AGENT0_DIR, safe_name)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(audio_path, media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
