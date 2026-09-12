import os
import sys
import glob
import json
import time
import shutil
from typing import Optional
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# Add workspace to sys.path
sys.path.insert(0, os.path.dirname(__file__))

from src.audio_processor import AudioProcessor
from src.models.rawnet2 import RawNet2
from src.models.aasist import AASIST

app = FastAPI(title="Altur Banking - Voice Anti-Spoofing Benchmark")

BASE_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(BASE_DIR, "demo_real_audios")
WEB_DIR = os.path.join(BASE_DIR, "web")

os.makedirs(RAW_DIR, exist_ok=True)

# Pre-instantiate Deep Learning Models
RAWNET2_MODEL = RawNet2()
RAWNET2_MODEL.eval()

AASIST_MODEL = AASIST()
AASIST_MODEL.eval()

def get_metadata_store():
    meta_path = os.path.join(RAW_DIR, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_metadata_store(meta: dict):
    meta_path = os.path.join(RAW_DIR, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def infer_ground_truth(key: str, filename: str, custom_gt: Optional[str] = None) -> str:
    if custom_gt and custom_gt in ("REAL_HUMAN", "FAKE_SPOOF"):
        return custom_gt
    
    k_lower = (key + " " + filename).lower()
    if any(w in k_lower for w in ["fake", "spoof", "synth", "deepfake", "vocoder", "clone"]):
        return "FAKE_SPOOF"
    elif any(w in k_lower for w in ["real", "bonafide", "human", "user", "rec1", "rec2"]):
        return "REAL_HUMAN"
    return "UNKNOWN"

def get_audio_catalog():
    """
    Scans RAW_DIR and returns all currently available audio files with metadata.
    """
    valid_extensions = ('.ogg', '.wav', '.mp3', '.flac', '.m4a')
    files = {}
    metadata = get_metadata_store().get("files", {})

    for fname in sorted(os.listdir(RAW_DIR)):
        fpath = os.path.join(RAW_DIR, fname)
        if os.path.isfile(fpath) and fname.lower().endswith(valid_extensions):
            key = os.path.splitext(fname)[0]
            
            meta_entry = metadata.get(key, {})
            label = meta_entry.get("label", key.replace("_", " ").title())
            gt = meta_entry.get("ground_truth", infer_ground_truth(key, fname))

            files[key] = {
                "key": key,
                "label": label,
                "filename": fname,
                "raw_path": fpath,
                "ground_truth": gt
            }

    return files

def process_all_audios():
    """
    Executes ASP and Deep Learning Anti-Spoofing models (RawNet2 & AASIST), evaluating accuracy against ground truth.
    """
    catalog = get_audio_catalog()
    processor = AudioProcessor(target_sr=16000)

    audios_data = []

    for key, item in catalog.items():
        raw_path = item["raw_path"]
        ground_truth = item["ground_truth"]
        
        # 1. Load & Normalize
        y_raw, sr = processor.load_and_normalize(raw_path)
        
        # 2. Spectral Denoising
        y_denoised = processor.apply_denoising(y_raw, prop_decrease=0.72)
        
        # 3. Bandpass Filter
        y_filtered = processor.apply_bandpass_filter(y_denoised, lowcut=90.0, highcut=3900.0)
        
        # 4. VAD Silence Trimming
        y_final = processor.apply_vad(y_filtered, top_db=24)

        # Prepare tensor for deep learning models
        tensor_in = torch.tensor(y_final, dtype=torch.float32).unsqueeze(0)

        # Model 1: RawNet2
        t0 = time.perf_counter()
        rawnet_res = RAWNET2_MODEL.predict_spoof(tensor_in)
        raw_latency = round((time.perf_counter() - t0) * 1000, 2)
        rawnet_res["latency_ms"] = raw_latency
        if ground_truth != "UNKNOWN":
            is_truth_real = (ground_truth == "REAL_HUMAN")
            rawnet_res["is_correct"] = (rawnet_res.get("is_real", not rawnet_res.get("is_spoof", False)) == is_truth_real)
        else:
            rawnet_res["is_correct"] = None

        # Model 2: AASIST-L
        t0 = time.perf_counter()
        aasist_res = AASIST_MODEL.predict_spoof(tensor_in)
        aasist_latency = round((time.perf_counter() - t0) * 1000, 2)
        aasist_res["latency_ms"] = aasist_latency
        if ground_truth != "UNKNOWN":
            is_truth_real = (ground_truth == "REAL_HUMAN")
            aasist_res["is_correct"] = (aasist_res.get("is_real", not aasist_res.get("is_spoof", False)) == is_truth_real)
        else:
            aasist_res["is_correct"] = None

        raw_dur = round(len(y_raw) / sr, 2)

        audios_data.append({
            "key": key,
            "label": item["label"],
            "filename": item["filename"],
            "raw_filename": item["filename"],
            "raw_url": f"/audio/raw/{item['filename']}",
            "raw_duration": raw_dur,
            "ground_truth": ground_truth,
            "rawnet2": rawnet_res,
            "aasist": aasist_res
        })

    return {
        "audios": audios_data
    }

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_path = os.path.join(WEB_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Dashboard UI not found")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/data")
async def get_dashboard_data():
    return process_all_audios()

@app.post("/api/upload")
async def upload_audio_file(
    label: str = Form(...),
    file: UploadFile = File(...),
    ground_truth: str = Form("AUTO")
):
    safe_label = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in label.strip().lower())
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ('.ogg', '.wav', '.mp3', '.flac', '.m4a'):
        raise HTTPException(status_code=400, detail="Invalid audio format")

    dest_filename = f"{safe_label}{ext}"
    dest_path = os.path.join(RAW_DIR, dest_filename)

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Save custom metadata if provided
    meta = get_metadata_store()
    if "files" not in meta:
        meta["files"] = {}
    
    resolved_gt = infer_ground_truth(safe_label, dest_filename, ground_truth)
    meta["files"][safe_label] = {
        "filename": dest_filename,
        "label": label.strip(),
        "ground_truth": resolved_gt
    }
    save_metadata_store(meta)

    return process_all_audios()

@app.delete("/api/audio/{filename}")
async def delete_single_audio(filename: str):
    raw_path = os.path.join(RAW_DIR, filename)
    if os.path.exists(raw_path):
        os.remove(raw_path)
    
    key = os.path.splitext(filename)[0]
    meta = get_metadata_store()
    if "files" in meta and key in meta["files"]:
        del meta["files"][key]
        save_metadata_store(meta)

    return process_all_audios()

@app.delete("/api/audios/all")
async def delete_all_audios():
    for f in glob.glob(os.path.join(RAW_DIR, "*.*")):
        if os.path.isfile(f):
            os.remove(f)

    return process_all_audios()

@app.get("/audio/raw/{filename}")
async def stream_raw_audio(filename: str):
    file_path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Raw audio file not found")
    media_type = "audio/ogg" if filename.endswith(".ogg") else ("audio/wav" if filename.endswith(".wav") else "audio/flac")
    return FileResponse(file_path, media_type=media_type)

if __name__ == "__main__":
    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", 8000))
    print(f"Starting Altur Voice Anti-Spoofing Benchmark Server on http://{HOST}:{PORT} ...")
    uvicorn.run(app, host=HOST, port=PORT)
