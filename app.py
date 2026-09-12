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
from src.biometrics import BiometricsEngine
from src.models.rawnet2 import RawNet2
from src.models.aasist import AASIST

app = FastAPI(title="Altur Banking - Voice Biometrics & Anti-Spoofing Benchmark")

BASE_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(BASE_DIR, "demo_real_audios")
PROCESSED_DIR = os.path.join(RAW_DIR, "processed_output")
SAMPLE_PACKS_DIR = os.path.join(BASE_DIR, "sample_packs")
WEB_DIR = os.path.join(BASE_DIR, "web")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SAMPLE_PACKS_DIR, exist_ok=True)

# Global active parameters
CURRENT_PARAMS = {
    "top_db": 24,
    "prop_decrease": 0.72,
    "lowcut": 90.0,
    "highcut": 3900.0,
    "similarity_threshold": 0.985
}

# Pre-instantiate Deep Learning Models
RAWNET2_MODEL = RawNet2()
RAWNET2_MODEL.eval()

AASIST_MODEL = AASIST()
AASIST_MODEL.eval()

class ParameterConfig(BaseModel):
    top_db: int = 24
    prop_decrease: float = 0.72
    lowcut: float = 90.0
    highcut: float = 3900.0
    similarity_threshold: float = 0.985

class SampleLoadRequest(BaseModel):
    pack: str = "asvspoof"

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

def process_all_audios(params: dict):
    """
    Executes ASP and all 3 Anti-Spoofing models, evaluating accuracy against ground truth.
    """
    catalog = get_audio_catalog()
    processor = AudioProcessor(target_sr=16000)
    engine = BiometricsEngine(speaker_similarity_threshold=params["similarity_threshold"])

    audios_data = []
    features_map = {}

    for key, item in catalog.items():
        raw_path = item["raw_path"]
        ground_truth = item["ground_truth"]
        
        # 1. Load & Normalize
        y_raw, sr = processor.load_and_normalize(raw_path)
        
        # 2. Spectral Denoising
        y_denoised = processor.apply_denoising(y_raw, prop_decrease=params["prop_decrease"])
        
        # 3. Bandpass Filter
        y_filtered = processor.apply_bandpass_filter(y_denoised, lowcut=params["lowcut"], highcut=params["highcut"])
        
        # 4. VAD Silence Trimming
        y_final = processor.apply_vad(y_filtered, top_db=params["top_db"])

        # 5. Export processed audio
        proc_filename = f"processed_{key}_denoised.wav"
        proc_path = os.path.join(PROCESSED_DIR, proc_filename)
        sf.write(proc_path, y_final, sr)

        # 6. Extract features for DSP & Biometrics
        mfcc = processor.extract_mfcc(y_final)
        
        # Model 1: DSP Baseline
        t0 = time.perf_counter()
        lfcc = processor.extract_lfcc(y_final)
        spec = processor.compute_spectral_features(y_final)
        dsp_res = engine.assess_anti_spoofing(lfcc, spec)
        dsp_latency = round((time.perf_counter() - t0) * 1000, 2)
        dsp_res["latency_ms"] = dsp_latency
        
        # Accuracy check for DSP
        if ground_truth != "UNKNOWN":
            is_truth_spoof = (ground_truth == "FAKE_SPOOF")
            dsp_res["is_correct"] = (dsp_res["is_spoof"] == is_truth_spoof)
        else:
            dsp_res["is_correct"] = None

        # Prepare tensor for deep learning models
        tensor_in = torch.tensor(y_final, dtype=torch.float32).unsqueeze(0)

        # Model 2: RawNet2
        t0 = time.perf_counter()
        rawnet_res = RAWNET2_MODEL.predict_spoof(tensor_in)
        raw_latency = round((time.perf_counter() - t0) * 1000, 2)
        rawnet_res["latency_ms"] = raw_latency
        if ground_truth != "UNKNOWN":
            is_truth_spoof = (ground_truth == "FAKE_SPOOF")
            rawnet_res["is_correct"] = (rawnet_res["is_spoof"] == is_truth_spoof)
        else:
            rawnet_res["is_correct"] = None

        # Model 3: AASIST-L
        t0 = time.perf_counter()
        aasist_res = AASIST_MODEL.predict_spoof(tensor_in)
        aasist_latency = round((time.perf_counter() - t0) * 1000, 2)
        aasist_res["latency_ms"] = aasist_latency
        if ground_truth != "UNKNOWN":
            is_truth_spoof = (ground_truth == "FAKE_SPOOF")
            aasist_res["is_correct"] = (aasist_res["is_spoof"] == is_truth_spoof)
        else:
            aasist_res["is_correct"] = None

        raw_dur = round(len(y_raw) / sr, 2)
        proc_dur = round(len(y_final) / sr, 2)

        features_map[key] = mfcc

        audios_data.append({
            "key": key,
            "label": item["label"],
            "filename": item["filename"],
            "raw_filename": item["filename"],
            "processed_filename": proc_filename,
            "raw_url": f"/audio/raw/{item['filename']}",
            "processed_url": f"/audio/processed/{proc_filename}",
            "raw_duration": raw_dur,
            "processed_duration": proc_dur,
            "ground_truth": ground_truth,
            "spectral": spec,
            "anti_spoof": dsp_res,
            "rawnet2": rawnet_res,
            "aasist": aasist_res
        })

    # Compute Biometric Verification matrix against the first enrolled recording
    enrolled_key = list(features_map.keys())[0] if features_map else ""
    biometrics_list = []

    if enrolled_key and enrolled_key in features_map:
        enrolled_mfcc = features_map[enrolled_key]
        enrolled_label = catalog[enrolled_key]["label"]

        for key, item in catalog.items():
            if key in features_map and key != enrolled_key:
                cand_mfcc = features_map[key]
                ver_res = engine.verify_speaker(enrolled_mfcc, cand_mfcc)
                
                # Biometrics Ground Truth: Is it the same person as enrolled?
                cand_gt = item["ground_truth"]
                is_same_expected = (cand_gt == "REAL_HUMAN" and "rec2" in key) or (key == enrolled_key)
                gt_bio_label = "SAME_SPEAKER" if is_same_expected else "DIFFERENT_SPEAKER"

                biometrics_list.append({
                    "candidate_key": key,
                    "candidate_label": item["label"],
                    "similarity_score": ver_res["similarity_score"],
                    "confidence_percentage": ver_res["confidence_percentage"],
                    "threshold_used": ver_res["threshold_used"],
                    "is_match": ver_res["is_match"],
                    "ground_truth": gt_bio_label,
                    "is_correct": (ver_res["is_match"] == is_same_expected)
                })

    return {
        "params": params,
        "enrolled_key": enrolled_key,
        "audios": audios_data,
        "biometrics": biometrics_list
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
    return process_all_audios(CURRENT_PARAMS)

@app.post("/api/reprocess")
async def reprocess_audio_endpoint(config: ParameterConfig):
    global CURRENT_PARAMS
    CURRENT_PARAMS = config.dict()
    return process_all_audios(CURRENT_PARAMS)

@app.post("/api/samples/load")
async def load_sample_pack(req: SampleLoadRequest):
    pack_name = req.pack.lower().strip()
    pack_path = os.path.join(SAMPLE_PACKS_DIR, pack_name)
    if not os.path.exists(pack_path):
        raise HTTPException(status_code=404, detail=f"Sample pack '{pack_name}' not found")

    # Clear current audio directory
    for f in glob.glob(os.path.join(RAW_DIR, "*.*")):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(os.path.join(PROCESSED_DIR, "*.*")):
        if os.path.isfile(f):
            os.remove(f)

    # Copy files from sample pack
    for item in os.listdir(pack_path):
        s_item = os.path.join(pack_path, item)
        d_item = os.path.join(RAW_DIR, item)
        if os.path.isfile(s_item):
            shutil.copy2(s_item, d_item)

    return process_all_audios(CURRENT_PARAMS)

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

    return process_all_audios(CURRENT_PARAMS)

@app.delete("/api/audio/{filename}")
async def delete_single_audio(filename: str):
    raw_path = os.path.join(RAW_DIR, filename)
    if os.path.exists(raw_path):
        os.remove(raw_path)
    
    key = os.path.splitext(filename)[0]
    for proc_file in glob.glob(os.path.join(PROCESSED_DIR, f"*{key}*")):
        if os.path.isfile(proc_file):
            os.remove(proc_file)

    meta = get_metadata_store()
    if "files" in meta and key in meta["files"]:
        del meta["files"][key]
        save_metadata_store(meta)

    return process_all_audios(CURRENT_PARAMS)

@app.delete("/api/audios/all")
async def delete_all_audios():
    for f in glob.glob(os.path.join(RAW_DIR, "*.*")):
        if os.path.isfile(f):
            os.remove(f)
    for f in glob.glob(os.path.join(PROCESSED_DIR, "*.*")):
        if os.path.isfile(f):
            os.remove(f)

    return process_all_audios(CURRENT_PARAMS)

@app.get("/audio/raw/{filename}")
async def stream_raw_audio(filename: str):
    file_path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Raw audio file not found")
    media_type = "audio/ogg" if filename.endswith(".ogg") else ("audio/wav" if filename.endswith(".wav") else "audio/flac")
    return FileResponse(file_path, media_type=media_type)

@app.get("/audio/processed/{filename}")
async def stream_processed_audio(filename: str):
    file_path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Processed audio file not found")
    return FileResponse(file_path, media_type="audio/wav")

if __name__ == "__main__":
    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", 8000))
    print(f"Starting Altur Voice Biometrics & Anti-Spoofing Benchmark Server on http://{HOST}:{PORT} ...")
    uvicorn.run(app, host=HOST, port=PORT)
