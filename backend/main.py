import os
import sys
import uuid
import io
import json
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Setup paths for fashn_vton
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FASHN_SRC_PATH = os.path.join(PROJECT_ROOT, "fashnvton", "src")
sys.path.append(FASHN_SRC_PATH)

from fashn_vton.pipeline import TryOnPipeline

app = FastAPI(title="Couture AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None
upscaler = None
jobs: dict = {}  # job_id → {status, step, image_urls, error}

STORAGE_DIR = os.path.join(PROJECT_ROOT, "storage", "vto_results")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
GARMENTS_FILE = os.path.join(DATA_DIR, "garments.json")
os.makedirs(STORAGE_DIR, exist_ok=True)

# Debug: Log paths on startup
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"GARMENTS_FILE: {GARMENTS_FILE}")
print(f"GARMENTS_FILE exists: {os.path.exists(GARMENTS_FILE)}")


@app.on_event("startup")
async def startup_event():
    global pipeline, upscaler
    weights_dir = os.path.join(PROJECT_ROOT, "fashnvton", "weights")
    print(f"📦 Loading VTO Pipeline from {weights_dir}...")
    
    # Detect device availability
    if torch.cuda.is_available():
        device = "cuda"
        device_info = "GPU (CUDA)"
    else:
        device = "cpu"
        device_info = "CPU"
    
    print(f"Using device: {device_info}")
    pipeline = TryOnPipeline(weights_dir=weights_dir, device=device)
    print(f"✅ Pipeline loaded on {device_info}")

    print("✨ Loading Real-ESRGAN Upscaler...")
    torch_device = torch.device(device)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upscaler = RealESRGANer(
        scale=4,
        model_path=os.path.join(weights_dir, 'RealESRGAN_x4plus.pth'),
        model=model,
        tile=256,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=torch_device
    )
    print(f"✅ Real-ESRGAN loaded on {device_info}")


def _run_tryon_job(job_id: str, person_img: Image.Image, garment_img: Image.Image,
                   category: str, num_samples: int):
    """Sync function — runs in BackgroundTasks thread pool."""
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["step"] = "generating"
        print(f"🚀 [{job_id[:8]}] category={category} samples={num_samples}")

        result = pipeline(
            person_image=person_img,
            garment_image=garment_img,
            category=category,
            num_timesteps=20,
            num_samples=num_samples,
        )

        jobs[job_id]["step"] = "upscaling"
        print(f"✨ [{job_id[:8]}] Upscaling {len(result.images)} image(s)...")

        image_urls = []
        for result_img in result.images:
            img_np = np.array(result_img)
            high_res_np, _ = upscaler.enhance(img_np, outscale=4)
            high_res_img = Image.fromarray(high_res_np)
            filename = f"result_{uuid.uuid4()}.png"
            high_res_img.save(os.path.join(STORAGE_DIR, filename))
            image_urls.append(f"/results/{filename}")

        jobs[job_id].update({"status": "done", "step": "done", "image_urls": image_urls})
        print(f"✅ [{job_id[:8]}] Done — {len(image_urls)} result(s)")

    except Exception as e:
        print(f"❌ [{job_id[:8]}] {str(e)}")
        jobs[job_id].update({"status": "failed", "error": str(e)})


@app.post("/tryon")
async def run_tryon(
    background_tasks: BackgroundTasks,
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: str = Form("tops"),
    num_samples: int = Form(1),
):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    # Read files before returning (UploadFile is not safe to use after response)
    person_bytes = await person_image.read()
    garment_bytes = await garment_image.read()
    person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
    garment_img = Image.open(io.BytesIO(garment_bytes)).convert("RGB")

    num_samples = max(1, min(num_samples, 4))  # cap at 4 for GPU processing

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "step": "queued", "image_urls": [], "error": None}

    background_tasks.add_task(_run_tryon_job, job_id, person_img, garment_img, category, num_samples)

    return {"job_id": job_id, "status": "pending"}


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/garments")
async def get_garments():
    if not os.path.exists(GARMENTS_FILE):
        print(f"❌ Garments file not found: {GARMENTS_FILE}")
        raise HTTPException(status_code=404, detail=f"garments.json not found at {GARMENTS_FILE}")
    try:
        with open(GARMENTS_FILE) as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} garments from {GARMENTS_FILE}")
        return data
    except Exception as e:
        print(f"❌ Error loading garments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/{filename}")
async def get_result(filename: str):
    file_path = os.path.join(STORAGE_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(file_path)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "active_jobs": len([j for j in jobs.values() if j["status"] in ("pending", "processing")]),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
