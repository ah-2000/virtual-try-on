import os
import sys
import uuid
import io
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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

app = FastAPI(title="Virtual Try-On API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None
upscaler = None

STORAGE_DIR = os.path.join(PROJECT_ROOT, "storage", "vto_results")
os.makedirs(STORAGE_DIR, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    global pipeline, upscaler
    weights_dir = os.path.join(PROJECT_ROOT, "fashnvton", "weights")
    print(f"📦 Loading VTO Pipeline from {weights_dir}...")
    pipeline = TryOnPipeline(weights_dir=weights_dir, device="cpu")
    print("✅ Pipeline loaded successfully on CPU")
    
    print("✨ Loading Real-ESRGAN Upscaler...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upscaler = RealESRGANer(
        scale=4,
        model_path=os.path.join(weights_dir, 'RealESRGAN_x4plus.pth'),
        model=model,
        tile=256,  # Tiling prevents OOM and OS killing process on CPU
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device
    )
    print("✅ Real-ESRGAN loaded successfully")

@app.post("/tryon")
async def run_tryon(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: str = Form("tops") # tops, bottoms, one-pieces
):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    try:
        # Load images
        person_bytes = await person_image.read()
        garment_bytes = await garment_image.read()
        
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        garment_img = Image.open(io.BytesIO(garment_bytes)).convert("RGB")

        print(f"🚀 Processing try-on for category: {category}")
        
        # Run inference
        result = pipeline(
            person_image=person_img,
            garment_image=garment_img,
            category=category,
            num_timesteps=20, # Fast mode for API
        )

        # Upscale result
        print("✨ Upscaling the result...")
        result_img = result.images[0]
        img_np = np.array(result_img)
        high_res_np, _ = upscaler.enhance(img_np, outscale=4)
        high_res_img = Image.fromarray(high_res_np)

        # Save result
        result_id = str(uuid.uuid4())
        output_filename = f"result_{result_id}.png"
        output_path = os.path.join(STORAGE_DIR, output_filename)
        high_res_img.save(output_path)

        return {
            "status": "success",
            "result_id": result_id,
            "image_url": f"/results/{output_filename}"
        }

    except Exception as e:
        print(f"❌ Error during try-on: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{filename}")
async def get_result(filename: str):
    file_path = os.path.join(STORAGE_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(file_path)

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": pipeline is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
