import sys
import os
import torch
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# 1. Setup paths to find the 'fashn_vton' package inside 'fashnvton/src'
project_root = os.path.dirname(os.path.abspath(__file__))
fashn_src_path = os.path.join(project_root, "fashnvton", "src")
sys.path.append(fashn_src_path)

# 2. Import the correct class from the package
from fashn_vton.pipeline import TryOnPipeline

# 3. Initialize pipeline
weights_dir = os.path.join(project_root, "fashnvton", "weights")
# The class in your source is named TryOnPipeline
pipeline = TryOnPipeline(weights_dir=weights_dir, device="cpu")

# 4. Load your images
person = Image.open(os.path.join(project_root, "..", "assets", "WhatsApp Image 2026-03-11 at 12.11.42 PM.jpeg")).convert("RGB")
# Use one of the new Khaadi dresses
garment = Image.open(os.path.join(project_root, "..", "frontend", "public", "garments", "Screenshot 2026-03-16 at 1.16.22 PM.png")).convert("RGB")

# 5. Run try-on
print("🚀 Running Virtual Try-On (this may take a few minutes on CPU)...")
result = pipeline(
    person_image=person,
    garment_image=garment,
    category="tops",  # "tops" | "bottoms" | "one-pieces"
)

# 6. Initialize Real-ESRGAN
print("✨ Upscaling the result with Real-ESRGAN...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# FASHN TryOn model weights are small, we can use RRDBNet for ESRGAN
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upscaler = RealESRGANer(
    scale=4,
    model_path=os.path.join(weights_dir, 'RealESRGAN_x4plus.pth'),
    model=model,
    tile=256,  # Use tiling to prevent Out-Of-Memory (OOM) / "Killed" on CPU
    tile_pad=10,
    pre_pad=0,
    half=False, # Use fp32 for CPU/MPS or older GPUs
    device=device
)

# 7. Upscale output (Convert PIL to Numpy for ESRGAN, then back)
result_image = result.images[0]
img_np = np.array(result_image)
high_res_np, _ = upscaler.enhance(img_np, outscale=4)
high_res_image = Image.fromarray(high_res_np)

# 8. Save output
output_dir = os.path.join(project_root, "..", "assets")
high_res_image.save(os.path.join(output_dir, "output4_upscaled.png"))
result_image.save(os.path.join(output_dir, "output4_original.png"))
print("✅ Done! Saved original to assets/output4_original.png and upscaled to assets/output4_upscaled.png")
