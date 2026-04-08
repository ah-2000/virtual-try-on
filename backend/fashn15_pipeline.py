"""
FASHN VTON v1.5 inference pipeline wrapper.
Primary model — 972M params, fits in 8GB VRAM, maskless architecture.

Phase 1: Optimized parameters for best quality on consumer GPU.
Phase 2: Smart category auto-detection from garment image.
"""
import os
import torch
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FASHN_DIR = os.path.join(PROJECT_ROOT, "fashn_vton15")
WEIGHTS_DIR = os.path.join(FASHN_DIR, "weights")

_pipeline = None
_device = "cpu"


def is_available() -> bool:
    """Check if FASHN VTON v1.5 weights are ready."""
    model_path = os.path.join(WEIGHTS_DIR, "model.safetensors")
    dwpose_dir = os.path.join(WEIGHTS_DIR, "dwpose")
    yolox_path = os.path.join(dwpose_dir, "yolox_l.onnx")
    dwpose_path = os.path.join(dwpose_dir, "dw-ll_ucoco_384.onnx")

    if not os.path.isfile(model_path):
        print(f"[FASHN] model.safetensors not found at {model_path}")
        return False
    if not os.path.isfile(yolox_path) or not os.path.isfile(dwpose_path):
        print(f"[FASHN] DWPose weights not found at {dwpose_dir}")
        return False
    return True


def load(device: str = "cuda") -> None:
    """Load FASHN VTON v1.5 pipeline."""
    global _pipeline, _device

    if _pipeline is not None:
        return

    if not is_available():
        raise RuntimeError(
            "FASHN VTON v1.5 weights not found. Run:\n"
            "  python fashn_vton15/scripts/download_weights.py --weights-dir fashn_vton15/weights"
        )

    print("[FASHN] Loading FASHN VTON v1.5 pipeline...")

    from fashn_vton import TryOnPipeline

    _pipeline = TryOnPipeline(weights_dir=WEIGHTS_DIR, device=device)
    _device = device

    vram = torch.cuda.memory_allocated(0) / 1024**3 if device == "cuda" else 0
    print(f"[FASHN] Pipeline ready on {device} ({vram:.1f}GB VRAM)")


def get_pipeline():
    """Expose pipeline for pose validation etc."""
    return _pipeline


# ---------------------------------------------------------------------------
# Phase 2: Smart category auto-detection
# ---------------------------------------------------------------------------

def _detect_garment_category(garment_img: Image.Image) -> str:
    """
    Auto-detect garment category using FashnHumanParser segmentation.

    Analyzes the garment image to determine if it's:
    - "tops"       : mostly upper body coverage (shirt, kurta top, blouse)
    - "bottoms"    : mostly lower body coverage (pants, shalwar, skirt)
    - "one-pieces" : full body coverage (dress, shalwar kameez set, jumpsuit)

    This is critical for Pakistani clothing where users often upload
    full outfit photos but select "tops" — causing only upper body to change.
    """
    if _pipeline is None:
        return "tops"

    garment_np = np.array(garment_img.convert("RGB"))
    seg_pred = _pipeline.hp_model.predict(garment_np)

    from fashn_human_parser import LABELS_TO_IDS

    h, w = seg_pred.shape[:2]
    total_pixels = h * w

    # Count pixels for each body region
    top_ids = [LABELS_TO_IDS.get("top", -1), LABELS_TO_IDS.get("scarf", -1)]
    bottom_ids = [LABELS_TO_IDS.get("pants", -1), LABELS_TO_IDS.get("skirt", -1)]
    dress_ids = [LABELS_TO_IDS.get("dress", -1)]

    top_pixels = sum(np.sum(seg_pred == i) for i in top_ids if i >= 0)
    bottom_pixels = sum(np.sum(seg_pred == i) for i in bottom_ids if i >= 0)
    dress_pixels = sum(np.sum(seg_pred == i) for i in dress_ids if i >= 0)

    # Calculate ratios
    top_ratio = top_pixels / total_pixels
    bottom_ratio = bottom_pixels / total_pixels
    dress_ratio = dress_pixels / total_pixels
    clothing_ratio = top_ratio + bottom_ratio + dress_ratio

    # Decision logic
    # If dress pixels present at all → one-pieces (dresses are clearly detected)
    if dress_ratio > 0.03:
        category = "one-pieces"
    # If both top AND bottom are present → full outfit → one-pieces
    # (shalwar kameez: top detected + pants/skirt detected = full outfit)
    elif top_ratio > 0.03 and bottom_ratio > 0.03:
        category = "one-pieces"
    # If significant clothing but mostly bottom → bottoms
    elif bottom_ratio > top_ratio and bottom_ratio > 0.05:
        category = "bottoms"
    # Default to tops
    else:
        category = "tops"

    print(f"[FASHN] Auto-detected category: {category} "
          f"(top={top_ratio:.1%}, bottom={bottom_ratio:.1%}, dress={dress_ratio:.1%})")

    return category


def run_tryon(
    person_img: Image.Image,
    garment_img: Image.Image,
    category: str = "tops",
    garment_photo_type: str = "model",
    num_samples: int = 1,
    num_timesteps: int = 30,
    guidance_scale: float = 2.0,
    seed: int = 42,
) -> list[Image.Image]:
    """
    Run FASHN VTON v1.5 try-on with optimized parameters.

    Phase 1 optimizations:
        - num_timesteps=30 (was 25) — better quality, ~35s total
        - guidance_scale=2.0 (was 1.5) — stronger garment adherence
        - segmentation_free=True — preserves body, allows garment volume
        - skip_cfg_last_n_steps=1 — prevents color saturation at end

    Phase 2: auto-detects category if garment is a full outfit.
    """
    if _pipeline is None:
        raise RuntimeError("FASHN VTON not loaded. Call load() first.")

    # Phase 2: Smart category detection
    # If user selected "tops" but garment is actually a full outfit,
    # override to "one-pieces" for better results
    detected_category = _detect_garment_category(garment_img)

    # Only override if detected is MORE coverage than user selected
    # (e.g., user said "tops" but it's a full dress → override to "one-pieces")
    # Never downgrade (user said "one-pieces" → keep it even if detected as "tops")
    COVERAGE_RANK = {"tops": 1, "bottoms": 1, "one-pieces": 2}
    if COVERAGE_RANK.get(detected_category, 0) > COVERAGE_RANK.get(category, 0):
        print(f"[FASHN] Category override: {category} → {detected_category} (full outfit detected)")
        category = detected_category

    # For "tops" use segmentation_free=True (preserve lower body)
    # For "one-pieces"/"bottoms" use segmentation_free=False (proper full masking
    # gives better garment length accuracy for long dresses/kurtas)
    seg_free = category == "tops"

    result = _pipeline(
        person_image=person_img.convert("RGB"),
        garment_image=garment_img.convert("RGB"),
        category=category,
        garment_photo_type=garment_photo_type,
        num_samples=num_samples,
        num_timesteps=num_timesteps,
        guidance_scale=guidance_scale,
        seed=seed,
        segmentation_free=seg_free,
        skip_cfg_last_n_steps=1,      # Phase 1: prevent color saturation
    )

    return result.images
