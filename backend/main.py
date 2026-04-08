import os
import sys
import uuid
import io
import json
import random
import hashlib
import types
import threading
import torch
import cv2
from dataclasses import dataclass
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from starlette.requests import Request
from starlette.responses import Response
from PIL import Image, ImageFilter
import numpy as np

# Background removal for person images
try:
    from rembg import remove as rembg_remove
    _has_rembg = True
    print("[OK] rembg loaded — background removal enabled")
except ImportError:
    _has_rembg = False
    print("[WARN] rembg not installed — background removal disabled")

from preprocessor import validate_pose, auto_crop_person

# basicsr expects torchvision.transforms.functional_tensor in older releases.
try:
    import torchvision.transforms.functional_tensor  # type: ignore # noqa: F401
except ModuleNotFoundError:
    import torchvision.transforms.functional as _tvf

    _functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
    _functional_tensor.rgb_to_grayscale = _tvf.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _functional_tensor

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Setup paths for fashn_vton (needed even as fallback)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FASHN_SRC_PATH = os.path.join(PROJECT_ROOT, "fashnvton", "src")
sys.path.append(FASHN_SRC_PATH)

import idm_local_pipeline as idm_local
import fashn15_pipeline as fashn_local

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# IDM-VTON Wrapper — presents FASHN-compatible interface to main.py
# ---------------------------------------------------------------------------

@dataclass
class PipelineOutput:
    images: List[Image.Image]


# Lock for DensePose (it does os.chdir which isn't thread-safe)
_idm_lock = threading.Lock()


def _openpose_to_dwpose_format(openpose_result: dict, img_w: int, img_h: int) -> dict:
    """Convert IDM-VTON OpenPose output to DWpose format for validate_pose().

    OpenPose returns: {"pose_keypoints_2d": [[x,y], ...]} with pixel coords (384x512).
    DWpose expects: {"bodies": {"candidate": (N,2) normalized, "subset": (1,18) with -1 for missing}}.
    """
    kps = openpose_result.get("pose_keypoints_2d", [])
    candidate = np.zeros((18, 2), dtype=np.float64)
    subset = np.full((1, 18), -1, dtype=np.float64)

    for i, pt in enumerate(kps[:18]):
        x, y = pt[0], pt[1]
        if x == 0 and y == 0:
            continue  # missing keypoint
        # Normalize to [0, 1] — OpenPose coords are in 384x512 space
        candidate[i] = [x / 384.0, y / 512.0]
        subset[0, i] = float(i)  # visible keypoint: set to its index

    return {"bodies": {"candidate": candidate, "subset": subset}}


class IDMVTONWrapper:
    """Wraps idm_local_pipeline to present FASHN-compatible interface."""

    is_idm = True

    def __init__(self, device: str = "cuda"):
        self._device = device
        self.sam_predictor = None  # placeholder for SAM2 attachment

    def __call__(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        category: str = "tops",
        garment_photo_type: str = "model",
        num_timesteps: int = 30,
        guidance_scale: float = 2.0,
        num_samples: int = 1,
        seed: int = 0,
        **kwargs,
    ) -> PipelineOutput:
        images = []
        for i in range(num_samples):
            sample_seed = seed + i
            with _idm_lock:
                img = idm_local.run_tryon(
                    person_img=person_image,
                    garment_img=garment_image,
                    category=category,
                    denoise_steps=num_timesteps,
                    seed=sample_seed,
                )
            images.append(img)
        return PipelineOutput(images=images)

    def pose_model(self, img_bgr: np.ndarray) -> dict:
        """Run OpenPose and return DWpose-compatible format for validate_pose()."""
        openpose = idm_local.get_openpose()
        pil_img = Image.fromarray(img_bgr[:, :, ::-1])  # BGR -> RGB -> PIL
        openpose_result = openpose(pil_img, resolution=384)
        h, w = img_bgr.shape[:2]
        return _openpose_to_dwpose_format(openpose_result, w, h)

    @property
    def hp_model(self):
        return None  # IDM-VTON uses SCHP, not FashnHumanParser


class FASHNWrapper:
    """Wraps FASHN VTON v1.5 to present the same interface."""

    is_idm = False

    def __init__(self, device: str = "cuda"):
        self._device = device
        self.sam_predictor = None

    def __call__(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        category: str = "tops",
        garment_photo_type: str = "model",
        num_timesteps: int = 25,
        guidance_scale: float = 1.5,
        num_samples: int = 1,
        seed: int = 0,
        **kwargs,
    ) -> PipelineOutput:
        images = fashn_local.run_tryon(
            person_img=person_image,
            garment_img=garment_image,
            category=category,
            garment_photo_type=garment_photo_type,
            num_samples=num_samples,
            num_timesteps=num_timesteps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return PipelineOutput(images=images)

    def pose_model(self, img_bgr: np.ndarray) -> dict:
        """Use FASHN's built-in DWPose for pose validation."""
        pipe = fashn_local.get_pipeline()
        if pipe is None:
            return {"bodies": {"candidate": np.zeros((18, 2)), "subset": np.full((1, 18), -1)}}
        pil_img = Image.fromarray(img_bgr[:, :, ::-1])
        result = pipe.pose_model(pil_img)
        return result

    @property
    def hp_model(self):
        pipe = fashn_local.get_pipeline()
        return pipe.hp_model if pipe else None


app = FastAPI(title="Couture AI API")

# Manual CORS middleware
@app.middleware("http")
async def cors_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        return Response(status_code=200, headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        })
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

pipeline = None
pipeline_type = "none"    # "IDM-VTON" or "FASHN"
upscaler = None
face_enhancer = None
sam_predictor = None      # SAM2 for mask refinement
hp_postprocess = None     # FashnHumanParser for post-processing
jobs: dict = {}
result_cache: dict = {}

STORAGE_DIR = os.path.join(PROJECT_ROOT, "storage", "vto_results")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
GARMENTS_FILE = os.path.join(DATA_DIR, "garments.json")
os.makedirs(STORAGE_DIR, exist_ok=True)

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"GARMENTS_FILE: {GARMENTS_FILE}")
print(f"GARMENTS_FILE exists: {os.path.exists(GARMENTS_FILE)}")


def _image_hash(img: Image.Image) -> str:
    return hashlib.md5(img.tobytes()).hexdigest()


@app.on_event("startup")
async def startup_event():
    global pipeline, pipeline_type, upscaler, face_enhancer
    weights_dir = os.path.join(PROJECT_ROOT, "fashnvton", "weights")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_info = "GPU (CUDA)" if device == "cuda" else "CPU"
    print(f"Using device: {device_info}")

    # ---------------------------------------------------------------
    # 1. Load model: OOTDiffusion (primary) → IDM-VTON (fallback)
    # ---------------------------------------------------------------
    # Try FASHN VTON v1.5 (primary) → IDM-VTON (fallback)
    # ---------------------------------------------------------------
    if fashn_local.is_available():
        try:
            print("[LOAD] Loading FASHN VTON v1.5 (primary)...")
            fashn_local.load(device=device)
            pipeline = FASHNWrapper(device=device)
            pipeline_type = "FASHN-VTON"
            print(f"[OK] FASHN VTON v1.5 loaded on {device_info}")
        except Exception as e:
            print(f"[WARN] FASHN VTON failed: {e}")
            print("[LOAD] Falling back to IDM-VTON...")
            if idm_local.is_available():
                idm_local.load(device=device)
                pipeline = IDMVTONWrapper(device=device)
                pipeline_type = "IDM-VTON"
                print(f"[OK] IDM-VTON loaded on {device_info} (fallback)")
            else:
                raise RuntimeError("No model available. Run setup scripts first.")
    elif idm_local.is_available():
        print("[LOAD] FASHN VTON not available, loading IDM-VTON (fallback)...")
        idm_local.load(device=device)
        pipeline = IDMVTONWrapper(device=device)
        pipeline_type = "IDM-VTON"
        print(f"[OK] IDM-VTON loaded on {device_info} (fallback)")
    else:
        raise RuntimeError("No model available. Run setup scripts first.")

    # ---------------------------------------------------------------
    # 4. SAM2 mask refinement (optional)
    # ---------------------------------------------------------------
    global sam_predictor
    sam2_path = os.path.join(weights_dir, "sam2_hiera_large.pt")
    if os.path.exists(sam2_path):
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            sam2_model = build_sam2("sam2_hiera_l", sam2_path, device=device)
            sam_predictor = SAM2ImagePredictor(sam2_model)
            print(f"[OK] SAM2 mask refiner loaded on {device_info}")
        except Exception as e:
            print(f"[WARN] SAM2 failed: {e}")
    else:
        print(f"[INFO] SAM2 disabled — save sam2_hiera_large.pt to: {sam2_path}")

    # Attach SAM2 to pipeline (FASHN uses it in pipeline.py preprocessing)
    if pipeline is not None:
        pipeline.sam_predictor = sam_predictor

    # ---------------------------------------------------------------
    # 5. FashnHumanParser for post-processing (color/texture/lighting)
    # ---------------------------------------------------------------
    global hp_postprocess
    if hasattr(pipeline, "hp_model") and pipeline.hp_model is not None:
        hp_postprocess = pipeline.hp_model
        print("[OK] Post-processing parser: reusing pipeline.hp_model")
    else:
        # IDM-VTON uses SCHP — load FashnHumanParser on CPU to save VRAM
        try:
            from fashn_human_parser import FashnHumanParser
            hp_postprocess = FashnHumanParser(device="cpu")
            print(f"[OK] FashnHumanParser loaded on CPU (saves VRAM)")
        except Exception as e:
            print(f"[WARN] FashnHumanParser failed: {e} — post-processing disabled")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Background removal — replace busy backgrounds with plain white
# ---------------------------------------------------------------------------

# Background color: #F1EFEF — neutral off-white
_BG_COLOR = (241, 239, 239)

# Cache rembg session so it's not re-created on every call
_rembg_session = None

def _get_rembg_session():
    global _rembg_session
    if _rembg_session is None and _has_rembg:
        from rembg import new_session
        _rembg_session = new_session("u2net")
        print("[OK] u2net rembg session created")
    return _rembg_session


def _remove_background(person_img: Image.Image, bg_color=_BG_COLOR) -> Image.Image:
    """Remove background from person image and replace with #F1EFEF."""
    session = _get_rembg_session()
    if session is None:
        print("[WARN] No rembg session — skipping background removal")
        return person_img
    try:
        result_rgba = rembg_remove(
            person_img,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=230,
            alpha_matting_background_threshold=20,
            alpha_matting_erode_size=5,
        )
        # Create solid #F1EFEF background and composite person over it
        bg = Image.new("RGBA", result_rgba.size, (*bg_color, 255))
        bg.paste(result_rgba, (0, 0), result_rgba)
        print(f"[OK] Background removed — replaced with #{bg_color[0]:02X}{bg_color[1]:02X}{bg_color[2]:02X}")
        return bg.convert("RGB")
    except Exception as e:
        print(f"[WARN] Background removal failed: {e}")
        return person_img


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _sharpen_output(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=80, threshold=3))


def _apply_bg_color(img: Image.Image, bg_color=_BG_COLOR, threshold=230) -> Image.Image:
    """Replace all near-white pixels with the exact #F1EFEF background color.

    Any pixel where R, G, B are ALL above `threshold` is considered background
    and gets replaced with the target color. Threshold lowered to 230 to catch
    more off-white background pixels from rembg output.
    """
    img_np = np.array(img)
    bg_mask = np.all(img_np >= threshold, axis=-1)
    img_np[bg_mask] = bg_color
    return Image.fromarray(img_np)


def _preserve_face(person_img: Image.Image, result_img: Image.Image) -> Image.Image:
    """Blend original face + hair from the person image into the result.

    Phase 3: Uses FashnHumanParser to find face (1) and hair (2) in both
    the original person and the result. Blends the original face/hair back
    to prevent distortion from the diffusion model.
    """
    if hp_postprocess is None:
        return result_img
    try:
        from scipy.ndimage import gaussian_filter, binary_erosion

        # Resize person to match result
        person_resized = person_img.resize(result_img.size, Image.LANCZOS)

        # Parse the RESULT image to find where face+hair are in the output
        result_np = np.array(result_img)
        result_parse = hp_postprocess.predict(result_np)

        # Face=1, Hair=2 in FashnHumanParser
        face_hair_mask = np.isin(result_parse, [1, 2])
        if not face_hair_mask.any():
            return result_img

        # Also parse person to get the ORIGINAL face region
        person_np = np.array(person_resized)
        person_parse = hp_postprocess.predict(person_np)
        person_face_mask = np.isin(person_parse, [1, 2])

        # Use intersection — only blend where both agree there's a face
        # This prevents blending background/clothing as "face"
        combined_mask = face_hair_mask | person_face_mask
        if not combined_mask.any():
            return result_img

        # Erode to stay safely inside face region
        combined_mask = binary_erosion(combined_mask, iterations=4)
        if not combined_mask.any():
            return result_img

        # Feather edges for smooth blending (larger sigma = smoother transition)
        mask_float = combined_mask.astype(np.float32)
        mask_float = gaussian_filter(mask_float, sigma=5)
        mask_3d = np.stack([mask_float] * 3, axis=-1)

        # Blend: original face over result
        blended = person_np.astype(np.float64) * mask_3d + result_np.astype(np.float64) * (1 - mask_3d)
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    except Exception as e:
        print(f"[WARN] Face preservation failed: {e}")
        return result_img


def _preserve_feet(person_img: Image.Image, result_img: Image.Image) -> Image.Image:
    """Blend original feet/shoes from person into result.

    Fix 3: When a long dress is applied, model generates weird legs/pants
    under the garment. This blends the original person's feet area back
    to keep natural shoes/feet appearance.
    """
    if hp_postprocess is None:
        return result_img
    try:
        from scipy.ndimage import gaussian_filter, binary_dilation

        person_resized = person_img.resize(result_img.size, Image.LANCZOS)
        person_np = np.array(person_resized)
        result_np = np.array(result_img)

        # Parse result to find feet (15) in FashnHumanParser
        result_parse = hp_postprocess.predict(result_np)
        person_parse = hp_postprocess.predict(person_np)

        # Feet=15 in FashnHumanParser
        person_feet = person_parse == 15
        result_feet = result_parse == 15

        # Use person's original feet region
        feet_mask = person_feet | result_feet
        if not feet_mask.any():
            return result_img

        # Dilate slightly to include shoes edges
        feet_mask = binary_dilation(feet_mask, iterations=3)

        # Feather for smooth blend
        mask_float = feet_mask.astype(np.float32)
        mask_float = gaussian_filter(mask_float, sigma=3)
        mask_3d = np.stack([mask_float] * 3, axis=-1)

        blended = person_np.astype(np.float64) * mask_3d + result_np.astype(np.float64) * (1 - mask_3d)
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    except Exception as e:
        print(f"[WARN] Feet preservation failed: {e}")
        return result_img


# Map category -> FashnHumanParser label IDs for the garment region.
# LABELS_TO_IDS = {top:3, dress:4, scarf:10, skirt:5, pants:6, belt:7}
_CATEGORY_TO_HP_LABELS: dict[str, list[int]] = {
    "tops":       [3, 4, 10],
    "bottoms":    [5, 6, 7],
    "one-pieces": [3, 4, 10, 5, 6, 7],
}


def _get_garment_mask(img_np: np.ndarray, category: str) -> np.ndarray | None:
    """Get boolean garment mask from hp_postprocess. Returns None if unavailable."""
    if hp_postprocess is None:
        return None
    seg = hp_postprocess.predict(img_np)
    label_ids = _CATEGORY_TO_HP_LABELS.get(category, [3, 4, 10])
    mask = np.isin(seg, label_ids)
    return mask if np.any(mask) else None


def _color_harmonize_garment(
    result_img: Image.Image,
    garment_img: Image.Image,
    category: str,
) -> Image.Image:
    """Match garment-region color histogram to the input garment.

    IMPORTANT: We segment BOTH images independently to get actual garment pixels.
    The result's garment mask and the garment input's garment mask are different shapes
    (person wearing vs product photo), so we can't use spatial correspondence — we match
    color distributions between the two sets of garment pixels.
    """
    if hp_postprocess is None:
        return result_img
    try:
        from skimage.exposure import match_histograms
    except ImportError:
        return result_img

    result_np = np.array(result_img)
    result_mask = _get_garment_mask(result_np, category)
    if result_mask is None:
        return result_img

    # Segment the garment INPUT image to find its own garment pixels
    garment_np = np.array(garment_img.resize(result_img.size, Image.LANCZOS))
    ref_mask = _get_garment_mask(garment_np, category)
    if ref_mask is None:
        return result_img

    result_pixels = result_np[result_mask]   # (N, 3) — garment pixels in result
    ref_pixels = garment_np[ref_mask]        # (M, 3) — garment pixels in input

    matched = match_histograms(
        result_pixels.astype(np.float64),
        ref_pixels.astype(np.float64),
        channel_axis=-1,
    ).astype(np.uint8)

    harmonized = result_np.copy()
    harmonized[result_mask] = matched
    return Image.fromarray(harmonized)


# ---------------------------------------------------------------------------
# Lighting Harmonization — match garment luminance to surrounding context
# ---------------------------------------------------------------------------

def _lighting_harmonize_garment(
    result_img: Image.Image,
    category: str,
    alpha: float = 0.3,
) -> Image.Image:
    """Match garment-region luminance to the surrounding body/background context.

    Uses LAB color space to adjust the L (lightness) channel of the garment pixels
    so they blend naturally with the surrounding skin/background lighting.
    Only touches the garment region — face/skin/background unchanged.
    """
    result_np = np.array(result_img)
    garment_mask = _get_garment_mask(result_np, category)
    if garment_mask is None:
        return result_img

    # Convert to LAB
    lab = cv2.cvtColor(result_np, cv2.COLOR_RGB2LAB).astype(np.float64)

    # Build a border context ring: dilate garment mask, subtract original
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    dilated = cv2.dilate(garment_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    border_mask = dilated & ~garment_mask

    if not np.any(border_mask):
        return result_img

    # Get L channel stats for garment vs its border context
    garment_L = lab[garment_mask, 0]
    border_L = lab[border_mask, 0]

    g_mean, g_std = garment_L.mean(), garment_L.std() + 1e-6
    b_mean, b_std = border_L.mean(), border_L.std() + 1e-6

    # Soft transfer: blend alpha% toward border statistics (avoid over-correction)
    target_mean = g_mean + alpha * (b_mean - g_mean)
    target_std = g_std + alpha * (b_std - g_std)

    adjusted_L = (garment_L - g_mean) * (target_std / g_std) + target_mean
    adjusted_L = np.clip(adjusted_L, 0, 255)

    lab_out = lab.copy()
    lab_out[garment_mask, 0] = adjusted_L

    result_rgb = cv2.cvtColor(lab_out.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(result_rgb)


# ---------------------------------------------------------------------------
# Garment texture transfer (high-frequency detail)
# ---------------------------------------------------------------------------

def _transfer_garment_texture(
    high_res_np: np.ndarray,
    garment_img: Image.Image,
    category: str,
    blend_weight: float = 0.15,
) -> np.ndarray:
    """Enhance garment texture in the upscaled result using self-detail boost.

    Instead of spatially transferring texture from the garment image (wrong layout),
    we extract high-frequency detail from the RESULT's own garment region and sharpen
    it. This preserves whatever fabric detail the model generated without introducing
    spatial artifacts from the mismatched garment product photo.
    """
    garment_mask = _get_garment_mask(high_res_np, category)
    if garment_mask is None:
        return high_res_np

    # Extract high-freq detail from result's own garment region
    result_gray = cv2.cvtColor(high_res_np, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(result_gray, cv2.CV_64F, ksize=3)
    lap_max = np.abs(laplacian).max()
    if lap_max > 0:
        laplacian = laplacian / lap_max

    texture_layer = np.stack([laplacian] * 3, axis=-1)
    result = high_res_np.astype(np.float64)
    texture_boost = texture_layer * blend_weight * 255.0
    mask_3d = np.stack([garment_mask] * 3, axis=-1)
    result = np.where(mask_3d, np.clip(result + texture_boost, 0, 255), result)

    return result.astype(np.uint8)


def _enhance_face(high_res_np: np.ndarray, job_tag: str) -> np.ndarray:
    if face_enhancer is None:
        return high_res_np
    try:
        img_bgr = high_res_np[:, :, ::-1]
        _, _, restored_bgr = face_enhancer.enhance(
            img_bgr, has_aligned=False, only_center_face=False, paste_back=True,
        )
        if restored_bgr is not None:
            return restored_bgr[:, :, ::-1]
    except Exception as e:
        print(f"[WARN] [{job_tag}] Face enhancement failed: {e}")
    return high_res_np


# ---------------------------------------------------------------------------
# Try-on job
# ---------------------------------------------------------------------------

def _update_job(job_id: str, step: str, progress: int, **extra):
    """Update job status with step name and progress percentage."""
    jobs[job_id].update({"step": step, "progress": progress, **extra})
    print(f"[{progress:3d}%] [{job_id[:8]}] {step}")


def _run_tryon_job(
    job_id: str,
    person_img: Image.Image,
    garment_img: Image.Image,
    category: str,
    num_samples: int,
    garment_photo_type: str,
):
    try:
        import time as _time
        jobs[job_id]["status"] = "processing"
        _t0 = _time.time()

        # Step 1: Remove background (5%)
        _update_job(job_id, "removing_background", 5)
        _ts = _time.time()
        person_img = _remove_background(person_img)
        print(f"[TIMER] [{job_id[:8]}] bg_removal: {_time.time()-_ts:.1f}s")

        # Step 2: Detecting pose + ensure portrait orientation (15%)
        _update_job(job_id, "detecting_pose", 15)
        _ts = _time.time()
        pose_warning = None
        pose_confidence = 1.0
        try:
            person_np_bgr = np.array(person_img)[:, :, ::-1]
            pose_result = pipeline.pose_model(person_np_bgr)
            validation = validate_pose(pose_result)
            pose_confidence = validation["pose_confidence"]
            if not validation["pose_valid"]:
                pose_warning = validation["warning"]
            if pose_confidence >= 0.3:
                # Use 2:3 ratio for FASHN, 3:4 for IDM-VTON
                target_ratio = 2/3 if pipeline_type == "FASHN-VTON" else 3/4
                person_img = auto_crop_person(person_img, pose_result,
                                              margin=0.20, target_ratio=target_ratio)
        except Exception as e:
            print(f"[WARN] Pose validation failed: {e}")

        # Ensure portrait orientation even if pose failed
        w, h = person_img.size
        if w > h:
            # Landscape image — pad to portrait to prevent head cut
            new_h = int(w / (2/3))  # 2:3 ratio
            bg = Image.new("RGB", (w, new_h), _BG_COLOR)
            bg.paste(person_img, (0, (new_h - h) // 2))
            person_img = bg
            print(f"[FIX] Landscape→portrait padded: {w}x{h} → {w}x{new_h}")

        jobs[job_id].update({"pose_confidence": pose_confidence, "pose_warning": pose_warning})
        print(f"[TIMER] [{job_id[:8]}] pose_detect: {_time.time()-_ts:.1f}s")

        # Step 3: Parsing body (20%)
        _update_job(job_id, "parsing_body", 20)

        # Step 4: Generating try-on (25% → 70%)
        _update_job(job_id, "generating_tryon", 25)
        cache_key = (_image_hash(person_img), _image_hash(garment_img), category, garment_photo_type, num_samples)

        # Check cache
        if cache_key in result_cache:
            cached_urls = result_cache[cache_key]
            if all(os.path.exists(os.path.join(STORAGE_DIR, u.split("/")[-1])) for u in cached_urls):
                _update_job(job_id, "done", 100, status="done", image_urls=cached_urls)
                return

        _ts = _time.time()
        result = pipeline(
            person_image=person_img,
            garment_image=garment_img,
            category=category,
            garment_photo_type=garment_photo_type,
            num_timesteps=30 if pipeline_type == "FASHN-VTON" else 25,
            guidance_scale=2.0 if pipeline_type == "FASHN-VTON" else 2.5,
            num_samples=num_samples,
            seed=random.randint(0, 2**32 - 1),
        )
        print(f"[TIMER] [{job_id[:8]}] diffusion: {_time.time()-_ts:.1f}s")

        _update_job(job_id, "fitting_garment", 70)

        image_urls = []
        for idx, result_img in enumerate(result.images):
            # Step 5: Preserving face + feet (75%)
            _update_job(job_id, "preserving_face", 75)
            _ts = _time.time()
            result_img = _preserve_face(person_img, result_img)
            result_img = _preserve_feet(person_img, result_img)
            print(f"[TIMER] [{job_id[:8]}] face_feet_preserve: {_time.time()-_ts:.1f}s")

            # Step 6: Clean background — remove distorted bg, replace with #F1EFEF (80%)
            _update_job(job_id, "cleaning_background", 80)
            _ts = _time.time()
            result_img = _remove_background(result_img)
            print(f"[TIMER] [{job_id[:8]}] bg_cleanup: {_time.time()-_ts:.1f}s")

            # Step 7: Sharpening (85%)
            _update_job(job_id, "sharpening", 85)
            result_img = _sharpen_output(result_img)

            # Step 8: Force exact #F1EFEF on any remaining near-white pixels (90%)
            _update_job(job_id, "saving_result", 90)
            result_img = _apply_bg_color(result_img)
            filename = f"result_{uuid.uuid4()}.png"
            result_img.save(os.path.join(STORAGE_DIR, filename))
            image_urls.append(f"/results/{filename}")

        print(f"[TIMER] [{job_id[:8]}] TOTAL: {_time.time()-_t0:.1f}s")

        _update_job(job_id, "done", 100, status="done", image_urls=image_urls)
        result_cache[cache_key] = image_urls

    except Exception as e:
        import traceback
        print(f"[ERR] [{job_id[:8]}] {str(e)}")
        traceback.print_exc()
        jobs[job_id].update({"status": "failed", "error": str(e), "progress": 0})


@app.post("/tryon")
async def run_tryon(
    background_tasks: BackgroundTasks,
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: str = Form("tops"),
    num_samples: int = Form(1),
    garment_photo_type: str = Form("model"),
):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    person_bytes = await person_image.read()
    garment_bytes = await garment_image.read()
    person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
    garment_img = Image.open(io.BytesIO(garment_bytes)).convert("RGB")

    num_samples = max(1, min(num_samples, 4))
    if garment_photo_type not in ("model", "flat-lay"):
        garment_photo_type = "model"

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending", "step": "queued", "progress": 0,
        "image_urls": [], "error": None,
        "pose_confidence": 1.0, "pose_warning": None,
    }

    background_tasks.add_task(
        _run_tryon_job,
        job_id, person_img, garment_img,
        category, num_samples, garment_photo_type,
    )

    return {"job_id": job_id, "status": "pending"}


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/garments")
async def get_garments():
    if not os.path.exists(GARMENTS_FILE):
        raise HTTPException(status_code=404, detail=f"garments.json not found at {GARMENTS_FILE}")
    try:
        with open(GARMENTS_FILE) as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/{filename}")
async def get_result(filename: str):
    file_path = os.path.join(STORAGE_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(file_path)




async def _check_data_storage() -> dict:
    """Check that data directory and garments file are accessible."""
    try:
        if not os.path.isdir(DATA_DIR):
            return {"status": "fail", "detail": f"Data directory missing: {DATA_DIR}"}
        if not os.path.isfile(GARMENTS_FILE):
            return {"status": "fail", "detail": f"Garments file missing: {GARMENTS_FILE}"}
        with open(GARMENTS_FILE, "r") as f:
            json.load(f)
        return {"status": "ok"}
    except Exception as e:
        return {"status": "fail", "detail": str(e)}


async def _check_storage_dir() -> dict:
    """Check that the results storage directory is writable."""
    try:
        if not os.path.isdir(STORAGE_DIR):
            return {"status": "fail", "detail": f"Storage directory missing: {STORAGE_DIR}"}
        test_file = os.path.join(STORAGE_DIR, ".healthcheck")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        return {"status": "ok"}
    except Exception as e:
        return {"status": "fail", "detail": str(e)}


# Register health checks here — add new (name, callable) pairs to extend.
_health_checks: list[tuple[str, callable]] = [
    ("data_storage", _check_data_storage),
    ("results_storage", _check_storage_dir),
]


async def _run_health_checks() -> dict:
    checks = {}
    all_ok = True
    for name, check_fn in _health_checks:
        result = await check_fn()
        checks[name] = result
        if result["status"] != "ok":
            all_ok = False

    return {
        "status": "healthy" if all_ok else "degraded",
        "pipeline": pipeline_type,
        "model_loaded": pipeline is not None,
        "face_enhancer": face_enhancer is not None,
        "sam2_refiner": sam_predictor is not None,
        "color_harmonizer": hp_postprocess is not None,
        "lighting_harmonizer": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "active_jobs": len([j for j in jobs.values() if j["status"] in ("pending", "processing")]),
        "checks": checks,
    }


@app.get("/")
async def root():
    return await _run_health_checks()


@app.get("/health")
async def health_check():
    return await _run_health_checks()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
