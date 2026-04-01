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
    # 1. Try IDM-VTON first (higher quality), fallback to FASHN VTON
    # ---------------------------------------------------------------
    idm_loaded = False
    if idm_local.is_available():
        try:
            print("[LOAD] Loading IDM-VTON as primary pipeline...")
            idm_local.load(device=device)
            pipeline = IDMVTONWrapper(device=device)
            pipeline_type = "IDM-VTON"
            idm_loaded = True
            print(f"[OK] IDM-VTON loaded on {device_info}")
        except Exception as e:
            import traceback
            print(f"[WARN] IDM-VTON load failed: {e}")
            traceback.print_exc()

    if not idm_loaded:
        from fashn_vton.pipeline import TryOnPipeline
        print(f"[LOAD] Loading FASHN VTON (fallback) from {weights_dir}...")
        pipeline = TryOnPipeline(weights_dir=weights_dir, device=device)
        pipeline_type = "FASHN"

        # bfloat16 + torch.compile for FASHN only
        if device == "cuda" and torch.cuda.is_bf16_supported():
            pipeline.tryon_model = pipeline.tryon_model.to(torch.bfloat16)
            print("[FAST] Model cast to bfloat16")
        if device == "cuda" and hasattr(torch, "compile"):
            print("[FAST] Compiling model with torch.compile (first run will be slow)...")
            pipeline.tryon_model = torch.compile(pipeline.tryon_model, mode="reduce-overhead")
            print("[OK] Model compiled")

        print(f"[OK] FASHN VTON loaded on {device_info}")

    # ---------------------------------------------------------------
    # 2. Real-ESRGAN upscaler
    # ---------------------------------------------------------------
    print("[LOAD] Loading Real-ESRGAN Upscaler...")
    torch_device = torch.device(device)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upscaler = RealESRGANer(
        scale=4,
        model_path=os.path.join(weights_dir, 'RealESRGAN_x4plus.pth'),
        model=model,
        tile=512 if device == "cuda" else 256,
        tile_pad=10, pre_pad=0, half=False, device=torch_device
    )
    print(f"[OK] Real-ESRGAN loaded on {device_info}")

    # ---------------------------------------------------------------
    # 3. Face enhancement (CodeFormer preferred, GFPGAN fallback)
    # ---------------------------------------------------------------
    codeformer_path = os.path.join(weights_dir, "codeformer.pth")
    gfpgan_path = os.path.join(weights_dir, "GFPGANv1.4.pth")

    if os.path.exists(codeformer_path):
        try:
            from codeformer import CodeFormer as _CF
            face_enhancer = _CF(model_path=codeformer_path, upscale=1, device=torch_device)
            print(f"[OK] CodeFormer loaded on {device_info}")
        except Exception as e:
            print(f"[WARN] CodeFormer failed: {e}")
            if os.path.exists(gfpgan_path):
                try:
                    from gfpgan import GFPGANer
                    face_enhancer = GFPGANer(
                        model_path=gfpgan_path, upscale=1, arch="clean",
                        channel_multiplier=2, bg_upsampler=None, device=torch_device,
                    )
                    print(f"[OK] GFPGAN loaded (CodeFormer fallback) on {device_info}")
                except Exception as e2:
                    print(f"[WARN] GFPGAN also failed: {e2}")
    elif os.path.exists(gfpgan_path):
        try:
            from gfpgan import GFPGANer
            face_enhancer = GFPGANer(
                model_path=gfpgan_path, upscale=1, arch="clean",
                channel_multiplier=2, bg_upsampler=None, device=torch_device,
            )
            print(f"[OK] GFPGAN loaded on {device_info}")
        except Exception as e:
            print(f"[WARN] GFPGAN failed: {e}")
    else:
        print(
            "[INFO] No face enhancer weights found.\n"
            f"   Save codeformer.pth or GFPGANv1.4.pth to: {weights_dir}"
        )

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
        # IDM-VTON uses SCHP — load FashnHumanParser separately for post-processing
        try:
            from fashn_human_parser import FashnHumanParser
            hp_postprocess = FashnHumanParser(device=device)
            print(f"[OK] FashnHumanParser loaded separately for post-processing on {device_info}")
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
        _rembg_session = new_session("birefnet-general")
        print("[OK] BiRefNet-general session created")
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


def _apply_bg_color(img: Image.Image, bg_color=_BG_COLOR, threshold=240) -> Image.Image:
    """Replace all near-white pixels with the exact #F1EFEF background color.

    Any pixel where R, G, B are ALL above `threshold` is considered background
    and gets replaced with the target color. This ensures the exact background
    color regardless of what rembg or the diffusion model outputs.
    """
    img_np = np.array(img)
    # Mask: all channels above threshold = background pixel
    bg_mask = np.all(img_np >= threshold, axis=-1)
    img_np[bg_mask] = bg_color
    return Image.fromarray(img_np)


def _preserve_face(person_img: Image.Image, result_img: Image.Image) -> Image.Image:
    """Blend original face + hair from the bg-removed person image into the result.

    Uses the RESULT's parse map to find face (1) and hair (2) regions.
    Since person_img is already bg-removed (white background, no hoodie collar
    visible after IDM-VTON mask expansion), it's safe to include hair.
    Erodes the mask slightly to avoid edge artifacts, then feathers for smooth blend.
    """
    if hp_postprocess is None:
        return result_img
    try:
        from scipy.ndimage import gaussian_filter, binary_erosion

        # Parse the RESULT image to find face + hair locations
        result_pil_384 = result_img.resize((384, 512), Image.LANCZOS)
        result_parse = hp_postprocess(result_pil_384)
        if hasattr(result_parse, 'cpu'):
            result_parse = result_parse.cpu().numpy()
        else:
            result_parse = np.array(result_parse)
        result_parse_full = np.array(
            Image.fromarray(result_parse.astype(np.uint8)).resize(result_img.size, Image.NEAREST)
        )

        # Face=1, Hair=2 in FashnHumanParser
        face_hair_mask = np.isin(result_parse_full, [1, 2])
        if not face_hair_mask.any():
            return result_img

        # Erode slightly to stay inside the safe region
        face_hair_mask = binary_erosion(face_hair_mask, iterations=3)
        if not face_hair_mask.any():
            return result_img

        # Feather edges for smooth blending
        mask_float = face_hair_mask.astype(np.float32)
        mask_float = gaussian_filter(mask_float, sigma=4)
        mask_3d = np.stack([mask_float] * 3, axis=-1)

        person_resized = person_img.resize(result_img.size, Image.LANCZOS)
        result_np = np.array(result_img)
        person_np = np.array(person_resized)

        blended = person_np.astype(np.float64) * mask_3d + result_np.astype(np.float64) * (1 - mask_3d)
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    except Exception as e:
        print(f"[WARN] Face preservation failed: {e}")
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
        jobs[job_id]["status"] = "processing"

        # Step 1: Remove background (5%)
        _update_job(job_id, "removing_background", 5)
        person_img = _remove_background(person_img)

        # Step 2: Detecting pose (15%)
        _update_job(job_id, "detecting_pose", 15)
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
                person_img = auto_crop_person(person_img, pose_result)
        except Exception as e:
            print(f"[WARN] Pose validation failed: {e}")
        jobs[job_id].update({"pose_confidence": pose_confidence, "pose_warning": pose_warning})

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

        result = pipeline(
            person_image=person_img,
            garment_image=garment_img,
            category=category,
            garment_photo_type=garment_photo_type,
            num_timesteps=40 if pipeline_type == "IDM-VTON" else 25,
            guidance_scale=2.5 if pipeline_type == "IDM-VTON" else 1.5,
            num_samples=num_samples,
            seed=random.randint(0, 2**32 - 1),
        )

        _update_job(job_id, "fitting_garment", 70)

        image_urls = []
        for idx, result_img in enumerate(result.images):
            # Step 5: Preserving face (75%)
            _update_job(job_id, "preserving_face", 75)
            result_img = _preserve_face(person_img, result_img)

            # Step 6: Cleaning background (80%)
            _update_job(job_id, "cleaning_background", 80)
            result_img = _remove_background(result_img)

            # Step 7: Sharpening (85%)
            _update_job(job_id, "sharpening", 85)
            result_img = _sharpen_output(result_img)

            img_np = np.array(result_img)

            # Step 8: Upscaling (88%)
            _update_job(job_id, "upscaling", 88)
            high_res_np, _ = upscaler.enhance(img_np, outscale=2)

            # Step 9: Face restoration (93%)
            _update_job(job_id, "restoring_face", 93)
            high_res_np = _enhance_face(high_res_np, job_id[:8])

            # Step 10: Apply exact #F1EFEF background + save (97%)
            _update_job(job_id, "saving_result", 97)
            high_res_img = Image.fromarray(high_res_np)
            high_res_img = _apply_bg_color(high_res_img)
            filename = f"result_{uuid.uuid4()}.png"
            high_res_img.save(os.path.join(STORAGE_DIR, filename))
            image_urls.append(f"/results/{filename}")

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


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "pipeline": pipeline_type,
        "model_loaded": pipeline is not None,
        "face_enhancer": face_enhancer is not None,
        "sam2_refiner": sam_predictor is not None,
        "color_harmonizer": hp_postprocess is not None,
        "lighting_harmonizer": True,  # always available (LAB-based, no model needed)
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "active_jobs": len([j for j in jobs.values() if j["status"] in ("pending", "processing")]),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
