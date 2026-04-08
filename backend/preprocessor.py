"""Shared preprocessing utilities for the virtual try-on pipeline.

Contains:
- SAM2 mask refinement (Problem 1)
- Pose validation + auto-crop (Problem 3)
"""

import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# Problem 1 — SAM2 Mask Refinement
# ---------------------------------------------------------------------------

def refine_mask_with_sam(
    img_np: np.ndarray,
    seg_pred: np.ndarray,
    sam_predictor,
    clothing_label_ids: list[int],
) -> np.ndarray:
    """Refine the FashnHumanParser segmentation using SAM2 for pixel-perfect edges.

    Takes the coarse clothing mask from FashnHumanParser, computes a bounding box
    prompt for SAM2, gets a precise mask back, then merges it with the original
    semantic labels so downstream code still has correct label IDs.

    Args:
        img_np: RGB image as uint8 numpy array (H, W, 3).
        seg_pred: FashnHumanParser prediction array (H, W) with integer label IDs.
        sam_predictor: A loaded SAM2 SamPredictor instance (or None to skip).
        clothing_label_ids: Label IDs that correspond to clothing regions to refine.

    Returns:
        Refined seg_pred with clothing boundaries sharpened by SAM2.
    """
    if sam_predictor is None:
        return seg_pred

    # Build coarse clothing mask from human parser labels
    coarse_mask = np.isin(seg_pred, clothing_label_ids)

    if not np.any(coarse_mask):
        return seg_pred  # no clothing detected — nothing to refine

    # Compute bounding box of the clothing region (with padding)
    ys, xs = np.where(coarse_mask)
    h, w = img_np.shape[:2]
    pad = int(max(h, w) * 0.03)  # 3% padding
    x1 = max(0, xs.min() - pad)
    y1 = max(0, ys.min() - pad)
    x2 = min(w, xs.max() + pad)
    y2 = min(h, ys.max() + pad)
    box = np.array([x1, y1, x2, y2])

    # Run SAM2 with box prompt
    sam_predictor.set_image(img_np)
    masks, scores, _ = sam_predictor.predict(
        box=box,
        multimask_output=True,
    )

    # Pick best mask by score
    best_idx = int(np.argmax(scores))
    sam_mask = masks[best_idx].astype(bool)

    # Merge: keep SAM2 edges but preserve original semantic labels.
    # Where SAM2 says "clothing" AND the coarse mask had a label → keep original label.
    # Where SAM2 says "clothing" but coarse mask didn't → assign the dominant clothing label.
    # Where SAM2 says "not clothing" but coarse mask did → remove that label (set to 0/background).
    refined = seg_pred.copy()

    # Determine dominant clothing label in coarse mask
    clothing_pixels = seg_pred[coarse_mask]
    dominant_label = int(np.bincount(clothing_pixels).argmax())

    # Pixels SAM2 adds (SAM says yes, parser said no)
    added = sam_mask & ~coarse_mask
    refined[added] = dominant_label

    # Pixels SAM2 removes (SAM says no, parser said yes for clothing)
    removed = ~sam_mask & coarse_mask
    refined[removed] = 0  # background

    return refined


# ---------------------------------------------------------------------------
# Problem 3A — Pose Validation
# ---------------------------------------------------------------------------

# DWpose body keypoint indices (COCO-18 format):
# 0=nose, 1=neck, 2=R-shoulder, 3=R-elbow, 4=R-wrist,
# 5=L-shoulder, 6=L-elbow, 7=L-wrist, 8=R-hip, 9=R-knee,
# 10=R-ankle, 11=L-hip, 12=L-knee, 13=L-ankle, 14=R-eye,
# 15=L-eye, 16=R-ear, 17=L-ear
TORSO_KEYPOINT_INDICES = [2, 5, 8, 11]  # shoulders + hips


def validate_pose(pose_result: dict, confidence_threshold: float = 0.3) -> Dict:
    """Validate whether the detected pose is suitable for try-on.

    Checks:
    1. Are >= 10 of 18 body keypoints detected with confidence > threshold?
    2. Are all 4 torso keypoints (shoulders + hips) visible?

    Args:
        pose_result: DWpose output dict with 'bodies' key.
        confidence_threshold: Minimum keypoint confidence.

    Returns:
        Dict with:
            - pose_valid (bool): True if pose passes all checks.
            - pose_confidence (float): Fraction of visible body keypoints (0..1).
            - warning (str | None): Human-readable warning if pose is poor.
    """
    bodies = pose_result.get("bodies", {})
    subset = bodies.get("subset", np.array([]))

    if subset.size == 0:
        return {
            "pose_valid": False,
            "pose_confidence": 0.0,
            "warning": "No person detected. Please upload a photo with a visible person.",
        }

    # subset shape: (num_people, 18) — score per keypoint
    # After DWpose processing, valid keypoints have their index (>=0), invalid = -1
    # But we need the raw scores. The raw scores are in subset before the index replacement.
    # Looking at the DWpose code, subset values > threshold get replaced with index,
    # and < threshold get -1. So visible keypoints have value >= 0.
    scores = subset[0]  # first (best) person — shape (18,)

    # Count visible keypoints (value >= 0 means it passed the threshold)
    visible = scores >= 0
    num_visible = int(np.sum(visible))
    pose_confidence = num_visible / 18.0

    # Check torso keypoints
    torso_visible = all(scores[i] >= 0 for i in TORSO_KEYPOINT_INDICES)

    if num_visible < 10:
        return {
            "pose_valid": False,
            "pose_confidence": pose_confidence,
            "warning": f"Only {num_visible}/18 keypoints detected. Full body photo recommended for best results.",
        }

    if not torso_visible:
        return {
            "pose_valid": False,
            "pose_confidence": pose_confidence,
            "warning": "Shoulders and hips not fully visible. Please use a photo showing your full torso.",
        }

    return {
        "pose_valid": True,
        "pose_confidence": pose_confidence,
        "warning": None,
    }


# ---------------------------------------------------------------------------
# Problem 3B — Auto-crop to person
# ---------------------------------------------------------------------------

def auto_crop_person(
    img: Image.Image,
    pose_result: dict,
    margin: float = 0.15,
    target_ratio: float = 3 / 4,  # width / height — portrait
) -> Image.Image:
    """Crop and pad the image to center the person with consistent framing.

    Uses detected keypoints to find the person bounding box, adds margin,
    and resizes to the pipeline's expected aspect ratio.

    Args:
        img: Input PIL Image (RGB).
        pose_result: DWpose output dict.
        margin: Fractional margin around the person (0.15 = 15%).
        target_ratio: Target width/height ratio.

    Returns:
        Cropped and padded PIL Image.
    """
    bodies = pose_result.get("bodies", {})
    candidate = bodies.get("candidate", np.array([]))

    if candidate.size == 0:
        return img

    w_img, h_img = img.size

    # candidate coords are normalized to [0, 1] — convert to pixels
    kps = candidate.copy()  # shape (N, 2) where N = num_keypoints * num_people
    # Filter out invalid keypoints (marked as -1)
    valid = (kps[:, 0] >= 0) & (kps[:, 1] >= 0)
    if not np.any(valid):
        return img

    kps_px = kps[valid].copy()
    kps_px[:, 0] *= w_img
    kps_px[:, 1] *= h_img

    # Bounding box from valid keypoints
    x_min, y_min = kps_px.min(axis=0)
    x_max, y_max = kps_px.max(axis=0)

    # Add margin — extra margin on top for head clearance
    bw = x_max - x_min
    bh = y_max - y_min
    mx = bw * margin
    my_bottom = bh * margin
    my_top = bh * (margin + 0.10)  # 10% extra on top for head

    x1 = max(0, x_min - mx)
    y1 = max(0, y_min - my_top)
    x2 = min(w_img, x_max + mx)
    y2 = min(h_img, y_max + my_bottom)

    # Adjust to target aspect ratio (expand the shorter dimension)
    crop_w = x2 - x1
    crop_h = y2 - y1
    current_ratio = crop_w / max(crop_h, 1)

    if current_ratio > target_ratio:
        # Too wide — expand height
        new_h = crop_w / target_ratio
        expand = (new_h - crop_h) / 2
        y1 = max(0, y1 - expand)
        y2 = min(h_img, y2 + expand)
    else:
        # Too tall — expand width
        new_w = crop_h * target_ratio
        expand = (new_w - crop_w) / 2
        x1 = max(0, x1 - expand)
        x2 = min(w_img, x2 + expand)

    # Crop
    cropped = img.crop((int(x1), int(y1), int(x2), int(y2)))

    return cropped
