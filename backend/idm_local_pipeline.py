"""
Local IDM-VTON inference pipeline.
Used by main.py when local weights are present (setup_idm_local.py was run).

Architecture (mirrors gradio_demo/app.py exactly):
  - Human parsing  : SCHP (preprocess/humanparsing) → mask for inpainting
  - OpenPose       : preprocess/openpose → keypoints (used for mask only)
  - DensePose      : apply_net + Detectron2 → UV surface map (pose image input)
  - Diffusion      : TryonPipeline (SDXL-based) + UNet2DConditionModel (garment encoder)

Requires:
  - setup_idm_local.py to have been run (clones repo, downloads weights, copies ckpt/, builds detectron2)
  - diffusers>=0.27.0
"""
import os
import sys
import random
import torch
import numpy as np
from PIL import Image

PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
IDM_SRC_DIR    = os.path.join(PROJECT_ROOT, "idm_vton_local")
IDM_DEMO_DIR   = os.path.join(IDM_SRC_DIR,  "gradio_demo")
IDM_WEIGHTS    = os.path.join(PROJECT_ROOT, "fashnvton", "weights", "IDM-VTON")

_pipe         = None
_parsing      = None
_openpose     = None
_device       = "cpu"


def is_available() -> bool:
    """True when local weights AND Detectron2 (for DensePose) are both installed."""
    weights_ok = (
        os.path.isdir(IDM_SRC_DIR) and
        os.path.isdir(os.path.join(IDM_WEIGHTS, "unet")) and
        os.path.isdir(os.path.join(IDM_WEIGHTS, "unet_encoder"))
    )
    if not weights_ok:
        print(f"[IDM-LOCAL] Weights missing at {IDM_WEIGHTS}")
        return False

    densepose_ckpt = os.path.join(IDM_SRC_DIR, "ckpt", "densepose", "model_final_162be9.pkl")
    if not os.path.isfile(densepose_ckpt):
        print(f"[IDM-LOCAL] DensePose checkpoint missing: {densepose_ckpt}")
        return False

    # Check detectron2 is actually importable (not just that the ckpt exists)
    try:
        import detectron2  # noqa: F401
    except ImportError:
        print("[IDM-LOCAL] detectron2 not installed — run: pip install detectron2 or setup_idm_local.py")
        return False

    return True


def load(device: str = "cuda") -> None:
    """Load the full IDM-VTON pipeline into VRAM. Call once at startup."""
    global _pipe, _parsing, _openpose, _device
    if _pipe is not None:
        return

    if not is_available():
        raise RuntimeError(
            "IDM-VTON local not fully set up. Run setup_idm_local.py first "
            "(it clones the repo, copies weights, and builds Detectron2)."
        )

    # Add all necessary paths
    for p in (IDM_SRC_DIR, IDM_DEMO_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)

    print("[IDM-LOCAL] Loading IDM-VTON pipeline...")

    from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
    from src.unet_hacked_tryon   import UNet2DConditionModel
    from src.tryon_pipeline      import StableDiffusionXLInpaintPipeline as TryonPipeline
    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose    import OpenPose

    from diffusers    import DDPMScheduler, AutoencoderKL
    from transformers import (
        AutoTokenizer,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPVisionModelWithProjection,
        CLIPImageProcessor,
    )

    dtype = torch.float16

    print("[IDM-LOCAL] Loading tokenizers and text encoders...")
    tokenizer_one = AutoTokenizer.from_pretrained(IDM_WEIGHTS, subfolder="tokenizer",   use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(IDM_WEIGHTS, subfolder="tokenizer_2", use_fast=False)

    text_encoder_one = CLIPTextModel.from_pretrained(
        IDM_WEIGHTS, subfolder="text_encoder",   torch_dtype=dtype
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        IDM_WEIGHTS, subfolder="text_encoder_2", torch_dtype=dtype
    )

    print("[IDM-LOCAL] Loading UNets + VAE + image encoder...")
    unet = UNet2DConditionModel.from_pretrained(
        IDM_WEIGHTS, subfolder="unet",         torch_dtype=dtype
    )
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        IDM_WEIGHTS, subfolder="unet_encoder", torch_dtype=dtype
    )
    vae = AutoencoderKL.from_pretrained(
        IDM_WEIGHTS, subfolder="vae", torch_dtype=dtype
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        IDM_WEIGHTS, subfolder="image_encoder", torch_dtype=dtype
    )
    noise_scheduler = DDPMScheduler.from_pretrained(IDM_WEIGHTS, subfolder="scheduler")

    print("[IDM-LOCAL] Building TryonPipeline...")
    _pipe = TryonPipeline.from_pretrained(
        IDM_WEIGHTS,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=dtype,
    )
    _pipe.unet_encoder = unet_encoder

    # Preprocessing on CPU
    _parsing  = Parsing(-1)   # CPU
    _openpose = OpenPose(-1)  # CPU

    _device = device
    print(f"[IDM-LOCAL] Pipeline ready (inference on {device})")


def get_openpose():
    """Expose OpenPose detector for pose validation."""
    return _openpose


def get_device():
    return _device


# ---------------------------------------------------------------------------
# Category mapping
# ---------------------------------------------------------------------------
_CAT_MAP = {
    "tops":       "upper_body",
    "bottoms":    "lower_body",
    "one-pieces": "dresses",
}


def _get_densepose_image(human_img_np: np.ndarray) -> Image.Image:
    """
    Run DensePose via apply_net (Detectron2) to produce the pose conditioning image.
    Working directory must be IDM_DEMO_DIR so configs/ and ckpt/ paths resolve.
    """
    import apply_net
    from detectron2.data.detection_utils import (
        convert_PIL_to_numpy,
        _apply_exif_orientation,
    )

    pil_input = Image.fromarray(human_img_np).resize((384, 512))
    pil_input = _apply_exif_orientation(pil_input)
    bgr_input  = convert_PIL_to_numpy(pil_input, format="BGR")

    old_cwd = os.getcwd()
    os.chdir(IDM_DEMO_DIR)   # apply_net uses relative paths for config + ckpt
    try:
        args = apply_net.create_argument_parser().parse_args([
            "show",
            "./configs/densepose_rcnn_R_50_FPN_s1x.yaml",
            "./ckpt/densepose/model_final_162be9.pkl",
            "dp_segm",
            "-v",
            "--opts", "MODEL.DEVICE", "cpu",
        ])
        pose_bgr = args.func(args, bgr_input)
    finally:
        os.chdir(old_cwd)

    pose_rgb = pose_bgr[:, :, ::-1]   # BGR → RGB
    return Image.fromarray(pose_rgb).resize((768, 1024))


def run_tryon(
    person_img:    Image.Image,
    garment_img:   Image.Image,
    category:      str = "tops",
    denoise_steps: int = 30,
    seed:          int | None = None,
) -> Image.Image:
    """
    Run IDM-VTON locally and return the result as a PIL Image.
    load() must be called first.
    """
    if _pipe is None:
        raise RuntimeError("IDM-VTON pipeline not loaded. Call load() first.")

    for p in (IDM_SRC_DIR, IDM_DEMO_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)

    from utils_mask  import get_mask_location
    from torchvision import transforms
    from torchvision.transforms.functional import to_pil_image
    from typing      import List

    if seed is None:
        seed = random.randint(0, 2 ** 31 - 1)

    cat_label = _CAT_MAP.get(category, "upper_body")
    W, H = 768, 1024

    # --- Resize inputs -------------------------------------------------------
    garment_img = garment_img.convert("RGB").resize((W, H))
    human_img   = person_img.convert("RGB").resize((W, H))

    # --- Mask (OpenPose keypoints + SCHP parsing) ----------------------------
    keypoints    = _openpose(human_img.resize((384, 512)))
    model_parse, _ = _parsing(human_img.resize((384, 512)))
    mask, mask_gray = get_mask_location("hd", cat_label, model_parse, keypoints)
    mask      = mask.resize((W, H))

    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    # --- DensePose pose image ------------------------------------------------
    human_np = np.array(human_img)
    pose_img = _get_densepose_image(human_np)

    # --- Move pipeline to GPU ---------------------------------------------------
    _pipe.to(_device)
    _pipe.unet_encoder.to(_device)

    # --- Encode prompts -------------------------------------------------------
    prompt          = f"model is wearing {cat_label.replace('_', ' ')}"
    negative_prompt = ("monochrome, lowres, bad anatomy, worst quality, low quality, "
                       "blurry, deformed, disfigured, watermark, text, extra limbs, "
                       "missing limbs, floating garment, unnatural pose, overexposed, "
                       "underexposed, plastic skin, unrealistic lighting")
    prompt_list     = [prompt]          if not isinstance(prompt, List) else prompt
    neg_list        = [negative_prompt] if not isinstance(negative_prompt, List) else negative_prompt

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with torch.inference_mode():
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = _pipe.encode_prompt(
                    prompt_list,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=neg_list,
                )

            cloth_prompt  = f"a photo of {cat_label.replace('_', ' ')}"
            cloth_prompt_list = [cloth_prompt] if not isinstance(cloth_prompt, List) else cloth_prompt
            with torch.inference_mode():
                (prompt_embeds_c, _, _, _) = _pipe.encode_prompt(
                    cloth_prompt_list,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=neg_list,
                )

            pose_tensor  = tensor_transform(pose_img).unsqueeze(0).to(_device, torch.float16)
            garm_tensor  = tensor_transform(garment_img).unsqueeze(0).to(_device, torch.float16)
            generator    = torch.Generator(_device).manual_seed(seed)

            images = _pipe(
                prompt_embeds=prompt_embeds.to(_device, torch.float16),
                negative_prompt_embeds=negative_prompt_embeds.to(_device, torch.float16),
                pooled_prompt_embeds=pooled_prompt_embeds.to(_device, torch.float16),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(_device, torch.float16),
                num_inference_steps=denoise_steps,
                generator=generator,
                strength=0.85,
                pose_img=pose_tensor.to(_device, torch.float16),
                text_embeds_cloth=prompt_embeds_c.to(_device, torch.float16),
                cloth=garm_tensor.to(_device, torch.float16),
                mask_image=mask,
                image=human_img,
                height=H,
                width=W,
                ip_adapter_image=garment_img.resize((W, H)),
                guidance_scale=2.5,
            )[0]

    return images[0]
