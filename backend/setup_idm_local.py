#!/usr/bin/env python3
"""
One-time setup script for local IDM-VTON GPU inference.

IMPORTANT: Close the backend server before running this script.
Run from the backend directory:
    python setup_idm_local.py

What it does:
  1. Clones the IDM-VTON source code into backend/idm_vton_local/
  2. Downloads model weights (~7 GB) into backend/fashnvton/weights/IDM-VTON/
  3. Copies preprocessing weights to idm_vton_local/ckpt/ (where the repo expects them)
  4. Builds Detectron2 from the bundled source (needed for DensePose)
  5. Installs diffusers>=0.27.0

Detectron2 build requirements (Windows):
  - Visual Studio 2019 or 2022 Build Tools with C++ workload
  - CUDA Toolkit 12.1 (matching your PyTorch build)
  Download VS Build Tools: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
"""
import os
import shutil
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IDM_SRC_DIR  = os.path.join(PROJECT_ROOT, "idm_vton_local")
IDM_DEMO_DIR = os.path.join(IDM_SRC_DIR,  "gradio_demo")
IDM_WEIGHTS  = os.path.join(PROJECT_ROOT, "fashnvton", "weights", "IDM-VTON")
IDM_CKPT_DIR = os.path.join(IDM_SRC_DIR,  "ckpt")


def run(cmd, cwd=None, **kwargs):
    print(f"  >> {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd, **kwargs)


# ---------------------------------------------------------------------------
# Step 1 – Clone IDM-VTON source
# ---------------------------------------------------------------------------
def clone_source():
    if os.path.isdir(os.path.join(IDM_SRC_DIR, ".git")):
        print("[SKIP] IDM-VTON source already cloned -- pulling latest...")
        run(["git", "-C", IDM_SRC_DIR, "pull", "--ff-only"])
    else:
        print("[STEP 1] Cloning IDM-VTON source code...")
        run(["git", "clone", "https://github.com/yisol/IDM-VTON.git", IDM_SRC_DIR])
    print("[OK] Source ready\n")


# ---------------------------------------------------------------------------
# Step 2 – Download HuggingFace model weights (~7 GB)
# ---------------------------------------------------------------------------
def download_weights():
    unet_dir = os.path.join(IDM_WEIGHTS, "unet")
    if os.path.isdir(unet_dir):
        print("[SKIP] IDM-VTON weights already downloaded\n")
        return

    print("[STEP 2] Downloading IDM-VTON model weights (~7 GB)...")
    print(f"   Destination: {IDM_WEIGHTS}\n")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    os.makedirs(IDM_WEIGHTS, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")
    snapshot_download(
        repo_id="yisol/IDM-VTON",
        local_dir=IDM_WEIGHTS,
        token=hf_token,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "*.ot"],
    )
    print("\n[OK] Weights downloaded\n")


# ---------------------------------------------------------------------------
# Step 3 – Copy preprocessing weights to ckpt/ (where the repo expects them)
# ---------------------------------------------------------------------------
def copy_ckpt_weights():
    print("[STEP 3] Copying preprocessing weights to ckpt/ folder...")

    copies = [
        # (source in IDM_WEIGHTS,              destination in IDM_CKPT_DIR)
        (os.path.join("densepose", "model_final_162be9.pkl"),  os.path.join("densepose", "model_final_162be9.pkl")),
        (os.path.join("humanparsing", "parsing_atr.onnx"),     os.path.join("humanparsing", "parsing_atr.onnx")),
        (os.path.join("humanparsing", "parsing_lip.onnx"),     os.path.join("humanparsing", "parsing_lip.onnx")),
        (os.path.join("openpose", "ckpts", "body_pose_model.pth"),
                                                                os.path.join("openpose", "ckpts", "body_pose_model.pth")),
    ]

    for src_rel, dst_rel in copies:
        src = os.path.join(IDM_WEIGHTS, src_rel)
        dst = os.path.join(IDM_CKPT_DIR, dst_rel)

        if not os.path.isfile(src):
            print(f"   [WARN] Source not found: {src_rel} -- skipping")
            continue

        if os.path.isfile(dst):
            print(f"   [SKIP] Already exists: {dst_rel}")
            continue

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"   [COPY] {src_rel}")

    # Also ensure the gradio_demo/ckpt symlink/copy is in place
    demo_ckpt = os.path.join(IDM_DEMO_DIR, "ckpt")
    if not os.path.exists(demo_ckpt):
        try:
            # Prefer a symlink (saves disk space)
            os.symlink(IDM_CKPT_DIR, demo_ckpt)
            print(f"   [LINK] gradio_demo/ckpt -> {IDM_CKPT_DIR}")
        except (OSError, NotImplementedError):
            # Symlinks need Developer Mode on Windows; fall back to copy
            shutil.copytree(IDM_CKPT_DIR, demo_ckpt)
            print(f"   [COPY] gradio_demo/ckpt/ (symlink failed, used copy)")

    print("[OK] ckpt/ ready\n")


# ---------------------------------------------------------------------------
# Step 4 – Build Detectron2 from bundled source
# ---------------------------------------------------------------------------
def build_detectron2():
    try:
        import detectron2  # noqa: F401
        print("[SKIP] Detectron2 already installed\n")
        return
    except ImportError:
        pass

    # Check if CUDA Toolkit (nvcc) is actually installed
    nvcc_found = shutil.which("nvcc") is not None
    cuda_home_candidates = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
        r"C:\CUDA\v12.1",
    ]
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        for c in cuda_home_candidates:
            if os.path.isdir(c):
                cuda_home = c
                break

    if not nvcc_found and not cuda_home:
        print("[WARN] CUDA Toolkit NOT installed (nvcc not found).")
        print("   Detectron2 build SKIPPED.")
        print("")
        print("   Your GPU (RTX 3060 Ti, 8 GB) also has insufficient VRAM for")
        print("   local IDM-VTON inference (needs 12-14 GB).")
        print("")
        print("   RECOMMENDATION: Use the HuggingFace Space for Advanced mode.")
        print("   Set HF_TOKEN in backend/.env to avoid quota limits.")
        print("")
        return   # non-fatal -- HF Space fallback will be used

    # CUDA Toolkit found -- attempt build
    detectron2_clone = os.path.join(PROJECT_ROOT, "detectron2_src")
    print("[STEP 4] Building Detectron2 for Windows + CUDA...")
    print("   Build time: ~10-15 minutes\n")

    if not os.path.isdir(os.path.join(detectron2_clone, ".git")):
        print("   Cloning Detectron2...")
        run(["git", "clone", "https://github.com/facebookresearch/detectron2.git",
             detectron2_clone, "--depth", "1"])
    else:
        print("   [SKIP] Detectron2 source already cloned")

    env = os.environ.copy()
    if cuda_home:
        env["CUDA_HOME"] = cuda_home
        env["CUDA_PATH"] = cuda_home
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")   # Ampere (RTX 30xx)
    env.setdefault("FORCE_CUDA", "1")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", detectron2_clone],
        check=True,
        env=env,
    )
    print("[OK] Detectron2 installed\n")


# ---------------------------------------------------------------------------
# Step 5 – Install remaining Python deps
# ---------------------------------------------------------------------------
def install_deps():
    print("[STEP 5] Installing Python dependencies...")
    run([sys.executable, "-m", "pip", "install",
         "diffusers>=0.27.0",
         "einops",
    ])
    print("[OK] Dependencies installed\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 62)
    print(" IDM-VTON Local GPU Setup")
    print("=" * 62)
    print("IMPORTANT: Make sure the backend server is NOT running.\n")

    clone_source()
    download_weights()
    copy_ckpt_weights()
    build_detectron2()
    install_deps()

    # Final status check
    try:
        from idm_local_pipeline import is_available
        local_ready = is_available()
    except Exception:
        local_ready = False

    try:
        import detectron2  # noqa: F401
        d2_ok = True
    except ImportError:
        d2_ok = False

    print("=" * 62)
    if local_ready:
        print(" Setup COMPLETE! IDM-VTON is ready for local GPU inference.")
        print(" Restart the backend -- Advanced mode will run on your GPU.")
    else:
        print(" Setup DONE (partial).")
        print(f"  Weights copied : [OK]")
        print(f"  Detectron2     : {'[OK]' if d2_ok else '[NOT BUILT] -- HF Space fallback active'}")
        print("")
        print(" Advanced mode will use the HuggingFace Space (same model,")
        print(" runs on A100). Set HF_TOKEN in backend/.env to avoid quotas.")
    print("=" * 62)


if __name__ == "__main__":
    main()
