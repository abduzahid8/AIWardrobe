"""
Modal deployment for Mobile-VTON (Virtual Try-On) service.

Architecture
------------
- The SD3.5-Medium checkpoint (3.5 GB) is stored in a Modal Volume so it is
  downloaded ONCE and persisted across all container restarts.
- On first deploy, run:
      modal run modal_app.py::download_model
  to pre-populate the volume.  Subsequent cold starts load from disk in <10s.
- The FastAPI service mounts the Volume read-only so inference containers spin
  up immediately without any HuggingFace network I/O.

GPU: A10G (24 GB VRAM) — required to keep SD3.5 Medium + DeepLabV3 fully in
CUDA without enable_model_cpu_offload (which adds ~10s latency on T4).

Prerequisites:
  pip install modal
  modal setup                          # Authenticate
  modal run modal_app.py::download_model   # Pre-populate checkpoint volume
  modal deploy modal_app.py            # Deploy the serving app

Deploy:
  cd mobile-vton-service
  modal deploy modal_app.py
"""

import modal
import os

# ---------------------------------------------------------------------------
# Read HF_TOKEN from project .env (deploy-time only, not baked into image)
# ---------------------------------------------------------------------------
_HF_TOKEN = ""
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                _HF_TOKEN = line.split("=", 1)[1].strip()
                break

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APP_NAME = "aiwardrobe-mobile-vton"

# A10G has 24 GB VRAM — required to hold SD3.5 Medium + DeepLabV3 fully in
# CUDA without enable_model_cpu_offload (which caused 5x latency on T4).
GPU_TYPE = "A10G"

# Modal Volume: checkpoint is stored here persistently across container restarts.
# First-time setup: modal run modal_app.py::download_model
VOLUME_NAME = "aiwardrobe-vton-checkpoint"
CHECKPOINT_REMOTE_DIR = "/checkpoints/sd35medium"
HF_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"

# ---------------------------------------------------------------------------
# Modal Volume (persistent checkpoint storage)
# ---------------------------------------------------------------------------
checkpoint_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ---------------------------------------------------------------------------
# Modal Image (container environment)
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.32.0",
        # torch>=2.3.0 for CUDA 12 + BF16 stability on A10G
        "torch>=2.3.0",
        # torchvision>=0.18.0 ships DeepLabV3_ResNet50_Weights enum
        "torchvision>=0.18.0",
        "diffusers>=0.32.2",
        "transformers>=4.42.0",
        "accelerate>=1.12.0",
        "safetensors>=0.4.5",
        "huggingface-hub>=0.24.0",
        "pillow>=10.0.0",
        "numpy>=1.26.4",
        "scipy>=1.14.1",
        "einops>=0.8.0",
        "tqdm>=4.66.5",
        "requests>=2.31.0",
        "omegaconf>=2.3.0",
        "matplotlib>=3.8.0",
    )
    # Copy only main.py into the image
    .add_local_file(os.path.join(os.path.dirname(__file__), "main.py"), remote_path="/app/main.py")
)

# ---------------------------------------------------------------------------
# Modal App
# ---------------------------------------------------------------------------
app = modal.App(APP_NAME, image=image)

# ---------------------------------------------------------------------------
# One-time model download function
# Run this ONCE before deploying:
#   modal run modal_app.py::download_model
# ---------------------------------------------------------------------------
@app.function(
    volumes={"/checkpoints": checkpoint_volume},
    timeout=900,        # 15 min — enough to download 3.5 GB on any connection
    memory=4096,
    env={"HF_TOKEN": _HF_TOKEN},
)
def download_model():
    """
    Downloads SD3.5-Medium from HuggingFace into the persistent Modal Volume.
    Run once: modal run modal_app.py::download_model
    """
    import os
    from huggingface_hub import snapshot_download

    dest = CHECKPOINT_REMOTE_DIR
    hf_token = os.environ.get("HF_TOKEN") or None

    # Check if already downloaded
    if os.path.isdir(dest) and len(os.listdir(dest)) > 5:
        print(f"✓ Checkpoint already present at {dest} ({len(os.listdir(dest))} files)")
        checkpoint_volume.commit()
        return

    print(f"Downloading {HF_MODEL_ID} → {dest} ...")
    os.makedirs(dest, exist_ok=True)
    snapshot_download(
        repo_id=HF_MODEL_ID,
        local_dir=dest,
        token=hf_token,
        resume_download=True,
    )
    # Commit to volume so other containers see the files immediately
    checkpoint_volume.commit()
    print(f"✓ Checkpoint saved to volume '{VOLUME_NAME}' at {dest}")

# ---------------------------------------------------------------------------
# ASGI wrapper for the FastAPI app
# ---------------------------------------------------------------------------
@app.function(
    gpu=GPU_TYPE,
    memory=32768,       # 32 GB RAM — prevent OOM crashes during heavy load
    timeout=180,        # 3 min — warm inference completes in <30s
    volumes={"/checkpoints": checkpoint_volume},   # Mount checkpoint volume
    env={
        "HF_TOKEN": _HF_TOKEN,
        "VTON_VERSION": "3",
        "MOBILE_VTON_DTYPE": "fp16",    # fp16 for A10G speed/VRAM balance
        "MOBILE_VTON_DEVICE": "cuda",
        # Point main.py at the volume-backed checkpoint (no HF download at runtime)
        "MOBILE_VTON_CHECKPOINT": CHECKPOINT_REMOTE_DIR,
        # Disable HuggingFace online mode — checkpoint already on disk
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    },
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def fastapi_app():
    """
    Returns the FastAPI application configured for Modal.
    Checkpoint is loaded from the pre-populated Modal Volume — no download needed.
    """
    import sys
    sys.path.insert(0, "/app")

    from main import app as _fastapi_app
    return _fastapi_app

# ---------------------------------------------------------------------------
# Health check endpoint (standalone, for monitoring)
# ---------------------------------------------------------------------------
@app.function(
    gpu=GPU_TYPE,
    memory=32768,
    timeout=60,
    volumes={"/checkpoints": checkpoint_volume},
    env={
        "HF_TOKEN": _HF_TOKEN,
        "VTON_VERSION": "3",
        "MOBILE_VTON_CHECKPOINT": CHECKPOINT_REMOTE_DIR,
    },
)
@modal.fastapi_endpoint(method="GET")
def health():
    import torch
    import os
    ckpt = CHECKPOINT_REMOTE_DIR
    ckpt_files = len(os.listdir(ckpt)) if os.path.isdir(ckpt) else 0
    return {
        "status": "ok",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint_path": ckpt,
        "checkpoint_files": ckpt_files,
        "volume": VOLUME_NAME,
    }
