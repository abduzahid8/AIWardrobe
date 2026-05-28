"""
Modal deployment for Mobile-VTON (Virtual Try-On) service.

This script deploys the Mobile-VTON FastAPI service to Modal's serverless GPU platform.
The 3.5GB checkpoint is baked into the Docker image for fast cold starts.

Prerequisites:
  pip install modal
  modal setup  # Authenticate with your Modal token

Deploy:
  cd mobile-vton-service
  modal deploy modal_app.py
"""

import modal
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APP_NAME = "aiwardrobe-mobile-vton"
GPU_TYPE = "T4"    # T4 is ~3x cheaper than A10G

# Local checkpoint directory (must exist before deploying)
LOCAL_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoint", "checkpoint")

# Remote checkpoint directory (downloaded at runtime on first request)
CHECKPOINT_REMOTE_DIR = "/app/checkpoint/checkpoint"

# ---------------------------------------------------------------------------
# Modal Image (container environment)
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.32.0",
        "torch>=2.0.1",
        "torchvision>=0.15.2",
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
    # Copy only main.py into the image (not the whole directory)
    .add_local_file(os.path.join(os.path.dirname(__file__), "main.py"), remote_path="/app/main.py")
)

# ---------------------------------------------------------------------------
# Modal App
# ---------------------------------------------------------------------------
app = modal.App(APP_NAME, image=image)

# ---------------------------------------------------------------------------
# ASGI wrapper for the FastAPI app
# ---------------------------------------------------------------------------
@app.function(
    gpu=GPU_TYPE,
    memory=16384,      # 16 GB RAM — enough for SD3.5 inference
    timeout=300,       # 5 minutes
    # min_containers removed — cold start on first request saves money
    env={"HF_TOKEN": os.environ.get("HF_TOKEN", "")},
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def fastapi_app():
    """
    Returns the FastAPI application configured for Modal.
    Checkpoint is downloaded at runtime on first request if not present.
    """
    import sys
    sys.path.insert(0, "/app")

    # Patch environment before importing main.py
    os.environ["MOBILE_VTON_CHECKPOINT"] = CHECKPOINT_REMOTE_DIR
    os.environ["MOBILE_VTON_DEVICE"] = "cuda"
    os.environ["MOBILE_VTON_DTYPE"] = "fp16"

    from main import app as fastapi_app
    return fastapi_app

# ---------------------------------------------------------------------------
# Health check endpoint (standalone, for simple monitoring)
# ---------------------------------------------------------------------------
@app.function(
    gpu=GPU_TYPE,
    memory=16384,
    timeout=300,
    env={"HF_TOKEN": os.environ.get("HF_TOKEN", "")},
)
@modal.fastapi_endpoint(method="GET")
def health():
    import torch
    return {
        "status": "ok",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint_path": CHECKPOINT_REMOTE_DIR,
    }

