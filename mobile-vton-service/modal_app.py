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
GPU_TYPE = "A10G"  # Options: T4, A10G, A100, H100, L4

# Local checkpoint directory (must exist before deploying)
LOCAL_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoint", "checkpoint")

# ---------------------------------------------------------------------------
# Modal Image (container environment with checkpoint baked in)
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
    # Copy application code into the image
    .add_local_dir(".", remote_path="/app")
    # Copy checkpoint into the image (3.5GB, but ensures fast cold starts)
    .add_local_dir(LOCAL_CHECKPOINT_DIR, remote_path="/app/checkpoint/checkpoint")
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
    memory=32768,      # 32 GB RAM
    timeout=600,       # 10 minutes
    min_containers=1,  # Keep 1 container warm (~$0.50/hr for A10G)
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def fastapi_app():
    """
    Returns the FastAPI application configured for Modal.
    The checkpoint is baked into the image at /app/checkpoint/checkpoint.
    """
    import sys
    sys.path.insert(0, "/app")

    # Patch environment before importing main.py
    os.environ["MOBILE_VTON_CHECKPOINT"] = "/app/checkpoint/checkpoint"
    os.environ["MOBILE_VTON_DEVICE"] = "cuda"
    os.environ["MOBILE_VTON_DTYPE"] = "bf16"

    from main import app as fastapi_app
    return fastapi_app

# ---------------------------------------------------------------------------
# Health check endpoint (standalone, for simple monitoring)
# ---------------------------------------------------------------------------
@app.function(
    gpu=GPU_TYPE,
    memory=32768,
    timeout=600,
)
@modal.fastapi_endpoint(method="GET")
def health():
    import torch
    return {
        "status": "ok",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint_path": "/app/checkpoint/checkpoint",
    }
