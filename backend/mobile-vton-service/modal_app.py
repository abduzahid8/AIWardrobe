"""
Modal deployment for Mobile-VTON (Virtual Try-On) service.

This script deploys the Mobile-VTON FastAPI service to Modal's serverless GPU platform.
Uses SD1.5 Inpaint + IP-Adapter for high-fidelity garment-aware virtual try-on.

Prerequisites:
  pip install modal
  modal setup  # Authenticate with your Modal token

Deploy:
  cd backend/mobile-vton-service
  modal deploy modal_app.py
"""

import modal
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APP_NAME = "aiwardrobe-mobile-vton"
GPU_TYPE = "T4"

# Remote checkpoint directories — NFS is mounted at /app/checkpoint.
CHECKPOINT_REMOTE_DIR = "/app/checkpoint/checkpoint"
IP_ADAPTER_REMOTE_DIR = "/app/checkpoint/ip_adapter"

# Root .env file path (contains HF_TOKEN and other secrets)
_DOTENV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", ".env")

# Render API base URL — kept alive by the cron below
RENDER_API_URL = os.environ.get(
    "RENDER_API_URL",
    "https://aiwardrobe-api.onrender.com",
)

# ---------------------------------------------------------------------------
# Build-time download functions
# ---------------------------------------------------------------------------
def download_sd15_inpaint():
    """Pre-download runwayml/stable-diffusion-inpainting checkpoint during image build."""
    import os
    from huggingface_hub import snapshot_download

    checkpoint_dir = "/app/checkpoint/checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)

    hf_model_id = "runwayml/stable-diffusion-inpainting"
    hf_token = os.environ.get("HF_TOKEN")

    print(f"Downloading {hf_model_id} during image build...")
    snapshot_download(
        repo_id=hf_model_id,
        local_dir=checkpoint_dir,
        token=hf_token,
        ignore_patterns=[
            "*.ckpt",
            "*.onnx",
            "*.msgpack",
        ],
    )
    print("SD1.5 Inpaint download complete!")


def download_ip_adapter():
    """Pre-download h94/IP-Adapter SD1.5 weights and image encoder during image build."""
    import os
    from huggingface_hub import snapshot_download

    ip_adapter_dir = "/app/checkpoint/ip_adapter"
    os.makedirs(ip_adapter_dir, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")

    print("Downloading h94/IP-Adapter (ip-adapter_sd15.bin + image_encoder)...")
    snapshot_download(
        repo_id="h94/IP-Adapter",
        local_dir=ip_adapter_dir,
        token=hf_token,
        allow_patterns=[
            "models/ip-adapter_sd15.bin",
            "models/image_encoder/*",
        ],
    )
    print("IP-Adapter download complete!")


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
        "bitsandbytes>=0.43.0",
        "peft>=0.14.0",
    )
    .run_function(
        download_sd15_inpaint,
        secrets=[modal.Secret.from_dotenv(_DOTENV_PATH)],
    )
    .run_function(
        download_ip_adapter,
        secrets=[modal.Secret.from_dotenv(_DOTENV_PATH)],
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
    memory=16384,      # 16 GB RAM
    timeout=300,       # 5 minutes
    min_containers=1,
    secrets=[modal.Secret.from_dotenv(_DOTENV_PATH)],
    env={
        "RENDER_API_URL": RENDER_API_URL,
    },
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def fastapi_app():
    """
    Returns the FastAPI application configured for Modal.
    """
    import sys
    sys.path.insert(0, "/app")

    os.environ["MOBILE_VTON_CHECKPOINT"] = CHECKPOINT_REMOTE_DIR
    os.environ["MOBILE_VTON_IP_ADAPTER_DIR"] = IP_ADAPTER_REMOTE_DIR
    os.environ["MOBILE_VTON_DEVICE"] = "cuda"
    os.environ["MOBILE_VTON_DTYPE"] = "fp16"

    from main import app as fastapi_app
    return fastapi_app


# ---------------------------------------------------------------------------
# Keep-alive cron — pings Render every 5 minutes so it never sleeps.
# ---------------------------------------------------------------------------
@app.function(
    schedule=modal.Cron("*/5 * * * *"),
    image=image,
    timeout=30,
    env={"RENDER_API_URL": RENDER_API_URL},
)
def keep_render_alive():
    """Ping the Render API health endpoint to prevent cold-start 502/503 errors."""
    import requests
    import time

    render_url = os.environ.get("RENDER_API_URL", "https://aiwardrobe-api.onrender.com")
    url = f"{render_url}/health"

    try:
        start = time.time()
        resp = requests.get(url, timeout=25)
        elapsed = round((time.time() - start) * 1000)
        print(f"[keep_alive] Render ping OK — {resp.status_code} in {elapsed}ms")
    except Exception as exc:
        print(f"[keep_alive] Render ping failed: {exc}")