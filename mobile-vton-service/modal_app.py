"""
Modal deployment for Mobile-VTON (Virtual Try-On) service.

This script deploys the Mobile-VTON FastAPI service to Modal's serverless GPU platform.
The 3.5GB checkpoint is downloaded at runtime on first request.

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

# Remote checkpoint directory (downloaded at runtime on first request)
CHECKPOINT_REMOTE_DIR = "/app/checkpoint/checkpoint"

# Render API base URL — kept alive by the cron below
RENDER_API_URL = os.environ.get(
    "RENDER_API_URL",
    "https://aiwardrobe-api.onrender.com",
)

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
    timeout=300,       # 5 minutes — covers model download + inference
    # min_containers=1 keeps one GPU container warm at all times.
    # This eliminates cold-start model downloads (~60–120s) that cause 502/503.
    # Cost: ~$0.60/hr on T4. Remove to save money (accept cold starts).
    min_containers=1,
    env={
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "RENDER_API_URL": RENDER_API_URL,
    },
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def fastapi_app():
    """
    Returns the FastAPI application configured for Modal.
    Checkpoint is downloaded at runtime on first request if not present.
    min_containers=1 ensures this container is always warm.
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
# Keep-alive cron — pings Render every 5 minutes so it never sleeps.
#
# Render's free/starter plan spins down after 15 minutes of inactivity.
# A sleeping Render server causes 502/503 on the FIRST request because
# Render's own load-balancer times out before the app wakes up (~30–60s).
# This cron prevents that by sending a lightweight /health ping from Modal.
# ---------------------------------------------------------------------------
@app.function(
    schedule=modal.Cron("*/5 * * * *"),  # every 5 minutes
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
        # Not fatal — just log and let the next cron try again
        print(f"[keep_alive] Render ping failed: {exc}")
