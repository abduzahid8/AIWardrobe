"""
IDM-VTON Colab Setup Script
============================
Run this FIRST in a Google Colab cell with T4 GPU runtime.
It installs all dependencies, clones IDM-VTON, and downloads model checkpoints.

Usage (in Colab cell):
    !python colab_setup.py
"""

import subprocess
import sys
import os
import shutil


def run(cmd, desc=""):
    """Run a shell command with logging."""
    if desc:
        print(f"\n{'='*60}")
        print(f"  {desc}")
        print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"⚠️  Command returned non-zero exit code: {result.returncode}")
    return result.returncode


def check_gpu():
    """Verify GPU is available."""
    import torch

    if not torch.cuda.is_available():
        print("❌ No GPU detected! Go to Runtime → Change runtime type → T4 GPU")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✅ GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")

    if vram_gb < 14:
        print("⚠️  Less than 14 GB VRAM — may run out of memory. T4 (16 GB) recommended.")


def main():
    print("🚀 IDM-VTON Colab Setup Starting...")
    print("=" * 60)

    # ── Step 1: Install core Python packages ─────────────────
    # Use Colab's pre-installed PyTorch (already has CUDA support)
    run(
        "pip install -q torch torchvision",
        "Step 1/7: Ensuring PyTorch is installed",
    )

    # ── Step 2: Install critical version-pinned packages first ─
    # These MUST be specific versions for IDM-VTON compatibility
    run(
        "pip install -q diffusers==0.25.0 transformers==4.36.2 accelerate==0.25.0",
        "Step 2/7: Installing version-pinned diffusers/transformers",
    )

    # ── Step 3: Install remaining dependencies ───────────────
    packages = [
        "safetensors",
        "einops==0.7.0",
        "opencv-python",
        "gradio==4.24.0",
        "onnxruntime-gpu",
        "scipy==1.11.1",
        "fvcore",
        "cloudpickle",
        "omegaconf",
        "pycocotools",
        "basicsr",
        "av",
        # API server deps
        "fastapi",
        "uvicorn[standard]",
        "pyngrok",
        "python-multipart",
        "nest_asyncio",
    ]
    run(
        f"pip install -q {' '.join(packages)}",
        "Step 3/7: Installing remaining dependencies",
    )

    # ── Step 4: Install detectron2 (for DensePose) ───────────
    # Use archive URL instead of git clone (more reliable on Colab)
    run(
        "pip install -q 'detectron2 @ https://github.com/facebookresearch/detectron2/archive/refs/heads/main.zip'",
        "Step 4/7: Installing detectron2 (DensePose)",
    )

    # ── Step 5: Clone IDM-VTON repo ──────────────────────────
    if not os.path.exists("/content/IDM-VTON"):
        run(
            "git clone https://github.com/yisol/IDM-VTON.git /content/IDM-VTON",
            "Step 5/7: Cloning IDM-VTON repository",
        )
    else:
        print("\n✅ IDM-VTON already cloned at /content/IDM-VTON")

    # ── Step 6: Download model checkpoints ───────────────────
    print(f"\n{'='*60}")
    print("  Step 6/7: Downloading model checkpoints from HuggingFace")
    print("  This downloads ~15 GB of models. Be patient...")
    print(f"{'='*60}")

    # Download the main model (will be cached by HF)
    run("pip install -q huggingface_hub", "")

    download_script = '''
import os
from huggingface_hub import snapshot_download

# Download the main IDM-VTON model (~12 GB)
print("📦 Downloading IDM-VTON model weights...")
snapshot_download(
    repo_id="yisol/IDM-VTON",
    local_dir="/content/IDM-VTON-model",
    local_dir_use_symlinks=False,
)
print("✅ Main model downloaded")

# Download preprocessing checkpoints
ckpt_dir = "/content/IDM-VTON/ckpt"
os.makedirs(f"{ckpt_dir}/densepose", exist_ok=True)
os.makedirs(f"{ckpt_dir}/humanparsing", exist_ok=True)
os.makedirs(f"{ckpt_dir}/openpose/ckpts", exist_ok=True)

from huggingface_hub import hf_hub_download

# DensePose model
print("📦 Downloading DensePose model...")
hf_hub_download(
    repo_id="yisol/IDM-VTON",
    filename="ckpt/densepose/model_final_162be9.pkl",
    local_dir="/content/IDM-VTON",
    local_dir_use_symlinks=False,
    repo_type="space",
)

# Human parsing models
print("📦 Downloading Human Parsing models...")
for f in ["parsing_atr.onnx", "parsing_lip.onnx"]:
    hf_hub_download(
        repo_id="yisol/IDM-VTON",
        filename=f"ckpt/humanparsing/{f}",
        local_dir="/content/IDM-VTON",
        local_dir_use_symlinks=False,
        repo_type="space",
    )

# OpenPose model
print("📦 Downloading OpenPose model...")
hf_hub_download(
    repo_id="yisol/IDM-VTON",
    filename="ckpt/openpose/ckpts/body_pose_model.pth",
    local_dir="/content/IDM-VTON",
    local_dir_use_symlinks=False,
    repo_type="space",
)

print("✅ All preprocessing checkpoints downloaded")
'''

    # Write and run the download script
    with open("/tmp/download_models.py", "w") as f:
        f.write(download_script)
    run("python /tmp/download_models.py", "")

    # ── Step 7: Verify everything ────────────────────────────
    print(f"\n{'='*60}")
    print("  Step 7/7: Verifying installation")
    print(f"{'='*60}")

    check_gpu()

    # Verify diffusers version is correct
    try:
        import diffusers
        print(f"  diffusers version: {diffusers.__version__}")
        if diffusers.__version__ != "0.25.0":
            print(f"  ⚠️  Expected diffusers==0.25.0, got {diffusers.__version__}")
            print(f"  Running: pip install diffusers==0.25.0 --force-reinstall")
            run("pip install -q diffusers==0.25.0 --force-reinstall", "")
    except ImportError:
        print("  ❌ diffusers not installed!")

    # Check key files exist
    checks = [
        ("/content/IDM-VTON/src/tryon_pipeline.py", "IDM-VTON pipeline"),
        ("/content/IDM-VTON/gradio_demo/app.py", "Gradio demo"),
        ("/content/IDM-VTON/ckpt/densepose/model_final_162be9.pkl", "DensePose checkpoint"),
        ("/content/IDM-VTON/ckpt/humanparsing/parsing_atr.onnx", "Human parsing (ATR)"),
        ("/content/IDM-VTON/ckpt/openpose/ckpts/body_pose_model.pth", "OpenPose checkpoint"),
    ]

    all_ok = True
    for path, name in checks:
        if os.path.exists(path):
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name} — MISSING at {path}")
            all_ok = False

    if all_ok:
        print("\n" + "=" * 60)
        print("  🎉 SETUP COMPLETE!")
        print("  Now run: !python tryon_api.py")
        print("=" * 60)
    else:
        print("\n⚠️ Some files are missing. Check the errors above.")


if __name__ == "__main__":
    main()
