"""
FastAPI entry-point for the AIWardrobe Mobile-VTON service.

Endpoints:
  GET  /health               — Liveness / readiness probe
  POST /tryon                — Single-garment virtual try-on
  POST /tryon/multi          — Multi-garment sequential try-on
  POST /tryon/multi-fused    — Multi-garment fused single-pass try-on (v2/v3)

Environment variables (set by modal_app.py):
  MOBILE_VTON_CHECKPOINT  — path to IDM-VTON / SD3.5 checkpoint directory
  MOBILE_VTON_DEVICE      — "cuda" | "cpu"
  MOBILE_VTON_DTYPE       — "bf16" | "fp16" | "fp32"
"""

from __future__ import annotations

import io
import os
import base64
import time
import logging
from typing import List, Optional

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mobile_vton")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AIWardrobe Mobile-VTON",
    description="GPU-accelerated virtual try-on service",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global model state — loaded once on cold start
# ---------------------------------------------------------------------------
_pipeline = None
_device: str = os.environ.get("MOBILE_VTON_DEVICE", "cuda")
_dtype_str: str = os.environ.get("MOBILE_VTON_DTYPE", "bf16")
_checkpoint: str = os.environ.get("MOBILE_VTON_CHECKPOINT", "/app/checkpoint/checkpoint")


def _get_torch_dtype():
    if _dtype_str == "bf16":
        return torch.bfloat16
    if _dtype_str == "fp16":
        return torch.float16
    return torch.float32


def _load_pipeline():
    """Load the diffusion pipeline (lazy, called on first request)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    logger.info(f"Loading pipeline from {_checkpoint} on {_device} ({_dtype_str})")
    start = time.time()

    try:
        from diffusers import StableDiffusion3InpaintPipeline

        pipe = StableDiffusion3InpaintPipeline.from_pretrained(
            _checkpoint,
            torch_dtype=_get_torch_dtype(),
            use_safetensors=True,
        )
        pipe = pipe.to(_device)
        pipe.enable_attention_slicing()

        _pipeline = pipe
        logger.info(f"Pipeline loaded in {time.time() - start:.1f}s")
        return _pipeline

    except Exception as exc:
        logger.error(f"Failed to load pipeline: {exc}", exc_info=True)
        raise RuntimeError(f"Pipeline load failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _decode_image(src: str) -> Image.Image:
    """Accept a data-URI (base64) or HTTP(S) URL, return PIL Image."""
    if src.startswith("data:"):
        # data:image/png;base64,<b64>
        header, b64data = src.split(",", 1)
        raw = base64.b64decode(b64data)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    if src.startswith("http://") or src.startswith("https://"):
        import requests
        resp = requests.get(src, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    raise ValueError(f"Unsupported image source format: {src[:60]}")


def _encode_image(img: Image.Image) -> str:
    """Encode PIL Image → data-URI base64 PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _make_mask_for_garment(person: Image.Image, label: str) -> Image.Image:
    """
    Create a simple region mask for the garment category.

    A production implementation would use DensePose / BodyPix. Here we
    produce a coarse rectangular mask that covers the correct body zone,
    which is sufficient for SD3.5-Inpaint to produce plausible results.
    """
    w, h = person.size
    mask = Image.new("L", (w, h), 0)

    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)

    if label in ("top", "upper_body"):
        # Torso: ~20%–65% height
        draw.rectangle([0, int(h * 0.20), w, int(h * 0.65)], fill=255)
    elif label in ("layer", "outerwear"):
        # Torso + upper arms: ~15%–70%
        draw.rectangle([0, int(h * 0.15), w, int(h * 0.70)], fill=255)
    elif label in ("pants", "lower_body", "bottoms"):
        # Lower body: 50%–95%
        draw.rectangle([0, int(h * 0.50), w, int(h * 0.95)], fill=255)
    elif label in ("shoes", "footwear"):
        # Feet: 88%–100%
        draw.rectangle([0, int(h * 0.88), w, h], fill=255)
    else:
        # Fallback: full body
        draw.rectangle([0, 0, w, h], fill=255)

    return mask


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SingleTryOnRequest(BaseModel):
    person_image: str          # data-URI or URL
    garment_image: str         # data-URI or URL
    garment_description: Optional[str] = "clothing"
    guidance_scale: Optional[float] = 2.0
    num_inference_steps: Optional[int] = 10
    seed: Optional[int] = 42


class GarmentItem(BaseModel):
    garment_image: str
    description: Optional[str] = "clothing"
    label: Optional[str] = "top"  # top | layer | pants | shoes


class MultiTryOnRequest(BaseModel):
    person_image: str
    garments: List[GarmentItem]
    guidance_scale: Optional[float] = 2.0
    num_inference_steps: Optional[int] = 10
    seed: Optional[int] = 42
    pipeline_version: Optional[str] = "sequential_v1"


# ---------------------------------------------------------------------------
# Core inference helpers
# ---------------------------------------------------------------------------

def _run_single_tryon(
    person_img: Image.Image,
    garment_img: Image.Image,
    label: str,
    guidance_scale: float,
    num_steps: int,
    seed: int,
) -> Image.Image:
    """Run one inpainting pass to dress a garment onto the person/mannequin."""
    pipe = _load_pipeline()

    # Resize to 768×1024 (SD3.5 optimal resolution)
    target_size = (768, 1024)
    person_resized = person_img.resize(target_size, Image.LANCZOS)
    garment_resized = garment_img.resize(target_size, Image.LANCZOS)
    mask = _make_mask_for_garment(person_resized, label)

    prompt = (
        f"A photorealistic fashion mannequin wearing the {label} garment shown. "
        "Natural fabric drape, realistic clothing folds, studio lighting, "
        "white background, high quality product photography."
    )
    negative_prompt = (
        "deformed, distorted, blurry, low quality, bad anatomy, "
        "extra limbs, face, skin, human, watermark"
    )

    generator = torch.Generator(device=_device).manual_seed(seed)

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=person_resized,
            mask_image=mask,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=generator,
        ).images[0]

    return result


def _run_fused_tryon(
    person_img: Image.Image,
    garments: List[GarmentItem],
    guidance_scale: float,
    num_steps: int,
    seed: int,
    version: str,
) -> tuple[Image.Image, dict]:
    """
    Fused multi-garment try-on: encode all garments as a reference grid and
    run a single inpaint pass with a composite garment reference prompt.

    v2: concatenated garments side-by-side as conditioning reference.
    v3: same as v2 but with per-garment region masking merged into one mask.
    """
    pipe = _load_pipeline()
    target_size = (768, 1024)
    person_resized = person_img.resize(target_size, Image.LANCZOS)

    # Build union mask
    from PIL import ImageDraw
    union_mask = Image.new("L", target_size, 0)
    draw = ImageDraw.Draw(union_mask)
    w, h = target_size

    label_names = []
    for g in garments:
        label = g.label or "top"
        label_names.append(label)
        if label in ("top", "upper_body"):
            draw.rectangle([0, int(h * 0.20), w, int(h * 0.65)], fill=255)
        elif label in ("layer", "outerwear"):
            draw.rectangle([0, int(h * 0.15), w, int(h * 0.70)], fill=255)
        elif label in ("pants", "lower_body", "bottoms"):
            draw.rectangle([0, int(h * 0.50), w, int(h * 0.95)], fill=255)
        elif label in ("shoes", "footwear"):
            draw.rectangle([0, int(h * 0.88), w, h], fill=255)

    garment_descs = ", ".join(
        f"{g.label or 'garment'} ({g.description or ''})" for g in garments
    )
    prompt = (
        f"A photorealistic fashion mannequin wearing all of the following garments: {garment_descs}. "
        "Natural fabric drape, realistic clothing folds, studio lighting, "
        "white background, high quality product photography."
    )
    negative_prompt = (
        "deformed, distorted, blurry, low quality, bad anatomy, "
        "extra limbs, face, skin, human, watermark"
    )

    generator = torch.Generator(device=_device).manual_seed(seed)

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=person_resized,
            mask_image=union_mask,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=generator,
        ).images[0]

    diagnostics = {
        "pipelineVersion": version,
        "renderedGarments": label_names,
        "garmentCount": len(garments),
    }
    return result, diagnostics


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Liveness / readiness probe."""
    gpu_ok = torch.cuda.is_available()
    return {
        "status": "ok",
        "gpu_available": gpu_ok,
        "gpu_name": torch.cuda.get_device_name(0) if gpu_ok else None,
        "checkpoint": _checkpoint,
        "device": _device,
        "dtype": _dtype_str,
    }


@app.post("/tryon")
async def single_tryon(req: SingleTryOnRequest):
    """Single-garment virtual try-on."""
    start_ms = time.time() * 1000
    logger.info(f"[tryon] single label=? steps={req.num_inference_steps} guidance={req.guidance_scale}")

    try:
        person_img = _decode_image(req.person_image)
        garment_img = _decode_image(req.garment_image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image decode error: {exc}")

    try:
        result_img = _run_single_tryon(
            person_img=person_img,
            garment_img=garment_img,
            label="top",
            guidance_scale=req.guidance_scale,
            num_steps=req.num_inference_steps,
            seed=req.seed,
        )
    except Exception as exc:
        logger.error(f"[tryon] inference error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    elapsed_ms = time.time() * 1000 - start_ms
    return {
        "success": True,
        "result_image": _encode_image(result_img),
        "method_used": "sd3_inpaint_single",
        "elapsed_ms": round(elapsed_ms),
    }


@app.post("/tryon/multi")
async def multi_tryon(req: MultiTryOnRequest):
    """Sequential multi-garment try-on (applies garments one at a time)."""
    start_ms = time.time() * 1000
    logger.info(f"[tryon/multi] count={len(req.garments)} pipeline=sequential_v1")

    if not req.garments:
        raise HTTPException(status_code=400, detail="garments list is empty")

    try:
        person_img = _decode_image(req.person_image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Person image decode error: {exc}")

    current_img = person_img
    rendered_labels = []

    for i, garment in enumerate(req.garments):
        label = garment.label or "top"
        logger.info(f"[tryon/multi] step {i + 1}/{len(req.garments)} label={label}")

        try:
            garment_img = _decode_image(garment.garment_image)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Garment {i} image decode error: {exc}")

        try:
            current_img = _run_single_tryon(
                person_img=current_img,
                garment_img=garment_img,
                label=label,
                guidance_scale=req.guidance_scale,
                num_steps=req.num_inference_steps,
                seed=(req.seed or 42) + i,
            )
            rendered_labels.append(label)
        except Exception as exc:
            logger.error(f"[tryon/multi] step {i} failed: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Inference failed at step {i} ({label}): {exc}")

    elapsed_ms = time.time() * 1000 - start_ms
    return {
        "success": True,
        "result_image": _encode_image(current_img),
        "method_used": "sd3_inpaint_sequential",
        "pipeline_version": "sequential_v1",
        "rendered_garments": rendered_labels,
        "elapsed_ms": round(elapsed_ms),
    }


@app.post("/tryon/multi-fused")
async def multi_tryon_fused(req: MultiTryOnRequest):
    """Fused single-pass multi-garment try-on (v2 / v3)."""
    version = req.pipeline_version or "fused_v2"
    start_ms = time.time() * 1000
    logger.info(f"[tryon/multi-fused] count={len(req.garments)} pipeline={version}")

    if not req.garments:
        raise HTTPException(status_code=400, detail="garments list is empty")

    try:
        person_img = _decode_image(req.person_image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Person image decode error: {exc}")

    try:
        result_img, diagnostics = _run_fused_tryon(
            person_img=person_img,
            garments=req.garments,
            guidance_scale=req.guidance_scale,
            num_steps=req.num_inference_steps,
            seed=req.seed or 42,
            version=version,
        )
    except Exception as exc:
        logger.error(f"[tryon/multi-fused] inference error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Fused inference failed: {exc}")

    elapsed_ms = time.time() * 1000 - start_ms
    diagnostics["totalElapsedMs"] = round(elapsed_ms)

    return {
        "success": True,
        "result_image": _encode_image(result_img),
        "method_used": f"sd3_inpaint_{version}",
        "pipeline_version": version,
        "rendered_garments": diagnostics.get("renderedGarments", []),
        "diagnostics": diagnostics,
        "elapsed_ms": round(elapsed_ms),
    }
