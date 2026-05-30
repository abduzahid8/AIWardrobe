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
_segmenter = None
_device: str = os.environ.get("MOBILE_VTON_DEVICE", "cuda")
_dtype_str: str = os.environ.get("MOBILE_VTON_DTYPE", "bf16")
_checkpoint: str = os.environ.get("MOBILE_VTON_CHECKPOINT", "/app/checkpoint/checkpoint")


def _get_torch_dtype():
    if _dtype_str == "bf16":
        return torch.bfloat16
    if _dtype_str == "fp16":
        return torch.float16
    return torch.float32


def _ensure_checkpoint():
    """If local checkpoint doesn't exist, download from HuggingFace."""
    if os.path.isdir(_checkpoint) and len(os.listdir(_checkpoint)) > 0:
        logger.info(f"Checkpoint already present at {_checkpoint}")
        return

    hf_model_id = os.environ.get("HF_CHECKPOINT_ID", "stabilityai/stable-diffusion-3.5-medium")
    hf_token_raw = os.environ.get("HF_TOKEN", "")
    hf_token = hf_token_raw if hf_token_raw else None

    logger.info(f"Checkpoint not found at {_checkpoint}. Downloading {hf_model_id} from HuggingFace...")
    logger.info(f"HF_TOKEN present: {bool(hf_token)} (length={len(hf_token_raw) if hf_token_raw else 0})")
    start = time.time()

    try:
        from huggingface_hub import snapshot_download
        os.makedirs(_checkpoint, exist_ok=True)
        snapshot_download(
            repo_id=hf_model_id,
            local_dir=_checkpoint,
            token=hf_token,
            resume_download=True,
        )
        logger.info(f"Checkpoint downloaded in {time.time() - start:.1f}s")
    except Exception as exc:
        logger.error(f"Failed to download checkpoint: {exc}", exc_info=True)
        raise RuntimeError(f"Checkpoint download failed: {exc}") from exc


def _load_segmenter():
    """Lazy-load the torchvision DeepLabV3 segmenter for precise mannequin masking."""
    global _segmenter
    if _segmenter is not None:
        return _segmenter
    try:
        import torchvision.models.segmentation as segmentation
        from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
        logger.info("Loading torchvision DeepLabV3 ResNet50 segmenter...")
        seg_model = segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        seg_model.eval()
        if _device == "cuda":
            seg_model = seg_model.cuda()
        _segmenter = seg_model
        logger.info("DeepLabV3 segmenter loaded successfully.")
    except Exception as exc:
        logger.warning(f"Failed to load torchvision segmenter: {exc}. Using solid mask fallback.")
        _segmenter = False
    return _segmenter


def _load_pipeline():
    """Load the diffusion pipeline (lazy, called on first request)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    _ensure_checkpoint()

    logger.info(f"Loading pipeline from {_checkpoint} on {_device} ({_dtype_str})")
    start = time.time()

    try:
        from diffusers import StableDiffusion3InpaintPipeline

        pipe = StableDiffusion3InpaintPipeline.from_pretrained(
            _checkpoint,
            torch_dtype=_get_torch_dtype(),
            use_safetensors=True,
        )
        # Bypassing slow CPU offload to keep model loaded on CUDA for 5x faster renders
        pipe.to(_device)
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


def _encode_image(img: Image.Image, fmt: str = "jpeg") -> str:
    """
    Encode PIL Image → data-URI base64 string.
    Defaults to JPEG (q=92) which is ~6× smaller than PNG for photographic images.
    Falls back to PNG on any error.
    """
    fmt = fmt.lower().strip()
    if fmt in ("jpg", "jpeg"):
        buf = io.BytesIO()
        rgb = img.convert("RGB")  # JPEG does not support alpha
        rgb.save(buf, format="JPEG", quality=92, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"
    # PNG fallback
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _get_mannequin_silhouette(person: Image.Image) -> Image.Image:
    """
    Extract a precise semantic mask of the person/mannequin using DeepLabV3.
    Applies morphological dilation (+8 px) and Gaussian feathering (σ=4) to:
      - Catch edges that DeepLabV3 under-segments
      - Soften the mask boundary so inpainting blends naturally
    """
    seg_model = _load_segmenter()
    w, h = person.size

    if not seg_model:
        return Image.new("L", (w, h), 255)

    try:
        import torchvision.transforms as T
        import numpy as np
        from PIL import ImageFilter

        trf = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = trf(person).unsqueeze(0)
        if _device == "cuda":
            input_tensor = input_tensor.cuda()

        with torch.no_grad():
            out = seg_model(input_tensor)['out'][0]

        person_class_idx = 15  # COCO class 'person'
        classes_predictions = torch.argmax(out, dim=0).byte().cpu().numpy()

        # Binary mask at model resolution
        mask_np = (classes_predictions == person_class_idx).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_np, mode="L").resize((w, h), Image.BILINEAR)

        # Morphological dilation: expand silhouette by 8 px to catch DeepLabV3 under-segments
        mask_img = mask_img.filter(ImageFilter.MaxFilter(size=9))  # 9 = 2*4+1 → ~8px radius

        # Feather edges with Gaussian blur for smooth inpaint blend seam
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=4))

        return mask_img

    except Exception as exc:
        logger.warning(f"Failed to generate mannequin silhouette: {exc}. Using solid mask fallback.")
        return Image.new("L", (w, h), 255)


def _make_mask_for_garment(person: Image.Image, label: str) -> Image.Image:
    """
    Create a precise, body-locked mask for the garment category.
    Intersects the height-based zone with the mannequin's parsed silhouette
    to completely lock the white background.
    """
    w, h = person.size
    zone_mask = Image.new("L", (w, h), 0)

    from PIL import ImageDraw
    draw = ImageDraw.Draw(zone_mask)

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

    # Intersect height zone with the parsed mannequin silhouette
    silhouette = _get_mannequin_silhouette(person)
    from PIL import ImageChops
    final_mask = ImageChops.multiply(zone_mask, silhouette)

    return final_mask


def _get_mask_bbox(mask: Image.Image):
    """Return (left, upper, right, lower) of the mask's bounding box."""
    arr = list(mask.getdata())
    w, h = mask.size
    xs = [i % w for i, v in enumerate(arr) if v > 0]
    ys = [i // w for i, v in enumerate(arr) if v > 0]
    if not xs:
        return (0, 0, w, h)
    return (min(xs), min(ys), max(xs) + 1, max(ys) + 1)


def _composite_garment_onto_person(
    person: Image.Image,
    garment: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """
    Place the garment image into the mask region of the person image.
    Uses PIL corner flood-fill for high-quality, boundary-preserving
    background removal without white garment transparency bugs.
    """
    from PIL import ImageDraw

    composite = person.copy().convert("RGBA")
    garment_rgba = garment.convert("RGBA")

    # Bounding box of the mask region
    left, upper, right, lower = _get_mask_bbox(mask)
    bbox_w = right - left
    bbox_h = lower - upper

    # Resize garment to fit inside bbox (contain, no distortion)
    g_w, g_h = garment_rgba.size
    scale = min(bbox_w / g_w, bbox_h / g_h)
    new_w = max(1, int(g_w * scale))
    new_h = max(1, int(g_h * scale))
    garment_scaled = garment_rgba.resize((new_w, new_h), Image.LANCZOS)

    # Center within bbox
    paste_x = left + (bbox_w - new_w) // 2
    paste_y = upper + (bbox_h - new_h) // 2

    # High-quality corner flood-fill for background removal
    garment_clean = garment_scaled.copy()
    w_g, h_g = garment_clean.size
    corners = [(0, 0), (w_g - 1, 0), (0, h_g - 1), (w_g - 1, h_g - 1)]

    for pt in corners:
        try:
            pixel = garment_clean.getpixel(pt)
            if len(pixel) == 4 and pixel[3] > 0:
                ImageDraw.floodfill(garment_clean, pt, (255, 255, 255, 0), thresh=28)
        except Exception as exc:
            logger.warning(f"[_composite_garment_onto_person] Corner floodfill failed at {pt}: {exc}")

    composite.paste(garment_clean, (paste_x, paste_y), garment_clean)
    return composite.convert("RGB")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SingleTryOnRequest(BaseModel):
    person_image: str                        # data-URI or URL
    garment_image: str                       # data-URI or URL
    garment_description: Optional[str] = "clothing"
    guidance_scale: Optional[float] = 2.0
    num_inference_steps: Optional[int] = 10
    seed: Optional[int] = 42
    # "human" → model/person photo; "mannequin" → studio dummy
    subject_type: Optional[str] = "mannequin"
    # "jpeg" (default, small) or "png" (lossless)
    output_format: Optional[str] = "jpeg"


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
    subject_type: Optional[str] = "mannequin"
    output_format: Optional[str] = "jpeg"


# ---------------------------------------------------------------------------
# Core inference helpers
# ---------------------------------------------------------------------------

def _build_prompts(label: str, garment_desc: str, subject_type: str) -> tuple[str, str]:
    """
    Build prompt and negative prompt tuned to the garment type and subject.
    subject_type: "mannequin" | "human"
    """
    subject = "fashion mannequin" if subject_type != "human" else "fashion model"
    bg = "clean white studio background" if subject_type != "human" else "lifestyle background"

    prompt = (
        f"Ultra-high-quality fashion product photography. "
        f"A {subject} wearing a {garment_desc} ({label}). "
        "Perfectly draped fabric with natural creases and shadows. "
        f"Professional studio lighting, {bg}. "
        "Sharp details, vivid colors, 8K resolution."
    )
    negative_prompt = (
        "deformed body, distorted limbs, extra arms, extra legs, fused fingers, "
        "blurry, low resolution, pixelated, watermark, text, logo overlay, "
        "bad anatomy, plastic skin, oversaturated, jpeg artifacts, cropped"
    )
    return prompt, negative_prompt


def _run_single_tryon(
    person_img: Image.Image,
    garment_img: Image.Image,
    label: str,
    guidance_scale: float,
    num_steps: int,
    seed: int,
    garment_desc: str = "clothing",
    subject_type: str = "mannequin",
) -> Image.Image:
    """Run one inpainting pass to dress a garment onto the person/mannequin."""
    pipe = _load_pipeline()

    # Resize to 768×1024 (SD3.5 optimal resolution)
    target_size = (768, 1024)
    person_resized = person_img.resize(target_size, Image.LANCZOS)
    garment_resized = garment_img.resize(target_size, Image.LANCZOS)
    mask = _make_mask_for_garment(person_resized, label)

    # Composite the garment onto the person — gives the model a visual reference
    # of the actual garment shape/color, not just a text description.
    composite = _composite_garment_onto_person(person_resized, garment_resized, mask)

    prompt, negative_prompt = _build_prompts(label, garment_desc, subject_type)

    generator = torch.Generator(device=_device).manual_seed(seed)

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=composite,
            mask_image=mask,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            strength=0.68,  # Preserve garment textures, logos, and colors
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
    subject_type: str = "mannequin",
) -> tuple[Image.Image, dict]:
    """
    Fused multi-garment try-on: composite all garments onto the person,
    then run a single inpaint pass to blend everything together.

    v2/v3: garments composited into per-label mask regions.
    """
    pipe = _load_pipeline()
    target_size = (768, 1024)
    person_resized = person_img.resize(target_size, Image.LANCZOS)

    # Build union mask and composite garments onto person
    from PIL import ImageDraw
    union_mask = Image.new("L", target_size, 0)
    draw = ImageDraw.Draw(union_mask)
    w, h = target_size

    label_names = []
    current_composite = person_resized.copy().convert("RGBA")

    for g in garments:
        label = g.label or "top"
        label_names.append(label)

        # Build per-garment mask and add to union
        if label in ("top", "upper_body"):
            draw.rectangle([0, int(h * 0.20), w, int(h * 0.65)], fill=255)
        elif label in ("layer", "outerwear"):
            draw.rectangle([0, int(h * 0.15), w, int(h * 0.70)], fill=255)
        elif label in ("pants", "lower_body", "bottoms"):
            draw.rectangle([0, int(h * 0.50), w, int(h * 0.95)], fill=255)
        elif label in ("shoes", "footwear"):
            draw.rectangle([0, int(h * 0.88), w, h], fill=255)

        # Composite this garment onto the person within its region
        if g.garment_image:
            try:
                g_img = _decode_image(g.garment_image)
                g_resized = g_img.resize(target_size, Image.LANCZOS)
                g_mask = _make_mask_for_garment(person_resized, label)
                current_composite = _composite_garment_onto_person(
                    current_composite.convert("RGB"), g_resized, g_mask
                ).convert("RGBA")
            except Exception as exc:
                logger.warning(f"[_run_fused_tryon] could not composite garment {label}: {exc}")

    # Also intersect the entire union mask with the mannequin silhouette to lock background
    silhouette = _get_mannequin_silhouette(person_resized)
    from PIL import ImageChops
    union_mask = ImageChops.multiply(union_mask, silhouette)

    garment_descs = ", ".join(
        f"{g.label or 'garment'} ({g.description or ''})" for g in garments
    )
    subject = "fashion mannequin" if subject_type != "human" else "fashion model"
    bg = "clean white studio background" if subject_type != "human" else "lifestyle background"
    prompt = (
        f"Ultra-high-quality fashion product photography. "
        f"A {subject} wearing a complete outfit: {garment_descs}. "
        "Perfectly draped fabrics with natural creases and shadows. "
        f"Professional studio lighting, {bg}. "
        "Sharp details, vivid colors, 8K resolution."
    )
    negative_prompt = (
        "deformed body, distorted limbs, extra arms, extra legs, fused fingers, "
        "blurry, low resolution, pixelated, watermark, text, logo overlay, "
        "bad anatomy, plastic skin, oversaturated, jpeg artifacts, cropped"
    )

    generator = torch.Generator(device=_device).manual_seed(seed)

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=current_composite.convert("RGB"),
            mask_image=union_mask,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            strength=0.68,  # Maintain garment textures and logos perfectly
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
    fmt = req.output_format or "jpeg"
    logger.info(
        f"[tryon] single steps={req.num_inference_steps} guidance={req.guidance_scale} "
        f"subject={req.subject_type} fmt={fmt}"
    )

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
            garment_desc=req.garment_description or "clothing",
            subject_type=req.subject_type or "mannequin",
        )
    except Exception as exc:
        logger.error(f"[tryon] inference error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    elapsed_ms = time.time() * 1000 - start_ms
    return {
        "success": True,
        "result_image": _encode_image(result_img, fmt),
        "method_used": "sd3_inpaint_single",
        "output_format": fmt,
        "elapsed_ms": round(elapsed_ms),
    }


@app.post("/tryon/multi")
async def multi_tryon(req: MultiTryOnRequest):
    """Sequential multi-garment try-on (applies garments one at a time)."""
    start_ms = time.time() * 1000
    fmt = req.output_format or "jpeg"
    logger.info(f"[tryon/multi] count={len(req.garments)} pipeline=sequential_v1 subject={req.subject_type}")

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
                garment_desc=garment.description or "clothing",
                subject_type=req.subject_type or "mannequin",
            )
            rendered_labels.append(label)
        except Exception as exc:
            logger.error(f"[tryon/multi] step {i} failed: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Inference failed at step {i} ({label}): {exc}")

    elapsed_ms = time.time() * 1000 - start_ms
    return {
        "success": True,
        "result_image": _encode_image(current_img, fmt),
        "method_used": "sd3_inpaint_sequential",
        "pipeline_version": "sequential_v1",
        "output_format": fmt,
        "rendered_garments": rendered_labels,
        "elapsed_ms": round(elapsed_ms),
    }


@app.post("/tryon/multi-fused")
async def multi_tryon_fused(req: MultiTryOnRequest):
    """Fused single-pass multi-garment try-on (v2 / v3)."""
    version = req.pipeline_version or "fused_v2"
    fmt = req.output_format or "jpeg"
    start_ms = time.time() * 1000
    logger.info(
        f"[tryon/multi-fused] count={len(req.garments)} pipeline={version} "
        f"subject={req.subject_type} fmt={fmt}"
    )

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
            subject_type=req.subject_type or "mannequin",
        )
    except Exception as exc:
        logger.error(f"[tryon/multi-fused] inference error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Fused inference failed: {exc}")

    elapsed_ms = time.time() * 1000 - start_ms
    diagnostics["totalElapsedMs"] = round(elapsed_ms)

    return {
        "success": True,
        "result_image": _encode_image(result_img, fmt),
        "method_used": f"sd3_inpaint_{version}",
        "pipeline_version": version,
        "output_format": fmt,
        "rendered_garments": diagnostics.get("renderedGarments", []),
        "diagnostics": diagnostics,
        "elapsed_ms": round(elapsed_ms),
    }
