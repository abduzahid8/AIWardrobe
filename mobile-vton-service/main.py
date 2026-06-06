"""
FastAPI entry-point for the AIWardrobe Mobile-VTON service.

Uses Stable Diffusion 1.5 Inpainting + IP-Adapter for high-fidelity,
garment-aware virtual try-on. The IP-Adapter directly injects visual
features (color, texture, pattern, logos) from the garment photo into
the inpainting pipeline for exact pattern & texture mapping.

Endpoints:
  GET  /health         — Liveness / readiness probe
  POST /tryon          — Single-garment virtual try-on
  POST /tryon/multi    — Multi-garment sequential try-on
  POST /tryon/multi-fused — Multi-garment sequential dressing (ordered)

Environment variables (set by modal_app.py):
  MOBILE_VTON_CHECKPOINT     — path to SD1.5 Inpaint checkpoint directory
  MOBILE_VTON_IP_ADAPTER_DIR — path to IP-Adapter weights directory
  MOBILE_VTON_DEVICE         — "cuda" | "cpu"
  MOBILE_VTON_DTYPE          — "bf16" | "fp16" | "fp32"
"""

from __future__ import annotations

import io
import os
import base64
import time
import logging
from typing import List, Optional

import threading
import torch
from PIL import Image, ImageFilter
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
    description="GPU-accelerated virtual try-on service (SD1.5 + IP-Adapter)",
    version="2.0.0",
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
_pipeline_lock = threading.Lock()
_device: str = os.environ.get("MOBILE_VTON_DEVICE", "cuda")
_dtype_str: str = os.environ.get("MOBILE_VTON_DTYPE", "fp16")
_checkpoint: str = os.environ.get("MOBILE_VTON_CHECKPOINT", "/app/checkpoint/checkpoint")
_ip_adapter_dir: str = os.environ.get("MOBILE_VTON_IP_ADAPTER_DIR", "/app/checkpoint/ip_adapter")

IP_ADAPTER_SCALE = 0.85
MASK_BLUR_RADIUS = 20
SD15_TARGET_SIZE = (512, 512)

# Dressing order for sequential multi-garment try-on
DRESSING_ORDER = ["top", "layer", "pants", "shoes"]
DRESSING_PRIORITY = {label: i for i, label in enumerate(DRESSING_ORDER)}


def _normalize_garment_label(label: Optional[str], fallback_index: int = 0) -> str:
    """Normalize app/catalog labels into the four mask categories."""
    raw = (label or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in ("layer", "outerwear", "jacket", "coat", "cardigan", "hoodie", "blazer"):
        return "layer"
    if raw in ("pants", "lower_body", "bottom", "bottoms", "trousers", "jeans", "shorts"):
        return "pants"
    if raw in ("shoes", "shoe", "footwear", "sneakers", "loafers", "boots"):
        return "shoes"
    if raw in ("top", "upper_body", "shirt", "tshirt", "t_shirt", "tee", "polo", "sweater", "jumper"):
        return "layer" if raw == "upper_body" and fallback_index > 0 else "top"
    return DRESSING_ORDER[min(fallback_index, len(DRESSING_ORDER) - 1)]


def _get_torch_dtype():
    if _dtype_str == "bf16":
        return torch.bfloat16
    if _dtype_str == "fp16":
        return torch.float16
    return torch.float32


def _ensure_checkpoint():
    """
    Ensure the SD1.5 Inpaint model checkpoint is fully present at _checkpoint.
    If model_index.json is present, the weights were baked in during image build.
    """
    model_index = os.path.join(_checkpoint, "model_index.json")
    if os.path.isfile(model_index):
        logger.info(f"Checkpoint found at {_checkpoint} (model_index.json present).")
        return

    hf_model_id = os.environ.get("HF_CHECKPOINT_ID", "runwayml/stable-diffusion-inpainting")
    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    logger.info(f"Checkpoint not found at {_checkpoint}. Downloading {hf_model_id} from HuggingFace...")
    start = time.time()

    try:
        from huggingface_hub import snapshot_download
        os.makedirs(_checkpoint, exist_ok=True)
        snapshot_download(
            repo_id=hf_model_id,
            local_dir=_checkpoint,
            token=hf_token,
            ignore_patterns=["*.ckpt", "*.onnx", "*.msgpack", "*.pt"],
        )
        logger.info(f"Checkpoint downloaded in {time.time() - start:.1f}s")
    except Exception as exc:
        logger.error(f"Failed to download checkpoint: {exc}", exc_info=True)
        raise RuntimeError(f"Checkpoint download failed: {exc}") from exc


def _ensure_ip_adapter():
    """Ensure the IP-Adapter weights are present at _ip_adapter_dir."""
    ip_adapter_bin = os.path.join(_ip_adapter_dir, "models", "ip-adapter_sd15.bin")
    if os.path.isfile(ip_adapter_bin):
        logger.info(f"IP-Adapter weights found at {ip_adapter_bin}.")
        return

    logger.info(f"IP-Adapter weights not found. Downloading h94/IP-Adapter...")
    start = time.time()

    try:
        from huggingface_hub import snapshot_download
        os.makedirs(_ip_adapter_dir, exist_ok=True)
        hf_token = os.environ.get("HF_TOKEN", "").strip() or None
        snapshot_download(
            repo_id="h94/IP-Adapter",
            local_dir=_ip_adapter_dir,
            token=hf_token,
            allow_patterns=[
                "models/ip-adapter_sd15.bin",
                "models/image_encoder/*",
            ],
        )
        logger.info(f"IP-Adapter downloaded in {time.time() - start:.1f}s")
    except Exception as exc:
        logger.error(f"Failed to download IP-Adapter: {exc}", exc_info=True)
        raise RuntimeError(f"IP-Adapter download failed: {exc}") from exc


def _load_pipeline():
    """Load the SD1.5 Inpaint pipeline with IP-Adapter (lazy, called on first request)."""
    global _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        _ensure_checkpoint()
        _ensure_ip_adapter()

        logger.info(f"Loading SD1.5 Inpaint pipeline from {_checkpoint} on {_device} ({_dtype_str})")
        start = time.time()

        try:
            from diffusers import StableDiffusionInpaintPipeline

            if _device == "cuda":
                pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    _checkpoint,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                )
                pipe = pipe.to("cuda")
            else:
                pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    _checkpoint,
                    torch_dtype=torch.float32,
                    safety_checker=None,
                )
                pipe.enable_model_cpu_offload()

            # Load IP-Adapter — uses the diffusers IPAdapterMixin API
            # pretrained_model_name_or_path_or_dict points to local dir containing
            #   models/ip-adapter_sd15.bin  and  models/image_encoder/
            #
            # IMPORTANT: do NOT enable attention slicing with IP-Adapter.
            # enable_attention_slicing() calls set_attention_slicing(), which
            # REPLACES every UNet attention processor with a sliced one and
            # destroys the IP-Adapter cross-attention processors installed by
            # load_ip_adapter(). That breaks the pipeline two different ways:
            #   * slicing BEFORE load_ip_adapter ->
            #       "SlicedAttnProcessor.__init__() missing 1 required positional
            #        argument: 'slice_size'"
            #   * slicing AFTER load_ip_adapter -> the IP-Adapter feeds a
            #       (text_embeds, image_embeds) tuple into a plain processor ->
            #       "'tuple' object has no attribute 'shape'".
            # On a T4 (16 GB), SD1.5-inpaint @ 512x512 fp16 + IP-Adapter needs
            # only ~4-6 GB, so attention slicing is unnecessary. We rely on
            # fp16 (CUDA) / model CPU offload (CPU) for memory headroom instead.
            logger.info(f"Loading IP-Adapter from {_ip_adapter_dir}/models (scale={IP_ADAPTER_SCALE})...")
            pipe.load_ip_adapter(
                _ip_adapter_dir,
                subfolder="models",
                weight_name="ip-adapter_sd15.bin",
            )
            pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)

            _pipeline = pipe
            logger.info(f"Pipeline + IP-Adapter loaded in {time.time() - start:.1f}s")
            return _pipeline

        except Exception as exc:
            logger.error(f"Failed to load pipeline: {exc}", exc_info=True)
            raise RuntimeError(f"Pipeline load failed: {exc}") from exc


# Eagerly load the pipeline at container boot time. Unit tests can disable this
# to verify request normalization without downloading/loading model weights.
if os.environ.get("MOBILE_VTON_EAGER_LOAD", "1") != "0":
    try:
        _load_pipeline()
    except Exception as e:
        logger.error(f"Failed to eagerly load pipeline: {e}")


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _decode_image(src: str) -> Image.Image:
    """Accept a data-URI, raw base64, or HTTP(S) URL, return PIL Image."""
    if src.startswith("data:"):
        header, b64data = src.split(",", 1)
        raw = base64.b64decode(b64data)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    if src.startswith("http://") or src.startswith("https://"):
        import requests
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = requests.get(src, headers=headers, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    try:
        raw = base64.b64decode(src, validate=True)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Unsupported image source format: {src[:60]}") from exc


def _encode_image(img: Image.Image) -> str:
    """Encode PIL Image -> data-URI base64 PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _make_mask_for_garment(person: Image.Image, label: str) -> Image.Image:
    """
    Create a feathered region mask for the garment category.

    Uses Gaussian blur on the mask edges so fabric boundaries
    (collars, sleeves, hems) blend seamlessly with the mannequin.
    """
    w, h = person.size
    mask = Image.new("L", (w, h), 0)

    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)

    label = _normalize_garment_label(label)

    if label == "top":
        draw.rectangle([0, int(h * 0.20), w, int(h * 0.65)], fill=255)
    elif label == "layer":
        draw.rectangle([0, int(h * 0.15), w, int(h * 0.70)], fill=255)
    elif label == "pants":
        draw.rectangle([0, int(h * 0.50), w, int(h * 0.95)], fill=255)
    elif label == "shoes":
        draw.rectangle([0, int(h * 0.88), w, h], fill=255)
    else:
        draw.rectangle([0, 0, w, h], fill=255)

    # Feather the mask edges with Gaussian blur for seamless blending
    mask = mask.filter(ImageFilter.GaussianBlur(radius=MASK_BLUR_RADIUS))

    return mask


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SingleTryOnRequest(BaseModel):
    person_image: str
    garment_image: str
    garment_description: Optional[str] = "clothing"
    guidance_scale: Optional[float] = 7.5
    num_inference_steps: Optional[int] = 30
    seed: Optional[int] = 42


class GarmentItem(BaseModel):
    garment_image: str
    description: Optional[str] = "clothing"
    label: Optional[str] = "top"


class MultiTryOnRequest(BaseModel):
    person_image: str
    garments: List[GarmentItem]
    guidance_scale: Optional[float] = 7.5
    num_inference_steps: Optional[int] = 30
    seed: Optional[int] = 42
    pipeline_version: Optional[str] = "sequential_v1"


# ---------------------------------------------------------------------------
# Core inference helpers
# ---------------------------------------------------------------------------

def _build_tryon_prompt(label: str) -> str:
    """Build a studio-quality prompt for the given garment label."""
    return (
        f"A photorealistic fashion catalog photo of a mannequin wearing the exact {label} garment shown. "
        "The fabric color, texture, pattern, and design details match the reference garment precisely. "
        "Natural fabric drape, realistic clothing folds, soft contact shadows, "
        "studio lighting, clean white background, high quality product photography, 4k."
    )


_NEGATIVE_PROMPT = (
    "deformed, distorted, blurry, low quality, bad anatomy, "
    "extra limbs, face, skin, human, watermark, text, logo, "
    "oversaturated, undersaturated, worst quality, artificial, cartoon"
)


def _run_single_tryon(
    person_img: Image.Image,
    garment_img: Image.Image,
    label: str,
    guidance_scale: float,
    num_steps: int,
    seed: int,
) -> Image.Image:
    """Run one IP-Adapter inpainting pass to dress the exact garment onto the person/mannequin."""
    pipe = _load_pipeline()

    person_resized = person_img.resize(SD15_TARGET_SIZE, Image.LANCZOS)
    garment_resized = garment_img.resize(SD15_TARGET_SIZE, Image.LANCZOS)
    mask = _make_mask_for_garment(person_resized, label)

    prompt = _build_tryon_prompt(label)

    generator = torch.Generator(device="cpu").manual_seed(seed)

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=_NEGATIVE_PROMPT,
            image=person_resized,
            mask_image=mask,
            ip_adapter_image=garment_resized,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=generator,
        ).images[0]

    return result


def _sort_garments_by_dressing_order(garments: List[GarmentItem]) -> List[GarmentItem]:
    """Sort garments into dressing order: tops -> outerwear -> pants -> shoes."""
    def sort_key(g: GarmentItem) -> int:
        label = _normalize_garment_label(g.label)
        return DRESSING_PRIORITY.get(label, 99)
    return sorted(garments, key=sort_key)


def _run_fused_tryon(
    person_img: Image.Image,
    garments: List[GarmentItem],
    guidance_scale: float,
    num_steps: int,
    seed: int,
    version: str,
) -> tuple[Image.Image, dict]:
    """
    Sequential multi-garment try-on: dress garments one at a time in
    proper layering order (tops -> outerwear -> pants -> shoes).

    This prevents visual bleeding and artifacts between different clothing pieces.
    The endpoint remains fully compatible — callers don't need to change anything.
    """
    ordered = _sort_garments_by_dressing_order(garments)
    current_img = person_img
    rendered_labels = []

    for i, garment in enumerate(ordered):
        label = _normalize_garment_label(garment.label, i)
        logger.info(f"[fused_tryon] sequential step {i + 1}/{len(ordered)} label={label}")

        try:
            garment_img = _decode_image(garment.garment_image)
        except Exception as exc:
            raise RuntimeError(f"Garment {i} image decode error: {exc}")

        current_img = _run_single_tryon(
            person_img=current_img,
            garment_img=garment_img,
            label=label,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            seed=seed + i,
        )
        rendered_labels.append(label)

    diagnostics = {
        "pipelineVersion": version,
        "renderedGarments": rendered_labels,
        "garmentCount": len(garments),
        "dressingOrder": [_normalize_garment_label(g.label, i) for i, g in enumerate(ordered)],
    }
    return current_img, diagnostics


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
        "ip_adapter_dir": _ip_adapter_dir,
        "ip_adapter_scale": IP_ADAPTER_SCALE,
        "device": _device,
        "dtype": _dtype_str,
        "pipeline": "sd15_inpaint_ip_adapter",
    }


@app.post("/tryon")
async def single_tryon(req: SingleTryOnRequest):
    """Single-garment virtual try-on using IP-Adapter for exact garment mapping."""
    start_ms = time.time() * 1000
    logger.info(f"[tryon] single label=top steps={req.num_inference_steps} guidance={req.guidance_scale}")

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
        "method_used": "sd15_ip_adapter_single",
        "elapsed_ms": round(elapsed_ms),
    }


@app.post("/tryon/multi")
async def multi_tryon(req: MultiTryOnRequest):
    """Sequential multi-garment try-on (applies garments one at a time in dressing order)."""
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

    ordered = _sort_garments_by_dressing_order(req.garments)

    for i, garment in enumerate(ordered):
        label = _normalize_garment_label(garment.label, i)
        logger.info(f"[tryon/multi] step {i + 1}/{len(ordered)} label={label}")

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
        "method_used": "sd15_ip_adapter_sequential",
        "pipeline_version": "sequential_v1",
        "rendered_garments": rendered_labels,
        "elapsed_ms": round(elapsed_ms),
    }


@app.post("/tryon/multi-fused")
async def multi_tryon_fused(req: MultiTryOnRequest):
    """Multi-garment try-on with sequential dressing order (tops -> outerwear -> pants -> shoes)."""
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
        "method_used": f"sd15_ip_adapter_{version}",
        "pipeline_version": version,
        "rendered_garments": diagnostics.get("renderedGarments", []),
        "diagnostics": diagnostics,
        "elapsed_ms": round(elapsed_ms),
    }
