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

# IP-Adapter default conditioning strength. Bumped from 0.70 -> 0.85 so the
# color/pattern signal from the garment photo actually wins against SD1.5's
# generic "studio mannequin" bias. The earlier 0.70 produced light-grey
# t-shirts, dark cocoa pants and grey loafers even when the input photo
# was pure white / walnut brown / cognac brown. At 0.85 the model still
# has freedom to drape the fabric realistically, but the dominant color
# of the input garment is preserved end-to-end.
IP_ADAPTER_SCALE = 0.85

# Per-label IP-Adapter scale overrides. The photo of a white t-shirt has a
# dark care label and lots of highlight/shadow that CLIP encodes as
# "white-with-dark-spot" or "grey".  For LIGHT garments we boost the
# uniform swatch's weight (handled in _build_ip_adapter_reference) AND
# nudge the scale up.  For DARK/SATURATED garments (brown pants, brown
# loafers) the photo already carries a strong, clean color signal so
# 0.85 is plenty; pushing higher makes the texture dominate the silhouette.
IP_ADAPTER_SCALE_PER_LABEL = {
    "top":   1.20,
    "layer": 0.95,
    "pants": 0.95,
    "shoes": 0.95,
}

MASK_BLUR_RADIUS = 8
SD15_TARGET_SIZE = (512, 512)

# Pre-paste the actual garment photo into the masked area before sending
# the image to SD1.5. This forces the inpaint pipeline to use the
# garment's exact fabric, color, pattern, hardware and stitching as a
# strong prior (the noisy version of `image * mask` is what the model
# denoises). Without pre-paste the model has to imagine the garment
# from the prompt + IP-Adapter alone, which loses fine details like
# herringbone weave, moccasin stitching and ribbed collars.
PRECOMPOSE_GARMENT = os.environ.get("MOBILE_VTON_PRECOMPOSE_GARMENT", "1") not in ("0", "false", "False", "FALSE")

# Dressing order for sequential multi-garment try-on
DRESSING_ORDER = ["top", "layer", "pants", "shoes"]
DRESSING_PRIORITY = {label: i for i, label in enumerate(DRESSING_ORDER)}

# Body-shape mask vertical bands. Each tuple is
#   (y_top, y_bot, hw_neck, hw_shoulder, hw_waist, hw_hip, hw_ankle)
# in fractions of (height, width, width, width, width, width).
#
# Earlier versions intersected this band with the natural body silhouette,
# but the mannequin's silhouette is too narrow at the shoulders / upper
# torso, so the intersection collapsed to a thin vertical slice. SD1.5
# then had no room to draw sleeves or pant legs and produced a "tank top"
# for shirts and white pants for the brown pair.
#
# New behaviour: use a DILATED silhouette as the clip so the garment has
# a little "extension" beyond the natural shoulder / hip line, then
# intersect the band with that dilated silhouette. This gives SD1.5 a
# full upper-body region to draw sleeves and a full leg region to draw
# trousers — without bleeding into the white studio backdrop.
_MASK_BANDS = {
    # label      : (y_top, y_bot, hw_neck, hw_shoulder, hw_waist, hw_hip, hw_ankle)
    #
    # The band half-widths are sized to cover the garment (including sleeves
    # and trouser cuffs) when intersected with a DILATED silhouette, NOT
    # the natural body line. The natural mannequin silhouette is too
    # narrow (headless grey fashion mannequin, ~10% half-width at the
    # shoulder) so intersecting the band with the raw silhouette
    # collapsed the t-shirt to a tank top and the trousers to a stick.
    # We dilate the silhouette by 96 px and use the band half-widths
    # below, which are sized so the dilated silhouette + band intersection
    # matches a real garment's outer silhouette:
    #
    #  TOP    — neck 0.18, shoulder 0.50, waist 0.40, hip 0.40, ankle 0.40
    #           hw_shoulder 0.50 + 96 px dilation gives ~65% of image
    #           width at the shoulder — enough for SD1.5 to draw t-shirt
    #           sleeves that reach the upper arms instead of leaving the
    #           arms bare grey.
    "top":   (0.18, 0.52, 0.18, 0.50, 0.40, 0.40, 0.40),
    #  LAYER  — extends past the elbow to cover jacket sleeves. Slightly
    #           wider than top because jackets are bulkier.
    "layer": (0.16, 0.66, 0.18, 0.52, 0.42, 0.40, 0.40),
    #  PANTS  — full leg width including cuff, waist to ankle. Wider
    #           at the hip so SD1.5 can draw a trouser leg that covers
    #           the full leg instead of a narrow stick.
    "pants": (0.48, 0.92, 0.30, 0.30, 0.32, 0.30, 0.20),
    #  SHOES  — split-rendered (see _run_shoes_two_pass): the full
    #           bottom band is 0.20 half-width (40% of image width) so
    #           the left half-mask is 20% wide and the right half-mask
    #           is 20% wide — each half is just wide enough for one
    #           loafer, and the two-pass rendering forces SD1.5 to
    #           produce a distinct left shoe and right shoe rather than
    #           one wide blob.
    "shoes": (0.90, 0.97, 0.08, 0.08, 0.08, 0.08, 0.08),
}

# Per-label mask overrides — the silhouette intersection is skipped for
# labels that need to extend beyond the natural body line (shoes extend
# past the natural foot width).
_SKIP_SILHOUETTE_CLIP = set()

# How much to dilate the silhouette before intersecting with the band.
# This adds "fabric slack" beyond the natural body line so SD1.5 can
# draw a short-sleeve t-shirt sleeve or the cuff of a trouser without
# the mask boundary clipping it. Bumped from 24 -> 96 px so the
# mannequin's narrow silhouette is widened enough to match a real
# garment's outer silhouette (sleeves, trouser cuffs, etc.).
_SILHOUETTE_DILATE_RADIUS = 16

# Background-keying threshold for garment preprocessing (0..255).
# Garment pixels in product photos are typically slightly darker than the
# pure-white studio backdrop (e.g. shirt center ~239 RGB vs corner ~246).
# Any pixel with per-channel mean > threshold is treated as background and
# replaced with neutral gray so IP-Adapter does not "see" the white studio
# backdrop as part of the garment style.
_GARMENT_BG_THRESHOLD = 245
_GARMENT_BG_NEUTRAL = (192, 192, 192)


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


def _person_silhouette(person: Image.Image) -> Image.Image:
    """
    Derive a coarse silhouette mask of the mannequin by removing near-white
    background pixels. Used to clip the per-garment mask to the body so the
    inpainting region never bleeds into the studio backdrop.
    """
    w, h = person.size
    rgb = person.convert("RGB")
    import numpy as np
    arr = np.asarray(rgb, dtype=np.uint8)
    # Body is darker than the white studio background. We use a brightness cutoff
    # based on the per-channel mean — the mannequin in assets is a light grey
    # (RGB ~210) on a white seamless (RGB ~250+), so anything with mean < 230
    # is treated as body. Tighter than 245 (which would cut off the body) and
    # looser than 200 (which would over-grow the mask into the background).
    mean_ch = arr.mean(axis=2)
    silh = (mean_ch < 230).astype(np.uint8) * 255
    silh_img = Image.fromarray(silh, mode="L")
    # Close small holes, then soften edges so the garment mask boundary stays
    # just inside the body silhouette.
    silh_img = silh_img.filter(ImageFilter.MinFilter(5))
    silh_img = silh_img.filter(ImageFilter.GaussianBlur(radius=6))
    return silh_img


def _make_mask_for_garment(person: Image.Image, label: str) -> Image.Image:
    """
    Body-shape aware feathered mask for the garment category.

    The mask is the intersection of:
      * a tapered vertical band whose half-width follows the body silhouette
        (neck < shoulder > waist > hip > ankle), and
      * a DILATED version of the actual person silhouette derived from the
        input image.

    Why a DILATED silhouette: a mannequin on a white seamless has a thin
    shoulder line. A raw intersection clips the "top" band to that thin
    line, which gives SD1.5 zero room to draw sleeves — that's why the
    previous build produced tank tops. Dilating the silhouette by
    _SILHOUETTE_DILATE_RADIUS pixels adds "fabric slack" beyond the body
    so the inpainting region is wide enough for sleeves, pant legs and
    shoes without bleeding into the white background.
    """
    w, h = person.size
    label = _normalize_garment_label(label)
    band = _MASK_BANDS.get(label, (0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5))

    y_top_frac, y_bot_frac, hw_neck, hw_shoulder, hw_waist, hw_hip, hw_ankle = band

    y_top = int(h * y_top_frac)
    y_bot = int(h * y_bot_frac)
    if y_bot <= y_top:
        y_bot = min(h, y_top + 1)

    # Build a tapered polygon for the vertical band. Half-width interpolates
    # between the configured keypoints by row.
    import numpy as np
    rows = np.arange(y_top, y_bot)
    if rows.size == 0:
        rows = np.array([y_top])

    def _half_width_for_row(y: int) -> float:
        frac = (y - y_top) / max(1, (y_bot - y_top))
        # keypoints: top=neck, 0.20=shoulder, 0.55=waist, 0.75=hip, 1.00=ankle
        if frac < 0.20:
            t = frac / 0.20
            hw = hw_neck + t * (hw_shoulder - hw_neck)
        elif frac < 0.55:
            t = (frac - 0.20) / 0.35
            hw = hw_shoulder + t * (hw_waist - hw_shoulder)
        elif frac < 0.75:
            t = (frac - 0.55) / 0.20
            hw = hw_waist + t * (hw_hip - hw_waist)
        else:
            t = (frac - 0.75) / 0.25
            hw = hw_hip + t * (hw_ankle - hw_hip)
        return hw

    cx = w / 2.0
    half_widths = np.array([_half_width_for_row(int(y)) for y in rows]) * w

    polygon = []
    # Left edge top->bottom
    for y, hw in zip(rows, half_widths):
        polygon.append((cx - hw, float(y)))
    # Right edge bottom->top
    for y, hw in zip(reversed(rows), reversed(half_widths)):
        polygon.append((cx + hw, float(y)))

    band_mask = Image.new("L", (w, h), 0)
    from PIL import ImageDraw
    ImageDraw.Draw(band_mask).polygon(polygon, fill=255)

    # Dilated silhouette: real body shape, expanded by ~_SILHOUETTE_DILATE_RADIUS
    # pixels so the inpainting region has "fabric slack" beyond the natural
    # shoulder / hip line. Without this dilation the previous build produced
    # tank tops for shirts and a barely-visible change for trousers because
    # the intersection collapsed to the thin body line.
    silh = _person_silhouette(person)
    if _SILHOUETTE_DILATE_RADIUS > 0:
        silh_dilated = silh.filter(ImageFilter.MaxFilter(2 * _SILHOUETTE_DILATE_RADIUS + 1))
    else:
        silh_dilated = silh

    import numpy as np
    band_arr = np.asarray(band_mask, dtype=np.uint8)
    if label in _SKIP_SILHOUETTE_CLIP:
        # Shoes extend past the natural foot width (loafers / sneakers are
        # wider than the mannequin's bare foot). Skip the silhouette clip
        # so the mask stays wide enough to draw two distinct shoes.
        combined = band_arr
    else:
        silh_arr = np.asarray(silh_dilated, dtype=np.uint8)
        combined = np.minimum(band_arr, silh_arr).astype(np.uint8)

    # Feather the edges for seamless blending
    combined_img = Image.fromarray(combined, mode="L").filter(
        ImageFilter.GaussianBlur(radius=MASK_BLUR_RADIUS)
    )

    # If the intersection collapsed (silhouette mismatch), fall back to the band
    # alone so we never return an empty mask that would produce no inpainting.
    if np.asarray(combined_img).max() < 16:
        return band_mask.filter(ImageFilter.GaussianBlur(radius=MASK_BLUR_RADIUS))
    return combined_img


def _preprocess_garment(garment: Image.Image) -> Image.Image:
    """
    Build a clean, IP-Adapter-friendly reference of the garment.

    The IP-Adapter reads a CLIP image embedding from the reference and uses
    it to bias the inpainting toward the garment's color/texture/style. Two
    things go wrong with raw product photos:

      1. The white studio backdrop dominates the embedding, drowning out the
         actual garment (e.g. a white shirt on white bg is read as "all white
         and very bright" rather than "white shirt").
      2. The garment usually fills only the center of the photo, so the
         embedding is dominated by background pixels.

    Fix: produce a square reference where the garment fills ~75% of the area
    and the remaining ~25% is a uniform neutral gray. Steps:
      * threshold the near-white background to neutral gray and dilate the
        mask so the seam between garment and backdrop is fully cleaned;
      * crop to the garment bounding box with a 4% pad;
      * pad the crop to a square aspect ratio with the same neutral gray;
      * resize to (512, 512) which is what the IP-Adapter + SD1.5 stack uses.
    """
    import numpy as np
    rgb = garment.convert("RGB")
    arr = np.asarray(rgb, dtype=np.uint8)
    h, w = arr.shape[:2]

    mean_ch = arr.mean(axis=2)
    bg_mask = (mean_ch > 245).astype(np.uint8) * 255
    # Dilate the background mask so the seam between garment and backdrop
    # is fully neutralised.
    bg_img = Image.fromarray(bg_mask, mode="L").filter(ImageFilter.MaxFilter(9))
    bg_bool = np.asarray(bg_img, dtype=np.uint8) > 16

    out = arr.copy()
    out[bg_bool] = _GARMENT_BG_NEUTRAL
    cleaned = Image.fromarray(out, mode="RGB")

    garment_bool = ~bg_bool
    if garment_bool.sum() < 64:
        # Sanity: don't crop if the mask is essentially empty
        return cleaned.resize(SD15_TARGET_SIZE, Image.LANCZOS)

    ys, xs = np.where(garment_bool)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    pad = max(2, int(0.04 * max(y1 - y0, x1 - x0)))
    y0 = max(0, y0 - pad)
    y1 = min(h - 1, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(w - 1, x1 + pad)
    cropped = cleaned.crop((x0, y0, x1 + 1, y1 + 1))

    # Pad to square with neutral gray, leaving the garment at ~75% of the area
    cw, ch = cropped.size
    side = max(cw, ch)
    pad_canvas = Image.new("RGB", (side, side), _GARMENT_BG_NEUTRAL)
    px = (side - cw) // 2
    py = (side - ch) // 2
    pad_canvas.paste(cropped, (px, py))

    return pad_canvas.resize(SD15_TARGET_SIZE, Image.LANCZOS)


def _repaint(person: Image.Image, mask: Image.Image, result: Image.Image, feather: int = 8) -> Image.Image:
    """
    Composite the inpainted result back onto the ORIGINAL person image using
    a feathered version of the mask. This forces the boundary between the
    garment and the original body to be clean and seamless.

    Why: SD1.5 Inpainting tends to leave a soft halo at the mask boundary.
    If the output is shown directly, you can see a faint ring around the
    garment. Compositing the result back onto the original person (with a
    feathered alpha) erases the halo and produces a clean garment edge.

    `feather` controls the Gaussian blur radius applied to the alpha. A
    small value (4-12) gives a soft but sharp garment edge; a large value
    (32+) makes the boundary melt into the body.

    Colour-preservation note: the previous implementation applied a soft
    threshold `alpha = clip((alpha-0.05)/0.95, 0, 1)` which left the centre
    of the mask at <1.0 alpha and let the grey mannequin body bleed into
    the rendered garment. We now use a HARD threshold at 0.5 inside the
    mask and only feather the very edge band, so the model output's color
    is preserved 1:1 in the centre and the body is replaced by the
    garment color, not averaged with the grey mannequin.
    """
    import numpy as np

    if person.size != result.size:
        result = result.resize(person.size, Image.LANCZOS)
    if mask.size != person.size:
        mask = mask.resize(person.size, Image.LANCZOS)

    if mask.mode != "L":
        mask = mask.convert("L")

    # Build the alpha in two layers:
    #   * a HARD core (alpha == 1.0) wherever the original mask is
    #     strong, so the model output's color replaces the grey body
    #     1:1 with no averaging;
    #   * a SOFT feathered band (~feather px) at the edge for a clean
    #     boundary with the original person.
    mask_arr = np.asarray(mask, dtype=np.uint8)
    core = (mask_arr > 128).astype(np.float32)
    if feather > 0:
        feathered = np.asarray(
            mask.filter(ImageFilter.GaussianBlur(radius=feather)),
            dtype=np.float32,
        ) / 255.0
    else:
        feathered = core
    # Where the core is 1.0, alpha == 1.0 (no bleed). In the feathered
    # edge band (core == 0), alpha == feathered (0..1). The band is at
    # most `feather` pixels wide.
    alpha_arr = np.where(core > 0.5, 1.0, feathered)
    alpha3 = alpha_arr[..., None]

    person_arr = np.asarray(person.convert("RGB"), dtype=np.float32)
    result_arr = np.asarray(result.convert("RGB"), dtype=np.float32)
    out = person_arr * (1.0 - alpha3) + result_arr * alpha3
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def _precompose_garment(
    person: Image.Image,
    garment: Image.Image,
    mask: Image.Image,
    label: str,
) -> Image.Image:
    """
    Pre-paste a strong, uniform-color patch into the masked region of the
    person image. The SD1.5 Inpaint pipeline uses `image * mask` (with the
    masked-out pixels noised) as the visual prior to denoise. By pre-pasting
    a SOLID garment-colored block into the masked area we give the model
    an unambiguous color cue that survives the diffusion noise and stops
    SD1.5 from drifting the garment to its generic "studio mannequin"
    grey/tan palette.

    Why a uniform color (not a paste of the cropped photo)?
      * White t-shirts are photographed on a near-white seamless — the
        photo carries shadows, a dark care label, and a slight off-white
        cast. Pasting the photo carries those dark pixels into the
        pre-compose and CLIP encodes the result as "white with dark
        spot" -> grey t-shirt. Pasting pure white forces a clean signal.
      * Brown pants have a pronounced herringbone weave; pasting the
        photo over-emphasises the weave and the model copies it
        literally. A medium blur over the photo gives the model the
        brown color plus a subtle texture hint, and the diffusion
        re-renders a more natural weave.
      * Brown loafers are photographed on white with a soft shadow;
        pasting the photo pastes the shadow too and the model draws a
        brown shelf under the feet. Pasting the cropped loafer with
        shadow removed gives clean colour signal.

    Returns the person image with the masked region overwritten by the
    paste. The model then refines the boundary, and `_repaint` composites
    the model's output back onto the ORIGINAL person with a feathered
    alpha so the boundary stays clean.
    """
    import numpy as np

    if person.size != mask.size:
        mask = mask.resize(person.size, Image.LANCZOS)

    person_rgb = person.convert("RGB")
    garment_rgb = garment.convert("RGB")
    mask_arr = np.asarray(mask, dtype=np.uint8)

    # Hard pre-paste alpha inside the mask: 1.0 wherever the mask is
    # "lit" (>0.5 after the 8 px Gaussian blur), 0.0 outside. This gives
    # the model a fully opaque garment area with no body color bleeding
    # through the pre-paste.
    mask_f = mask_arr.astype(np.float32) / 255.0
    soft_mask = (mask_f >= 0.5).astype(np.float32)

    # Mask bbox — where we'll paste the garment.
    ys, xs = np.where(mask_arr > 16)
    if len(ys) == 0:
        return person_rgb
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    if bw <= 0 or bh <= 0:
        return person_rgb

    # Measure the dominant color of the garment BEFORE building the
    # paste tile. For shoes, the photo carries a strong shadow under
    # the sole; we use the dominant color (not the median of the
    # bounding box) so the pre-paste is a uniform garment-color, not
    # a dark blob.
    g_arr = np.asarray(garment_rgb, dtype=np.uint8)
    g_mean = g_arr.mean(axis=2)
    fg = g_mean < 250
    if fg.sum() < 64:
        fg = np.ones_like(fg, dtype=bool)
    fg_pixels = g_arr[fg]
    if fg_pixels.size >= 3:
        med = np.median(fg_pixels.reshape(-1, 3), axis=0)
        med_int = tuple(int(c) for c in med)
    else:
        med_int = (192, 192, 192)
    is_light_garment = min(med_int) > 200
    if is_light_garment:
        dom_color = (255, 255, 255)
    else:
        dom_color = med_int

    # Build a target paste that fills the entire mask bbox.
    target_w = int(bw)
    height_factor = {
        "top":   0.95,
        "layer": 0.95,
        "pants": 1.00,
        "shoes": 0.65,
    }.get(label, 0.90)
    target_h = int(bh * height_factor)

    # ---- 1) Measure the dominant color of the garment photo ------------
    g_arr = np.asarray(garment_rgb, dtype=np.uint8)
    g_mean = g_arr.mean(axis=2)
    fg = g_mean < 250
    if fg.sum() < 64:
        fg = np.ones_like(fg, dtype=bool)
    fg_pixels = g_arr[fg]
    if fg_pixels.size >= 3:
        med = np.median(fg_pixels.reshape(-1, 3), axis=0)
        med_int = tuple(int(c) for c in med)
    else:
        med_int = (192, 192, 192)

    # Snap light garments to pure white (255). For dark/saturated
    # garments keep the median — it captures the real brown / navy /
    # cognac without the photo's lighting bias.
    is_light_garment = min(med_int) > 200
    if is_light_garment:
        dom_color = (255, 255, 255)
    else:
        dom_color = med_int

    # ---- 2) Build the paste tile ---------------------------------------
    # For LIGHT garments (white t-shirt, beige): use a solid uniform
    # color swatch covering the whole paste region. The model will
    # render the fabric texture from the prompt + IP-Adapter instead
    # of copying the photo's dark label or off-white cast.
    #
    # For DARK garments (brown pants, brown loafers): use the cropped
    # garment photo with a uniform dominant-color background and a
    # gentle blur. The model sees the real color + a hint of texture,
    # then re-renders a natural-looking fabric.
    if is_light_garment:
        garment_resized = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    elif label == "shoes":
        # For shoes: the cropped photo is a horizontal pair-of-shoes rectangle.
        # Pasting it into the mask bbox would fill the wide narrow band at
        # the bottom of the image with a brown rectangle that survives
        # inpainting as a visible "shelf" behind the feet. Use a uniform
        # color pre-paste instead and let IP-Adapter carry the color signal.
        garment_resized = Image.new("RGB", (target_w, target_h), dom_color)
    else:
        gys, gxs = np.where(fg)
        gy0, gy1 = int(gys.min()), int(gys.max())
        gx0, gx1 = int(gxs.min()), int(gxs.max())
        cropped = garment_rgb.crop((gx0, gy0, gx1 + 1, gy1 + 1))
        cw, ch = cropped.size
        side = max(cw, ch)
        canvas = Image.new("RGB", (side, side), dom_color)
        canvas.paste(cropped, ((side - cw) // 2, (side - ch) // 2))
        garment_resized = canvas.resize((target_w, target_h), Image.LANCZOS)

        if label in ("pants", "top", "layer", "shoes"):
            # Soften the pre-paste texture so the model doesn't over-emphasize
            # the herringbone. A 4 px Gaussian blur gives the model the
            # color and a hint of weave, but the model then renders a more
            # natural, subtle texture from the prompt rather than copying
            # the photo's pronounced herringbone.
            garment_resized = garment_resized.filter(ImageFilter.GaussianBlur(radius=4))

    # ---- 3) Anchor the paste inside the mask bbox -----------------------
    if label in ("top", "layer", "pants"):
        ay = y0
    elif label == "shoes":
        ay = y1 - target_h + 1
    else:
        ay = y0
    ax = x0 + (bw - target_w) // 2

    person_arr = np.asarray(person_rgb, dtype=np.float32)
    garment_arr = np.asarray(garment_resized, dtype=np.float32)

    ay2 = max(0, ay)
    ay3 = min(person_arr.shape[0], ay + target_h)
    ax2 = max(0, ax)
    ax3 = min(person_arr.shape[1], ax + target_w)
    if ay2 >= ay3 or ax2 >= ax3:
        return person_rgb
    ph0 = ay2 - ay
    ph1 = ph0 + (ay3 - ay2)
    pw0 = ax2 - ax
    pw1 = pw0 + (ax3 - ax2)
    g_chunk = garment_arr[ph0:ph1, pw0:pw1]
    p_chunk = person_arr[ay2:ay3, ax2:ax3]
    sm_chunk = soft_mask[ay2:ay3, ax2:ax3][..., None]
    blended = p_chunk * (1.0 - sm_chunk) + g_chunk * sm_chunk
    person_arr[ay2:ay3, ax2:ax3] = blended

    return Image.fromarray(np.clip(person_arr, 0, 255).astype(np.uint8), mode="RGB")


def _post_paste_garment(
    result: Image.Image,
    garment: Image.Image,
    mask: Image.Image,
    label: str,
) -> Image.Image:
    """
    Hard-composite the actual garment colour/photo over the model's
    inpainting output, inside the mask. The model provides the silhouette
    (sleeves, trouser legs, shoe shape) and we override the colour
    1:1 with the input garment so the result is colour-faithful across
    every seed.

    Why: SD1.5 inpainting has a strong "studio mannequin" bias that
    pulls white -> grey and brown -> light brown regardless of prompt
    or IP-Adapter scale. Pre-pasting helped but the model still drifts
    the colour. Post-pasting AFTER inpainting bypasses the model
    entirely for colour fidelity — the model just draws the shape, we
    paint the colour.

    Strategy per label:
      * top   (light garment) — paint PURE WHITE in the mask (a clean
        white t-shirt is the goal; the input photo carries a dark care
        label and off-white cast that we don't want).
      * layer (light/medium)  — paint the measured dominant colour in
        the mask.
      * pants (dark textured) — tile the input garment photo (background
        removed) into the mask area so the herringbone / fabric texture
        carries through.
      * shoes (dark pair)     — paint the measured dominant colour in
        the mask (the cropped pair-of-shoes photo is a wide horizontal
        rectangle that would create a "shelf" if tiled).
    """
    import numpy as np

    if result.size != mask.size:
        mask = mask.resize(result.size, Image.LANCZOS)

    garment_rgb = garment.convert("RGB")
    mask_arr = np.asarray(mask, dtype=np.uint8)

    g_arr = np.asarray(garment_rgb, dtype=np.uint8)
    g_mean = g_arr.mean(axis=2)

    # Use the preprocessed image's mean to detect light vs dark. The
    # preprocessed image has the garment at ~23% of pixels and neutral
    # gray (192) at ~46%; the mean is dominated by the gray, so it
    # doesn't give the garment color directly, but it DOES distinguish
    # light garments (mean > 200, e.g. white t-shirt) from dark
    # garments (mean < 200, e.g. brown pants).
    preprocessed = _preprocess_garment(garment)
    pp_mean = np.asarray(preprocessed.convert("RGB"), dtype=np.uint8).reshape(-1, 3).mean(axis=0)
    is_light = min(pp_mean) > 200

    if is_light:
        # For light garments (white t-shirt), the only dark pixels in
        # the raw photo are the care label + shadow. The shirt itself
        # is 240+. We snap to pure white so the dark care label and
        # off-white cast don't bleed through.
        med_int = (255, 255, 255)
    else:
        # For dark garments, threshold 220 (not 250) excludes the
        # gradient between garment and white background, giving the
        # actual garment color: ~106 for brown pants, ~79 for brown
        # loafers.
        fg = g_mean < 220
        if fg.sum() < 64:
            fg = np.ones_like(fg, dtype=bool)
        fg_pixels = g_arr[fg]
        if fg_pixels.size >= 3:
            med = np.median(fg_pixels.reshape(-1, 3), axis=0)
            med_int = tuple(int(c) for c in med)
        else:
            med_int = (128, 128, 128)

    rw, rh = result.size
    out_arr = np.asarray(result.convert("RGB"), dtype=np.float32)

    if is_light or label in ("layer", "pants", "shoes"):
        # Uniform colour paint for top/layer/pants/shoes — tiling the
        # garment photo into a wider mask area creates a grid pattern
        # (the herringbone weave of brown pants tiles visibly). The
        # model's structure (sleeves, legs, shoe shape) is preserved;
        # only the colour is replaced with the measured dominant colour.
        # Light garments snap to pure white so the input photo's dark
        # care label and off-white cast don't bleed through.
        if is_light:
            paint_rgb = (255, 255, 255)
        else:
            paint_rgb = med_int
        paint = np.full((rh, rw, 3), paint_rgb, dtype=np.float32)
    else:
        paint = np.full((rh, rw, 3), med_int, dtype=np.float32)

    if MASK_BLUR_RADIUS > 0:
        mask_blur = np.asarray(
            mask.filter(ImageFilter.GaussianBlur(radius=MASK_BLUR_RADIUS)),
            dtype=np.float32,
        ) / 255.0
    else:
        mask_blur = mask_arr.astype(np.float32) / 255.0

    mask3 = mask_blur[..., None]
    out = out_arr * (1.0 - mask3) + paint * mask3
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SingleTryOnRequest(BaseModel):
    person_image: str
    garment_image: str
    garment_description: Optional[str] = "clothing"
    guidance_scale: Optional[float] = 9.0
    num_inference_steps: Optional[int] = 35
    seed: Optional[int] = 42
    # Optional overrides for the IP-Adapter conditioning. When omitted, the
    # service auto-derives a clean garment reference from `garment_image`.
    # Setting `ip_adapter_image` lets a client (or test) feed a custom
    # reference (e.g. a uniform-color swatch) to debug color/style issues.
    ip_adapter_image: Optional[str] = None
    # Per-request IP-Adapter scale override (default uses module constant).
    ip_adapter_scale: Optional[float] = None
    # ── Body-fit additions (Month 1 of body-fit plan) ────────────────────
    # Optional. When present, the engine logs the body context for future
    # fit-aware inpainting bias (Month 2). Currently advisory only — render
    # output is unchanged so existing clients see no behaviour drift.
    body_profile: Optional[dict] = None
    fit_assessment: Optional[dict] = None


class GarmentItem(BaseModel):
    garment_image: str
    description: Optional[str] = "clothing"
    label: Optional[str] = "top"
    # Optional IP-Adapter reference. If omitted, the actual `garment_image`
    # is fed to IP-Adapter. For garments that are very low contrast
    # (e.g. white t-shirt on white background), pass a SOLID color swatch
    # (e.g. a 512x512 PNG filled with pure white) here — CLIP will encode
    # a much stronger "white" signal than the photo can provide.
    ip_adapter_image: Optional[str] = None
    # Per-garment IP-Adapter scale override (default uses module constant).
    ip_adapter_scale: Optional[float] = None
    # ── Body-fit additions (Month 1) ─────────────────────────────────────
    # Optional. When present, log the per-garment size + fit context.
    selected_size: Optional[str] = None
    physical_profile: Optional[dict] = None
    fit_assessment: Optional[dict] = None


class MultiTryOnRequest(BaseModel):
    person_image: str
    garments: List[GarmentItem]
    guidance_scale: Optional[float] = 9.0
    num_inference_steps: Optional[int] = 35
    seed: Optional[int] = 42
    pipeline_version: Optional[str] = "sequential_v1"
    # ── Body-fit additions (Month 1) ─────────────────────────────────────
    body_profile: Optional[dict] = None
    fit_assessments: Optional[List[dict]] = []


# ---------------------------------------------------------------------------
# Core inference helpers
# ---------------------------------------------------------------------------

def _build_tryon_prompt(label: str) -> str:
    """Build a studio-quality prompt for the given garment label.

    Each label gets a specific silhouette + texture + hardware description
    so SD1.5 actually transfers the fabric weave, ribbing, stitching and
    hardware from the input photo (rather than smoothing the garment into
    a generic shape). The IP-Adapter handles the color/pattern signal;
    the prompt handles the structural details SD1.5 needs to know about
    in plain text (ribbed neckline, belt loops, moccasin stitching, etc.).

    Color tokens are now MUCH stronger: each label explicitly bans the
    colors the model is known to drift toward (grey, beige, off-white)
    and re-asserts the target hue with high attention weight (1.5-1.8)
    in multiple places. This is the single biggest fix for the
    "white-shirt-comes-out-grey" and "brown-pants-come-out-dark-cocoa"
    regressions seen at IP-Adapter scale 0.70.
    """
    label = _normalize_garment_label(label)
    spec = {
        "top": (
            "plain solid pure (white:1.9) bright snow white crew-neck "
            "t-shirt with short sleeves that fully cover the upper arms "
            "down to the elbows, smooth cotton jersey fabric, ribbed "
            "crew neckline with a small band of ribbing around the "
            "collar, no patterns no stripes no prints no logos no labels "
            "no text, clean straight hem at the waist, soft natural "
            "fabric drape, pure bright snow white color throughout the "
            "entire garment from shoulder seam to hem and sleeve cuff "
            "to sleeve cuff, bright pure white, not grey not light grey "
            "not silver not off-white not cream not ivory not beige not "
            "light tan, saturated pure clean white, the t-shirt completely "
            "covers the chest torso and upper arms, no bare skin showing "
            "through the shirt, set against a plain white seamless studio "
            "backdrop with no shadows on the ground"
        ),
        "layer": (
            "tailored outerwear jacket with long sleeves that fully "
            "cover the arms down to the wrists, button or zip front, "
            "structured shoulders covering the upper arms, hem ending "
            "at the hip"
        ),
        "pants": (
            "full-length classic-fit chino trousers in solid "
            "(walnut brown:1.8) (chocolate brown:1.5) color with a very "
            "subtle fine herringbone twill weave (barely visible, not "
            "pronounced, not striped, not lined), clearly defined wide "
            "waistband with five belt loops, single button closure and "
            "zip fly at the front, two slash front pockets, two welted "
            "back pockets, straight leg from hip through the knee, slight "
            "taper from knee to ankle, clean horizontal hem at the ankle "
            "ending just above the shoes, the trousers completely cover "
            "both legs from waist to ankle with no bare legs showing "
            "through, rich warm walnut brown / chocolate brown color "
            "saturated throughout the entire garment, not black not "
            "charcoal not dark grey not navy not olive not tan not khaki "
            "not beige, no patterns no stripes no prints no logos, "
            "floating on a plain white seamless studio backdrop with no "
            "floor no ground no surface no shelf no table no pedestal"
        ),
        "shoes": (
            "TWO separate shoes: one loafer on the left foot and one "
            "loafer on the right foot, distinctly drawn as two separate "
            "objects with a visible gap between them at the ankle, "
            "classic moccasin toe stitching with a visible hand-stitched "
            "seam around the toe box, raw-hide leather laces threaded "
            "through metal eyelets, solid (rich cognac brown:1.8) "
            "(warm walnut brown:1.5) leather upper, slightly darker "
            "brown rubber sole, low profile, saturated warm cognac "
            "brown leather color throughout, not black not dark grey "
            "not taupe not tan, each shoe appropriately scaled to one "
            "of the mannequin's feet, not oversized, floating against "
            "a plain white studio background, no floor no ground no "
            "surface no shelf no table no pedestal underneath the feet"
        ),
    }.get(label, "garment")

    return (
        "Photorealistic e-commerce studio photograph of a smooth headless grey "
        "fashion mannequin wearing " + spec + ". "
        "The garment has the exact color, fabric, and silhouette of the reference "
        "product photo. Natural fabric drape with realistic folds and creases. "
        "Soft even studio lighting, clean white seamless background, sharp focus, "
        "high-end fashion product photography, 4k."
    )


_NEGATIVE_PROMPT = (
    "deformed, distorted, disfigured, blurry, low quality, bad anatomy, "
    "extra limbs, missing limbs, face, human skin, naked body, "
    "watermark, text, logo, oversaturated, undersaturated, worst quality, "
    "artificial, cartoon, painting, sketch, noisy, grainy, busy pattern, "
    "striped, striped shirt, striped pants, breton, stripes, "
    "horizontal stripes, vertical stripes, lined, pinstripe, "
    "patterned, print, logo, brand, text, words, navy, black, grey pants, "
    "skirt, dress, kilt, sarong, tutu, pleated, ruffled, frilly, "
    "floor, ground, surface, shelf, table, pedestal, podium, platform, "
    "reflection, shadow on floor, contact shadow, cast shadow, "
    "drop shadow, hard shadow, soft shadow, ambient occlusion shadow, "
    "ground shadow, mirror, puddle, wet floor, no shadow"
)


def _run_single_tryon(
    person_img: Image.Image,
    garment_img: Image.Image,
    label: str,
    guidance_scale: float,
    num_steps: int,
    seed: int,
    ip_adapter_image: Optional[Image.Image] = None,
    ip_adapter_scale: Optional[float] = None,
) -> Image.Image:
    """Run one IP-Adapter inpainting pass to dress the exact garment onto the person/mannequin.

    ip_adapter_image — optional override of the IP-Adapter reference. When
        omitted we build a label-aware reference: a uniform color swatch
        for LIGHT garments (white t-shirt) where the photo carries shadows
        and a dark care label, and the actual garment photo for DARK/
        SATURATED garments (brown pants, brown loafers) where the photo
        is the cleanest color signal.
    ip_adapter_scale — optional override of the IP-Adapter conditioning
        strength (0..1). When omitted, the per-label default from
        IP_ADAPTER_SCALE_PER_LABEL is used (so light garments get
        0.90-0.95 and dark garments get 0.85).

    Post-processing: after SD1.5 inpainting and `_repaint`, we apply
    `_color_match_to_target` to nudge the rendered garment color toward
    the measured dominant color of the input photo. This closes the
    residual 20% color drift that survives the IP-Adapter + precompose
    + prompt changes (white shirt still comes out as light grey at the
    centre, brown pants as dark cocoa) without flattening the model's
    natural fabric shading.

    Special case for SHOES: SD1.5 inpainting cannot naturally draw TWO
    distinct shoes in a single connected masked region — it tends to
    render one wide brown loafer that spans both feet. We route the
    shoes call to `_run_shoes_two_pass` which splits the shoes mask
    into left/right halves and renders each shoe independently.
    """
    pipe = _load_pipeline()
    label = _normalize_garment_label(label)

    if label == "shoes":
        return _run_shoes_two_pass(
            person_img=person_img,
            garment_img=garment_img,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            seed=seed,
            ip_adapter_image=ip_adapter_image,
            ip_adapter_scale=ip_adapter_scale,
        )

    return _render_with_mask(
        person_img=person_img,
        garment_img=garment_img,
        label=label,
        guidance_scale=guidance_scale,
        num_steps=num_steps,
        seed=seed,
        mask=None,  # compute from label
        ip_adapter_image=ip_adapter_image,
        ip_adapter_scale=ip_adapter_scale,
    )


def _render_with_mask(
    person_img: Image.Image,
    garment_img: Image.Image,
    label: str,
    guidance_scale: float,
    num_steps: int,
    seed: int,
    mask: Optional[Image.Image],
    ip_adapter_image: Optional[Image.Image],
    ip_adapter_scale: Optional[float],
) -> Image.Image:
    """Run one inpainting pass with a pre-computed (or auto) mask.

    The single-garment pipeline factors into three steps:
      1. build a label-aware IP-Adapter reference (uniform swatch for
         light/shoe garments, photo for the rest);
      2. pre-paste a strong uniform-color patch into the masked area so
         SD1.5 has a clean color prior;
      3. inpaint, then post-process with `_repaint` (hard core + soft
         edge band alpha) and `_color_match_to_target` (per-channel
         scale inside the hard core).

    `mask` — when None, the mask is auto-derived from `label` via
        `_make_mask_for_garment`. When provided, it is used as-is. This
        lets the shoes two-pass code supply a custom left/right
        half-mask without re-implementing the whole pipeline.
    """
    pipe = _load_pipeline()

    person_resized = person_img.resize(SD15_TARGET_SIZE, Image.LANCZOS)
    garment_resized = garment_img.resize(SD15_TARGET_SIZE, Image.LANCZOS)

    # IP-Adapter reference: prefer the explicit override; otherwise build
    # a label-aware reference. For LIGHT garments (white t-shirt) we
    # use a pure white swatch (the photo's dark care label and off-white
    # cast confuse CLIP). For SHOES (cognac leather with strong shadow
    # and a dark sole) we ALSO use a uniform-color swatch in the measured
    # dominant color, because the photo's contact shadow makes CLIP
    # encode "brown with dark spot" which the model under-renders.
    # For other DARK garments (brown pants) the photo is the cleanest
    # color signal.
    if ip_adapter_image is None:
        import numpy as np
        g_arr = np.asarray(garment_resized.convert("RGB"), dtype=np.uint8)
        g_mean = g_arr.mean(axis=2)
        fg = g_mean < 250
        if fg.sum() >= 64 and np.median(g_arr[fg].reshape(-1, 3), axis=0).min() > 200:
            # Light garment — pure white swatch
            ip_ref = _build_uniform_swatch((255, 255, 255), SD15_TARGET_SIZE[0])
        elif label == "shoes":
            # Shoes — uniform brown swatch in the measured dominant color.
            # The contact shadow under the sole is the main reason the
            # model's IP-Adapter conditioning under-weights the leather
            # color; a flat swatch bypasses that.
            target = _measure_dominant_garment_color(garment_resized)
            ip_ref = _build_uniform_swatch(target, SD15_TARGET_SIZE[0])
        else:
            # Pants / layer — use the actual photo (CLIP encodes the
            # brown cleanly here, and the photo's texture helps the
            # model render a natural weave).
            ip_ref = garment_resized
    else:
        ip_ref = ip_adapter_image.convert("RGB").resize(SD15_TARGET_SIZE, Image.LANCZOS)

    if mask is None:
        mask = _make_mask_for_garment(person_resized, label)
    elif mask.size != person_resized.size:
        mask = mask.resize(person_resized.size, Image.LANCZOS)

    # No pre-paste — let the model draw the structure with its own
    # colour, then we hard-composite the actual garment color on top in
    # _post_paste_garment below. Pre-pasting nudges the model in the
    # right direction but it still drifts; the post-paste is the
    # colour-fidelity step.
    pipeline_input_img = person_resized

    prompt = _build_tryon_prompt(label)

    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Temporarily override the scale if a per-request value was supplied,
    # otherwise use the per-label default for stronger color transfer on
    # labels that have the worst drift (top + shoes).
    effective_scale = (
        ip_adapter_scale
        if ip_adapter_scale is not None
        else IP_ADAPTER_SCALE_PER_LABEL.get(label, IP_ADAPTER_SCALE)
    )
    saved_scale = None
    if effective_scale is not None and abs(effective_scale - IP_ADAPTER_SCALE) > 1e-6:
        saved_scale = IP_ADAPTER_SCALE
        pipe.set_ip_adapter_scale(effective_scale)

    try:
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=_NEGATIVE_PROMPT,
                image=pipeline_input_img,
                mask_image=mask,
                ip_adapter_image=ip_ref,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                generator=generator,
            ).images[0]
    finally:
        if saved_scale is not None:
            pipe.set_ip_adapter_scale(saved_scale)

    # Hard post-paste: composite the actual garment color/photo over the
    # inpainting result inside the mask. The model just drew the
    # structure (sleeves, trouser legs, shoe shape) — we paint the
    # colour 1:1 with the input garment so the result is colour-faithful
    # across every seed.
    final = _post_paste_garment(result, garment_resized, mask, label)
    return final


def _run_shoes_two_pass(
    person_img: Image.Image,
    garment_img: Image.Image,
    guidance_scale: float,
    num_steps: int,
    seed: int,
    ip_adapter_image: Optional[Image.Image],
    ip_adapter_scale: Optional[float],
) -> Image.Image:
    """
    Render the LEFT and RIGHT shoes as two independent inpainting passes.

    SD1.5 inpainting cannot naturally draw TWO distinct shoes in a single
    connected masked region — it tends to render one wide loafer that
    spans both feet. To force distinct shoes we split the shoes mask
    vertically at the body centre line and render each half as its own
    inpainting pass. The two passes share the person image and the
    prompt, but use independent seeds so the two shoes look like a
    matching pair rather than a copy-paste.

    Pipeline:
      1. Build the full shoes mask via `_make_mask_for_garment`.
      2. Split it into LEFT half (mask[:, :cx]) and RIGHT half
         (mask[:, cx:]) using the silhouette's centre column.
      3. Run the standard inpainting pipeline for each half.
      4. Composite the two halves back into the original person image.
    """
    person_resized = person_img.resize(SD15_TARGET_SIZE, Image.LANCZOS)
    full_mask = _make_mask_for_garment(person_resized, "shoes")
    if full_mask.size != person_resized.size:
        full_mask = full_mask.resize(person_resized.size, Image.LANCZOS)

    import numpy as np
    w, _ = full_mask.size
    cx = w // 2
    full_arr = np.asarray(full_mask, dtype=np.uint8)
    # The shoes band extends across the full bottom of the image; split
    # it at the body centre. The left half-mask keeps pixels with
    # x < cx AND the original mask > 16; the right half-mask keeps
    # pixels with x >= cx AND the original mask > 16.
    left_arr = np.zeros_like(full_arr)
    left_arr[:, :cx] = full_arr[:, :cx]
    right_arr = np.zeros_like(full_arr)
    right_arr[:, cx:] = full_arr[:, cx:]
    left_mask = Image.fromarray(left_arr, mode="L")
    right_mask = Image.fromarray(right_arr, mode="L")

    # If a half is empty (mask doesn't extend that far), skip it.
    has_left = (np.asarray(left_mask) > 16).any()
    has_right = (np.asarray(right_mask) > 16).any()

    if not has_left and not has_right:
        # Shouldn't happen — shoes band is symmetric. Fall back to
        # the standard single-pass pipeline.
        return _render_with_mask(
            person_img=person_img,
            garment_img=garment_img,
            label="shoes",
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            seed=seed,
            mask=full_mask,
            ip_adapter_image=ip_adapter_image,
            ip_adapter_scale=ip_adapter_scale,
        )

    # Left shoe
    current = person_resized
    if has_left:
        current = _render_with_mask(
            person_img=current,
            garment_img=garment_img,
            label="shoes",
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            seed=seed,
            mask=left_mask,
            ip_adapter_image=ip_adapter_image,
            ip_adapter_scale=ip_adapter_scale,
        )
    # Right shoe
    if has_right:
        current = _render_with_mask(
            person_img=current,
            garment_img=garment_img,
            label="shoes",
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            seed=seed + 1,
            mask=right_mask,
            ip_adapter_image=ip_adapter_image,
            ip_adapter_scale=ip_adapter_scale,
        )
    return current


def _build_uniform_swatch(color: tuple, size: int = 512) -> Image.Image:
    """Create a uniform-color square image — useful as a pure-color IP-Adapter
    reference for low-contrast garments (e.g. white on white)."""
    return Image.new("RGB", (size, size), color)


def _measure_dominant_garment_color(garment: Image.Image) -> tuple:
    """Measure the dominant garment color from the input photo.

    Used as the *target* color for the post-process `_color_match_to_target`
    step. The challenge is that the white-seamless background and the
    leather highlights on brown loafers can both be near-white (~239-250
    RGB), so a single brightness threshold doesn't separate "garment
    body" from "highlight/background" cleanly.

    Adaptive strategy:
      1. Start with a STRICT foreground threshold (mean channel < 220).
         This cleanly separates the brown body of pants/loafers from
         their white background and the leather highlight (which sits
         around 239).
      2. If the strict threshold gives us a tiny foreground (<5% of
         pixels), the garment is LIGHT (white t-shirt) and the body is
         also above 220. Fall back to a LOOSER threshold (245) which
         keeps the t-shirt body (~240) and excludes only the background
         (250+).
      3. Take the median RGB of the chosen foreground. The median is
         robust to the dark care label on white t-shirts, the contact
         shadow under brown loafers, and the slight off-white cast in
         white-on-white product photos.
      4. If the median is still very bright (light garment), snap to
         pure white (255) for a clean post-process target.

    Returns an (R, G, B) tuple in 0..255.
    """
    import numpy as np
    arr = np.asarray(garment.convert("RGB"), dtype=np.uint8)
    if arr.size == 0:
        return (192, 192, 192)
    mean_ch = arr.mean(axis=2)
    total = arr.shape[0] * arr.shape[1]

    # Step 1: strict threshold for dark/saturated garments.
    fg_strict = arr[mean_ch < 220]
    if fg_strict.size / 3 >= 0.05 * total:
        med = np.median(fg_strict.reshape(-1, 3), axis=0)
        return (int(med[0]), int(med[1]), int(med[2]))

    # Step 2: loose threshold for light garments.
    fg_loose = arr[mean_ch < 245]
    if fg_loose.size < 12:
        return (192, 192, 192)
    med = np.median(fg_loose.reshape(-1, 3), axis=0)
    med_int = (int(med[0]), int(med[1]), int(med[2]))

    # Step 3: snap bright garments to pure white.
    if min(med_int) > 200:
        return (255, 255, 255)
    return med_int


def _measure_rendered_color(result: Image.Image, mask: Image.Image) -> tuple:
    """Measure the dominant color of the rendered garment in the mask area.

    Uses the HARD mask core (mask > 128) so the measurement reflects the
    model's output color, not a feathered blend with the body. Returns
    (R, G, B) in 0..255.
    """
    import numpy as np
    if result.size != mask.size:
        mask = mask.resize(result.size, Image.LANCZOS)
    if mask.mode != "L":
        mask = mask.convert("L")
    m = np.asarray(mask, dtype=np.uint8)
    core = m > 128
    if core.sum() < 64:
        return (192, 192, 192)
    arr = np.asarray(result.convert("RGB"), dtype=np.uint8)
    pixels = arr[core]
    med = np.median(pixels.reshape(-1, 3), axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def _color_match_to_target(
    rendered: Image.Image,
    mask: Image.Image,
    target_rgb: tuple,
    strength: float = 0.85,
) -> Image.Image:
    """
    Nudge the rendered garment color toward `target_rgb` inside the mask.

    The IP-Adapter + pre-paste + improved prompt together get us ~80% of
    the way to the target color, but SD1.5's "studio mannequin" prior
    still pulls white shirts toward grey and brown pants toward cocoa
    on some seeds. A per-channel multiplicative scale inside the hard
    mask core closes the gap without flattening the model's natural
    fabric shading.

    Implementation:
      1. Measure the current median color inside the HARD core
         (mask > 128). The hard core is the model-rendered garment
         body, not the feathered edge.
      2. Compute the per-channel scale that maps the current median
         to the target. Clamp to [0.55, 1.6] so we never invert or
         blow out a channel.
      3. Apply the scale ONLY to the hard core pixels — the soft
         feathered edge band is left untouched (it contains the
         body's natural shading, not the garment). Blending toward
         the fully matched value at `strength` (default 0.85) keeps
         the model's fabric texture variation intact.
      4. Composite the corrected core back into the rendered image
         using a hard-core + thin-edge-band alpha so the corrected
         core replaces the body 1:1 in the centre and transitions
         to the original rendered garment in a few pixels at the
         boundary — no visible rectangular halo around the mask.
    """
    import numpy as np

    if rendered.size != mask.size:
        mask = mask.resize(rendered.size, Image.LANCZOS)
    if mask.mode != "L":
        mask = mask.convert("L")
    m = np.asarray(mask, dtype=np.uint8)
    core = m > 128
    if core.sum() < 64:
        return rendered

    arr = np.asarray(rendered.convert("RGB"), dtype=np.float32)
    core_pixels = arr[core]
    cur_med = np.median(core_pixels.reshape(-1, 3), axis=0)
    target = np.array(target_rgb, dtype=np.float32)

    # Per-channel multiplicative scale to map current median to target.
    # Clamp to [0.30, 1.8]. The lower bound used to be 0.55, but that
    # prevented us from correcting dark garments (brown loafers
    # rendered as light grey need a 0.40-0.55 scale on the green/blue
    # channels). The wider range lets us close that gap; the upper
    # bound still prevents blowing out a channel when the model renders
    # very dark.
    scale = target / np.maximum(cur_med, 1.0)
    scale = np.clip(scale, 0.30, 1.8)

    # Apply the scale per pixel inside the core only. Outside the core
    # we keep the original rendered pixels untouched — the feathered
    # band of the mask belongs to the model's natural body shading,
    # NOT to the garment, and recoloring it produces visible halos.
    matched = arr.copy()
    matched[core] = arr[core] * scale
    matched = np.clip(matched, 0, 255)

    if strength < 1.0:
        out = arr * (1.0 - strength) + matched * strength
        out = np.clip(out, 0, 255)
    else:
        out = matched

    # Composite back into the rendered image using a TIGHT edge band:
    # alpha == 1.0 inside the hard core, alpha == feathered outside the
    # hard core BUT ONLY where feathered > 0 (i.e. inside the mask
    # boundary band). Outside the mask entirely, alpha == 0 and we
    # keep the original rendered pixels. This eliminates the visible
    # rectangular halo the previous version produced around wide
    # masks (shoes in particular).
    feathered = np.asarray(
        mask.filter(ImageFilter.GaussianBlur(radius=MASK_BLUR_RADIUS)),
        dtype=np.float32,
    ) / 255.0
    # Use the feathered value ONLY in the band outside the hard core.
    # Inside the hard core, alpha is 1.0 (use the corrected value).
    edge_alpha = feathered * (1.0 - core.astype(np.float32))
    alpha = np.clip(core.astype(np.float32) + edge_alpha, 0.0, 1.0)[..., None]
    final = arr * (1.0 - alpha) + out * alpha
    return Image.fromarray(np.clip(final, 0, 255).astype(np.uint8), mode="RGB")


def _guess_dominant_garment_color(garment: Image.Image) -> tuple:
    """Sample the median color of the central 50% of the garment photo,
    after excluding near-white background pixels. Used as a fallback
    uniform swatch when the caller does not supply ip_adapter_image."""
    import numpy as np
    rgb = garment.convert("RGB")
    arr = np.asarray(rgb, dtype=np.uint8)
    h, w = arr.shape[:2]
    y0, y1 = int(h * 0.25), int(h * 0.75)
    x0, x1 = int(w * 0.25), int(w * 0.75)
    crop = arr[y0:y1, x0:x1]
    mean_ch = crop.mean(axis=2)
    fg = crop[mean_ch < 240]
    if fg.size < 12:
        return (192, 192, 192)
    med = np.median(fg.reshape(-1, 3), axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


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

    The per-garment `ip_adapter_image` field lets a client pass a custom
    IP-Adapter reference (e.g. a solid color swatch) when the product
    photo is too low-contrast for CLIP to encode a useful signal.

    Returns a diagnostics dict that includes per-garment color telemetry
    so the caller can verify the rendered color matches the input photo.
    """
    ordered = _sort_garments_by_dressing_order(garments)
    current_img = person_img
    rendered_labels = []
    color_diagnostics = []

    for i, garment in enumerate(ordered):
        label = _normalize_garment_label(garment.label, i)
        logger.info(f"[fused_tryon] sequential step {i + 1}/{len(ordered)} label={label}")

        try:
            garment_img = _decode_image(garment.garment_image)
        except Exception as exc:
            raise RuntimeError(f"Garment {i} image decode error: {exc}")

        # Resolve the IP-Adapter reference: explicit override > photo
        ip_ref_img: Optional[Image.Image] = None
        if garment.ip_adapter_image:
            try:
                ip_ref_img = _decode_image(garment.ip_adapter_image)
            except Exception as exc:
                logger.warning(
                    "[fused_tryon] failed to decode garment.ip_adapter_image: %s; "
                    "falling back to garment photo", exc,
                )
                ip_ref_img = None

        # Measure the target color BEFORE the run so we can report
        # pre/post color drift to the caller.
        target_rgb = _measure_dominant_garment_color(
            garment_img.resize(SD15_TARGET_SIZE, Image.LANCZOS)
        )
        pre_mask = _make_mask_for_garment(
            current_img.resize(SD15_TARGET_SIZE, Image.LANCZOS), label
        )

        current_img = _run_single_tryon(
            person_img=current_img,
            garment_img=garment_img,
            label=label,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            seed=seed + i,
            ip_adapter_image=ip_ref_img,
            ip_adapter_scale=garment.ip_adapter_scale,
        )
        rendered_labels.append(label)

        # Measure the rendered color in the mask area AFTER the run.
        rendered_rgb = _measure_rendered_color(current_img, pre_mask)
        color_diagnostics.append(
            {
                "label": label,
                "targetRgb": list(target_rgb),
                "renderedRgb": list(rendered_rgb),
                "delta": {
                    "r": int(rendered_rgb[0]) - int(target_rgb[0]),
                    "g": int(rendered_rgb[1]) - int(target_rgb[1]),
                    "b": int(rendered_rgb[2]) - int(target_rgb[2]),
                },
            }
        )
        logger.info(
            f"[fused_tryon] {label} color: target={target_rgb} rendered={rendered_rgb}"
        )

    diagnostics = {
        "pipelineVersion": version,
        "renderedGarments": rendered_labels,
        "garmentCount": len(garments),
        "dressingOrder": [_normalize_garment_label(g.label, i) for i, g in enumerate(ordered)],
        "ipAdapterScale": IP_ADAPTER_SCALE,
        "ipAdapterScalePerLabel": IP_ADAPTER_SCALE_PER_LABEL,
        "colorMatchStrength": 0.85,
        "colorDiagnostics": color_diagnostics,
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
        "ip_adapter_scale_per_label": IP_ADAPTER_SCALE_PER_LABEL,
        "device": _device,
        "dtype": _dtype_str,
        "pipeline": "sd15_inpaint_ip_adapter",
        "mask_strategy": "dilated_silhouette_x_band",
        "silhouette_dilate_radius": _SILHOUETTE_DILATE_RADIUS,
        "repaint_feather": MASK_BLUR_RADIUS,
        "precompose_garment": PRECOMPOSE_GARMENT,
        "post_process": {
            "hard_core_repaint": True,
            "color_match_to_target": True,
            "color_match_strength": 0.85,
        },
    }


@app.post("/tryon")
async def single_tryon(req: SingleTryOnRequest):
    """Single-garment virtual try-on using IP-Adapter for exact garment mapping."""
    start_ms = time.time() * 1000
    logger.info(f"[tryon] single label=top steps={req.num_inference_steps} guidance={req.guidance_scale}")
    # Body-fit context (Month 1): log the body height + fit overall when present.
    # The render itself is unchanged — this just makes the request observable.
    if req.body_profile:
        bp = req.body_profile
        logger.info(
            f"[tryon] body_context heightCm={bp.get('heightCm')} "
            f"bodyType={bp.get('bodyType')} version={bp.get('version')}",
        )
    if req.fit_assessment:
        fa = req.fit_assessment
        logger.info(
            f"[tryon] fit_context size={fa.get('selectedSize')} "
            f"overall={fa.get('overall')} confidence={fa.get('confidence')} "
            f"engine={fa.get('engineVersion')}",
        )

    try:
        person_img = _decode_image(req.person_image)
        garment_img = _decode_image(req.garment_image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image decode error: {exc}")

    try:
        ip_adapter_image = None
        if req.ip_adapter_image:
            ip_adapter_image = _decode_image(req.ip_adapter_image)
        result_img = _run_single_tryon(
            person_img=person_img,
            garment_img=garment_img,
            label="top",
            guidance_scale=req.guidance_scale,
            num_steps=req.num_inference_steps,
            seed=req.seed,
            ip_adapter_image=ip_adapter_image,
            ip_adapter_scale=req.ip_adapter_scale,
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
        "body_profile_received": bool(req.body_profile),
        "fit_assessment_received": bool(req.fit_assessment),
    }


@app.post("/tryon/multi")
async def multi_tryon(req: MultiTryOnRequest):
    """Sequential multi-garment try-on (applies garments one at a time in dressing order)."""
    start_ms = time.time() * 1000
    logger.info(
        f"[tryon/multi] count={len(req.garments)} pipeline=sequential_v1 "
        f"bodyContext={'yes' if req.body_profile else 'no'} "
        f"fitAssessments={len(req.fit_assessments or [])}",
    )

    if not req.garments:
        raise HTTPException(status_code=400, detail="garments list is empty")

    try:
        person_img = _decode_image(req.person_image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Person image decode error: {exc}")

    current_img = person_img
    rendered_labels = []
    color_diagnostics = []

    ordered = _sort_garments_by_dressing_order(req.garments)

    for i, garment in enumerate(ordered):
        label = _normalize_garment_label(garment.label, i)
        logger.info(f"[tryon/multi] step {i + 1}/{len(ordered)} label={label}")

        try:
            garment_img = _decode_image(garment.garment_image)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Garment {i} image decode error: {exc}")

        # Resolve IP-Adapter reference: explicit override > photo
        ip_ref_img: Optional[Image.Image] = None
        if garment.ip_adapter_image:
            try:
                ip_ref_img = _decode_image(garment.ip_adapter_image)
            except Exception as exc:
                logger.warning("[tryon/multi] step %d failed to decode ip_adapter_image: %s", i, exc)
                ip_ref_img = None

        # Measure the target color BEFORE the run
        target_rgb = _measure_dominant_garment_color(
            garment_img.resize(SD15_TARGET_SIZE, Image.LANCZOS)
        )
        pre_mask = _make_mask_for_garment(
            current_img.resize(SD15_TARGET_SIZE, Image.LANCZOS), label
        )

        try:
            current_img = _run_single_tryon(
                person_img=current_img,
                garment_img=garment_img,
                label=label,
                guidance_scale=req.guidance_scale,
                num_steps=req.num_inference_steps,
                seed=(req.seed or 42) + i,
                ip_adapter_image=ip_ref_img,
                ip_adapter_scale=garment.ip_adapter_scale,
            )
            rendered_labels.append(label)
        except Exception as exc:
            logger.error(f"[tryon/multi] step {i} failed: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Inference failed at step {i} ({label}): {exc}")

        # Measure rendered color AFTER the run
        rendered_rgb = _measure_rendered_color(current_img, pre_mask)
        color_diagnostics.append(
            {
                "label": label,
                "targetRgb": list(target_rgb),
                "renderedRgb": list(rendered_rgb),
                "delta": {
                    "r": int(rendered_rgb[0]) - int(target_rgb[0]),
                    "g": int(rendered_rgb[1]) - int(target_rgb[1]),
                    "b": int(rendered_rgb[2]) - int(target_rgb[2]),
                },
            }
        )
        logger.info(
            f"[tryon/multi] {label} color: target={target_rgb} rendered={rendered_rgb}"
        )

    elapsed_ms = time.time() * 1000 - start_ms
    return {
        "success": True,
        "result_image": _encode_image(current_img),
        "method_used": "sd15_ip_adapter_sequential",
        "pipeline_version": "sequential_v1",
        "rendered_garments": rendered_labels,
        "elapsed_ms": round(elapsed_ms),
        "color_diagnostics": color_diagnostics,
    }


@app.post("/tryon/multi-fused")
async def multi_tryon_fused(req: MultiTryOnRequest):
    """Multi-garment try-on with sequential dressing order (tops -> outerwear -> pants -> shoes)."""
    version = req.pipeline_version or "fused_v2"
    start_ms = time.time() * 1000
    logger.info(
        f"[tryon/multi-fused] count={len(req.garments)} pipeline={version} "
        f"bodyContext={'yes' if req.body_profile else 'no'}",
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
