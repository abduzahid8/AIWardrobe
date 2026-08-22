"""
Local test for the NEW body-aware mask generation + garment preprocessing v2.

Key improvements over v1:
  * Larger per-label dilation so sleeves / pant legs have room
  * Use the MASK (not the garment foreground) as the alpha when pasting
  * Garment fills the full mask bbox (stretched to fit) so the inpainting
    sees a complete garment, not a tiny centered crop
  * Per-label anchoring (top anchored to neck, pants anchored to waist,
    shoes anchored to foot bottom)
  * Two-pass shoes: split the shoes mask into left/right halves and paste
    each half of the shoes image into the corresponding foot mask
"""
import os
import sys
import types

def _install_stub(name, attrs=None):
    attrs = attrs or {}
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod

_torch_stub = _install_stub("torch")
_torch_stub.inference_mode = lambda: __import__("contextlib").nullcontext()
_torch_stub.Generator = lambda device=None: None
_torch_stub.manual_seed = lambda seed: None
_torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False, get_device_name=lambda i: None)
_torch_stub.bfloat16 = "bf16"
_torch_stub.float16 = "fp16"
_torch_stub.float32 = "fp32"

class _FastAPI:
    def __init__(self, *a, **kw): pass
    def add_middleware(self, *a, **kw): pass
    def get(self, *a, **kw):
        def deco(fn): return fn
        return deco
    def post(self, *a, **kw):
        def deco(fn): return fn
        return deco
_install_stub("fastapi", {"FastAPI": _FastAPI, "HTTPException": Exception})
class _CORS:
    def __init__(self, *a, **kw): pass
_install_stub("fastapi.middleware.cors", {"CORSMiddleware": _CORS})
class _BaseModel: pass
_install_stub("pydantic", {"BaseModel": _BaseModel})

os.environ.setdefault("MOBILE_VTON_EAGER_LOAD", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageFilter, ImageDraw
import numpy as np

import main  # noqa: E402

ROOT = "/Users/zohidvohidjonov/Desktop/AIWardrobe"
ASSETS = f"{ROOT}/assets/images"
ART = f"{ROOT}/mobile-vton-service/test_artifacts"
os.makedirs(ART, exist_ok=True)


# ---------------------------------------------------------------------------
# Body landmark detection
# ---------------------------------------------------------------------------
def analyze_body_geometry(silhouette_img):
    """Find key Y-coordinates and per-row edges on the body silhouette."""
    arr = np.asarray(silhouette_img.convert("L"), dtype=np.uint8)
    h, w = arr.shape
    body_rows = []
    for y in range(h):
        cols = np.where(arr[y] > 64)[0]
        if len(cols) > 0:
            body_rows.append((y, int(cols[0]), int(cols[-1])))
    if not body_rows:
        return None
    y_min = body_rows[0][0]
    y_max = body_rows[-1][0]
    span = max(1, y_max - y_min)

    def widest_in(lo, hi):
        a, b = y_min + int(lo * span), y_min + int(hi * span)
        cands = [r for r in body_rows if a <= r[0] <= b]
        return max(cands, key=lambda r: r[2] - r[1]) if cands else None

    def narrowest_in(lo, hi):
        a, b = y_min + int(lo * span), y_min + int(hi * span)
        cands = [r for r in body_rows if a <= r[0] <= b]
        return min(cands, key=lambda r: r[2] - r[1]) if cands else None

    return {
        "head": widest_in(0.0, 0.08),
        "shoulders": widest_in(0.15, 0.25),
        "chest": widest_in(0.22, 0.32),
        "waist": narrowest_in(0.30, 0.45),
        "hip": widest_in(0.48, 0.58),
        "knee": narrowest_in(0.58, 0.72),
        "ankle": narrowest_in(0.78, 0.92),
        "foot": widest_in(0.88, 1.00),
        "y_min": y_min, "y_max": y_max, "span": span,
        "body_rows": body_rows,
    }


# ---------------------------------------------------------------------------
# Mask generation v2
# ---------------------------------------------------------------------------
# Per-label Y-range (fraction of body span) and horizontal dilation (px)
# dilate=72 gives the model ~72 px of "fabric slack" past the body line
# on each side, which is enough to draw a t-shirt sleeve or a trouser
# cuff without bleeding into the white background.
_LABEL_CONFIG = {
    "top":   {"y_range": (0.17, 0.48), "dilate": 72, "feather": 12, "min_width_frac": 0.55},
    "layer": {"y_range": (0.15, 0.62), "dilate": 84, "feather": 12, "min_width_frac": 0.60},
    "pants": {"y_range": (0.46, 0.92), "dilate": 36, "feather": 12, "min_width_frac": 0.32},
    "shoes": {"y_range": (0.86, 0.99), "dilate": 20, "feather": 8,  "min_width_frac": 0.30},
}


def make_mask_v2(person_img, label):
    """
    Body-aware mask built by SLICING the silhouette.

    The mask uses the MAXIMUM width of the silhouette within the label's
    Y range, applied to ALL rows in that range. This gives a rectangular
    mask that follows the body's height but uses the widest point (so the
    garment fills the full mask, not just the narrow waist).

    The mask is dilated horizontally so the inpainting has room to draw
    the garment. The dilation is large enough for sleeves (top) and pant
    cuffs (pants) but small enough that the mask stays inside the image.
    """
    label_norm = main._normalize_garment_label(label)
    cfg = _LABEL_CONFIG[label_norm]
    silh = main._person_silhouette(person_img)
    geom = analyze_body_geometry(silh)
    if geom is None:
        return Image.new("L", person_img.size, 0)

    w, h = person_img.size
    span = geom["span"]
    y_lo = geom["y_min"] + int(cfg["y_range"][0] * span)
    y_hi = geom["y_min"] + int(cfg["y_range"][1] * span)
    y_hi = min(h - 1, y_hi)
    dil = cfg["dilate"]

    # Find the maximum width of the silhouette within the Y range
    rows_in_range = [r for r in geom["body_rows"] if y_lo <= r[0] <= y_hi]
    if not rows_in_range:
        return Image.new("L", (w, h), 0)

    # Use the maximum width row as the reference
    widest = max(rows_in_range, key=lambda r: r[2] - r[1])
    ref_lx, ref_rx = widest[1], widest[2]
    cx = (ref_lx + ref_rx) // 2

    # Build rectangular mask: all rows in Y range have the same width
    # (centered on the widest point, dilated)
    half_w = (ref_rx - ref_lx) // 2 + dil
    min_half = int(cfg["min_width_frac"] * w) // 2
    half_w = max(half_w, min_half)

    out = np.zeros((h, w), dtype=np.uint8)
    for y in range(y_lo, y_hi + 1):
        lx2 = max(0, cx - half_w)
        rx2 = min(w - 1, cx + half_w)
        out[y, lx2:rx2 + 1] = 255

    if out.max() < 16:
        return Image.fromarray(out, mode="L")

    mask_img = Image.fromarray(out, mode="L")
    if cfg["feather"] > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=cfg["feather"]))
    return mask_img


def make_shoes_halves(person_img):
    """
    Return (left_mask, right_mask) — the shoes mask split at the
    vertical centre line so each foot gets its own inpainting pass.
    """
    full = make_mask_v2(person_img, "shoes")
    arr = np.asarray(full, dtype=np.uint8)
    w = arr.shape[1]
    cx = w // 2
    left = np.zeros_like(arr)
    right = np.zeros_like(arr)
    left[:, :cx] = arr[:, :cx]
    right[:, cx:] = arr[:, cx:]
    return Image.fromarray(left, mode="L"), Image.fromarray(right, mode="L")


# ---------------------------------------------------------------------------
# Garment extraction
# ---------------------------------------------------------------------------
def extract_garment_fg(garment_img, bg_threshold=240, dilate=3):
    """Return (rgb, mask) — garment pixels separated from white background."""
    rgb = garment_img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.uint8)
    mean_ch = arr.mean(axis=2)
    fg = (mean_ch < bg_threshold).astype(np.uint8) * 255
    fg_img = Image.fromarray(fg, mode="L")
    if dilate > 0:
        fg_img = fg_img.filter(ImageFilter.MaxFilter(2 * dilate + 1))
    mask = np.asarray(fg_img, dtype=np.uint8) > 16
    return rgb, mask


def center_garment_in_bbox(garment_img, target_w, target_h):
    """
    Extract the garment foreground, CENTER it on a square canvas filled
    with the garment's dominant color, and resize to (target_w, target_h).

    Centering on a uniformly-colored canvas (rather than neutral gray)
    means the seam between garment and padding is the SAME COLOR as the
    garment — SD1.5 inpainting will not see a visible boundary between
    garment and "background" inside the mask.

    For LIGHT garments (white t-shirt) the canvas is pure white.
    For DARK garments (brown pants, brown loafers) the canvas is the
    measured dominant color of the garment foreground.
    """
    rgb, mask = extract_garment_fg(garment_img)
    arr = np.asarray(rgb, dtype=np.uint8)
    ys, xs = np.where(mask)
    if len(ys) < 64:
        return Image.new("RGB", (target_w, target_h), (192, 192, 192)), \
               np.zeros((target_h, target_w), dtype=np.float32)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    crop = arr[y0:y1 + 1, x0:x1 + 1]
    crop_mask = mask[y0:y1 + 1, x0:x1 + 1].astype(np.uint8) * 255

    # Measure the dominant color of the garment (median of foreground)
    fg_pixels = crop[crop_mask > 128]
    if len(fg_pixels) >= 3:
        med = np.median(fg_pixels.reshape(-1, 3), axis=0)
        bg_color = (int(med[0]), int(med[1]), int(med[2]))
    else:
        bg_color = (192, 192, 192)
    # Snap bright garments to pure white
    if min(bg_color) > 200:
        bg_color = (255, 255, 255)

    # Pad the crop to a square (centered) with the dominant color
    ch, cw = crop.shape[:2]
    side = max(cw, ch)
    canvas = np.full((side, side, 3), bg_color, dtype=np.uint8)
    canvas_mask = np.zeros((side, side), dtype=np.uint8)
    px = (side - cw) // 2
    py = (side - ch) // 2
    canvas[py:py + ch, px:px + cw] = crop
    canvas_mask[py:py + ch, px:px + cw] = crop_mask

    # Resize to target
    pil = Image.fromarray(canvas, mode="RGB").resize((target_w, target_h), Image.LANCZOS)
    mask_pil = Image.fromarray(canvas_mask, mode="L").resize((target_w, target_h), Image.LANCZOS)
    mask_arr = np.asarray(mask_pil, dtype=np.float32) / 255.0
    return pil, mask_arr


# ---------------------------------------------------------------------------
# Paste garment into mask area using the MASK as the alpha
# ---------------------------------------------------------------------------
def paste_garment_into_mask_v2(person_img, garment_img, mask_img, label,
                                stretch=True):
    """
    Paste the garment image into the masked region of `person_img`.

    Improvements over v1:
      * The garment is CENTERED in its own bounding box before scaling,
        so the result is always centered in the mask regardless of the
        garment's position in the product photo.
      * The MASK is the alpha — the garment is only visible where the
        mask is lit. This gives a proper "pants shape" (with a gap
        between the legs) instead of a uniform brown rectangle.
      * The garment fills the full mask bbox by default.
    """
    person_rgb = person_img.convert("RGB")
    mask_arr = np.asarray(mask_img.convert("L"), dtype=np.uint8)
    soft_mask = mask_arr.astype(np.float32) / 255.0
    core = (mask_arr > 128)
    if core.sum() < 64:
        return person_rgb

    ys, xs = np.where(core)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    if bw <= 0 or bh <= 0:
        return person_rgb

    new_w, new_h = bw, bh
    g_pil, g_mask_r = center_garment_in_bbox(garment_img, new_w, new_h)
    g_arr_r = np.asarray(g_pil, dtype=np.float32)

    # Anchor
    if label in ("top", "layer", "pants"):
        ay = y0
    else:
        ay = y1 - new_h + 1
    ax = x0 + (bw - new_w) // 2

    # Paste
    p_arr = np.asarray(person_rgb, dtype=np.float32)
    ay2 = max(0, ay)
    ay3 = min(p_arr.shape[0], ay + new_h)
    ax2 = max(0, ax)
    ax3 = min(p_arr.shape[1], ax + new_w)
    if ay2 >= ay3 or ax2 >= ax3:
        return person_rgb
    ph0 = ay2 - ay
    ph1 = ph0 + (ay3 - ay2)
    pw0 = ax2 - ax
    pw1 = pw0 + (ax3 - ax2)
    g_chunk = g_arr_r[ph0:ph1, pw0:pw1]
    gm_chunk = g_mask_r[ph0:ph1, pw0:pw1][..., None]
    p_chunk = p_arr[ay2:ay3, ax2:ax3]

    sm_chunk = soft_mask[ay2:ay3, ax2:ax3][..., None]
    # Where the mask is lit AND the garment has foreground, use the garment.
    # Where the mask is lit but the garment doesn't cover, use the garment
    # color (so the model sees a complete garment, not transparent holes).
    visible = np.maximum(sm_chunk, gm_chunk)
    blended = p_chunk * (1.0 - visible) + g_chunk * visible
    p_arr[ay2:ay3, ax2:ax3] = blended
    return Image.fromarray(np.clip(p_arr, 0, 255).astype(np.uint8), mode="RGB")


def paste_shoes_two_pass(person_img, garment_img):
    """
    Paste the shoes image using a two-pass approach.

    The product photo for shoes is typically a SINGLE shoe shown from
    the side. We can't split a single shoe at the centre — that would
    give us a heel and a toe, not two shoes. Instead, we paste the FULL
    shoe (centered on its own dominant-color canvas) into each foot mask.
    The shoe will be slightly compressed, but the model will see a
    complete shoe shape in each foot mask and draw two distinct shoes.

    The shoes mask is split into left/right halves at the body centre
    so each half-mask targets one foot.
    """
    left_mask, right_mask = make_shoes_halves(person_img)
    result = person_img.convert("RGB")
    for m in [left_mask, right_mask]:
        m_arr = np.asarray(m.convert("L"), dtype=np.uint8)
        core = (m_arr > 128)
        if core.sum() < 16:
            continue
        ys, xs = np.where(core)
        y0m, y1m = int(ys.min()), int(ys.max())
        x0m, x1m = int(xs.min()), int(xs.max())
        bw = x1m - x0m + 1
        bh = y1m - y0m + 1
        if bw <= 0 or bh <= 0:
            continue
        # Center the full shoe in the foot mask
        g_pil, g_mask_r = center_garment_in_bbox(garment_img, bw, bh)
        g_arr_r = np.asarray(g_pil, dtype=np.float32)
        # Paste centered in the mask
        ax = x0m
        ay = y0m
        p_arr = np.asarray(result, dtype=np.float32)
        ay2 = max(0, ay); ay3 = min(p_arr.shape[0], ay + bh)
        ax2 = max(0, ax); ax3 = min(p_arr.shape[1], ax + bw)
        if ay2 >= ay3 or ax2 >= ax3:
            continue
        ph0 = ay2 - ay; ph1 = ph0 + (ay3 - ay2)
        pw0 = ax2 - ax; pw1 = pw0 + (ax3 - ax2)
        g_chunk = g_arr_r[ph0:ph1, pw0:pw1]
        gm_chunk = g_mask_r[ph0:ph1, pw0:pw1][..., None]
        sm_chunk = m_arr[ay2:ay3, ax2:ax3].astype(np.float32) / 255.0
        sm_chunk = sm_chunk[..., None]
        p_chunk = p_arr[ay2:ay3, ax2:ax3]
        visible = np.maximum(sm_chunk, gm_chunk)
        blended = p_chunk * (1.0 - visible) + g_chunk * visible
        p_arr[ay2:ay3, ax2:ax3] = blended
        result = Image.fromarray(np.clip(p_arr, 0, 255).astype(np.uint8), mode="RGB")
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
person = Image.open(f"{ASSETS}/mannequin_front.png").convert("RGB")
tshirt = Image.open(f"{ASSETS}/basic_white_tshirt.png").convert("RGB")
pants  = Image.open(f"{ASSETS}/basic_brown_pants.png").convert("RGB")
shoes  = Image.open(f"{ASSETS}/basic_brown_loafers.png").convert("RGB")

# 1) Body analysis
silh = main._person_silhouette(person)
geom = analyze_body_geometry(silh)
print("=== BODY GEOMETRY ===")
for k, v in geom.items():
    if k == "body_rows":
        print(f"  {k}: {len(v)} rows")
    else:
        print(f"  {k}: {v}")

# 2) New masks
masks = {}
for label in ["top", "layer", "pants", "shoes"]:
    m = make_mask_v2(person, label)
    masks[label] = m
    coverage = (np.asarray(m) > 16).sum() / np.asarray(m).size
    print(f"  mask[{label:5s}]  coverage={coverage*100:5.1f}%  extrema={m.getextrema()}")

# 3) Save mask overlays
tints = {
    "top":   (255, 60, 60),
    "layer": (255, 140, 0),
    "pants": (60, 200, 60),
    "shoes": (60, 120, 255),
}
arr = np.asarray(person, dtype=np.float32).copy()
for label, tint in tints.items():
    m = np.asarray(masks[label], dtype=np.float32) / 255.0
    m3 = np.stack([m, m, m], axis=-1)
    color = np.array(tint, dtype=np.float32)
    arr = arr * (1 - 0.4 * m3) + color * (0.4 * m3)
canvas = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
canvas.save(f"{ART}/v3_mask_composite.png")
for label, m in masks.items():
    m.save(f"{ART}/v3_mask_{label}.png")
print(f"  saved v3_mask_*.png")

# 4) Pre-paste tests
pastes = {}
for label, garment in [("top", tshirt), ("pants", pants)]:
    paste = paste_garment_into_mask_v2(person, garment, masks[label], label)
    pastes[label] = paste
    paste.save(f"{ART}/v3_paste_{label}.png")
    print(f"  saved v3_paste_{label}.png")

# Shoes use two-pass
shoes_paste = paste_shoes_two_pass(person, shoes)
pastes["shoes"] = shoes_paste
shoes_paste.save(f"{ART}/v3_paste_shoes.png")
print(f"  saved v3_paste_shoes.png")

# 5) Cumulative paste
cum = person.copy()
cum = paste_garment_into_mask_v2(cum, tshirt, masks["top"], "top")
cum.save(f"{ART}/v3_paste_cum_top.png")
cum = paste_garment_into_mask_v2(cum, pants, masks["pants"], "pants")
cum.save(f"{ART}/v3_paste_cum_pants.png")
cum = paste_shoes_two_pass(cum, shoes)
cum.save(f"{ART}/v3_paste_cum_all.png")
print(f"  saved v3_paste_cum_*.png")

# 6) Strip composite
CELL = 320; PAD = 8; LABEL_H = 24
n = 6
W = CELL * n + PAD * (n + 1)
H = CELL + LABEL_H + PAD * 2
strip = Image.new("RGB", (W, H), (255, 255, 255))
draw = ImageDraw.Draw(strip)
cols = [
    ("mannequin", person),
    ("+top", pastes["top"]),
    ("+pants", paste_garment_into_mask_v2(pastes["top"], pants, masks["pants"], "pants")),
    ("+shoes", cum),
    ("mask top", Image.merge("RGB", (masks["top"], masks["top"], masks["top"]))),
    ("mask pants", Image.merge("RGB", (masks["pants"], masks["pants"], masks["pants"]))),
]
for i, (label, img) in enumerate(cols):
    x = PAD + i * (CELL + PAD)
    y = LABEL_H
    strip.paste(img.convert("RGB").resize((CELL, CELL), Image.LANCZOS), (x, y))
    draw.text((x + 4, 4), label, fill="black")
strip.save(f"{ART}/v3_paste_strip.png")
print(f"  saved v3_paste_strip.png")
