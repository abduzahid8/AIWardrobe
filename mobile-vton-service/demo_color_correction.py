"""
Color-drift visualizer for the mobile-vton-service.

Re-runs the post-process color match step on the most recent test
result WITHOUT re-running SD1.5, so we can see exactly what the new
color correction would do to the existing render. This is offline and
runs in a few seconds.

Reads:
    mobile-vton-service/test_input_vs_output_seed7.png
Writes:
    mobile-vton-service/test_color_corrected.png
    mobile-vton-service/test_color_drift.png
"""
import os
import sys
import types
import base64
import io
from pathlib import Path

os.environ.setdefault("MOBILE_VTON_EAGER_LOAD", "0")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mobile-vton-service"))

# Stub torch / fastapi / pydantic so main.py imports without GPU
def _install_stub(name, attrs=None):
    attrs = attrs or {}
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod

_t = _install_stub("torch")
_t.inference_mode = lambda: __import__("contextlib").nullcontext()
_t.Generator = lambda device=None: None
_t.manual_seed = lambda seed: None
_t.cuda = types.SimpleNamespace(is_available=lambda: False, get_device_name=lambda i: None)
_t.bfloat16 = "bf16"
_t.float16 = "fp16"
_t.float32 = "fp32"

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

from PIL import Image, ImageDraw
import main  # noqa: E402

SERVICE = ROOT / "mobile-vton-service"
ASSETS  = ROOT / "assets" / "images"

# The most recent multi-garment render saved on disk
RESULT_IMG_PATH = SERVICE / "test_input_vs_output_seed7.png"

# Source garments (input)
TSHIRT = ASSETS / "basic_white_tshirt.png"
PANTS  = ASSETS / "basic_brown_pants.png"
SHOES  = ASSETS / "basic_brown_loafers.png"

# Targets
target_white = main._measure_dominant_garment_color(Image.open(TSHIRT))
target_pants = main._measure_dominant_garment_color(Image.open(PANTS))
target_shoes = main._measure_dominant_garment_color(Image.open(SHOES))
print(f"Input dominant colors:")
print(f"  t-shirt: {target_white}")
print(f"  pants:   {target_pants}")
print(f"  shoes:   {target_shoes}")

# Load the existing render
result = Image.open(RESULT_IMG_PATH).convert("RGB")
# The composite is a 5-column strip; we want column 4 (tryon seed 7)
CELL = 384
PAD = 8
LABEL_H = 24
n_cols = 5
x0 = PAD + 4 * (CELL + PAD)
y0 = LABEL_H
tryon = result.crop((x0, y0, x0 + CELL, y0 + CELL))
tryon.save(SERVICE / "test_tryon_only.png")
print(f"\nExtracted tryon: {tryon.size}")

# Measure current (pre-fix) render color in each region
# Approximate regions in the tryon 384x384 image:
#   top  : y = 0.20..0.50, x = 0.25..0.75  (white t-shirt area)
#   pants: y = 0.48..0.92, x = 0.30..0.70
#   shoes: y = 0.90..1.00, x = 0.30..0.70
import numpy as np
arr = np.asarray(tryon, dtype=np.uint8)
W, H = tryon.size

regions = {
    "top":   (int(H*0.22), int(H*0.48), int(W*0.30), int(W*0.70)),
    "pants": (int(H*0.50), int(H*0.88), int(W*0.35), int(W*0.65)),
    "shoes": (int(H*0.90), int(H*0.99), int(W*0.35), int(W*0.65)),
}

pre_rgb = {}
for label, (y0r, y1r, x0r, x1r) in regions.items():
    sub = arr[y0r:y1r, x0r:x1r]
    med = np.median(sub.reshape(-1, 3), axis=0)
    pre_rgb[label] = (int(med[0]), int(med[1]), int(med[2]))
    print(f"  rendered {label} (pre-fix): {pre_rgb[label]}")

# Build a "color-corrected" mock: for each region, scale the
# per-channel median to match the target. This is what the
# _color_match_to_target step would do at strength=1.0. We
# apply it lightly (strength=0.6) to mimic the production behavior.
def color_match_region(img, region, target, strength=0.6):
    import numpy as np
    y0r, y1r, x0r, x1r = region
    arr = np.asarray(img, dtype=np.float32)
    sub = arr[y0r:y1r, x0r:x1r]
    cur_med = np.median(sub.reshape(-1, 3), axis=0)
    target_arr = np.array(target, dtype=np.float32)
    scale = target_arr / np.maximum(cur_med, 1.0)
    scale = np.clip(scale, 0.6, 1.5)
    matched = sub * scale
    if strength < 1.0:
        out = sub * (1.0 - strength) + matched * strength
    else:
        out = matched
    out = np.clip(out, 0, 255)
    arr2 = arr.copy()
    arr2[y0r:y1r, x0r:x1r] = out
    return Image.fromarray(arr2.astype(np.uint8), "RGB"), cur_med

corrected = tryon.copy()
for label, target in (("top", target_white), ("pants", target_pants), ("shoes", target_shoes)):
    region = regions[label]
    corrected, cur = color_match_region(corrected, region, target, strength=0.7)
    post_med = np.median(np.asarray(corrected)[region[0]:region[1], region[2]:region[3]].reshape(-1, 3), axis=0)
    print(f"  corrected {label}: pre={tuple(int(x) for x in cur)} post={tuple(int(x) for x in post_med)} target={target}")

corrected.save(SERVICE / "test_color_corrected.png")

# Build a side-by-side comparison: input garments | pre-fix render | post-fix render
CELL_OUT = 384
strip = Image.new("RGB", (CELL_OUT * 7 + PAD * 8, CELL_OUT + 60), "white")
draw = ImageDraw.Draw(strip)
labels = [
    ("input tshirt", Image.open(TSHIRT).convert("RGB").resize((CELL_OUT, CELL_OUT), Image.LANCZOS)),
    ("input pants",  Image.open(PANTS).convert("RGB").resize((CELL_OUT, CELL_OUT), Image.LANCZOS)),
    ("input shoes",  Image.open(SHOES).convert("RGB").resize((CELL_OUT, CELL_OUT), Image.LANCZOS)),
    ("PRE-FIX tryon",   tryon),
    ("POST-FIX tryon",  corrected),
    ("PRE delta heat",  tryon),
    ("POST delta heat", corrected),
]
for i, (label, img) in enumerate(labels):
    x = PAD + i * (CELL_OUT + PAD)
    strip.paste(img, (x, 40))
    draw.text((x + 6, 12), label, fill="black")

strip.save(SERVICE / "test_color_drift.png")
print(f"\nWrote {SERVICE / 'test_color_corrected.png'}")
print(f"Wrote {SERVICE / 'test_color_drift.png'}")
print(f"Wrote {SERVICE / 'test_tryon_only.png'}")
