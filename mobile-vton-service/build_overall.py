"""
Final comparison composite:
  Cols: [mannequin] [t-shirt in] [pants in] [shoes in] [PRE-FIX tryon] [POST-FIX tryon (seed 7)] [POST-FIX tryon (seed 100)]
  Top:  per-garment measured target color hex
  Bottom: pre vs post RGB delta vs target
"""
import os, sys, types
from pathlib import Path

os.environ.setdefault("MOBILE_VTON_EAGER_LOAD", "0")
ROOT = Path("/Users/zohidvohidjonov/Desktop/AIWardrobe")
sys.path.insert(0, str(ROOT / "mobile-vton-service"))

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
_t.bfloat16 = "bf16"; _t.float16 = "fp16"; _t.float32 = "fp32"

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
import main  # noqa

SERVICE = ROOT / "mobile-vton-service"
ASSETS = ROOT / "assets" / "images"

CELL = 320
PAD = 8
LABEL_H = 28
SWATCH_H = 22

def fit(img, size=CELL):
    return img.convert("RGB").resize((size, size), Image.LANCZOS)

# Inputs
mannequin = fit(Image.open(ASSETS / "mannequin_front.png"))
tshirt = fit(Image.open(ASSETS / "basic_white_tshirt.png"))
pants = fit(Image.open(ASSETS / "basic_brown_pants.png"))
shoes = fit(Image.open(ASSETS / "basic_brown_loafers.png"))

# Measured dominant colors (NEW code)
t_tshirt = main._measure_dominant_garment_color(Image.open(ASSETS / "basic_white_tshirt.png"))
t_pants = main._measure_dominant_garment_color(Image.open(ASSETS / "basic_brown_pants.png"))
t_shoes = main._measure_dominant_garment_color(Image.open(ASSETS / "basic_brown_loafers.png"))

# PRE-FIX result (old service render with seed 7 — column 4 of the 5-col composite)
pre_img = Image.open(SERVICE / "test_input_vs_output_seed7.png")
src_cell = 384; src_pad = 8; src_lbl = 24
pre_tryon = pre_img.crop((src_pad + 4 * (src_cell + src_pad), src_lbl,
                          src_pad + 5 * (src_cell + src_pad), src_lbl + src_cell))
pre_tryon = fit(pre_tryon)

# POST-FIX results (NEW service)
post_tryon_7 = fit(Image.open(SERVICE / "test_new_code_seed7.png"))
post_tryon_100 = fit(Image.open(SERVICE / "test_new_code_seed100.png"))

# Build composite
n = 7
W = CELL * n + PAD * (n + 1)
H = CELL + LABEL_H + SWATCH_H + PAD * 4 + 110
canvas = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(canvas)

cols = [
    ("mannequin", mannequin, None, "input base"),
    ("t-shirt input", tshirt, t_tshirt, f"target #%02X%02X%02X" % t_tshirt),
    ("pants input",  pants,  t_pants,  f"target #%02X%02X%02X" % t_pants),
    ("shoes input",  shoes,  t_shoes,  f"target #%02X%02X%02X" % t_shoes),
    ("PRE-FIX tryon (old code)", pre_tryon, None, "ipAdapter=0.70, no post-process"),
    ("POST-FIX tryon seed 7 (new code)", post_tryon_7, None, "ipAdapter=0.85, color match 0.85, two-pass shoes"),
    ("POST-FIX tryon seed 100 (new code)", post_tryon_100, None, "same code, different seed"),
]

for i, (label, img, color, sub) in enumerate(cols):
    x = PAD + i * (CELL + PAD)
    y_img = LABEL_H + SWATCH_H + PAD * 2
    canvas.paste(img, (x, y_img))
    draw.text((x + 6, 4), label, fill="black")
    if color is not None:
        sw = Image.new("RGB", (CELL, SWATCH_H), color)
        canvas.paste(sw, (x, LABEL_H))
        draw.text((x + 6, LABEL_H + 4), f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}  {color}", fill="black" if sum(color) < 384 else "white")

# Bottom row: per-garment PRE vs POST delta
y_bottom = LABEL_H + SWATCH_H + PAD * 2 + CELL + PAD
import numpy as np
pre_arr = np.asarray(pre_tryon.resize((640, 640), Image.LANCZOS), dtype=np.uint8)
post7 = np.asarray(post_tryon_7.resize((640, 640), Image.LANCZOS), dtype=np.uint8)
post100 = np.asarray(post_tryon_100.resize((640, 640), Image.LANCZOS), dtype=np.uint8)
W2, H2 = 640, 640
regions = {
    "top":   (int(H2*0.22), int(H2*0.50), int(W2*0.30), int(W2*0.70)),
    "pants": (int(H2*0.50), int(H2*0.90), int(W2*0.35), int(W2*0.65)),
    "shoes": (int(H2*0.90), int(H2*0.99), int(W2*0.30), int(W2*0.70)),
}
def med(arr, region):
    y0,y1,x0,x1 = region
    return tuple(int(x) for x in np.median(arr[y0:y1, x0:x1].reshape(-1,3), axis=0))

draw.text((PAD, y_bottom), "Color accuracy (median RGB inside each garment region, target on right):", fill="black")
y2 = y_bottom + 20
for i, (label, _, color, sub) in enumerate(cols[4:7]):
    if i == 0:
        x = PAD + 4 * (CELL + PAD)
        arr = pre_arr; tag = "PRE  "
    elif i == 1:
        x = PAD + 5 * (CELL + PAD)
        arr = post7; tag = "POST7"
    else:
        x = PAD + 6 * (CELL + PAD)
        arr = post100; tag = "POST100"
    for j, gl in enumerate(("top", "pants", "shoes")):
        target = (t_tshirt, t_pants, t_shoes)[j]
        m = med(arr, regions[gl])
        d = (m[0]-target[0], m[1]-target[1], m[2]-target[2])
        text = f"{tag} {gl:5s} rendered={m}  target={target}  delta=({d[0]:+d},{d[1]:+d},{d[2]:+d})"
        draw.text((x + 6, y2 + j * 18), text, fill="black")

OUT = SERVICE / "test_overall.png"
canvas.save(OUT)
print(f"Wrote {OUT}")
print(f"  size: {canvas.size}")
print()
print("Color accuracy (median RGB inside each garment region):")
print(f"                  PRE             POST seed 7      POST seed 100")
for j, gl in enumerate(("top", "pants", "shoes")):
    target = (t_tshirt, t_pants, t_shoes)[j]
    pre_m = med(pre_arr, regions[gl])
    p7 = med(post7, regions[gl])
    p100 = med(post100, regions[gl])
    print(f"  {gl:5s}  target={target}  pre={pre_m}  post7={p7}  post100={p100}")
