"""
Offline validation of the mobile-vton preprocessing pipeline.

Verifies (without torch / diffusers):
  * _person_silhouette extracts a body shape from the mannequin image
  * _make_mask_for_garment produces a body-shape aware mask per label
  * _preprocess_garment removes the white background from product photos

Outputs:
  mobile-vton-service/test_artifacts/silhouette.png
  mobile-vton-service/test_artifacts/mask_<label>.png
  mobile-vton-service/test_artifacts/garment_preprocessed.png
"""
import os
import sys
import types

os.environ.setdefault("MOBILE_VTON_EAGER_LOAD", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Stub torch (and any other heavy deps) so we can import main.py on a machine
# without a CUDA toolchain. Only the pure-PIL/numpy helpers are exercised.
def _install_stub(name, attrs=None):
    attrs = attrs or {}
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod

# torch stub: only the surface main.py uses at import time
_torch_stub = _install_stub("torch")
_torch_stub.inference_mode = lambda: __import__("contextlib").nullcontext()
_torch_stub.Generator = lambda device=None: None
_torch_stub.manual_seed = lambda seed: None
_torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False, get_device_name=lambda i: None)
_torch_stub.bfloat16 = "bf16"
_torch_stub.float16 = "fp16"
_torch_stub.float32 = "fp32"
_torch_stub.inference_mode = lambda: __import__("contextlib").nullcontext()

# fastapi / pydantic stubs so the import doesn't fail
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

# PIL and numpy are real
from PIL import Image  # noqa: E402

import main  # noqa: E402

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_artifacts")
os.makedirs(ART, exist_ok=True)

PERSON = "/Users/zohidvohidjonov/Desktop/AIWardrobe/assets/images/mannequin_front.png"
GARMENT = "/Users/zohidvohidjonov/Desktop/AIWardrobe/assets/images/basic_white_tshirt.png"

person_img = Image.open(PERSON).convert("RGB")
garment_img = Image.open(GARMENT).convert("RGB")

# 1) Silhouette ----------------------------------------------------------
silh = main._person_silhouette(person_img)
silh.save(os.path.join(ART, "silhouette.png"))
print(f"silhouette saved  size={silh.size}  max={silh.getextrema()[1]}")

# 2) Per-label masks ----------------------------------------------------
for label in ["top", "layer", "pants", "shoes"]:
    mask = main._make_mask_for_garment(person_img, label)
    out = os.path.join(ART, f"mask_{label}.png")
    mask.save(out)
    # Coverage check
    import numpy as np
    arr = np.asarray(mask)
    coverage = (arr > 16).sum() / arr.size
    print(f"mask[{label}]  saved  coverage={coverage*100:5.1f}%  extrema={mask.getextrema()}")

# 3) Garment pre-processing --------------------------------------------
processed = main._preprocess_garment(garment_img)
processed.save(os.path.join(ART, "garment_preprocessed.png"))
import numpy as np
src = np.asarray(garment_img)
dst = np.asarray(processed)
n_bg = (src.min(axis=2) > main._GARMENT_BG_THRESHOLD).sum()
print(f"garment preprocessed  saved  bg_pixels_removed={n_bg} of {src.shape[0]*src.shape[1]}")

print("\nAll preprocessing artifacts written to:", ART)

# 4) Composite — overlay each label mask (in a different tint) on the
# mannequin so the geometry can be visually verified.
import numpy as np
base = person_img.convert("RGB")
arr = np.asarray(base, dtype=np.float32)
tints = {
    "top":   (255, 80, 80),    # red
    "layer": (255, 165, 0),    # orange
    "pants": (80, 200, 80),    # green
    "shoes": (80, 130, 255),   # blue
}
overlay = arr.copy()
for label in ["top", "layer", "pants", "shoes"]:
    mask = np.asarray(main._make_mask_for_garment(person_img, label), dtype=np.float32) / 255.0
    mask3 = np.stack([mask, mask, mask], axis=-1)
    color = np.array(tints[label], dtype=np.float32)
    overlay = overlay * (1 - 0.35 * mask3) + color * (0.35 * mask3)

Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB").save(
    os.path.join(ART, "mask_composite.png")
)
print("composite saved →", os.path.join(ART, "mask_composite.png"))
