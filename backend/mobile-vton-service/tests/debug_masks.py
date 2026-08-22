"""
Debug script to check mask and garment dimensions.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['MOBILE_VTON_EAGER_LOAD'] = '0'

import types
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

import main
from PIL import Image
import numpy as np

ROOT = "/Users/zohidvohidjonov/Desktop/AIWardrobe"
ASSETS = f"{ROOT}/assets/images"

# Re-execute the test file to get the helper functions
test_code = open("test_v2_masks.py").read()
# Just get the function definitions before the test execution
funcs_section = test_code.split("# Tests")[0]
namespace = {"__name__": "__main__", "__file__": os.path.abspath("test_v2_masks.py"), "main": main}
exec(funcs_section, namespace)

person = Image.open(f"{ASSETS}/mannequin_front.png").convert("RGB")
tshirt = Image.open(f"{ASSETS}/basic_white_tshirt.png").convert("RGB")
pants = Image.open(f"{ASSETS}/basic_brown_pants.png").convert("RGB")

# Check TOP
m = namespace["make_mask_v2"](person, "top")
m_arr = np.asarray(m, dtype=np.uint8)
core = (m_arr > 128)
ys, xs = np.where(core)
y0, y1 = int(ys.min()), int(ys.max())
x0, x1 = int(xs.min()), int(xs.max())
bw, bh = x1 - x0 + 1, y1 - y0 + 1
print(f"TOP mask: y={y0}-{y1} ({bh}px)  x={x0}-{x1} ({bw}px)")

g_pil, g_mask = namespace["center_garment_in_bbox"](tshirt, bw, bh)
print(f"TOP garment: size={g_pil.size}")
g_arr = np.asarray(g_pil, dtype=np.uint8)
mean_ch = g_arr.mean(axis=2)
white_pct = (mean_ch > 200).sum() / mean_ch.size * 100
print(f"  white pixels: {white_pct:.1f}%")
print(f"  row 0 mean: {g_arr[0].mean(axis=0)}")
print(f"  row -1 mean: {g_arr[-1].mean(axis=0)}")
print(f"  col 0 mean: {g_arr[:, 0].mean(axis=0)}")
print(f"  col -1 mean: {g_arr[:, -1].mean(axis=0)}")

# Check PANTS
m2 = namespace["make_mask_v2"](person, "pants")
m2_arr = np.asarray(m2, dtype=np.uint8)
core2 = (m2_arr > 128)
ys2, xs2 = np.where(core2)
y0p, y1p = int(ys2.min()), int(ys2.max())
x0p, x1p = int(xs2.min()), int(xs2.max())
bwp, bhp = x1p - x0p + 1, y1p - y0p + 1
print(f"PANTS mask: y={y0p}-{y1p} ({bhp}px)  x={x0p}-{x1p} ({bwp}px)")

gp_pil, gp_mask = namespace["center_garment_in_bbox"](pants, bwp, bhp)
print(f"PANTS garment: size={gp_pil.size}")
gp_arr = np.asarray(gp_pil, dtype=np.uint8)
mean_ch_p = gp_arr.mean(axis=2)
brown_pct = ((mean_ch_p > 80) & (mean_ch_p < 160)).sum() / mean_ch_p.size * 100
print(f"  brown pixels: {brown_pct:.1f}%")
print(f"  row 0 mean: {gp_arr[0].mean(axis=0)}")
print(f"  row -1 mean: {gp_arr[-1].mean(axis=0)}")
print(f"  col 0 mean: {gp_arr[:, 0].mean(axis=0)}")
print(f"  col -1 mean: {gp_arr[:, -1].mean(axis=0)}")
