"""
Real end-to-end test of the NEW mobile-vton service code.

Calls the live Modal deployment with the standard 3-garment input
(white t-shirt, brown pants, brown loafers) and saves the actual
dressed-up result.
"""
import ssl
import urllib.request
import json
import base64
import time
from pathlib import Path

ROOT = Path("/Users/zohidvohidjonov/Desktop/AIWardrobe")
ASSETS = ROOT / "assets" / "images"
OUT = ROOT / "mobile-vton-service"
ENDPOINT = "https://karimdzanovzoha--aiwardrobe-mobile-vton-fastapi-app.modal.run/tryon/multi-fused"

# Use unverified SSL for the local Python install (certifi bundle mismatch)
ctx = ssl._create_unverified_context()


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def call(payload, timeout=600):
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
        return json.loads(r.read())


def save_result(resp, name):
    img_b64 = resp["result_image"]
    if img_b64.startswith("data:"):
        img_b64 = img_b64.split(",", 1)[1]
    (OUT / name).write_bytes(base64.b64decode(img_b64))
    print(f"  saved {name} ({len(img_b64)} chars b64)")


for seed in (7, 100):
    print(f"\n=== seed {seed} ===")
    payload = {
        "person_image": b64(ASSETS / "mannequin_front.png"),
        "garments": [
            {"label": "top",   "garment_image": b64(ASSETS / "basic_white_tshirt.png")},
            {"label": "pants", "garment_image": b64(ASSETS / "basic_brown_pants.png")},
            {"label": "shoes", "garment_image": b64(ASSETS / "basic_brown_loafers.png")},
        ],
        "seed": seed,
        "guidance_scale": 7.5,
        "num_inference_steps": 25,
        "pipeline_version": "fused_v2",
    }
    t0 = time.time()
    resp = call(payload)
    dt = time.time() - t0
    print(f"  success={resp.get('success')}  method={resp.get('method_used')}  "
          f"elapsed={resp.get('elapsed_ms')}ms  wall={dt:.1f}s")
    if resp.get("diagnostics"):
        diag = resp["diagnostics"]
        print(f"  rendered={diag.get('renderedGarments')}  order={diag.get('dressingOrder')}")
        print(f"  ipAdapterScale={diag.get('ipAdapterScale')}  ipAdapterScalePerLabel={diag.get('ipAdapterScalePerLabel')}")
        print(f"  colorMatchStrength={diag.get('colorMatchStrength')}")
        for cd in diag.get("colorDiagnostics", []):
            tgt = cd["targetRgb"]; rnd = cd["renderedRgb"]; d = cd["delta"]
            print(f"  {cd['label']:5s}: target={tgt}  rendered={rnd}  delta=({d['r']:+d},{d['g']:+d},{d['b']:+d})")
    save_result(resp, f"test_new_code_seed{seed}.png")
