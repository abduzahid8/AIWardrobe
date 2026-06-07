"""
Side-by-side comparison:
  [mannequin]  [t-shirt input]  [pants input]  [shoes input]  [tryon result]

Run multiple seeds and produce one composite per seed.
"""
import base64
import json
import time
import urllib.request
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets" / "images"
OUT_DIR = ROOT / "mobile-vton-service"
ENDPOINT = "https://karimdzanovzoha--aiwardrobe-mobile-vton-fastapi-app.modal.run/tryon/multi-fused"

CELL = 384
PAD = 8
LABEL_H = 24

def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()

def call(payload: dict) -> dict:
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())

def fit(img: Image.Image) -> Image.Image:
    return img.convert("RGB").resize((CELL, CELL), Image.LANCZOS)

mannequin = fit(Image.open(ASSETS / "mannequin_front.png"))
tshirt    = fit(Image.open(ASSETS / "basic_white_tshirt.png"))
pants     = fit(Image.open(ASSETS / "basic_brown_pants.png"))
shoes     = fit(Image.open(ASSETS / "basic_brown_loafers.png"))

cols_in = [("mannequin", mannequin), ("t-shirt", tshirt), ("pants", pants), ("shoes", shoes)]

def make_composite(result_img: Image.Image, out_path: Path, seed: int):
    n = 5
    W = CELL * n + PAD * (n + 1)
    H = CELL + PAD * 2 + LABEL_H
    strip = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(strip)
    cols = cols_in + [(f"tryon (seed {seed})", fit(result_img))]
    for i, (label, img) in enumerate(cols):
        x = PAD + i * (CELL + PAD)
        y = LABEL_H
        strip.paste(img, (x, y))
        draw.text((x + 6, 4), label, fill="black")
    strip.save(out_path)
    print(f"Wrote {out_path}")

for seed in (7, 100, 777):
    print(f"--- seed {seed} ---")
    payload = {
        "person_image": b64(ASSETS / "mannequin_front.png"),
        "garments": [
            {"label": "top",   "garment_image": b64(ASSETS / "basic_white_tshirt.png")},
            {"label": "pants", "garment_image": b64(ASSETS / "basic_brown_pants.png")},
            {"label": "shoes", "garment_image": b64(ASSETS / "basic_brown_loafers.png")},
        ],
        "seed": seed,
        "guidance_scale": 7.5,
        "num_steps": 25,
    }
    t0 = time.time()
    resp = call(payload)
    dt = time.time() - t0
    if not resp.get("success"):
        print("  FAILED:", resp)
        continue
    img_b64 = resp["result_image"]
    if img_b64.startswith("data:"):
        img_b64 = img_b64.split(",", 1)[1]
    result_img = Image.open(__import__("io").BytesIO(base64.b64decode(img_b64)))
    print(f"  ok in {dt:.1f}s")
    make_composite(result_img, OUT_DIR / f"test_input_vs_output_seed{seed}.png", seed)
