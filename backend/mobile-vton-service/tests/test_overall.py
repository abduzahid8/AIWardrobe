"""
Build a 4-column comparison strip:
  [mannequin]  [t-shirt input]  [pants input]  [final tryon result]

Reads assets/images/*.png and the most recent test_multi_seed*.png result,
and writes a single composite for easy evaluation.
"""
import base64
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent.parent.parent
MANNEQUIN = ROOT / "assets" / "images" / "mannequin_front.png"
TSHIRT    = ROOT / "assets" / "images" / "basic_white_tshirt.png"
PANTS     = ROOT / "assets" / "images" / "basic_brown_pants.png"
SHOES     = ROOT / "assets" / "images" / "basic_brown_loafers.png"
RESULT    = ROOT / "backend" / "mobile-vton-service" / "test_multi_seed7.png"
OUT       = ROOT / "backend" / "mobile-vton-service" / "test_overall.png"

CELL = 512

def fit(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    return img.resize((CELL, CELL), Image.LANCZOS)

cols = [
    ("mannequin",  fit(Image.open(MANNEQUIN))),
    ("t-shirt",    fit(Image.open(TSHIRT))),
    ("pants",      fit(Image.open(PANTS))),
    ("tryon",      fit(Image.open(RESULT))),
]

strip = Image.new("RGB", (CELL * 4 + 30, CELL + 60), "white")
draw = ImageDraw.Draw(strip)
for i, (label, img) in enumerate(cols):
    x = 6 + i * (CELL + 6)
    strip.paste(img, (x, 50))
    draw.text((x + 8, 16), label, fill="black")

strip.save(OUT)
print(f"Wrote {OUT}")
