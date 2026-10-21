#!/usr/bin/env python3
"""
Regenerate splash-screen assets with the W mark optically centered.

Produces:
  assets/splash-logo.png                                  (1024x1024, transparent, centered mark)
  assets/splash-icon.png                                  (full-bleed, brand bg, centered mark, no border)
  ios/.../SplashScreenLegacy.imageset/image{,@2x,@3x}.png (square transparent logo)
  android/.../drawable-{m,h,xh,xxh,xxxh}dpi/splash_logo.png (tightly cropped mark, 128dp wide)

Source mark is extracted from the existing splash-icon.png (navy W on light blue).
Run from repo root: python3 scripts/generate-splash-assets.py
"""

import os
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "assets", "splash-icon.png")

BRAND_BG = (173, 216, 230)  # #ADD8E6 — matches app.json / iOS colorset
NAVY = (16, 40, 82)  # #102852 — sampled mark color

LOGO_PX = 1024  # square canvas for splash-logo.png + iOS imageset
MARK_WIDTH_RATIO = 0.56  # mark width as fraction of canvas (matches iOS storyboard box)
ANDROID_MARK_DP = 128  # on-screen mark width on Android


def extract_mark(src_path):
    """Cut the navy W out of the legacy splash image as (RGBA, bbox)."""
    im = Image.open(src_path).convert("RGB")
    a = np.array(im).astype(np.float64)
    h, w = a.shape[:2]
    bg = np.array(Image.open(src_path).convert("RGB"))[100, 100].astype(np.float64)

    # Navy mark mask on the interior (ignore the legacy 1-6px black border).
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mask = (r < 80) & (g < 100) & (b < 160)
    mask[:10, :] = False
    mask[-10:, :] = False
    mask[:, :10] = False
    mask[:, -10:] = False

    ys, xs = np.where(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()

    # Anti-aliased cutout: derive alpha from how far each pixel is from the
    # flat background toward the mark color.
    pad = 6
    crop = a[max(0, y0 - pad) : y1 + pad + 1, max(0, x0 - pad) : x1 + pad + 1]
    bgv = bg.reshape(1, 1, 3)
    fgv = np.array(NAVY, dtype=np.float64).reshape(1, 1, 3)
    denom = np.where(np.abs(bgv - fgv) < 1e-6, 1e-6, bgv - fgv)
    alpha = np.clip(np.mean((bgv - crop) / denom, axis=2), 0.0, 1.0)

    rgba = np.zeros((*alpha.shape, 4), dtype=np.uint8)
    rgba[..., 0] = NAVY[0]
    rgba[..., 1] = NAVY[1]
    rgba[..., 2] = NAVY[2]
    rgba[..., 3] = (alpha * 255).round().astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def scale_mark(mark, target_w):
    w, h = mark.size
    target_h = max(1, round(h * target_w / w))
    return mark.resize((target_w, target_h), Image.LANCZOS)


def center_on(mark, canvas_w, canvas_h, fill=None):
    canvas = Image.new("RGBA", (canvas_w, canvas_h), fill if fill else (0, 0, 0, 0))
    x = (canvas_w - mark.size[0]) // 2
    y = (canvas_h - mark.size[1]) // 2
    canvas.paste(mark, (x, y), mark)
    return canvas


def save(img, *parts):
    path = os.path.join(ROOT, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    print("wrote", os.path.relpath(path, ROOT), img.size)


def main():
    mark = extract_mark(SRC)
    print("mark crop:", mark.size)

    # 1. Square transparent logo — centered mark (56% of canvas width).
    square = center_on(scale_mark(mark, round(LOGO_PX * MARK_WIDTH_RATIO)), LOGO_PX, LOGO_PX)
    save(square, "assets", "splash-logo.png")

    # 2. Full-bleed splash image — brand background, centered mark, no border.
    full_w, full_h = 1284, 2138  # legacy splash-icon dimensions
    full = center_on(
        scale_mark(mark, round(full_w * MARK_WIDTH_RATIO)), full_w, full_h, fill=BRAND_BG + (255,)
    )
    save(full, "assets", "splash-icon.png")

    # 3. iOS legacy imageset — the square transparent logo at all scales.
    for name in ("image.png", "image@2x.png", "image@3x.png"):
        save(square, "ios", "AIWardrobe", "Images.xcassets", "SplashScreenLegacy.imageset", name)

    # 4. Android density buckets — tightly cropped mark at 128dp wide.
    for folder, density in (
        ("drawable-mdpi", 1.0),
        ("drawable-hdpi", 1.5),
        ("drawable-xhdpi", 2.0),
        ("drawable-xxhdpi", 3.0),
        ("drawable-xxxhdpi", 4.0),
    ):
        save(
            scale_mark(mark, round(ANDROID_MARK_DP * density)),
            "android", "app", "src", "main", "res", folder, "splash_logo.png",
        )


if __name__ == "__main__":
    main()
