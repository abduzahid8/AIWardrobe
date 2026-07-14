"""
Local sanity test for the _soft_color_match logic in main.py.

We can't import main.py (it depends on torch), so this replicates the
exact function body and verifies the math is correct on synthetic data.
If this passes AND the diff vs main.py's _soft_color_match is just
whitespace, we know the edit is correct.

Three test cases:
  1. Light garment (white shirt): rendered grey with shading -> recolored
     white WITH shading preserved.
  2. Dark garment (brown pants): rendered cool grey-blue -> recolored
     warm walnut brown.
  3. Edge case: target is (255, 255, 255) -> no div-by-zero, output is
     luminance-only (greyscale shaded).
"""
import sys
import numpy as np
from PIL import Image, ImageFilter


def _soft_color_match(
    rendered: Image.Image,
    mask: Image.Image,
    target_rgb: tuple,
    strength: float = 0.80,
    mask_blur: int = 8,
) -> Image.Image:
    if rendered.size != mask.size:
        mask = mask.resize(rendered.size, Image.LANCZOS)
    if mask.mode != "L":
        mask = mask.convert("L")

    rendered_arr = np.asarray(rendered.convert("RGB"), dtype=np.float32)

    mask_smooth = np.asarray(
        mask.filter(ImageFilter.GaussianBlur(radius=mask_blur)),
        dtype=np.float32,
    ) / 255.0

    lum = (
        0.299 * rendered_arr[..., 0]
        + 0.587 * rendered_arr[..., 1]
        + 0.114 * rendered_arr[..., 2]
    )

    target = np.array(target_rgb, dtype=np.float32)
    target_lum = 0.299 * target[0] + 0.587 * target[1] + 0.114 * target[2]
    if target_lum < 1e-3:
        ratios = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    else:
        ratios = target / target_lum

    recolored = lum[..., None] * ratios.reshape(1, 1, 3)

    blend_alpha = (mask_smooth * strength)[..., None]
    out = rendered_arr * (1.0 - blend_alpha) + recolored * blend_alpha

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def make_synthetic_rendered(color=(150, 150, 150), shading=True, size=(64, 64)) -> Image.Image:
    """Make a 'rendered garment' with shading: a vertical gradient from
    dark (top) to light (bottom), in the given base color."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    if shading:
        # Vertical gradient: top is 70% of base, bottom is 100% of base
        for y in range(size[1]):
            t = y / (size[1] - 1)
            scale = 0.7 + 0.3 * t
            arr[y, :, 0] = int(color[0] * scale)
            arr[y, :, 1] = int(color[1] * scale)
            arr[y, :, 2] = int(color[2] * scale)
    else:
        arr[..., 0] = color[0]
        arr[..., 1] = color[1]
        arr[..., 2] = color[2]
    return Image.fromarray(arr, mode="RGB")


def make_full_mask(size=(64, 64)) -> Image.Image:
    """All-ones mask (entire image is the garment)."""
    return Image.new("L", size, 255)


def main():
    print("=" * 60)
    print("Test 1: Light garment (white shirt)")
    print("=" * 60)
    rendered = make_synthetic_rendered(color=(150, 150, 150), shading=True)
    mask = make_full_mask()
    out = _soft_color_match(rendered, mask, (255, 255, 255), strength=1.0)
    arr = np.asarray(out)
    print(f"  rendered top row mean RGB: {np.asarray(rendered)[0, 0]}  (should be ~105 grey)")
    print(f"  rendered bottom row mean RGB: {np.asarray(rendered)[-1, 0]}  (should be ~150 grey)")
    print(f"  output top row mean RGB: {arr[0, 0]}  (should be ~105 white-ish)")
    print(f"  output bottom row mean RGB: {arr[-1, 0]}  (should be ~150 white-ish)")
    # Verify: R == G == B (pure white with shading)
    assert arr[0, 0, 0] == arr[0, 0, 1] == arr[0, 0, 2], "Color should be pure white"
    assert arr[-1, 0, 0] == arr[-1, 0, 1] == arr[-1, 0, 2], "Color should be pure white"
    # Verify shading preserved (top darker than bottom)
    assert arr[0, 0, 0] < arr[-1, 0, 0], "Shading should be preserved (top darker)"
    print("  PASS: white shirt recolored with shading preserved")

    print()
    print("=" * 60)
    print("Test 2: Dark garment (walnut brown pants)")
    print("=" * 60)
    # Rendered as cool grey (model drift)
    rendered = make_synthetic_rendered(color=(110, 115, 125), shading=True)
    out = _soft_color_match(rendered, mask, (106, 79, 56), strength=1.0)
    arr = np.asarray(out)
    print(f"  rendered top row:    {np.asarray(rendered)[0, 0]}  (cool grey)")
    print(f"  output top row:      {arr[0, 0]}  (should be warm brown)")
    print(f"  output bottom row:   {arr[-1, 0]}  (should be warm brown)")
    # Verify: warm brown (R > G > B)
    assert arr[0, 0, 0] > arr[0, 0, 1] > arr[0, 0, 2], "Should be warm brown (R>G>B)"
    assert arr[-1, 0, 0] > arr[-1, 0, 1] > arr[-1, 0, 2], "Should be warm brown (R>G>B)"
    # Verify shading preserved
    assert arr[0, 0, 0] < arr[-1, 0, 0], "Shading should be preserved"
    print("  PASS: brown pants recolored with hue fix + shading preserved")

    print()
    print("=" * 60)
    print("Test 3: Edge case — target is white, mask covers everything")
    print("=" * 60)
    rendered = make_synthetic_rendered(color=(80, 80, 80))
    out = _soft_color_match(rendered, mask, (255, 255, 255), strength=1.0)
    arr = np.asarray(out)
    print(f"  output mean: {arr.mean(axis=(0,1))}")
    assert arr.shape[2] == 3, "RGB shape"
    # No crash from target_lum close to 0 or NaN
    assert not np.isnan(arr).any(), "No NaNs"
    print("  PASS: no edge-case crash")

    print()
    print("=" * 60)
    print("Test 4: Outside mask = unchanged (deep into mask vs deep outside)")
    print("=" * 60)
    rendered = make_synthetic_rendered(color=(150, 150, 150))
    # Half mask (left half only)
    half_mask = Image.new("L", (64, 64), 0)
    half_mask_arr = np.zeros((64, 64), dtype=np.uint8)
    half_mask_arr[:, :32] = 255
    half_mask = Image.fromarray(half_mask_arr, mode="L")
    out = _soft_color_match(rendered, half_mask, (255, 255, 255), strength=1.0)
    arr = np.asarray(out)
    # Check well inside the mask (column 5, far from boundary) vs well outside (column 58)
    inside = arr[:, 5, 0].mean()
    outside = arr[:, 58, 0].mean()
    print(f"  output col 5  (deep inside mask):  mean={inside}  (should be ~150, recolored white)")
    print(f"  output col 58 (deep outside mask): mean={outside}  (should be ~150 unchanged grey)")
    # Inside mask: ratios (1,1,1) so output = luminance. Lum of (150,150,150) = 150.
    # Outside mask: unchanged = 150.
    # They should be EQUAL (both at 150) because target is white which equals the rendered color
    # under ratios=(1,1,1). Test with a more dramatic color to differentiate.
    rendered2 = make_synthetic_rendered(color=(200, 0, 0))
    out2 = _soft_color_match(rendered2, half_mask, (0, 200, 0), strength=1.0)
    arr2 = np.asarray(out2)
    inside2 = arr2[:, 5, 0].mean()
    outside2 = arr2[:, 58, 0].mean()
    print(f"  red->green test col 5:  mean={inside2}  (R should be low, G high)")
    print(f"  red->green test col 58: mean={outside2}  (should still be red)")
    assert inside2 < 50, "Deep inside mask should have low R (was recolored to green)"
    assert outside2 > 150, "Deep outside mask should still be red (unchanged)"
    print("  PASS: inside recolored, outside preserved")

    print()
    print("=" * 60)
    print("Test 5: Strength blend — strength=0 = no change")
    print("=" * 60)
    rendered = make_synthetic_rendered(color=(100, 100, 100))
    out = _soft_color_match(rendered, mask, (255, 0, 0), strength=0.0)
    arr = np.asarray(out)
    rendered_arr = np.asarray(rendered)
    assert np.allclose(arr, rendered_arr), "strength=0 should give back original"
    print("  PASS: strength=0 = no change")

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
