// =============================================================================
// Anatomical anchor boxes for the FIXED mannequin (assets/images/mannequin_front.png).
// Coordinates are normalized [0..1] of a 1024x1024 working canvas. They were
// measured from the reference image and replace the previous hand-tuned
// smoothstep masks. Because the mannequin never changes pose, simple boxes
// (with feathered edges) give us deterministic, debug-friendly garment regions.
// =============================================================================

export type GarmentLabel = 'top' | 'layer' | 'pants' | 'shoes'

export interface AnchorBox {
  /** left edge, normalized [0..1] */ x0: number
  /** top  edge, normalized [0..1] */ y0: number
  /** right edge, normalized [0..1] */ x1: number
  /** bottom edge, normalized [0..1] */ y1: number
  /** feather radius in pixels (at 1024 canvas) for mask edge softening */
  feather: number
}

// Box layout for assets/images/mannequin_front.png.
// Slight overlap between layer and top is intentional so a jacket can be added
// over an existing shirt without erasing the collar.
export const ANCHOR_BOXES: Record<GarmentLabel, AnchorBox> = {
  top:   { x0: 0.33, y0: 0.19, x1: 0.67, y1: 0.49, feather: 18 },
  layer: { x0: 0.28, y0: 0.18, x1: 0.72, y1: 0.53, feather: 20 },
  pants: { x0: 0.39, y0: 0.51, x1: 0.61, y1: 0.90, feather: 16 },
  shoes: { x0: 0.38, y0: 0.90, x1: 0.62, y1: 0.985, feather: 10 },
}

export interface PixelBox {
  px0: number
  py0: number
  px1: number
  py1: number
  pw: number
  ph: number
  feather: number
}

export function boxToPixels(box: AnchorBox, width: number, height: number): PixelBox {
  const px0 = Math.round(box.x0 * width)
  const py0 = Math.round(box.y0 * height)
  const px1 = Math.round(box.x1 * width)
  const py1 = Math.round(box.y1 * height)
  return {
    px0,
    py0,
    px1,
    py1,
    pw: Math.max(1, px1 - px0),
    ph: Math.max(1, py1 - py0),
    feather: box.feather,
  }
}

export function normalizeGarmentLabel(label: string, step: number): GarmentLabel {
  const l = String(label || '').toLowerCase()
  if (l === 'layer' || l === 'outerwear' || l === 'jacket' || l === 'coat') return 'layer'
  if (l === 'pants' || l === 'lower_body' || l === 'trousers' || l === 'bottom') return 'pants'
  if (l === 'shoes' || l === 'footwear') return 'shoes'
  if (l === 'top' || l === 'shirt' || l === 'tee' || l === 't-shirt') return 'top'
  if (l === 'upper_body') return step <= 1 ? 'top' : 'layer'
  return 'top'
}

/**
 * Build a feathered rectangular mask (Float32 in [0..1]) sized W x H. The mask
 * is 1.0 inside the anchor box and falls off smoothly to 0.0 over `feather`
 * pixels around the edge using a smoothstep ramp.
 */
export function featheredRectMask(box: PixelBox, width: number, height: number): Float32Array {
  const mask = new Float32Array(width * height)
  const { px0, py0, px1, py1, feather } = box
  const f = Math.max(1, feather)

  for (let y = 0; y < height; y++) {
    const dyTop = y - py0
    const dyBot = py1 - y
    const yEdge = Math.min(dyTop, dyBot)
    if (yEdge < -f) continue
    const yWeight = smoothstep01(yEdge, f)

    for (let x = 0; x < width; x++) {
      const dxLeft = x - px0
      const dxRight = px1 - x
      const xEdge = Math.min(dxLeft, dxRight)
      if (xEdge < -f) continue
      const xWeight = smoothstep01(xEdge, f)
      const w = Math.min(yWeight, xWeight)
      if (w > 0) mask[y * width + x] = w
    }
  }

  return mask
}

function smoothstep01(distance: number, feather: number): number {
  if (distance >= 0) return 1
  const t = 1 + distance / feather // -feather..0 → 0..1
  if (t <= 0) return 0
  if (t >= 1) return 1
  return t * t * (3 - 2 * t)
}
