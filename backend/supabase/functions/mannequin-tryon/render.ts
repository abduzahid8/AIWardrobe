import { Image } from 'https://deno.land/x/imagescript@1.2.17/mod.ts'
import { boxToPixels, type AnchorBox, type GarmentLabel } from './anchors.ts'
import { decodeImage, encodePngDataUri, removeWhiteBackground } from './garmentPrep.ts'

export interface RenderResult {
  imageDataUri: string
  mask: Float32Array
  width: number
  height: number
}

function channel(pixel: number, shift: number): number {
  return (pixel >> shift) & 0xff
}

function rgba(r: number, g: number, b: number, a: number): number {
  return ((r & 0xff) << 24) | ((g & 0xff) << 16) | ((b & 0xff) << 8) | (a & 0xff)
}

function multiplyMask(mask: Float32Array, scalar: number): Float32Array {
  const out = new Float32Array(mask.length)
  for (let i = 0; i < mask.length; i++) out[i] = Math.max(0, Math.min(1, mask[i] * scalar))
  return out
}

function maxMask(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length)
  for (let i = 0; i < a.length; i++) out[i] = Math.max(a[i], b[i])
  return out
}

function subtractMask(base: Float32Array, cut: Float32Array, strength = 1): Float32Array {
  const out = new Float32Array(base.length)
  for (let i = 0; i < base.length; i++) out[i] = Math.max(0, base[i] - cut[i] * strength)
  return out
}

function featherMask(mask: Float32Array, width: number, height: number, radius: number): Float32Array {
  const out = new Float32Array(mask.length)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let best = 0
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const tx = x + dx
          const ty = y + dy
          if (tx < 0 || tx >= width || ty < 0 || ty >= height) continue
          const distance = Math.sqrt(dx * dx + dy * dy)
          if (distance > radius) continue
          const sample = mask[ty * width + tx] * (1 - distance / Math.max(1, radius))
          if (sample > best) best = sample
        }
      }
      out[y * width + x] = best
    }
  }
  return out
}

function buildPlacementMask(img: Image, offsetX: number, offsetY: number, width: number, height: number): Float32Array {
  const mask = new Float32Array(width * height)
  for (let y = 1; y <= img.height; y++) {
    for (let x = 1; x <= img.width; x++) {
      const alpha = channel(img.getPixelAt(x, y), 0)
      if (alpha <= 8) continue
      const tx = offsetX + (x - 1)
      const ty = offsetY + (y - 1)
      if (tx < 0 || tx >= width || ty < 0 || ty >= height) continue
      mask[ty * width + tx] = Math.max(mask[ty * width + tx], alpha / 255)
    }
  }
  return featherMask(mask, width, height, 4)
}

function cropToOpaqueBounds(img: Image): Image {
  let minX = img.width, minY = img.height, maxX = -1, maxY = -1
  for (let y = 1; y <= img.height; y++) {
    for (let x = 1; x <= img.width; x++) {
      if (channel(img.getPixelAt(x, y), 0) > 10) {
        if (x < minX) minX = x
        if (y < minY) minY = y
        if (x > maxX) maxX = x
        if (y > maxY) maxY = y
      }
    }
  }
  if (maxX < minX || maxY < minY) return img
  return img.crop(minX, minY, Math.max(1, maxX - minX + 1), Math.max(1, maxY - minY + 1))
}

function tintShadow(base: Image, mask: Float32Array, width: number, height: number, amount: number) {
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const m = mask[y * width + x]
      if (m <= 0.001) continue
      const px = base.getPixelAt(x + 1, y + 1)
      const shade = 1 - m * amount
      base.setPixelAt(
        x + 1,
        y + 1,
        rgba(
          Math.round(channel(px, 24) * shade),
          Math.round(channel(px, 16) * shade),
          Math.round(channel(px, 8) * shade),
          channel(px, 0),
        ),
      )
    }
  }
}

function pasteWithMask(base: Image, garment: Image, ox: number, oy: number, mask: Float32Array, width: number, height: number) {
  for (let y = 1; y <= garment.height; y++) {
    for (let x = 1; x <= garment.width; x++) {
      const tx = ox + (x - 1)
      const ty = oy + (y - 1)
      if (tx < 0 || tx >= width || ty < 0 || ty >= height) continue
      const alpha = mask[ty * width + tx]
      if (alpha <= 0.001) continue
      const src = garment.getPixelAt(x, y)
      const srcAlpha = channel(src, 0) / 255
      if (srcAlpha <= 0.01) continue
      const dst = base.getPixelAt(tx + 1, ty + 1)
      const a = Math.max(0, Math.min(1, alpha * srcAlpha))
      const inv = 1 - a
      base.setPixelAt(
        tx + 1,
        ty + 1,
        rgba(
          Math.round(channel(dst, 24) * inv + channel(src, 24) * a),
          Math.round(channel(dst, 16) * inv + channel(src, 16) * a),
          Math.round(channel(dst, 8) * inv + channel(src, 8) * a),
          255,
        ),
      )
    }
  }
}

function applyCategoryShape(mask: Float32Array, label: GarmentLabel, width: number, height: number): Float32Array {
  const out = new Float32Array(mask.length)
  for (let y = 0; y < height; y++) {
    const ny = y / Math.max(1, height - 1)
    for (let x = 0; x < width; x++) {
      const idx = y * width + x
      const nx = x / Math.max(1, width - 1)
      let weight = mask[idx]
      if (weight <= 0.001) continue

      if (label === 'top') {
        const center = 1 - Math.min(1, Math.abs(nx - 0.5) / 0.49)
        const torso = ny < 0.9 ? 1 : Math.max(0.55, 1 - (ny - 0.9) / 0.1)
        const shoulderLift = ny < 0.18 ? 0.9 + (ny / 0.18) * 0.1 : 1
        weight *= Math.max(0.72, Math.pow(center, 0.42)) * torso * shoulderLift
      } else if (label === 'layer') {
        const center = 1 - Math.min(1, Math.abs(nx - 0.5) / 0.54)
        const hem = ny < 0.94 ? 1 : Math.max(0.7, 1 - (ny - 0.94) / 0.06)
        weight *= (0.92 + center * 0.08) * hem
      } else if (label === 'pants') {
        const innerGap = nx > 0.46 && nx < 0.54 && ny > 0.16 ? 0.72 : 1
        const waist = ny < 0.12 ? 0.7 + (ny / 0.12) * 0.3 : 1
        weight *= Math.max(0.68, innerGap) * waist
      } else if (label === 'shoes') {
        const left = Math.abs(nx - 0.32) < 0.18
        const right = Math.abs(nx - 0.68) < 0.18
        weight *= left || right ? 1 : 0.08
      }

      out[idx] = Math.max(0, Math.min(1, weight))
    }
  }
  return featherMask(out, width, height, label === 'shoes' ? 3 : 4)
}

function placementForLabel(label: GarmentLabel, box: ReturnType<typeof boxToPixels>, prepared: Image) {
  if (label === 'top') {
    const scale = Math.min((box.pw * 1.2) / prepared.width, (box.ph * 1.16) / prepared.height)
    const gw = Math.max(1, Math.round(prepared.width * scale))
    const gh = Math.max(1, Math.round(prepared.height * scale))
    return { gw, gh, ox: box.px0 + Math.round((box.pw - gw) / 2), oy: box.py0 + Math.round((box.ph - gh) / 2) + 18 }
  }

  if (label === 'layer') {
    const scale = Math.min((box.pw * 1.22) / prepared.width, (box.ph * 1.2) / prepared.height)
    const gw = Math.max(1, Math.round(prepared.width * scale))
    const gh = Math.max(1, Math.round(prepared.height * scale))
    return { gw, gh, ox: box.px0 + Math.round((box.pw - gw) / 2), oy: box.py0 + Math.round((box.ph - gh) / 2) + 8 }
  }

  if (label === 'pants') {
    const scale = Math.min((box.pw * 1.28) / prepared.width, (box.ph * 1.08) / prepared.height)
    const gw = Math.max(1, Math.round(prepared.width * scale))
    const gh = Math.max(1, Math.round(prepared.height * scale))
    return { gw, gh, ox: box.px0 + Math.round((box.pw - gw) / 2), oy: box.py0 + Math.round((box.ph - gh) / 2) + 4 }
  }

  const scale = Math.min((box.pw * 1.05) / prepared.width, (box.ph * 1.05) / prepared.height)
  const gw = Math.max(1, Math.round(prepared.width * scale))
  const gh = Math.max(1, Math.round(prepared.height * scale))
  return { gw, gh, ox: box.px0 + Math.round((box.pw - gw) / 2), oy: box.py0 + Math.round((box.ph - gh) / 2) }
}

export async function renderDeterministicGarment(
  mannequinSrc: string,
  garmentSrc: string,
  anchor: AnchorBox,
  label: GarmentLabel,
): Promise<RenderResult> {
  const [mannequin, garmentRaw] = await Promise.all([
    decodeImage(mannequinSrc),
    removeWhiteBackground(garmentSrc),
  ])

  const W = 1024
  const H = 1024
  const base = mannequin.width === W && mannequin.height === H ? mannequin.clone() : mannequin.resize(W, H)
  const prepared = cropToOpaqueBounds(garmentRaw)
  const box = boxToPixels(anchor, W, H)
  const placement = placementForLabel(label, box, prepared)
  const fitted = prepared.resize(placement.gw, placement.gh)

  let garmentMask = buildPlacementMask(fitted, placement.ox, placement.oy, W, H)
  garmentMask = applyCategoryShape(garmentMask, label, W, H)

  if (label === 'layer') {
    tintShadow(base, multiplyMask(garmentMask, 0.35), W, H, 0.08)
  } else if (label === 'pants') {
    tintShadow(base, multiplyMask(garmentMask, 0.3), W, H, 0.05)
  } else if (label === 'shoes') {
    tintShadow(base, multiplyMask(garmentMask, 0.45), W, H, 0.12)
  }

  pasteWithMask(base, fitted, placement.ox, placement.oy, garmentMask, W, H)

  let outputMask = garmentMask
  if (label === 'layer') {
    const hemMask = new Float32Array(W * H)
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const idx = y * W + x
        const ny = y / Math.max(1, H - 1)
        if (garmentMask[idx] > 0 && ny > 0.72) hemMask[idx] = garmentMask[idx]
      }
    }
    outputMask = subtractMask(outputMask, featherMask(hemMask, W, H, 12), 0.45)
  }

  if (label === 'top') {
    outputMask = multiplyMask(outputMask, 1)
  }

  if (label === 'pants') {
    const waistCut = new Float32Array(W * H)
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const idx = y * W + x
        const ny = y / Math.max(1, H - 1)
        if (outputMask[idx] > 0 && ny < 0.44) waistCut[idx] = outputMask[idx]
      }
    }
    outputMask = subtractMask(outputMask, featherMask(waistCut, W, H, 8), 0.55)
  }

  if (label === 'shoes') {
    outputMask = maxMask(outputMask, multiplyMask(garmentMask, 0.95))
  }

  return {
    imageDataUri: await encodePngDataUri(base),
    mask: outputMask,
    width: W,
    height: H,
  }
}
