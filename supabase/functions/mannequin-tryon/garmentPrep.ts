import { Image } from 'https://deno.land/x/imagescript@1.2.17/mod.ts'
import { boxToPixels, type AnchorBox } from './anchors.ts'

function stripDataUri(s: string): string {
  return s.startsWith('data:') ? (s.split(',')[1] ?? s) : s
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  let binary = ''
  const bytes = new Uint8Array(buffer)
  const chunk = 8192
  for (let i = 0; i < bytes.byteLength; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk))
  }
  return btoa(binary)
}

const toUint8 = (b64: string): Uint8Array => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))

function channel(pixel: number, shift: number): number {
  return (pixel >> shift) & 0xff
}

function rgba(r: number, g: number, b: number, a: number): number {
  return ((r & 0xff) << 24) | ((g & 0xff) << 16) | ((b & 0xff) << 8) | (a & 0xff)
}

function luminance(pixel: number): number {
  const r = channel(pixel, 24)
  const g = channel(pixel, 16)
  const b = channel(pixel, 8)
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

async function imageToBase64(src: string): Promise<{ b64: string; mime: string }> {
  let b64: string
  if (src.startsWith('data:')) {
    b64 = stripDataUri(src)
  } else {
    const res = await fetch(src)
    if (!res.ok) throw new Error(`Failed to fetch image (${res.status}): ${src.slice(0, 100)}`)
    b64 = arrayBufferToBase64(await res.arrayBuffer())
  }

  const header = atob(b64.slice(0, 16))
  const b0 = header.charCodeAt(0), b1 = header.charCodeAt(1), b2 = header.charCodeAt(2), b3 = header.charCodeAt(3)
  let mime = 'image/jpeg'
  if (b0 === 0xFF && b1 === 0xD8 && b2 === 0xFF) mime = 'image/jpeg'
  else if (b0 === 0x89 && b1 === 0x50 && b2 === 0x4E && b3 === 0x47) mime = 'image/png'
  else if (b0 === 0x52 && b1 === 0x49 && b2 === 0x46 && b3 === 0x46) mime = 'image/webp'
  return { b64, mime }
}

export async function decodeImage(src: string): Promise<Image> {
  const { b64 } = await imageToBase64(src)
  return await Image.decode(toUint8(b64))
}

export async function encodePngDataUri(img: Image): Promise<string> {
  const png = await img.encode()
  return `data:image/png;base64,${arrayBufferToBase64(png.buffer as ArrayBuffer)}`
}

export interface RoughDressResult {
  imageDataUri: string
  mask: Float32Array
  width: number
  height: number
}

export async function removeWhiteBackground(src: string): Promise<Image> {
  const img = await decodeImage(src)
  const out = img.clone()
  const w = out.width
  const h = out.height

  const bg = sampleCornerBackground(out)
  const threshold = 38

  for (let y = 1; y <= h; y++) {
    for (let x = 1; x <= w; x++) {
      const px = out.getPixelAt(x, y)
      const r = channel(px, 24)
      const g = channel(px, 16)
      const b = channel(px, 8)
      const dist = Math.max(Math.abs(r - bg.r), Math.abs(g - bg.g), Math.abs(b - bg.b))
      const lum = luminance(px)
      if (dist <= threshold && lum >= 210) {
        out.setPixelAt(x, y, rgba(r, g, b, 0))
      }
    }
  }

  trimFringeAlpha(out)
  suppressLikelyHumanPixels(out)
  keepLargestOpaqueRegion(out)
  return out
}

function sampleCornerBackground(img: Image): { r: number; g: number; b: number } {
  const clampX = (x: number) => Math.max(1, Math.min(img.width, x))
  const clampY = (y: number) => Math.max(1, Math.min(img.height, y))
  const points = [
    img.getPixelAt(clampX(4), clampY(4)),
    img.getPixelAt(clampX(img.width - 5), clampY(4)),
    img.getPixelAt(clampX(4), clampY(img.height - 5)),
    img.getPixelAt(clampX(img.width - 5), clampY(img.height - 5)),
  ]
  const avg = points.reduce((acc, px) => ({
    r: acc.r + channel(px, 24),
    g: acc.g + channel(px, 16),
    b: acc.b + channel(px, 8),
  }), { r: 0, g: 0, b: 0 })

  return { r: Math.round(avg.r / points.length), g: Math.round(avg.g / points.length), b: Math.round(avg.b / points.length) }
}

function trimFringeAlpha(img: Image): void {
  const w = img.width
  const h = img.height
  const alpha = new Uint8Array(w * h)
  for (let y = 1; y <= h; y++) {
    for (let x = 1; x <= w; x++) {
      alpha[(y - 1) * w + (x - 1)] = channel(img.getPixelAt(x, y), 0)
    }
  }

  for (let y = 2; y < h; y++) {
    for (let x = 2; x < w; x++) {
      const idx = (y - 1) * w + (x - 1)
      if (alpha[idx] === 0) continue
      let transparentNeighbors = 0
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue
          if (alpha[(y - 1 + dy) * w + (x - 1 + dx)] === 0) transparentNeighbors++
        }
      }
      if (transparentNeighbors >= 6) {
        const px = img.getPixelAt(x, y)
        img.setPixelAt(x, y, rgba(channel(px, 24), channel(px, 16), channel(px, 8), Math.round(alpha[idx] * 0.35)))
      }
    }
  }
}

function suppressLikelyHumanPixels(img: Image): void {
  for (let y = 1; y <= img.height; y++) {
    for (let x = 1; x <= img.width; x++) {
      const px = img.getPixelAt(x, y)
      const a = channel(px, 0)
      if (a <= 8) continue
      const r = channel(px, 24)
      const g = channel(px, 16)
      const b = channel(px, 8)
      const max = Math.max(r, g, b)
      const min = Math.min(r, g, b)
      const skinLike = r > 150 && g > 95 && b > 70 && r > g && g > b && max - min > 12
      const hairLike = r > 35 && r < 140 && g > 20 && g < 110 && b > 10 && b < 90 && r >= g && g >= b
      if (skinLike || hairLike) {
        img.setPixelAt(x, y, rgba(r, g, b, 0))
      }
    }
  }
}

function keepLargestOpaqueRegion(img: Image): void {
  const w = img.width
  const h = img.height
  const visited = new Uint8Array(w * h)
  let best: number[] = []

  for (let y = 1; y <= h; y++) {
    for (let x = 1; x <= w; x++) {
      const idx = (y - 1) * w + (x - 1)
      if (visited[idx]) continue
      visited[idx] = 1
      if (channel(img.getPixelAt(x, y), 0) <= 12) continue

      const queue = [idx]
      const region: number[] = [idx]
      for (let qi = 0; qi < queue.length; qi++) {
        const current = queue[qi]
        const cx = current % w
        const cy = Math.floor(current / w)
        const neighbors = [
          [cx - 1, cy],
          [cx + 1, cy],
          [cx, cy - 1],
          [cx, cy + 1],
        ]
        for (const [nx, ny] of neighbors) {
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue
          const ni = ny * w + nx
          if (visited[ni]) continue
          visited[ni] = 1
          if (channel(img.getPixelAt(nx + 1, ny + 1), 0) <= 12) continue
          queue.push(ni)
          region.push(ni)
        }
      }

      if (region.length > best.length) best = region
    }
  }

  if (!best.length) return
  const keep = new Uint8Array(w * h)
  for (const idx of best) keep[idx] = 1
  for (let y = 1; y <= h; y++) {
    for (let x = 1; x <= w; x++) {
      const idx = (y - 1) * w + (x - 1)
      if (keep[idx]) continue
      const px = img.getPixelAt(x, y)
      if (channel(px, 0) > 0) {
        img.setPixelAt(x, y, rgba(channel(px, 24), channel(px, 16), channel(px, 8), 0))
      }
    }
  }
}

export async function buildRoughDressedImage(mannequinSrc: string, garmentSrc: string, anchor: AnchorBox): Promise<RoughDressResult> {
  const [mannequin, garment] = await Promise.all([
    decodeImage(mannequinSrc),
    removeWhiteBackground(garmentSrc),
  ])

  const W = 1024
  const H = 1024
  const base = mannequin.width === W && mannequin.height === H ? mannequin.clone() : mannequin.resize(W, H)
  const prepared = cropToOpaqueBounds(garment)
  const box = boxToPixels(anchor, W, H)

  const scale = Math.min(box.pw / prepared.width, box.ph / prepared.height)
  const gw = Math.max(1, Math.round(prepared.width * scale))
  const gh = Math.max(1, Math.round(prepared.height * scale))
  const fitted = prepared.resize(gw, gh)

  const ox = box.px0 + Math.round((box.pw - gw) / 2)
  const oy = box.py0 + Math.round((box.ph - gh) / 2)
  base.composite(fitted, ox, oy)

  const garmentMask = buildPlacementMask(fitted, ox, oy, W, H)

  return {
    imageDataUri: await encodePngDataUri(base),
    mask: garmentMask,
    width: W,
    height: H,
  }
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

  return featherMask(mask, width, height, 6)
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
          const weight = 1 - distance / Math.max(1, radius)
          const sample = mask[ty * width + tx] * weight
          if (sample > best) best = sample
        }
      }
      out[y * width + x] = best
    }
  }
  return out
}
