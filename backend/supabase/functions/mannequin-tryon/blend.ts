import { Image } from 'https://deno.land/x/imagescript@1.2.17/mod.ts'

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

function blendPixel(basePixel: number, editedPixel: number, alpha: number): number {
  const clamped = Math.max(0, Math.min(1, alpha))
  const inv = 1 - clamped
  return rgba(
    Math.round(channel(basePixel, 24) * inv + channel(editedPixel, 24) * clamped),
    Math.round(channel(basePixel, 16) * inv + channel(editedPixel, 16) * clamped),
    Math.round(channel(basePixel, 8) * inv + channel(editedPixel, 8) * clamped),
    Math.round(channel(basePixel, 0) * inv + channel(editedPixel, 0) * clamped),
  )
}

export async function decodeDataUri(src: string): Promise<Image> {
  const raw = stripDataUri(src)
  return await Image.decode(toUint8(raw))
}

export async function encodePngDataUri(img: Image): Promise<string> {
  const png = await img.encode()
  return `data:image/png;base64,${arrayBufferToBase64(png.buffer as ArrayBuffer)}`
}

export async function maskedBlend(baseDataUri: string, editedDataUri: string, mask: Float32Array, width: number, height: number): Promise<string> {
  const [baseIn, editedIn] = await Promise.all([decodeDataUri(baseDataUri), decodeDataUri(editedDataUri)])
  const base = baseIn.width === width && baseIn.height === height ? baseIn : baseIn.resize(width, height)
  const edited = editedIn.width === width && editedIn.height === height ? editedIn : editedIn.resize(width, height)
  const out = base.clone()

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const a = mask[y * width + x]
      if (a <= 0.0001) continue
      out.setPixelAt(x, y, blendPixel(base.getPixelAt(x, y), edited.getPixelAt(x, y), a))
    }
  }

  return await encodePngDataUri(out)
}
