import { PNG } from 'npm:pngjs@7.0.0'
import jpeg from 'npm:jpeg-js@0.4.4'
import { Buffer } from 'node:buffer'

type Pixels = { width: number; height: number; data: Uint8Array }
const MAX_PIXELS = 6_000_000
function decode(value: string): Pixels {
  const raw = value.startsWith('data:') ? value.slice(value.indexOf(',') + 1) : value
  if (!raw || raw.length > 14 * 1024 * 1024 || raw.length % 4 === 1 || !/^[A-Za-z0-9+/]*={0,2}$/.test(raw)) throw new Error('Invalid image encoding')
  const bytes = Buffer.from(raw, 'base64')
  if (bytes.length > 10 * 1024 * 1024) throw new Error('Image exceeds 10 MB')
  if (bytes.length >= 33 && bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) {
    const width = bytes.readUInt32BE(16), height = bytes.readUInt32BE(20)
    if (!width || !height || width * height > MAX_PIXELS) throw new Error('Image resolution exceeds six megapixels')
    // pngjs bounds decompression for noninterlaced PNGs; its interlaced path
    // has no output-size bound, so do not admit that input to the Edge worker.
    if (bytes[28] !== 0) throw new Error('Interlaced PNG is unsupported')
    return PNG.sync.read(bytes)
  }
  if (bytes[0] === 255 && bytes[1] === 216) {
    return jpeg.decode(bytes, { useTArray: true, maxResolutionInMP: 6, maxMemoryUsageInMB: 96 })
  }
  throw new Error('Only PNG and JPEG images are supported')
}

/** Fit both references onto a white canvas without warping or discarding either. */
export function composeTryOnImages(person: string, garment: string, width = 1536, height = 1024): string {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 2 || width % 2 || height < 1 || width * height > 1536 * 1024) throw new Error('Invalid composite size')
  const canvas = new PNG({ width, height })
  canvas.data.fill(255)
  const paint = (source: Pixels, offset: number) => {
    const panelWidth = width / 2
    const scale = Math.min(panelWidth / source.width, height / source.height)
    const fittedWidth = Math.max(1, Math.floor(source.width * scale))
    const fittedHeight = Math.max(1, Math.floor(source.height * scale))
    const left = offset + Math.floor((panelWidth - fittedWidth) / 2)
    const top = Math.floor((height - fittedHeight) / 2)
    for (let y = 0; y < fittedHeight; y++) {
      const sy = Math.min(source.height - 1, Math.floor(y / scale))
      for (let x = 0; x < fittedWidth; x++) {
        const sx = Math.min(source.width - 1, Math.floor(x / scale))
        const input = (sy * source.width + sx) * 4
        const output = ((top + y) * width + left + x) * 4
        const alpha = source.data[input + 3] / 255
        for (let channel = 0; channel < 3; channel++) canvas.data[output + channel] = Math.round(source.data[input + channel] * alpha + 255 * (1 - alpha))
      }
    }
  }
  paint(decode(person), 0)
  paint(decode(garment), width / 2)
  return `data:image/png;base64,${PNG.sync.write(canvas, { deflateLevel: 3 }).toString('base64')}`
}
