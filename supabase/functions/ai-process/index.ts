import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import { PNG } from 'npm:pngjs@7.0.0'
import jpeg from 'npm:jpeg-js@0.4.4'
import { Buffer } from 'node:buffer'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

// NVIDIA NIM API endpoint (correct base URL from build.nvidia.com)
const NVIDIA_API_URL = 'https://integrate.api.nvidia.com/v1/chat/completions'
const NVIDIA_GROUNDING_DINO_URL = 'https://ai.api.nvidia.com/v1/cv/nvidia/nv-grounding-dino'
const NVIDIA_STATUS_URL = 'https://ai.api.nvidia.com/v1/status'

type RgbColor = {
  r: number
  g: number
  b: number
  brightness: number
}

type DetectionBox = {
  x: number
  y: number
  width: number
  height: number
  confidence: number
  phrase: string
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

function stripDataUri(value: string): string {
  return value.startsWith('data:') ? (value.split(',')[1] ?? value) : value
}

function getMimeTypeFromDataUri(value: string): string {
  const match = value.match(/^data:([^;]+);base64,/)
  return match?.[1] ?? 'image/jpeg'
}

function normalizeMimeType(value: string): 'image/png' | 'image/jpeg' {
  const mime = getMimeTypeFromDataUri(value).toLowerCase()
  return mime.includes('png') ? 'image/png' : 'image/jpeg'
}

function base64ToBytes(value: string): Uint8Array {
  return Uint8Array.from(atob(stripDataUri(value)), (c) => c.charCodeAt(0))
}

function bytesToBase64(bytes: Uint8Array): string {
  const chunkSize = 0x8000
  let binary = ''
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize))
  }
  return btoa(binary)
}

type DecodedImage = {
  width: number
  height: number
  data: Uint8ClampedArray
  mime: 'image/png' | 'image/jpeg'
}

function decodeImageDataUrl(image: string): DecodedImage {
  const mime = normalizeMimeType(image)
  const bytes = base64ToBytes(image)

  if (mime === 'image/png') {
    const png = PNG.sync.read(Buffer.from(bytes))
    return {
      width: png.width,
      height: png.height,
      data: new Uint8ClampedArray(png.data),
      mime,
    }
  }

  const decoded = jpeg.decode(bytes, { useTArray: true })
  return {
    width: decoded.width,
    height: decoded.height,
    data: new Uint8ClampedArray(decoded.data),
    mime,
  }
}

function encodePngDataUrl(width: number, height: number, data: Uint8ClampedArray): string {
  const png = new PNG({ width, height })
  png.data = Buffer.from(data)
  const encoded = PNG.sync.write(png)
  return `data:image/png;base64,${bytesToBase64(encoded)}`
}

function getBrightness(r: number, g: number, b: number): number {
  return 0.299 * r + 0.587 * g + 0.114 * b
}

function colorDistance(r: number, g: number, b: number, bg: RgbColor): number {
  const dr = r - bg.r
  const dg = g - bg.g
  const db = b - bg.b
  return Math.sqrt(dr * dr + dg * dg + db * db)
}

function median(values: number[]): number {
  if (values.length === 0) return 255
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 1
    ? sorted[mid]
    : Math.round((sorted[mid - 1] + sorted[mid]) / 2)
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function buildClothingClassificationPrompt(): string {
  return `Analyze only the primary clothing item in this image.
Ignore price tags, sale badges, screenshot UI, text overlays, icons, mannequins, hangers, hands, and background props.
If multiple objects are visible, describe only the main garment.
Return ONLY a valid JSON object:
{"category":"t-shirt|shirt|blouse|sweater|hoodie|jacket|coat|cardigan|dress|skirt|pants|jeans|shorts|sneakers|boots|sandals|bag|hat|scarf|belt|clothing","section":"tops|bottoms|dresses|outerwear|shoes|accessories|other","style":"casual|formal|sport|semi_classic|elegant|minimalist|other","color":"main color","material":"fabric or null","description":"1-2 sentence description of the garment only"}
Return valid JSON only.`
}

function parseNvidiaResponseText(responseText: string): any {
  try {
    return JSON.parse(responseText)
  } catch {
    const sseMatch = responseText.match(/data:\s*(\{[\s\S]*\})/)
    if (sseMatch) {
      try {
        return JSON.parse(sseMatch[1])
      } catch {
        // fall through to broad JSON extraction
      }
    }

    const jsonMatch = responseText.match(/\{[\s\S]*\}/)
    if (jsonMatch) {
      try {
        return JSON.parse(jsonMatch[0])
      } catch {
        return {}
      }
    }
  }

  return {}
}

function parseClassificationContent(content: string): any {
  try {
    const jsonMatch = content.match(/\{[\s\S]*\}/)
    if (jsonMatch) {
      return JSON.parse(jsonMatch[0])
    }
  } catch {
    // fall through to line parsing
  }

  const parsedData: any = {}
  const lines = content.split('\n')
  for (const line of lines) {
    if (line.toLowerCase().includes('category')) {
      parsedData.category = line.split(':')[1]?.trim().replace(/["',]/g, '') || 'clothing'
    }
    if (line.toLowerCase().includes('section')) {
      parsedData.section = line.split(':')[1]?.trim().replace(/["',]/g, '') || 'other'
    }
    if (line.toLowerCase().includes('style')) {
      parsedData.style = line.split(':')[1]?.trim().replace(/["',]/g, '') || 'casual'
    }
    if (line.toLowerCase().includes('color')) {
      parsedData.color = line.split(':')[1]?.trim().replace(/["',]/g, '') || 'various'
    }
    if (line.toLowerCase().includes('material')) {
      parsedData.material = line.split(':')[1]?.trim().replace(/["',]/g, '') || null
    }
  }
  return parsedData
}

function mapCategoryToSection(rawCategory: string, rawSection?: string): string {
  const normalizedSection = String(rawSection || '').toLowerCase().trim()
  if (['tops', 'bottoms', 'dresses', 'outerwear', 'shoes', 'accessories', 'other'].includes(normalizedSection)) {
    return normalizedSection
  }

  const labelToSection: Record<string, string> = {
    't-shirt': 'tops',
    shirt: 'tops',
    blouse: 'tops',
    sweater: 'tops',
    hoodie: 'tops',
    cardigan: 'tops',
    jacket: 'outerwear',
    coat: 'outerwear',
    dress: 'dresses',
    skirt: 'bottoms',
    pants: 'bottoms',
    trousers: 'bottoms',
    jeans: 'bottoms',
    shorts: 'bottoms',
    sneakers: 'shoes',
    boots: 'shoes',
    sandals: 'shoes',
    bag: 'accessories',
    hat: 'accessories',
    scarf: 'accessories',
    belt: 'accessories',
  }

  return labelToSection[String(rawCategory || '').toLowerCase()] || 'other'
}

async function classifyClothingWithNvidia(
  nvidiaToken: string,
  imageDataUrl: string,
): Promise<{
  classification: {
    category: string
    section: string
    confidence: number
    attributes: {
      style: string
      color?: string
      material?: string | null
    }
  }
  description: string
  errorInfo?: { status: number; body: string }
}> {
  const nvidiaResponse = await fetch(NVIDIA_API_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${nvidiaToken}`,
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify({
      model: 'nvidia/llama-3.1-nemotron-nano-vl-8b-v1',
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'image_url',
              image_url: { url: imageDataUrl },
            },
            {
              type: 'text',
              text: buildClothingClassificationPrompt(),
            },
          ],
        },
      ],
      max_tokens: 300,
      temperature: 0.1,
    }),
  })

  if (!nvidiaResponse.ok) {
    const errorText = await nvidiaResponse.text()
    console.error('NVIDIA API error:', nvidiaResponse.status, errorText.slice(0, 500))
    return {
      classification: {
        category: 'clothing',
        section: 'other',
        confidence: 0.5,
        attributes: {
          style: 'casual',
          color: 'various',
          material: null,
        },
      },
      description: 'Clothing item (AI classification unavailable)',
      errorInfo: { status: nvidiaResponse.status, body: errorText.slice(0, 200) },
    }
  }

  const responseText = await nvidiaResponse.text()
  console.log('NVIDIA raw response (first 500 chars):', responseText.slice(0, 500))

  const nvidiaData = parseNvidiaResponseText(responseText)
  const content = nvidiaData.choices?.[0]?.message?.content || ''
  const parsedData = parseClassificationContent(content)
  const category = String(parsedData.category || 'clothing')
  const section = mapCategoryToSection(category, parsedData.section)

  return {
    classification: {
      category,
      section,
      confidence: 0.88,
      attributes: {
        style: parsedData.style || 'casual',
        color: parsedData.color || 'various',
        material: parsedData.material || null,
      },
    },
    description: parsedData.description || `${parsedData.color || 'Various'} ${category}`,
  }
}

function estimateBorderBackground(
  data: Uint8ClampedArray,
  width: number,
  height: number,
): RgbColor {
  const rs: number[] = []
  const gs: number[] = []
  const bs: number[] = []
  const step = Math.max(1, Math.floor(Math.min(width, height) / 160))

  const sample = (x: number, y: number) => {
    const idx = (y * width + x) * 4
    if ((data[idx + 3] ?? 0) < 250) return
    rs.push(data[idx] ?? 255)
    gs.push(data[idx + 1] ?? 255)
    bs.push(data[idx + 2] ?? 255)
  }

  for (let x = 0; x < width; x += step) {
    sample(x, 0)
    if (height > 1) sample(x, height - 1)
  }
  for (let y = step; y < height - 1; y += step) {
    sample(0, y)
    if (width > 1) sample(width - 1, y)
  }

  const r = median(rs)
  const g = median(gs)
  const b = median(bs)
  return { r, g, b, brightness: getBrightness(r, g, b) }
}

function estimateBackgroundThreshold(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  bg: RgbColor,
): number {
  const distances: number[] = []
  const step = Math.max(1, Math.floor(Math.min(width, height) / 180))

  const sample = (x: number, y: number) => {
    const idx = (y * width + x) * 4
    if ((data[idx + 3] ?? 0) < 250) return
    distances.push(colorDistance(data[idx] ?? 255, data[idx + 1] ?? 255, data[idx + 2] ?? 255, bg))
  }

  for (let x = 0; x < width; x += step) {
    sample(x, 0)
    if (height > 1) sample(x, height - 1)
  }
  for (let y = step; y < height - 1; y += step) {
    sample(0, y)
    if (width > 1) sample(width - 1, y)
  }

  const spread = median(distances)
  const baseThreshold = spread * 3
  const brightnessFloor = bg.brightness > 220 ? 42 : bg.brightness > 180 ? 36 : 28
  return clamp(Math.round(Math.max(baseThreshold, brightnessFloor)), 24, 72)
}

function isLikelyBackgroundPixel(
  data: Uint8ClampedArray,
  pixelIndex: number,
  bg: RgbColor,
  threshold: number,
): boolean {
  const idx = pixelIndex * 4
  const alpha = data[idx + 3] ?? 0
  if (alpha < 16) return true

  const r = data[idx] ?? 255
  const g = data[idx + 1] ?? 255
  const b = data[idx + 2] ?? 255
  const pixelBrightness = getBrightness(r, g, b)

  if (bg.brightness > 180 && pixelBrightness < bg.brightness - 65) {
    return false
  }

  return colorDistance(r, g, b, bg) <= threshold
}

function buildBackgroundMask(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  bg: RgbColor,
  threshold: number,
): Uint8Array {
  const total = width * height
  const visited = new Uint8Array(total)
  const background = new Uint8Array(total)
  const queue = new Uint32Array(total)
  let head = 0
  let tail = 0

  const enqueue = (index: number) => {
    if (visited[index]) return
    visited[index] = 1
    if (!isLikelyBackgroundPixel(data, index, bg, threshold)) return
    background[index] = 1
    queue[tail++] = index
  }

  for (let x = 0; x < width; x++) {
    enqueue(x)
    if (height > 1) enqueue((height - 1) * width + x)
  }
  for (let y = 1; y < height - 1; y++) {
    enqueue(y * width)
    if (width > 1) enqueue(y * width + (width - 1))
  }

  while (head < tail) {
    const current = queue[head++]
    const x = current % width
    const y = Math.floor(current / width)

    if (x > 0) enqueue(current - 1)
    if (x < width - 1) enqueue(current + 1)
    if (y > 0) enqueue(current - width)
    if (y < height - 1) enqueue(current + width)
  }

  return background
}

type ComponentInfo = {
  id: number
  area: number
  sumX: number
  sumY: number
  minX: number
  maxX: number
  minY: number
  maxY: number
  sumBrightness: number
  brightPixels: number
}

function keepPrimaryForegroundComponents(
  data: Uint8ClampedArray,
  foreground: Uint8Array,
  width: number,
  height: number,
): Uint8Array {
  const total = width * height
  const visited = new Uint8Array(total)
  const componentMap = new Uint16Array(total)
  const queue = new Uint32Array(total)
  const components: ComponentInfo[] = []
  const minArea = Math.max(64, Math.floor(total * 0.00035))
  let nextId = 1

  for (let index = 0; index < total; index++) {
    if (!foreground[index] || visited[index]) continue

    let head = 0
    let tail = 0
    let area = 0
    let sumX = 0
    let sumY = 0
    let minX = width
    let maxX = 0
    let minY = height
    let maxY = 0
    let sumBrightness = 0
    let brightPixels = 0

    visited[index] = 1
    queue[tail++] = index

    while (head < tail) {
      const current = queue[head++]
      componentMap[current] = nextId

      const x = current % width
      const y = Math.floor(current / width)

      area += 1
      sumX += x
      sumY += y
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y

      const pixelOffset = current * 4
      const brightness = getBrightness(
        data[pixelOffset] ?? 255,
        data[pixelOffset + 1] ?? 255,
        data[pixelOffset + 2] ?? 255,
      )
      sumBrightness += brightness
      if (brightness >= 220) brightPixels += 1

      const neighbors = [
        current - 1,
        current + 1,
        current - width,
        current + width,
        current - width - 1,
        current - width + 1,
        current + width - 1,
        current + width + 1,
      ]

      for (const neighbor of neighbors) {
        if (neighbor < 0 || neighbor >= total) continue

        const nx = neighbor % width
        const ny = Math.floor(neighbor / width)
        if (Math.abs(nx - x) > 1 || Math.abs(ny - y) > 1) continue
        if (!foreground[neighbor] || visited[neighbor]) continue

        visited[neighbor] = 1
        queue[tail++] = neighbor
      }
    }

    if (area >= minArea) {
      components.push({ id: nextId, area, sumX, sumY, minX, maxX, minY, maxY, sumBrightness, brightPixels })
    }

    nextId += 1
  }

  if (components.length === 0) return foreground

  const centerX = (width - 1) / 2
  const centerY = (height - 1) / 2

  const main = components.reduce((best, component) => {
    const cx = component.sumX / component.area
    const cy = component.sumY / component.area
    const dx = (cx - centerX) / Math.max(1, width / 2)
    const dy = (cy - centerY) / Math.max(1, height / 2)
    const distancePenalty = Math.sqrt(dx * dx + dy * dy)
    const score = component.area / (1 + distancePenalty * 1.4)

    if (!best || score > best.score) {
      return { component, score }
    }
    return best
  }, null as { component: ComponentInfo; score: number } | null)?.component

  if (!main) return foreground

  const mainCenterX = main.sumX / main.area
  const mainCenterY = main.sumY / main.area
  const mainWidth = main.maxX - main.minX + 1
  const mainHeight = main.maxY - main.minY + 1
  const margin = Math.round(Math.min(width, height) * 0.04) + 6
  const keepIds = new Set<number>([main.id])

  for (const component of components) {
    if (component.id === main.id) continue

    const cx = component.sumX / component.area
    const cy = component.sumY / component.area
    const areaRatio = component.area / main.area
    const widthSpan = component.maxX - component.minX + 1
    const heightSpan = component.maxY - component.minY + 1
    const aspectRatio = Math.max(widthSpan, heightSpan) / Math.max(1, Math.min(widthSpan, heightSpan))
    const meanBrightness = component.sumBrightness / component.area
    const brightPixelRatio = component.brightPixels / component.area
    const fillRatio = component.area / Math.max(1, widthSpan * heightSpan)
    const overlapsMainBounds = !(
      component.maxX < main.minX - margin ||
      component.minX > main.maxX + margin ||
      component.maxY < main.minY - margin ||
      component.minY > main.maxY + margin
    )
    const isCentral =
      cx > width * 0.1 &&
      cx < width * 0.9 &&
      cy > height * 0.08 &&
      cy < height * 0.92
    const dx = (cx - centerX) / Math.max(1, width / 2)
    const dy = (cy - centerY) / Math.max(1, height / 2)
    const centerDistance = Math.sqrt(dx * dx + dy * dy)
    const sitsLowerThanMainCore = cy > mainCenterY + mainHeight * 0.1
    const sitsRightOfMainCore = cx > mainCenterX + mainWidth * 0.08
    const suspiciousOverlay =
      meanBrightness > 150 &&
      brightPixelRatio > 0.34 &&
      aspectRatio > 1.25 &&
      fillRatio > 0.18 &&
      areaRatio < 0.18 &&
      (centerDistance > 0.18 || (sitsLowerThanMainCore && sitsRightOfMainCore))

    if (suspiciousOverlay) {
      continue
    }

    if (overlapsMainBounds || (areaRatio >= 0.12 && isCentral) || areaRatio >= 0.35) {
      keepIds.add(component.id)
    }
  }

  const filtered = new Uint8Array(total)
  for (let i = 0; i < total; i++) {
    if (foreground[i] && keepIds.has(componentMap[i])) {
      filtered[i] = 1
    }
  }
  return filtered
}

function getGroundingPrompts(category?: string, section?: string): Array<{ prompt: string; keywords: string[] }> {
  const c = (category || '').toLowerCase().trim()
  const s = (section || '').toLowerCase().trim()

  const prompts: Array<{ prompt: string; keywords: string[] }> = []

  if (c.includes('jacket') || c.includes('coat') || s === 'outerwear') {
    prompts.push({
      prompt: 'jacket, coat, puffer jacket, outerwear, clothing item',
      keywords: ['jacket', 'coat', 'puffer', 'outerwear', 'clothing'],
    })
  } else if (c.includes('dress') || s === 'dresses') {
    prompts.push({
      prompt: 'dress, outfit, clothing item',
      keywords: ['dress', 'outfit', 'clothing'],
    })
  } else if (c.includes('pant') || c.includes('jean') || c.includes('skirt') || c.includes('short') || s === 'bottoms') {
    prompts.push({
      prompt: 'pants, jeans, skirt, shorts, trousers, clothing item',
      keywords: ['pants', 'jeans', 'skirt', 'shorts', 'trousers', 'clothing'],
    })
  } else if (c.includes('shoe') || c.includes('sneaker') || c.includes('boot') || c.includes('sandal') || s === 'shoes') {
    prompts.push({
      prompt: 'shoe, sneaker, boot, sandal, footwear',
      keywords: ['shoe', 'sneaker', 'boot', 'sandal', 'footwear'],
    })
  } else if (c.includes('bag') || c.includes('hat') || c.includes('scarf') || s === 'accessories') {
    prompts.push({
      prompt: 'bag, hat, scarf, accessory, clothing item',
      keywords: ['bag', 'hat', 'scarf', 'accessory', 'clothing'],
    })
  } else if (c || s === 'tops') {
    prompts.push({
      prompt: 'shirt, t-shirt, blouse, sweater, hoodie, top, clothing item',
      keywords: ['shirt', 't-shirt', 'blouse', 'sweater', 'hoodie', 'top', 'clothing'],
    })
  }

  prompts.push(
    {
      prompt: 'clothing item, garment, apparel',
      keywords: ['clothing', 'garment', 'apparel'],
    },
    {
      prompt: 'jacket, coat, shirt, pants, dress, skirt, jeans, shoes, bag, hat',
      keywords: ['jacket', 'coat', 'shirt', 'pants', 'dress', 'skirt', 'jeans', 'shoes', 'bag', 'hat'],
    },
  )

  return prompts
}

function getOverlayGroundingPrompts(): Array<{ prompt: string; keywords: string[] }> {
  return [
    {
      prompt: 'price tag, sale badge, discount label, sticker, product label',
      keywords: ['price', 'sale', 'discount', 'label', 'sticker', 'tag'],
    },
    {
      prompt: 'hanging tag, clothing tag, product tag, retail label',
      keywords: ['tag', 'label', 'retail'],
    },
    {
      prompt: 'text label, price label, badge, sticker',
      keywords: ['text', 'label', 'badge', 'sticker'],
    },
  ]
}

function getIntersectionArea(a: DetectionBox, b: DetectionBox): number {
  const x1 = Math.max(a.x, b.x)
  const y1 = Math.max(a.y, b.y)
  const x2 = Math.min(a.x + a.width, b.x + b.width)
  const y2 = Math.min(a.y + a.height, b.y + b.height)
  return Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
}

function isNearGarmentBox(a: DetectionBox, b: DetectionBox, margin: number): boolean {
  return !(
    a.x + a.width < b.x - margin ||
    a.x > b.x + b.width + margin ||
    a.y + a.height < b.y - margin ||
    a.y > b.y + b.height + margin
  )
}

function getDetectionBoxes(
  response: any,
  preferredKeywords: string[],
): Array<DetectionBox & { score: number }> {
  const content = response?.choices?.[0]?.message?.content
  const frameWidth = Number(content?.frameWidth || 0)
  const frameHeight = Number(content?.frameHeight || 0)
  const frameArea = Math.max(frameWidth * frameHeight, 1)
  const groups = Array.isArray(content?.boundingBoxes) ? content.boundingBoxes : []

  const detections: Array<DetectionBox & { score: number }> = []

  for (const group of groups) {
    const phrase = String(group?.phrase || '').toLowerCase()
    const bboxes = Array.isArray(group?.bboxes) ? group.bboxes : []
    const confidences = Array.isArray(group?.confidence) ? group.confidence : []

    for (let i = 0; i < bboxes.length; i++) {
      const bbox = bboxes[i]
      if (!Array.isArray(bbox) || bbox.length < 4) continue

      const [x, y, width, height] = bbox.map((value: unknown) => Number(value))
      if (![x, y, width, height].every(Number.isFinite)) continue
      if (width <= 0 || height <= 0) continue

      const confidence = Number(confidences[i] ?? 0)
      const areaRatio = (width * height) / frameArea
      const keywordMatch = preferredKeywords.some((keyword) => phrase.includes(keyword))
      const score = confidence * 2 + areaRatio * 3 + (keywordMatch ? 1.25 : 0)

      detections.push({
        x,
        y,
        width,
        height,
        confidence,
        phrase,
        score,
      })
    }
  }

  detections.sort((a, b) => b.score - a.score)
  return detections
}

function selectBestDetectionBox(
  response: any,
  preferredKeywords: string[],
): DetectionBox | null {
  const detections = getDetectionBoxes(response, preferredKeywords)
  if (detections.length === 0) return null

  const { score: _score, ...best } = detections[0]
  return best
}

function dedupeDetectionBoxes(boxes: DetectionBox[]): DetectionBox[] {
  const deduped: DetectionBox[] = []

  for (const box of boxes.sort((a, b) => b.confidence - a.confidence)) {
    const overlapsExisting = deduped.some((existing) => {
      const intersection = getIntersectionArea(existing, box)
      const minArea = Math.min(existing.width * existing.height, box.width * box.height)
      return minArea > 0 && intersection / minArea > 0.55
    })

    if (!overlapsExisting) {
      deduped.push(box)
    }
  }

  return deduped
}

async function fetchNvidiaGroundingDino(
  nvidiaToken: string,
  imageDataUrl: string,
  prompt: string,
): Promise<any | null> {
  const payload = {
    model: 'Grounding-Dino',
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: prompt },
          { type: 'media_url', media_url: { url: imageDataUrl } },
        ],
      },
    ],
    threshold: 0.22,
  }

  const response = await fetch(NVIDIA_GROUNDING_DINO_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${nvidiaToken}`,
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify(payload),
  })

  if (response.status === 200) {
    return await response.json()
  }

  if (response.status !== 202) {
    const body = await response.text()
    console.warn('NVIDIA Grounding DINO failed:', response.status, body.slice(0, 300))
    return null
  }

  const requestId = response.headers.get('NVCF-REQID')
  if (!requestId) {
    console.warn('NVIDIA Grounding DINO returned 202 without request id')
    return null
  }

  for (let attempt = 0; attempt < 8; attempt++) {
    await sleep(1200)

    const pollResponse = await fetch(`${NVIDIA_STATUS_URL}/${requestId}`, {
      headers: {
        Authorization: `Bearer ${nvidiaToken}`,
        Accept: 'application/json',
      },
    })

    if (pollResponse.status === 202) {
      continue
    }

    if (!pollResponse.ok) {
      const body = await pollResponse.text()
      console.warn('NVIDIA Grounding DINO polling failed:', pollResponse.status, body.slice(0, 300))
      return null
    }

    return await pollResponse.json()
  }

  console.warn('NVIDIA Grounding DINO polling timed out')
  return null
}

async function detectGarmentBoxWithNvidia(
  nvidiaToken: string,
  imageDataUrl: string,
  category?: string,
  section?: string,
): Promise<DetectionBox | null> {
  const prompts = getGroundingPrompts(category, section)

  for (const candidate of prompts) {
    const response = await fetchNvidiaGroundingDino(nvidiaToken, imageDataUrl, candidate.prompt)
    const detection = selectBestDetectionBox(response, candidate.keywords)
    if (detection) {
      console.log('NVIDIA Grounding DINO detection:', {
        prompt: candidate.prompt,
        phrase: detection.phrase,
        confidence: detection.confidence,
        box: [detection.x, detection.y, detection.width, detection.height],
      })
      return detection
    }
  }

  return null
}

async function detectOverlayBoxesWithNvidia(
  nvidiaToken: string,
  imageDataUrl: string,
  garmentBox?: DetectionBox | null,
): Promise<DetectionBox[]> {
  const prompts = getOverlayGroundingPrompts()
  const overlayBoxes: DetectionBox[] = []

  for (const candidate of prompts) {
    const response = await fetchNvidiaGroundingDino(nvidiaToken, imageDataUrl, candidate.prompt)
    const detections = getDetectionBoxes(response, candidate.keywords)

    for (const detection of detections) {
      const area = detection.width * detection.height
      const aspectRatio = Math.max(detection.width, detection.height) / Math.max(1, Math.min(detection.width, detection.height))
      const overlapsGarment = garmentBox ? getIntersectionArea(detection, garmentBox) > 0 : true
      const nearGarment = garmentBox ? isNearGarmentBox(detection, garmentBox, 120) : true
      const isReasonableSize = area >= 300 && area <= 120000
      const isLabelLike = aspectRatio >= 0.5 && aspectRatio <= 5

      if ((overlapsGarment || nearGarment) && isReasonableSize && isLabelLike) {
        overlayBoxes.push(detection)
      }
    }
  }

  return dedupeDetectionBoxes(overlayBoxes)
}

function expandDetectionBox(
  box: DetectionBox,
  imageWidth: number,
  imageHeight: number,
  mode: 'classification' | 'cutout' | 'overlay' = 'cutout',
): DetectionBox {
  const horizontalFactor = mode === 'classification' ? 0.12 : mode === 'overlay' ? 0.06 : 0.08
  const topFactor = mode === 'classification' ? 0.12 : mode === 'overlay' ? 0.06 : 0.08
  const bottomFactor = mode === 'classification' ? 0.08 : mode === 'overlay' ? 0.06 : 0.02
  const minHorizontalPad = mode === 'classification' ? 14 : mode === 'overlay' ? 6 : 12
  const minTopPad = mode === 'classification' ? 16 : mode === 'overlay' ? 6 : 12
  const minBottomPad = mode === 'classification' ? 10 : mode === 'overlay' ? 6 : 4
  const padLeft = Math.max(minHorizontalPad, Math.round(box.width * horizontalFactor))
  const padRight = Math.max(minHorizontalPad, Math.round(box.width * horizontalFactor))
  const padTop = Math.max(minTopPad, Math.round(box.height * topFactor))
  const padBottom = Math.max(minBottomPad, Math.round(box.height * bottomFactor))
  const x = clamp(Math.floor(box.x - padLeft), 0, Math.max(0, imageWidth - 1))
  const y = clamp(Math.floor(box.y - padTop), 0, Math.max(0, imageHeight - 1))
  const maxWidth = imageWidth - x
  const maxHeight = imageHeight - y
  const width = clamp(Math.ceil(box.width + padLeft + padRight), 1, maxWidth)
  const height = clamp(Math.ceil(box.height + padTop + padBottom), 1, maxHeight)

  return { ...box, x, y, width, height }
}

async function cropImageToDetectionBox(image: string, box?: DetectionBox | null): Promise<string | null> {
  try {
    if (!box) {
      return null
    }

    const decoded = decodeImageDataUrl(image)
    const cropBox = expandDetectionBox(box, decoded.width, decoded.height, 'classification')
    const cropped = new Uint8ClampedArray(cropBox.width * cropBox.height * 4)

    for (let y = 0; y < cropBox.height; y++) {
      for (let x = 0; x < cropBox.width; x++) {
        const srcX = cropBox.x + x
        const srcY = cropBox.y + y
        const srcOffset = (srcY * decoded.width + srcX) * 4
        const dstOffset = (y * cropBox.width + x) * 4
        cropped[dstOffset] = decoded.data[srcOffset]
        cropped[dstOffset + 1] = decoded.data[srcOffset + 1]
        cropped[dstOffset + 2] = decoded.data[srcOffset + 2]
        cropped[dstOffset + 3] = decoded.data[srcOffset + 3]
      }
    }

    return encodePngDataUrl(cropBox.width, cropBox.height, cropped)
  } catch (error) {
    console.warn('Crop image to detection box failed:', error)
    return null
  }
}

function isInsideDetectionBox(x: number, y: number, box: DetectionBox): boolean {
  return x >= box.x && x < box.x + box.width && y >= box.y && y < box.y + box.height
}

type LocalCutoutResult = {
  dataUrl: string | null
  error: string | null
}

async function createLocalStudioCutout(
  image: string,
  focusBox?: DetectionBox | null,
  overlayBoxes: DetectionBox[] = [],
): Promise<LocalCutoutResult> {
  try {
    const decoded = decodeImageDataUrl(image)
    const imageWidth = decoded.width
    const imageHeight = decoded.height
    const focusRegion = focusBox
      ? expandDetectionBox(focusBox, imageWidth, imageHeight, 'cutout')
      : null
    const overlayRegions = overlayBoxes.map((box) => expandDetectionBox(box, imageWidth, imageHeight, 'overlay'))
    const width = imageWidth
    const height = imageHeight
    const data = new Uint8ClampedArray(decoded.data)
    const bg = estimateBorderBackground(data, width, height)
    const threshold = estimateBackgroundThreshold(data, width, height, bg)
    const background = buildBackgroundMask(data, width, height, bg, threshold)
    const foreground = new Uint8Array(width * height)

    for (let i = 0; i < foreground.length; i++) {
      foreground[i] = background[i] ? 0 : 1
    }

    const filteredForeground = keepPrimaryForegroundComponents(data, foreground, width, height)

    for (let i = 0; i < filteredForeground.length; i++) {
      const x = i % width
      const y = Math.floor(i / width)
      const outsideFocusRegion = focusRegion
        ? (
          x < focusRegion.x ||
          x >= focusRegion.x + focusRegion.width ||
          y < focusRegion.y ||
          y >= focusRegion.y + focusRegion.height
        )
        : false
      const insideOverlayRegion = overlayRegions.some((box) => isInsideDetectionBox(x, y, box))

      if (!filteredForeground[i] || outsideFocusRegion || insideOverlayRegion) {
        data[i * 4 + 3] = 0
      }
    }

    return {
      dataUrl: encodePngDataUrl(width, height, data),
      error: null,
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    console.warn('Local studio cutout failed:', message)
    return {
      dataUrl: null,
      error: message,
    }
  }
}

serve(async (req) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Create Supabase client with service role (admin access)
    // Using SB_URL and SB_SERVICE_KEY instead of SUPABASE_ prefix (reserved by Supabase)
    const supabaseAdmin = createClient(
      Deno.env.get('SB_URL') ?? '',
      Deno.env.get('SB_SERVICE_KEY') ?? ''
    )

    // Get API keys from secure config table
    const { data: nvidiaConfig, error: nvidiaError } = await supabaseAdmin
      .from('app_config')
      .select('value')
      .eq('key', 'nvidia_token')
      .single()
    
    console.log('NVIDIA config query result:', { nvidiaConfig, nvidiaError, hasValue: !!nvidiaConfig?.value })
    
    const { data: replicateConfig } = await supabaseAdmin
      .from('app_config')
      .select('value')
      .eq('key', 'replicate_token')
      .single()

    const NVIDIA_TOKEN = nvidiaConfig?.value
    const REPLICATE_TOKEN = replicateConfig?.value

    console.log('Tokens loaded:', { hasNvidia: !!NVIDIA_TOKEN, hasReplicate: !!REPLICATE_TOKEN })

    if (!NVIDIA_TOKEN) {
      console.error('NVIDIA token missing from app_config')
      return new Response(
        JSON.stringify({ error: 'NVIDIA API key not configured. Add to app_config table with key "nvidia_token"' }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    const { image, operation } = await req.json()

    if (!image) {
      return new Response(
        JSON.stringify({ error: 'Image is required' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    // FIX 6: Validate image size (max 10MB base64 = ~7.5MB raw)
    // Strip data URI prefix before calculating size
    const base64Data = stripDataUri(image)
    const imageSizeBytes = (base64Data.length * 3) / 4
    if (imageSizeBytes > 10 * 1024 * 1024) {
      return new Response(
        JSON.stringify({ error: 'Image too large. Max 10MB.' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    const imageDataUrl = image.startsWith('data:') ? image : `data:image/jpeg;base64,${image}`

    const shouldClassify =
      operation === 'classify' ||
      operation === 'describe' ||
      operation === 'all' ||
      operation === 'studio_photo'
    const shouldNormalizeAngle = operation === 'remove_bg' || operation === 'all'
    const shouldRemoveBackground =
      operation === 'remove_bg' ||
      operation === 'all' ||
      operation === 'studio_photo'
    const shouldEnhanceClothing = operation === 'remove_bg' || operation === 'all'

    let result: any = {}
    let initialDetectedBox: DetectionBox | null = null
    let classificationImageDataUrl = imageDataUrl

    if (operation === 'studio_photo') {
      try {
        initialDetectedBox = await detectGarmentBoxWithNvidia(NVIDIA_TOKEN, imageDataUrl)
        const croppedClassificationImage = await cropImageToDetectionBox(imageDataUrl, initialDetectedBox)
        if (croppedClassificationImage) {
          classificationImageDataUrl = croppedClassificationImage
        }
      } catch (preDetectionError) {
        console.warn('Studio photo pre-detection failed:', preDetectionError)
      }
    }

    // Classification + Description - NVIDIA Granite Vision (FREE TIER - 10,000 requests/month)
    if (shouldClassify) {
      const classificationResult = await classifyClothingWithNvidia(
        NVIDIA_TOKEN,
        classificationImageDataUrl,
      )

      if (classificationResult.errorInfo) {
        result._nvidiaError = classificationResult.errorInfo
      }

      result.classification = classificationResult.classification
      result.description = classificationResult.description
    }

    // Angle Normalization - Convert any angle to standard flat lay view
    let normalizedImageUrl: string | null = null
    
    if (shouldNormalizeAngle && REPLICATE_TOKEN) {
      console.log('Starting angle normalization...')
      try {
        const imageDataUrl = image.startsWith('data:') ? image : `data:image/jpeg;base64,${image}`
        
        // Use image-to-image model to normalize angle to flat lay
        const normalizeResponse = await fetch('https://api.replicate.com/v1/predictions', {
          method: 'POST',
          headers: {
            'Authorization': `Token ${REPLICATE_TOKEN}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            // SDXL img2img for viewpoint transformation
            version: '39ed52f2a78e934b3ba6e2a89ff5fe015787ab9b000a95e0e5d5ee92b3455a51',
            input: {
              image: imageDataUrl,
              prompt: 'clothing item in flat lay photography style, viewed directly from above, 90 degree top-down angle, garment laid flat on surface, straight overhead perspective, no perspective distortion, parallel to camera, e-commerce product photography',
              negative_prompt: 'hanging clothes, mannequin, model wearing, side view, angled view, perspective, 3d render, tilted, folded, crumpled, shadow on clothes',
              strength: 0.55,
              num_inference_steps: 30,
              guidance_scale: 8.0,
            },
          }),
        })

        const normalizePrediction = await normalizeResponse.json()

        // Poll for normalization result
        let normalizeStatus = normalizePrediction.status
        let normalizeOutput = null
        let normalizePolls = 0
        const MAX_NORMALIZE_POLLS = 50 // Allow up to 50 seconds

        while (normalizeStatus !== 'succeeded' && normalizeStatus !== 'failed' && normalizePolls < MAX_NORMALIZE_POLLS) {
          await new Promise(r => setTimeout(r, 1000))
          const pollResponse = await fetch(`https://api.replicate.com/v1/predictions/${normalizePrediction.id}`, {
            headers: { 'Authorization': `Token ${REPLICATE_TOKEN}` },
          })
          const poll = await pollResponse.json()
          normalizeStatus = poll.status
          normalizeOutput = poll.output
          normalizePolls++
        }

        if (normalizeStatus === 'succeeded' && normalizeOutput) {
          console.log('Angle normalization completed successfully')
          normalizedImageUrl = normalizeOutput
        } else {
          console.warn(`Angle normalization ${normalizeStatus === 'failed' ? 'failed' : 'timed out'}, using original image`)
          normalizedImageUrl = imageDataUrl
        }
      } catch (normalizeError) {
        console.warn('Angle normalization error:', normalizeError)
        normalizedImageUrl = image.startsWith('data:') ? image : `data:image/jpeg;base64,${image}`
      }
    }

    // Background Removal - local studio cutout for AI Studio, Replicate rembg for legacy flows
    let processedImageUrl: string | null = null
    
    if (shouldRemoveBackground) {
      if (operation === 'studio_photo') {
        let detectedBox: DetectionBox | null = initialDetectedBox
        let overlayBoxes: DetectionBox[] = []
        let localCutoutError: string | null = null
        try {
          const refinedBox = await detectGarmentBoxWithNvidia(
            NVIDIA_TOKEN,
            imageDataUrl,
            result.classification?.category,
            result.classification?.section,
          )
          if (refinedBox) {
            detectedBox = refinedBox
          }
        } catch (groundingError) {
          console.warn('NVIDIA Grounding DINO refinement failed:', groundingError)
        }

        try {
          overlayBoxes = await detectOverlayBoxesWithNvidia(
            NVIDIA_TOKEN,
            imageDataUrl,
            detectedBox,
          )
          if (overlayBoxes.length > 0) {
            console.log('NVIDIA overlay detections:', overlayBoxes.map((box) => ({
              phrase: box.phrase,
              confidence: box.confidence,
              box: [box.x, box.y, box.width, box.height],
            })))
          }
        } catch (overlayError) {
          console.warn('NVIDIA overlay detection failed:', overlayError)
        }

        const localCutout = await createLocalStudioCutout(imageDataUrl, detectedBox, overlayBoxes)
        localCutoutError = localCutout.error
        if (localCutout.dataUrl) {
          result.cutoutUrl = localCutout.dataUrl
          processedImageUrl = localCutout.dataUrl
          console.log('✅ Local studio cutout succeeded')
        } else {
          console.warn('❌ Local studio cutout failed:', localCutoutError)
          // Fallback to original image as data URL so client has something to show
          result.cutoutUrl = imageDataUrl
          processedImageUrl = null // Signal that we need Replicate fallback
        }

        if (localCutoutError) {
          result.localCutoutError = localCutoutError
        }
      }

      // If local cutout failed and we have Replicate token, try Replicate fallback
      if (!processedImageUrl && REPLICATE_TOKEN) {
        console.log('🔄 Falling back to Replicate for background removal...')
        // Use normalized image if available, otherwise use original
        const imageToProcess = normalizedImageUrl || (image.startsWith('data:') ? image : `data:image/jpeg;base64,${image}`)
        
        const replicateResponse = await fetch('https://api.replicate.com/v1/predictions', {
          method: 'POST',
          headers: {
            'Authorization': `Token ${REPLICATE_TOKEN}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            version: 'fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003',
            input: { image: imageToProcess },
          }),
        })
        
        const prediction = await replicateResponse.json()
        
        let status = prediction.status
        let output = null
        let polls = 0
        const MAX_POLLS = 30
        
        while (status !== 'succeeded' && status !== 'failed' && polls < MAX_POLLS) {
          await new Promise(r => setTimeout(r, 1000))
          const pollResponse = await fetch(`https://api.replicate.com/v1/predictions/${prediction.id}`, {
            headers: { 'Authorization': `Token ${REPLICATE_TOKEN}` },
          })
          const poll = await pollResponse.json()
          status = poll.status
          output = poll.output
          polls++
        }
        
        if (status === 'failed' || polls >= MAX_POLLS) {
          console.warn(`Replicate bg removal ${status === 'failed' ? 'failed' : 'timed out'} after ${polls}s`)
        }
        
        result.cutoutUrl = output
        result.normalizedUrl = normalizedImageUrl // Return the normalized angle image
        processedImageUrl = output
      }
    }

    // Clothing Enhancement - "Iron" the clothes (smooth, flatten, professional look)
    if (shouldEnhanceClothing && processedImageUrl && REPLICATE_TOKEN) {
      console.log('Starting clothing enhancement (ironing)...')
      try {
        // Use image-to-image model to smooth/iron the clothing
        const enhanceResponse = await fetch('https://api.replicate.com/v1/predictions', {
          method: 'POST',
          headers: {
            'Authorization': `Token ${REPLICATE_TOKEN}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            // Using SDXL for image-to-image enhancement
            version: '39ed52f2a78e934b3ba6e2a89ff5fe015787ab9b000a95e0e5d5ee92b3455a51',
            input: {
              image: processedImageUrl,
              prompt: 'professional flat lay clothing product photo, perfectly ironed garment, smooth fabric, no wrinkles, clean edges, white background, studio lighting, high quality e-commerce photography, crisp and clean',
              negative_prompt: 'wrinkled, creased, folded, messy, shadow, dark, low quality, blurry, distorted',
              strength: 0.35,
              num_inference_steps: 25,
              guidance_scale: 7.5,
            },
          }),
        })

        const enhancePrediction = await enhanceResponse.json()

        // Poll for enhancement result
        let enhanceStatus = enhancePrediction.status
        let enhanceOutput = null
        let enhancePolls = 0
        const MAX_ENHANCE_POLLS = 45 // Allow up to 45 seconds for enhancement

        while (enhanceStatus !== 'succeeded' && enhanceStatus !== 'failed' && enhancePolls < MAX_ENHANCE_POLLS) {
          await new Promise(r => setTimeout(r, 1000))
          const pollResponse = await fetch(`https://api.replicate.com/v1/predictions/${enhancePrediction.id}`, {
            headers: { 'Authorization': `Token ${REPLICATE_TOKEN}` },
          })
          const poll = await pollResponse.json()
          enhanceStatus = poll.status
          enhanceOutput = poll.output
          enhancePolls++
        }

        if (enhanceStatus === 'succeeded' && enhanceOutput) {
          console.log('Clothing enhancement (ironing) completed successfully')
          result.enhancedUrl = enhanceOutput
          // Use enhanced image as the final cutout
          result.cutoutUrl = enhanceOutput
        } else {
          console.warn(`Clothing enhancement ${enhanceStatus === 'failed' ? 'failed' : 'timed out'}, using original cutout`)
        }
      } catch (enhanceError) {
        console.warn('Clothing enhancement error:', enhanceError)
        // Continue with original cutout if enhancement fails
      }
    }

    // Final safety check: ensure cutoutUrl is always set
    if (!result.cutoutUrl) {
      console.warn('⚠️ cutoutUrl not set, falling back to original image')
      result.cutoutUrl = imageDataUrl
    }

    console.log('✅ AI processing complete:', {
      operation,
      hasCutoutUrl: !!result.cutoutUrl,
      hasClassification: !!result.classification,
      hasEnhancedUrl: !!result.enhancedUrl,
    })

    return new Response(
      JSON.stringify({ success: true, ...result }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )

  } catch (error) {
    // FIX 5: TypeScript safe error handling (error is 'unknown' type)
    const message = error instanceof Error ? error.message : 'Internal server error'
    return new Response(
      JSON.stringify({ error: message }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )
  }
})
