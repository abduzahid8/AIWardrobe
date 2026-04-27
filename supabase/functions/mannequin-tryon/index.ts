// =============================================================================
// Virtual Try-On — Flux.1 Kontext-dev via NVIDIA NIM ONLY
// =============================================================================
// Auth:   Bearer ${nvidia_token} (stored in app_config)
// Endpoint: https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux-1-kontext-dev
// Mode:   Synchronous — returns base64 image in artifacts[0].base64
//
// Pipeline per garment step:
//   1) Build a 1024x1024 side-by-side composite: [ mannequin LEFT | garment RIGHT ]
//   2) POST to NVIDIA NIM with body-anchored transfer prompt.
//   3) Smart-crop the LEFT half of the result and return as data URI.
//
// Replicate and BFL direct API are NOT used.
// =============================================================================

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import { Image } from 'https://deno.land/x/imagescript@1.2.17/mod.ts'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

// Correct hosted endpoint (model name uses dot, not hyphen).
const NVIDIA_NIM_URL = 'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev'
const NVCF_ASSETS_URL = 'https://api.nvcf.nvidia.com/v2/nvcf/assets'

// =============================================================================
// Image utilities
// =============================================================================
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

const toUint8 = (b64: string): Uint8Array => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))

// Side-by-side composite: mannequin LEFT | garment RIGHT (1024x1024 PNG)
async function buildComposite(mannequinSrc: string, garmentSrc: string): Promise<string> {
  const [mann, garm] = await Promise.all([imageToBase64(mannequinSrc), imageToBase64(garmentSrc)])
  const [mannImg, garmImg] = await Promise.all([
    Image.decode(toUint8(mann.b64)),
    Image.decode(toUint8(garm.b64)),
  ])

  const W = 1024, H = 1024, half = 512
  const canvas = new Image(W, H)
  canvas.fill(0xFFFFFFFF)

  // Left panel — full-bleed mannequin
  const ms = Math.min(half / mannImg.width, H / mannImg.height)
  const mw = Math.round(mannImg.width * ms), mh = Math.round(mannImg.height * ms)
  canvas.composite(mannImg.resize(mw, mh), Math.round((half - mw) / 2), Math.round((H - mh) / 2))

  // Right panel — garment with margin
  const gs = Math.min((half - 40) / garmImg.width, (H - 80) / garmImg.height)
  const gw = Math.round(garmImg.width * gs), gh = Math.round(garmImg.height * gs)
  canvas.composite(garmImg.resize(gw, gh), half + Math.round((half - gw) / 2), Math.round((H - gh) / 2))

  // Soft vertical divider
  for (let y = 0; y < H; y++) {
    for (let dx = -1; dx <= 1; dx++) {
      canvas.setPixelAt(half + dx, y + 1, 0xCFD6E0FF)
    }
  }

  const png = await canvas.encode()
  return `data:image/png;base64,${arrayBufferToBase64(png.buffer as ArrayBuffer)}`
}

// Crop the left half of a side-by-side output (the dressed mannequin).
async function smartCropLeft(dataUri: string): Promise<string> {
  try {
    const raw = stripDataUri(dataUri)
    const img = await Image.decode(toUint8(raw))
    if (img.width >= img.height * 1.4) {
      const half = Math.floor(img.width / 2)
      const cropped = img.crop(0, 0, half, img.height)
      const png = await cropped.encode()
      return `data:image/png;base64,${arrayBufferToBase64(png.buffer as ArrayBuffer)}`
    }
    return dataUri
  } catch (e: any) {
    console.warn(`smartCropLeft passthrough: ${e.message}`)
    return dataUri
  }
}

function normalizeGarmentLabel(garmentLabel: string, step: number): string {
  return garmentLabel === 'upper_body' ? (step === 1 ? 'layer' : 'top') : garmentLabel
}

function channel(pixel: number, shift: number): number {
  return (pixel >> shift) & 0xff
}

function rgba(r: number, g: number, b: number, a: number): number {
  return ((r & 0xff) << 24) | ((g & 0xff) << 16) | ((b & 0xff) << 8) | (a & 0xff)
}

function blendPixel(basePixel: number, generatedPixel: number, alpha: number): number {
  const clamped = Math.max(0, Math.min(1, alpha))
  const inv = 1 - clamped
  return rgba(
    Math.round(channel(basePixel, 24) * inv + channel(generatedPixel, 24) * clamped),
    Math.round(channel(basePixel, 16) * inv + channel(generatedPixel, 16) * clamped),
    Math.round(channel(basePixel, 8) * inv + channel(generatedPixel, 8) * clamped),
    Math.round(channel(basePixel, 0) * inv + channel(generatedPixel, 0) * clamped),
  )
}

async function preserveReferenceRegions(
  baseMannequinDataUri: string,
  generatedDataUri: string,
  garmentLabel: string,
  step: number,
): Promise<string> {
  try {
    const normalized = normalizeGarmentLabel(garmentLabel, step)
    const baseRaw = stripDataUri(baseMannequinDataUri)
    const generatedRaw = stripDataUri(generatedDataUri)
    const [baseImg, generatedImg] = await Promise.all([
      Image.decode(toUint8(baseRaw)),
      Image.decode(toUint8(generatedRaw)),
    ])

    const W = 1024
    const H = 1024
    const fittedBase = baseImg.width === W && baseImg.height === H ? baseImg : baseImg.resize(W, H)
    const fittedGenerated = generatedImg.width === W && generatedImg.height === H ? generatedImg : generatedImg.resize(W, H)
    if (normalized !== 'pants' && normalized !== 'lower_body' && normalized !== 'shoes') {
      return generatedDataUri
    }

    const merged = fittedBase.clone()
    const blendStart = normalized === 'shoes' ? 700 : 360
    const blendEnd = normalized === 'shoes' ? 860 : 540

    for (let y = 0; y < H; y++) {
      let alpha = 0
      if (y >= blendEnd) alpha = 1
      else if (y > blendStart) alpha = (y - blendStart) / (blendEnd - blendStart)

      if (alpha <= 0) continue

      for (let x = 0; x < W; x++) {
        const basePixel = fittedBase.getPixelAt(x, y)
        const generatedPixel = fittedGenerated.getPixelAt(x, y)
        merged.setPixelAt(x, y, blendPixel(basePixel, generatedPixel, alpha))
      }
    }

    const png = await merged.encode()
    return `data:image/png;base64,${arrayBufferToBase64(png.buffer as ArrayBuffer)}`
  } catch (e: any) {
    console.warn(`preserveReferenceRegions passthrough: ${e?.message ?? e}`)
    return generatedDataUri
  }
}

// =============================================================================
// Prompt builder — body-anchored garment transfer prompt for the LEFT panel.
// =============================================================================
function buildPrompt(
  garmentLabel: string,
  step: number,
  garmentName: string,
  garmentDesc: string,
  alreadyWearing: string[],
  isComposite: boolean,
): string {
  const fullDesc = [garmentName, garmentDesc].filter(Boolean).join(' — ').trim() || 'the clothing item'
  const label = normalizeGarmentLabel(garmentLabel, step)
  const wornText = alreadyWearing.join(', ')

  let bodyAnchor = ''
  if (label === 'top') {
    bodyAnchor =
      'Fit the top onto the torso from the shoulders to the hip, with sleeves, neckline and length matching the reference. ' +
      'Do not modify the mannequin head shape, neck stump above the collar, hands below the wrists, wooden stand, hips, legs, or feet.'
  } else if (label === 'layer') {
    bodyAnchor =
      'Fit this outer layer onto the torso and arms as a jacket, coat, cardigan, hoodie, or overshirt depending on the reference. ' +
      'Keep it naturally open or closed according to the reference garment. Do not alter the pants, shoes, mannequin head shape, neck stump, hands, or wooden stand.'
  } else if (label === 'pants' || label === 'lower_body') {
    bodyAnchor =
      'Place the waistband on the natural waist and fit the pants down to the ankle, matching the reference rise, leg cut, and length. ' +
      'Do not modify anything above the waistband or the shoes/feet. The shirt, jacket, sleeves, collar, cuffs, torso silhouette, mannequin chest, arms, hands, neck stump, and all upper-body garments must remain unchanged.'
  } else if (label === 'shoes') {
    bodyAnchor =
      'Fit the shoes onto BOTH feet from the ankle down only. Do not modify the pants hem, legs, or anything above the ankles. All pants, upper-body garments, mannequin body shape, hands, neck stump, and wooden stand must remain unchanged.'
  }

  const preserve = alreadyWearing.length
    ? `The mannequin is already wearing these garments and they must stay visible and unchanged: ${wornText}. Preserve every existing garment and only edit the body zone required for the new item.`
    : ''

  const zoneGuard =
    label === 'pants' || label === 'lower_body'
      ? 'Edit only the lower body from waist to ankles. Freeze the entire upper body, headless neck stump, hands, mannequin stand, and framing exactly as they currently appear.'
      : label === 'shoes'
        ? 'Edit only the feet and shoes area. Freeze the pants, shirt, jacket, torso, sleeves, hands, mannequin stand, and framing exactly as they currently appear.'
        : label === 'layer'
          ? 'Edit only the outer layer region over the torso and arms. Keep the underlying shirt visible where appropriate. Do not change the mannequin silhouette, stand, or framing.'
          : 'Edit only the top garment region on the torso and arms. Do not change the mannequin silhouette, stand, or framing.'

  const universal =
    "Keep the mannequin's pose, body shape, shoulder width, torso width, arm position, hand shape, neck stump, wooden stand, plain white studio background, camera framing, crop, and lighting unchanged. " +
    'Do not zoom out, zoom in, widen the body, replace limbs, or regenerate the mannequin. Match the true silhouette, color, fabric, and details of the reference garment exactly.'

  if (isComposite) {
    return [
      `The input image has two panels separated by a thin vertical line. LEFT panel: a fashion mannequin. RIGHT panel: a reference garment on a white background — ${fullDesc}.`,
      `Transfer the garment from the RIGHT panel onto the mannequin in the LEFT panel.`,
      bodyAnchor,
      preserve,
      zoneGuard,
      `Output: the LEFT panel only — the dressed mannequin on a plain white background, in the same pose and framing. The RIGHT panel and divider must be removed.`,
      universal,
    ].filter(Boolean).join(' ')
  }

  return [
    step === 1
      ? `Dress this fashion mannequin in: ${fullDesc}.`
      : `Add to the mannequin: ${fullDesc}.`,
    bodyAnchor,
    preserve,
    zoneGuard,
    universal,
  ].filter(Boolean).join(' ')
}

// =============================================================================
// NVIDIA NIM Flux.1 Kontext-dev call (synchronous, with retry on 5xx/429)
//
// Protocol (verified empirically against NVIDIA's hosted NIM):
//   1. POST  https://api.nvcf.nvidia.com/v2/nvcf/assets    { contentType, description }
//      → { assetId, uploadUrl }
//   2. PUT   <uploadUrl>   <raw image bytes>               (Content-Type matches)
//   3. POST  https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev
//      Headers: Authorization: Bearer <key>, NVCF-INPUT-ASSET-REFERENCES: <assetId>
//      Body:    { prompt, image: "data:image/png;example_id,<assetId>" }
//      → { artifacts: [{ base64 }] }   (or { image })
//
// NOTE: As of this writing the hosted endpoint returns 500 Internal Server
//       Error with no body for valid requests. The flow below is correct per
//       NVIDIA's schema; see NVIDIA_SUPPORT_TICKET.md in this folder.
// =============================================================================
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

async function uploadAssetToNvcf(apiKey: string, dataUri: string): Promise<string> {
  const raw = stripDataUri(dataUri)
  const bytes = Uint8Array.from(atob(raw), (c) => c.charCodeAt(0))
  // Detect mime
  let contentType: 'image/png' | 'image/jpeg' = 'image/png'
  if (bytes[0] === 0xFF && bytes[1] === 0xD8 && bytes[2] === 0xFF) contentType = 'image/jpeg'

  const r1 = await fetch(NVCF_ASSETS_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      accept: 'application/json',
    },
    body: JSON.stringify({ contentType, description: 'mannequin-tryon-input' }),
  })
  if (!r1.ok) throw new Error(`NVCF asset create failed (${r1.status}): ${(await r1.text()).slice(0, 200)}`)
  const { assetId, uploadUrl } = await r1.json() as { assetId: string; uploadUrl: string }

  const r2 = await fetch(uploadUrl, {
    method: 'PUT',
    headers: {
      'Content-Type': contentType,
      'x-amz-meta-nvcf-asset-description': 'mannequin-tryon-input',
    },
    body: bytes,
  })
  if (!r2.ok) throw new Error(`NVCF asset PUT failed (${r2.status}): ${(await r2.text()).slice(0, 200)}`)

  return assetId
}

async function callNvidiaKontext(
  apiKey: string,
  inputImageDataUri: string,
  prompt: string,
): Promise<{ resultDataUri: string | null; error: string }> {
  // 1) Upload to NVCF assets so we can pass it as `data:<mime>;example_id,<id>`.
  let assetId: string
  try {
    assetId = await uploadAssetToNvcf(apiKey, inputImageDataUri)
  } catch (e: any) {
    return { resultDataUri: null, error: `Asset upload failed: ${e?.message ?? e}` }
  }

  // The `image` value uses a fake data URI whose payload is the *index* into
  // the comma-separated NVCF-INPUT-ASSET-REFERENCES header, NOT the asset UUID.
  // Schema: data:<mime>;example_id,<index>   (we only send one asset → index 0)
  const mime = inputImageDataUri.startsWith('data:image/jpeg') ? 'image/jpeg' : 'image/png'
  const imageRef = `data:${mime};example_id,0`
  const payload = {
    prompt,
    image: imageRef,
    aspect_ratio: 'match_input_image',
    steps: 30,
    cfg_scale: 3.5,
    seed: 42,
  }

  const maxAttempts = 4
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const res = await fetch(NVIDIA_NIM_URL, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
          Accept: 'application/json',
          'NVCF-INPUT-ASSET-REFERENCES': assetId,
        },
        body: JSON.stringify(payload),
      })

      if (res.status === 429 || res.status >= 500) {
        const retryAfterHdr = res.headers.get('retry-after')
        const baseMs = retryAfterHdr ? Number(retryAfterHdr) * 1000 : 2000 * Math.pow(2, attempt - 1)
        const wait = baseMs + Math.floor(Math.random() * 800)
        const bodyText = await res.text().catch(() => '')
        console.warn(`NVIDIA ${res.status} attempt ${attempt}/${maxAttempts}, sleeping ${wait}ms — ${bodyText.slice(0, 200)}`)
        if (attempt === maxAttempts) {
          return { resultDataUri: null, error: `NVIDIA NIM HTTP ${res.status}: ${bodyText.slice(0, 200)}` }
        }
        await sleep(wait)
        continue
      }

      if (!res.ok) {
        const t = await res.text()
        return { resultDataUri: null, error: `NVIDIA NIM HTTP ${res.status}: ${t.slice(0, 300)}` }
      }

      const data = await res.json()

      // NVIDIA returns { artifacts: [{ base64: "..." }], ... } for image gen.
      // The artifact is JPEG (header /9j/...), so detect mime from the magic bytes.
      if (data.artifacts?.[0]?.base64) {
        const b64 = data.artifacts[0].base64 as string
        const head = atob(b64.slice(0, 8))
        const outMime = (head.charCodeAt(0) === 0xFF && head.charCodeAt(1) === 0xD8) ? 'image/jpeg' : 'image/png'
        return { resultDataUri: `data:${outMime};base64,${b64}`, error: '' }
      }
      if (data.image) {
        const img = typeof data.image === 'string' ? data.image : ''
        if (img) return { resultDataUri: img.startsWith('data:') ? img : `data:image/png;base64,${img}`, error: '' }
      }
      if (data.output?.image) {
        return { resultDataUri: data.output.image, error: '' }
      }

      console.error('NVIDIA NIM unexpected response shape:', JSON.stringify(data).slice(0, 400))
      return { resultDataUri: null, error: `NVIDIA NIM unexpected response: ${JSON.stringify(data).slice(0, 200)}` }
    } catch (err: any) {
      console.warn(`callNvidiaKontext attempt ${attempt} threw:`, err?.message)
      if (attempt === maxAttempts) {
        return { resultDataUri: null, error: err?.message ?? String(err) }
      }
      await sleep(2000 * Math.pow(2, attempt - 1))
    }
  }

  return { resultDataUri: null, error: 'Unknown NVIDIA NIM failure' }
}

// =============================================================================
// Edge handler — synchronous: action=submit returns the final image directly.
// (action=poll is kept for client back-compat but always reports success.)
// =============================================================================
serve(async (req) => {
  if (req.method === 'OPTIONS') return new Response('ok', { headers: corsHeaders })

  const respond = (body: unknown, status = 200) =>
    new Response(JSON.stringify(body), {
      status,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    })

  try {
    const supabaseAdmin = createClient(
      Deno.env.get('SB_URL') ?? Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SB_SERVICE_KEY') ?? Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? '',
    )

    const { data: nvidiaRow } = await supabaseAdmin
      .from('app_config')
      .select('value')
      .eq('key', 'nvidia_token')
      .single()

    const NVIDIA_KEY = nvidiaRow?.value ?? Deno.env.get('NVIDIA_TOKEN')
    if (!NVIDIA_KEY) {
      return respond({ success: false, error: 'No nvidia_token in app_config or env.' })
    }

    const body = await req.json()
    const action: string = body.action ?? 'submit'

    // ── POLL (sync — already done; returned for client compatibility) ───────
    if (action === 'poll') {
      // The new pipeline is synchronous. If a client calls poll, we cannot
      // recover a prior result, so return a clear error.
      return respond({
        success: false,
        status: 'failed',
        error: 'Sync pipeline: results are returned in the submit response, no polling needed.',
      })
    }

    // ── SUBMIT ──────────────────────────────────────────────────────────────
    const mannequin_image: string | undefined = body.mannequin_image
    const garment_image: string | undefined = body.garment_image ?? body.garment?.image ?? body.garment?.imageUrl ?? body.garment?.url
    const garment_label: string = body.garment?.label ?? body.garment?.type ?? 'top'
    const garment_name: string = body.garment?.name ?? ''
    const garment_desc: string = body.garment?.description ?? ''
    const already_wearing: string[] = Array.isArray(body.already_wearing) ? body.already_wearing : []
    const step: number = body.step ?? 1
    const total: number = body.total ?? 1

    if (!mannequin_image) return respond({ success: false, error: 'mannequin_image is required' }, 400)
    if (!garment_image) return respond({ success: false, error: 'garment_image is required for FLUX.1-Kontext-dev try-on' }, 400)

    const inputImage = await buildComposite(mannequin_image, garment_image)
    const isComposite = true

    const prompt = buildPrompt(garment_label, step, garment_name, garment_desc, already_wearing, isComposite)
    console.log(
      `mannequin-tryon submit: step=${step}/${total} label=${garment_label} composite=${isComposite} ` +
      `prompt="${prompt.slice(0, 160)}…"`,
    )

    const { resultDataUri, error } = await callNvidiaKontext(NVIDIA_KEY, inputImage, prompt)
    if (!resultDataUri) {
      return respond({ success: false, error: error || 'NVIDIA NIM failed' })
    }

    const croppedImage = isComposite ? await smartCropLeft(resultDataUri) : resultDataUri
    const finalImage = await preserveReferenceRegions(mannequin_image, croppedImage, garment_label, step)

    return respond({
      success: true,
      mode: 'sync',
      resultUrl: finalImage,
      wasComposite: isComposite,
      methodUsed: 'nvidia_flux_kontext_dev',
      step, total, garmentLabel: garment_label,
    })
  } catch (err: any) {
    console.error('mannequin-tryon error:', err)
    return respond({ success: false, error: err?.message ?? 'Unknown error' })
  }
})
