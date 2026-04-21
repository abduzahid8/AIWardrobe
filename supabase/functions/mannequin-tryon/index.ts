import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

// =============================================================================
// PATH A — NVIDIA FLUX.1-Kontext-dev
// Correct endpoint per NVIDIA NIM docs (underscore in model id)
// =============================================================================
const NVIDIA_URL = 'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux_1-kontext-dev'

// Prompt for composite image (mannequin LEFT | garment RIGHT)
function buildCompositeTryOnPrompt(garmentType: string): string {
  const base =
    'This image contains two panels side by side: LEFT = a mannequin, RIGHT = a garment item. ' +
    'Your task: dress the mannequin in the LEFT panel with the exact garment shown in the RIGHT panel. ' +
    'Preserve the mannequin\'s pose, body proportions, skin tone, and background exactly. ' +
    'Only replace the relevant clothing region. Do NOT include the right panel in the output — show only the dressed mannequin. ' +
    'Professional studio lighting, clean white background.'
  if (garmentType === 'lower_body') return base + ' The garment is trousers/pants — apply to the lower body only.'
  if (garmentType === 'dresses')    return base + ' The garment is a dress — replace the full outfit.'
  if (garmentType === 'shoes')      return base + ' The garment is footwear — replace the shoes only.'
  if (garmentType === 'outfit')     return base + ' This is a complete outfit — replace all clothing.'
  return base + ' The garment is an upper-body item — apply to the torso only.'
}


function stripDataUri(s: string): string {
  return s.startsWith('data:') ? (s.split(',')[1] ?? s) : s
}

// =============================================================================
// COMPOSITE — side-by-side image (mannequin left | garment right) for NVIDIA
// =============================================================================
async function createCompositeB64(mannequinB64: string, garmentB64: string): Promise<string> {
  const toBytes = (b64: string) =>
    Uint8Array.from(atob(stripDataUri(b64)), (c) => c.charCodeAt(0))

  const [mannBitmap, garmBitmap] = await Promise.all([
    createImageBitmap(new Blob([toBytes(mannequinB64)], { type: 'image/png' })),
    createImageBitmap(new Blob([toBytes(garmentB64)], { type: 'image/png' })),
  ])

  const h = mannBitmap.height
  const garmW = Math.round(garmBitmap.width * (h / garmBitmap.height))
  const divider = 4

  const canvas = new OffscreenCanvas(mannBitmap.width + divider + garmW, h)
  const ctx = canvas.getContext('2d')!
  ctx.fillStyle = '#ffffff'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  ctx.drawImage(mannBitmap, 0, 0)
  ctx.drawImage(garmBitmap, mannBitmap.width + divider, 0, garmW, h)

  const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.92 })
  const ab = await blob.arrayBuffer()
  return `data:image/jpeg;base64,${btoa(String.fromCharCode(...new Uint8Array(ab)))}`
}

async function callNvidiaKontext(
  nvidiaKey: string,
  mannequinImage: string,
  garmentImage: string,
  garmentType: string,
): Promise<{ result: string | null; error: string }> {
  try {
    // Build composite: mannequin left | garment right
    let compositeB64: string
    let prompt: string
    try {
      compositeB64 = await createCompositeB64(mannequinImage, garmentImage)
      prompt = buildCompositeTryOnPrompt(garmentType)
      console.log('NVIDIA: composite image created')
    } catch (e) {
      console.warn('NVIDIA: composite failed, using mannequin only:', e)
      compositeB64 = mannequinImage
      prompt = buildCompositeTryOnPrompt(garmentType)
    }

    // Convert data URI to Blob for multipart upload
    const b64Data = stripDataUri(compositeB64)
    const mime = compositeB64.startsWith('data:image/jpeg') ? 'image/jpeg' : 'image/png'
    const bytes = Uint8Array.from(atob(b64Data), (c) => c.charCodeAt(0))
    const imageBlob = new Blob([bytes], { type: mime })

    const form = new FormData()
    form.append('prompt', prompt)
    form.append('image', imageBlob, 'composite.jpg')
    form.append('width', '800')
    form.append('height', '1328')
    form.append('steps', '28')
    form.append('cfg_scale', '3.5')
    form.append('seed', '42')

    const res = await fetch(NVIDIA_URL, {
      method: 'POST',
      headers: { Authorization: `Bearer ${nvidiaKey}`, Accept: 'application/json' },
      body: form,
    })

    if (!res.ok) {
      const t = await res.text()
      return { result: null, error: `NVIDIA HTTP ${res.status}: ${t.slice(0, 300)}` }
    }

    const data = await res.json()
    console.log('NVIDIA response keys:', Object.keys(data))

    if (data.artifacts?.[0]?.base64) return { result: `data:image/png;base64,${data.artifacts[0].base64}`, error: '' }
    if (data.image) {
      const img = data.image.startsWith('data:') ? data.image : `data:image/png;base64,${data.image}`
      return { result: img, error: '' }
    }
    if (data.output?.image) return { result: data.output.image, error: '' }
    if (data.b64_json) return { result: `data:image/png;base64,${data.b64_json}`, error: '' }

    return { result: null, error: `NVIDIA unexpected response: ${JSON.stringify(data).slice(0, 200)}` }
  } catch (err: any) {
    return { result: null, error: err?.message ?? String(err) }
  }
}


// =============================================================================
// Edge Function handler
// =============================================================================
serve(async (req) => {
  if (req.method === 'OPTIONS') return new Response('ok', { headers: corsHeaders })

  try {
    const supabaseAdmin = createClient(
      Deno.env.get('SB_URL') ?? Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SB_SERVICE_KEY') ?? Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? '',
    )

    const { data: nvidiaRow } = await supabaseAdmin.from('app_config').select('value').eq('key', 'nvidia_token').single()

    const NVIDIA_KEY = nvidiaRow?.value ?? Deno.env.get('NVIDIA_TOKEN') ?? Deno.env.get('NVIDIA_API_KEY')

    console.log('Config resolved:', { nvidia: !!NVIDIA_KEY })

    if (!NVIDIA_KEY) {
      return new Response(
        JSON.stringify({ success: false, error: 'No NVIDIA API key found. Add nvidia_token to app_config table or set NVIDIA_TOKEN env var.' }),
        { status: 200, headers: { ...corsHeaders, 'Content-Type': 'application/json' } },
      )
    }

    const { mannequin_image, garment_image, garment_type } = await req.json()
    if (!mannequin_image || !garment_image) {
      return new Response(
        JSON.stringify({ success: false, error: 'mannequin_image and garment_image are required' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } },
      )
    }

    const gType = garment_type || 'upper_body'

    // ── NVIDIA FLUX.1-Kontext-dev — only path ─────────────────────────────
    const { result, error } = await callNvidiaKontext(NVIDIA_KEY, mannequin_image, garment_image, gType)
    if (result) return new Response(
      JSON.stringify({ success: true, resultUrl: result, methodUsed: 'nvidia_flux_kontext' }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } },
    )

    throw new Error(`NVIDIA FLUX.1-Kontext-dev failed: ${error}`)
  } catch (err: any) {
    console.error('mannequin-tryon error:', err)
    return new Response(
      JSON.stringify({ success: false, error: err.message || 'Unknown error' }),
      { status: 200, headers: { ...corsHeaders, 'Content-Type': 'application/json' } },
    )
  }
})
