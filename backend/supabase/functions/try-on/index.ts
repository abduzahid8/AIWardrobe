import { composeTryOnImages } from '../_shared/tryOnComposite.ts'
import { enforceRateLimit } from '../_shared/rateLimit.ts'
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

// ── NVIDIA FLUX.1-Kontext-dev NIM endpoint ────────────────────────────────
const NVIDIA_NIM_URL =
    'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev'

// ── Helpers ───────────────────────────────────────────────────────────────

function stripDataUri(dataUri: string): string {
    if (dataUri.startsWith('data:')) return dataUri.split(',')[1] ?? dataUri
    return dataUri
}

function buildPersonTryOnPrompt(garmentType: string): string {
    const base =
        'The left half of this image shows a person and the right half shows a garment. ' +
        'Dress the person in the exact garment from the right half. ' +
        'Keep the person\'s face, hair, pose, and background identical. ' +
        'CRITICAL: Preserve and keep all existing clothing the person is already wearing (e.g., pants, shoes) UNLESS it is being explicitly replaced by the new garment. Do not generate a black screen. ' +
        'Output ONLY the person wearing the new garment combined with their existing outfit, photorealistic, high resolution.'

    if (garmentType === 'lower_body') {
        return base + ' The garment is pants / trousers — apply it to the lower body.'
    }
    if (garmentType === 'dresses') {
        return base + ' The garment is a dress — apply it as a full-body outfit.'
    }
    return base + ' The garment is an upper-body item (shirt, jacket, etc.).'
}

// ── NVIDIA FLUX.1-Kontext-dev call ────────────────────────────────────────
async function callNvidiaKontext(
    nvidiaKey: string,
    personImage: string,
    garmentImage: string,
    garmentType: string,
): Promise<string | null> {
    try {
        const compositeDataUri = composeTryOnImages(personImage, garmentImage)
        const imagePayload = compositeDataUri.startsWith('data:')
            ? compositeDataUri
            : `data:image/png;base64,${stripDataUri(compositeDataUri)}`

        const res = await fetch(NVIDIA_NIM_URL, {
            signal: AbortSignal.timeout(45000),
            method: 'POST',
            headers: {
                Authorization: `Bearer ${nvidiaKey}`,
                'Content-Type': 'application/json',
                Accept: 'application/json',
            },
            body: JSON.stringify({
                prompt: buildPersonTryOnPrompt(garmentType),
                image: imagePayload,
                cfg_scale: 4.0,
                steps: 30,
                samples: 1,
                seed: 42,
                height: 1024,
                width: 768,
            }),
        })

        if (!res.ok) {
            console.error('NVIDIA NIM error:', res.status)
            return null
        }

        const data = await res.json()
        if (data.artifacts?.[0]?.base64) return `data:image/png;base64,${data.artifacts[0].base64}`
        if (data.image) return data.image.startsWith('data:') ? data.image : `data:image/png;base64,${data.image}`
        if (data.output?.image) return data.output.image

        console.error('NVIDIA NIM response missing image')
        return null
    } catch (err) {
        console.error('NVIDIA NIM call failed')
        return null
    }
}

// ── Replicate IDM-VTON call (fallback) ────────────────────────────────────
async function callReplicateIDMVTON(
    replicateToken: string,
    personImage: string,
    garmentImage: string,
    garmentType: string,
): Promise<string> {
    const category = garmentType === 'lower_body' ? 'lower_body' : 'upper_body'
    const deadline = AbortSignal.timeout(90000)
    const response = await fetch("https://api.replicate.com/v1/predictions", {
        signal: deadline,
        method: "POST",
        headers: {
            "Authorization": `Token ${replicateToken}`,
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            version: "0513734a452173b8173e907e3a59d19a36266e55b48528559432bd21c7d7e985",
            input: {
                human_img: personImage,
                garm_img: garmentImage,
                garment_des: "clothing",
                category,
                n_samples: 1,
                seed: 42,
            },
        }),
    })

    if (!response.ok) throw new Error("Replicate API Error: " + await response.text())

    let result = await response.json()
    for (let i = 0; i < 40; i++) {
        if (result.status === 'succeeded' || result.status === 'failed') break
        await new Promise(r => setTimeout(r, 2000))
        const pollRes = await fetch(result.urls.get, {
            signal: deadline,
            headers: { "Authorization": `Token ${replicateToken}` },
        })
        result = await pollRes.json()
    }

    if (result.status === "failed") throw new Error("Replicate prediction failed: " + (result.error || ''))
    const outputImage = Array.isArray(result.output) ? result.output[0] : result.output
    if (!outputImage) throw new Error('No output from Replicate')
    return outputImage
}

// ── Edge Function ─────────────────────────────────────────────────────────
serve(async (req) => {
    if (req.method === 'OPTIONS') {
        return new Response('ok', { headers: corsHeaders })
    }

    if (req.method !== 'POST') return new Response(JSON.stringify({success:false,error:'Method not allowed'}), {status:405,headers:{...corsHeaders,'Content-Type':'application/json'}})
    const rateLimitResponse = await enforceRateLimit(req, { feature: 'try-on', limit: 5, headers: corsHeaders })
    if (rateLimitResponse) return rateLimitResponse


    try {
        // Authenticated client — verify the caller
        const supabaseClient = createClient(
            Deno.env.get('SUPABASE_URL') ?? '',
            Deno.env.get('SUPABASE_ANON_KEY') ?? '',
            { global: { headers: { Authorization: req.headers.get('Authorization')! } } }
        )

        const { data: { user } } = await supabaseClient.auth.getUser()
        if (!user) return new Response(JSON.stringify({success:false,error:'Unauthorized'}),{status:401,headers:{...corsHeaders,'Content-Type':'application/json'}})

        const {person_image,garment_image,garment_type='upper_body'}=await req.json()
        if (![person_image,garment_image].every(image=>typeof image==='string'&&image.trim().length>0&&image.length<=14*1024*1024)||!['upper_body','lower_body','dresses'].includes(garment_type)) {
            return new Response(JSON.stringify({success:false,error:'Valid images and garment type are required'}),{status:400,headers:{...corsHeaders,'Content-Type':'application/json'}})
        }

        // Service-role client to read app_config securely
        const supabaseAdmin = createClient(
            Deno.env.get('SB_URL') ?? Deno.env.get('SUPABASE_URL') ?? '',
            Deno.env.get('SB_SERVICE_KEY') ?? Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? '',
        )

        const [{ data: nvidiaRow }, { data: replicateRow }] = await Promise.all([
            supabaseAdmin.from('app_config').select('value').eq('key', 'nvidia_token').single(),
            supabaseAdmin.from('app_config').select('value').eq('key', 'replicate_token').single(),
        ])

        const NVIDIA_KEY = nvidiaRow?.value
        const REPLICATE_TOKEN = replicateRow?.value


        // ── Path A: NVIDIA FLUX.1-Kontext-dev (primary) ───────────────────
        if (NVIDIA_KEY) {
            const result = await callNvidiaKontext(NVIDIA_KEY, person_image, garment_image, garment_type || 'upper_body')
            if (result) {
                return new Response(
                    JSON.stringify({ success: true, resultImage: result, methodUsed: 'nvidia_kontext' }),
                    { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 200 },
                )
            }
            console.warn('NVIDIA Kontext failed — falling back to Replicate IDM-VTON')
        }

        // ── Path B: Replicate IDM-VTON (fallback) ─────────────────────────
        if (REPLICATE_TOKEN) {
            const resultImage = await callReplicateIDMVTON(REPLICATE_TOKEN, person_image, garment_image, garment_type || 'upper_body')
            return new Response(
                JSON.stringify({ success: true, resultImage, methodUsed: 'replicate_idm_vton' }),
                { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 200 },
            )
        }

        return new Response(
            JSON.stringify({ success: false, error: 'Try-on provider is not configured' }),
            { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 503 },
        )

    } catch (error: any) {
        return new Response(
            JSON.stringify({ success: false, error: error instanceof SyntaxError?'Invalid JSON request':'Try-on unavailable' }),
            { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: error instanceof SyntaxError?400:503 },
        )
    }
})
