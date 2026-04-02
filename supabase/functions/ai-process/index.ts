import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

// NVIDIA NIM API endpoint (correct base URL from build.nvidia.com)
const NVIDIA_API_URL = 'https://integrate.api.nvidia.com/v1/chat/completions'

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
    const imageSizeBytes = (image.length * 3) / 4
    if (imageSizeBytes > 10 * 1024 * 1024) {
      return new Response(
        JSON.stringify({ error: 'Image too large. Max 10MB.' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    let result: any = {}

    // Classification + Description - NVIDIA Granite Vision (FREE TIER - 10,000 requests/month)
    if (operation === 'classify' || operation === 'describe' || operation === 'all') {
      // FIX 1: Correct NVIDIA VLM multimodal message format
      const imageDataUrl = image.startsWith('data:') ? image : `data:image/jpeg;base64,${image}`
      const nvidiaResponse = await fetch(NVIDIA_API_URL, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${NVIDIA_TOKEN}`,
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({
          model: 'nvidia/llama-3.1-nemotron-nano-vl-8b-v1',
          messages: [
            {
              role: 'user',
              content: [
                {
                  type: 'image_url',
                  image_url: { url: imageDataUrl }
                },
                {
                  type: 'text',
                  text: `Analyze this clothing image and return ONLY a JSON object:
{"category":"t-shirt|jeans|dress|jacket|coat|sneakers|boots|bag|hat|etc","style":"casual|formal|sport|streetwear","color":"main color","material":"fabric or null","description":"1-2 sentence description"}
Return valid JSON only. No extra text.`
                }
              ]
            }
          ],
          max_tokens: 300,
          temperature: 0.1,
        }),
      })
      
      // Check response status first
      if (!nvidiaResponse.ok) {
        const errorText = await nvidiaResponse.text()
        console.error('NVIDIA API error:', nvidiaResponse.status, errorText.slice(0, 500))
        result._nvidiaError = { status: nvidiaResponse.status, body: errorText.slice(0, 200) }
        // Fallback: return mock classification instead of failing
        console.log('Falling back to mock classification')
        result.classification = {
          category: 'clothing',
          section: 'other',
          confidence: 0.5,
          attributes: {
            style: 'casual',
            color: 'various',
            material: null,
          },
        }
        result.description = 'Clothing item (AI classification unavailable)'
      } else {
        // Get response text first to debug
        const responseText = await nvidiaResponse.text()
        console.log('NVIDIA raw response (first 500 chars):', responseText.slice(0, 500))
        
        // Handle SSE format (data: {...}) or regular JSON
        let nvidiaData: any = {}
        try {
          // Try parsing as regular JSON first
          nvidiaData = JSON.parse(responseText)
        } catch {
          // Try extracting JSON from SSE format (data: {...})
          const sseMatch = responseText.match(/data:\s*(\{[\s\S]*\})/)
          if (sseMatch) {
            try {
              nvidiaData = JSON.parse(sseMatch[1])
            } catch {
              // If that fails, try finding any JSON object
              const jsonMatch = responseText.match(/\{[\s\S]*\}/)
              if (jsonMatch) {
                nvidiaData = JSON.parse(jsonMatch[0])
              }
            }
          }
        }
        
        // Parse NVIDIA response
        const content = nvidiaData.choices?.[0]?.message?.content || ''
        
        // FIX 4: Use greedy regex so nested JSON doesn't break
        let parsedData: any = {}
        try {
          const jsonMatch = content.match(/\{[\s\S]*\}/)
          if (jsonMatch) {
            parsedData = JSON.parse(jsonMatch[0])
          }
        } catch (e) {
          // Fallback: extract from text response
          const lines = content.split('\n')
          for (const line of lines) {
            if (line.toLowerCase().includes('category')) {
              parsedData.category = line.split(':')[1]?.trim().replace(/["',]/g, '') || 'clothing'
            }
            if (line.toLowerCase().includes('style')) {
              parsedData.style = line.split(':')[1]?.trim().replace(/["',]/g, '') || 'casual'
            }
            if (line.toLowerCase().includes('color')) {
              parsedData.color = line.split(':')[1]?.trim().replace(/["',]/g, '') || 'various'
            }
          }
        }

        const labelToSection: Record<string, string> = {
          't-shirt': 'tops', 'shirt': 'tops', 'blouse': 'tops', 'sweater': 'tops',
          'hoodie': 'tops', 'jacket': 'outerwear', 'coat': 'outerwear',
          'dress': 'dresses', 'skirt': 'bottoms', 'pants': 'bottoms',
          'jeans': 'bottoms', 'shorts': 'bottoms', 'sneakers': 'shoes',
          'boots': 'shoes', 'sandals': 'shoes', 'bag': 'accessories',
        }

        result.classification = {
          category: parsedData.category || 'clothing',
          section: labelToSection[(parsedData.category || '').toLowerCase()] || 'other',
          confidence: 0.88, // NVIDIA Granite is high quality
          attributes: {
            style: parsedData.style || 'casual',
            color: parsedData.color || 'various',
            material: parsedData.material || null,
          },
        }
        
        result.description = parsedData.description || 
          `${parsedData.color || 'Various'} ${parsedData.category || 'clothing item'}`
      }
    }

    // Angle Normalization - Convert any angle to standard flat lay view
    let normalizedImageUrl: string | null = null
    
    if ((operation === 'remove_bg' || operation === 'all') && REPLICATE_TOKEN) {
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

    // Background Removal - Replicate rembg (using normalized image if available)
    let processedImageUrl: string | null = null
    
    if (operation === 'remove_bg' || operation === 'all') {
      if (!REPLICATE_TOKEN) {
        console.warn('Replicate token not set, skipping background removal')
      } else {
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
    if ((operation === 'remove_bg' || operation === 'all') && processedImageUrl && REPLICATE_TOKEN) {
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
