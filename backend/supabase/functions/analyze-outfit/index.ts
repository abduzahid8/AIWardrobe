import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

const NVIDIA_API_URL = 'https://integrate.api.nvidia.com/v1/chat/completions'

function stripDataUri(value: string): string {
  return value.startsWith('data:') ? (value.split(',')[1] ?? value) : value
}

// ── Outfit Analysis Prompt ──────────────────────────────────────────────────

function buildOutfitAnalysisPrompt(): string {
  return `Analyze the outfit in this image. Identify each distinct clothing item the person is wearing.
Ignore the person's body, background, and any non-clothing objects.
Return ONLY a valid JSON array where each element describes one clothing item:
[
  {
    "category": "t-shirt|shirt|blouse|sweater|hoodie|jacket|coat|cardigan|dress|skirt|pants|jeans|shorts|sneakers|boots|sandals|bag|hat|scarf|belt|clothing",
    "section": "tops|bottoms|dresses|outerwear|shoes|accessories|other",
    "specificType": "brief description like 'oversized white t-shirt' or 'slim black jeans'",
    "primaryColor": "main color name",
    "colorHex": "hex code like #1A1A2E",
    "style": "casual|formal|sport|semi_classic|elegant|minimalist|other",
    "material": "fabric type or null",
    "pattern": "solid|striped|plaid|graphic|other",
    "fit": "slim|regular|oversized|fitted|loose",
    "description": "1-2 sentence description of the item"
  }
]
Return valid JSON only. Identify ALL visible clothing items (typically 3-6 items).`
}

// ── Similar Item Search Prompt ──────────────────────────────────────────────

function buildSimilarItemPrompt(detectedItem: any): string {
  return `A user wants to find clothing similar to or better than this item:
- Type: ${detectedItem.specificType || detectedItem.category}
- Color: ${detectedItem.primaryColor}
- Style: ${detectedItem.style}
- Material: ${detectedItem.material || 'unknown'}
- Pattern: ${detectedItem.pattern || 'solid'}
- Fit: ${detectedItem.fit || 'regular'}

Suggest 3 alternative items that are similar or an upgrade. For each, provide:
Return ONLY a valid JSON array:
[
  {
    "name": "item name",
    "category": "same category as input",
    "primaryColor": "suggested color",
    "style": "casual|formal|sport|semi_classic|elegant|minimalist",
    "reason": "why this is similar or better",
    "upgradeLevel": "same|slight_upgrade|significant_upgrade"
  }
]
Return valid JSON only.`
}

// ── NVIDIA Vision Call ──────────────────────────────────────────────────────

async function callNvidiaVision(
  nvidiaToken: string,
  imageDataUrl: string,
  prompt: string,
): Promise<string> {
  const response = await fetch(NVIDIA_API_URL, {
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
            { type: 'image_url', image_url: { url: imageDataUrl } },
            { type: 'text', text: prompt },
          ],
        },
      ],
      max_tokens: 1024,
      temperature: 0.2,
    }),
  })

  if (!response.ok) {
    const errorText = await response.text()
    console.error('NVIDIA API error:', response.status, errorText.slice(0, 500))
    throw new Error(`NVIDIA API error: ${response.status}`)
  }

  const responseText = await response.text()
  return responseText
}

function parseNvidiaResponseText(responseText: string): any {
  try {
    return JSON.parse(responseText)
  } catch {
    const sseMatch = responseText.match(/data:\s*(\{[\s\S]*\})/)
    if (sseMatch) {
      try { return JSON.parse(sseMatch[1]) } catch {}
    }
    const jsonMatch = responseText.match(/\{[\s\S]*\}/)
    if (jsonMatch) {
      try { return JSON.parse(jsonMatch[0]) } catch {}
    }
  }
  return {}
}

function extractContentFromResponse(responseText: string): string {
  const nvidiaData = parseNvidiaResponseText(responseText)
  return nvidiaData.choices?.[0]?.message?.content || ''
}

function parseJsonArrayFromContent(content: string): any[] {
  try {
    const arrayMatch = content.match(/\[[\s\S]*\]/)
    if (arrayMatch) {
      return JSON.parse(arrayMatch[0])
    }
  } catch {}
  return []
}

// ── Wardrobe Similarity Search ──────────────────────────────────────────────

function itemSimilarityScore(detected: any, wardrobe: any): number {
  let score = 0
  const dCat = (detected.category || '').toLowerCase()
  const wCat = (wardrobe.category || wardrobe.type || wardrobe.itemType || '').toLowerCase()
  const dSection = (detected.section || '').toLowerCase()
  const wSection = (wardrobe.section || '').toLowerCase()

  // Category match (most important)
  if (dCat === wCat || dSection === wSection) score += 40
  else if (dCat.includes(wCat) || wCat.includes(dCat)) score += 25

  // Color similarity
  const dColor = (detected.primaryColor || '').toLowerCase()
  const wColor = (wardrobe.color || wardrobe.primaryColor || '').toLowerCase()
  if (dColor && wColor) {
    if (dColor === wColor) score += 25
    else {
      const neutralColors = ['black', 'white', 'grey', 'gray', 'navy', 'beige', 'cream', 'brown']
      const dNeutral = neutralColors.some(c => dColor.includes(c))
      const wNeutral = neutralColors.some(c => wColor.includes(c))
      if (dNeutral && wNeutral) score += 15
    }
  }

  // Style match
  const dStyle = (detected.style || '').toLowerCase()
  const wStyle = (wardrobe.style || '').toLowerCase()
  if (dStyle && wStyle && dStyle === wStyle) score += 15

  // Pattern match
  const dPattern = (detected.pattern || 'solid').toLowerCase()
  const wPattern = (wardrobe.pattern || 'solid').toLowerCase()
  if (dPattern === wPattern) score += 10

  // Material match
  const dMaterial = (detected.material || '').toLowerCase()
  const wMaterial = (wardrobe.material || '').toLowerCase()
  if (dMaterial && wMaterial && dMaterial === wMaterial) score += 10

  return score
}

function shopItemSimilarityScore(detected: any, shop: any): number {
  let score = 0
  const dCat = (detected.category || '').toLowerCase()
  const sGarmentType = (shop.garment_type || '').toLowerCase()
  const sCategory = (shop.category || '').toLowerCase()

  // Category match
  if (dCat === sGarmentType || dCat === sCategory) score += 40
  else if (sGarmentType.includes(dCat) || dCat.includes(sGarmentType)) score += 25

  // Color
  const dColor = (detected.primaryColor || '').toLowerCase()
  const sName = (shop.name || shop.description || '').toLowerCase()
  if (dColor && sName.includes(dColor)) score += 20

  // Style
  const dStyle = (detected.style || '').toLowerCase()
  if (dStyle && sName.includes(dStyle)) score += 15

  // Description overlap
  const dDesc = (detected.specificType || '').toLowerCase()
  if (dDesc) {
    const words = dDesc.split(/\s+/).filter(w => w.length > 3)
    for (const word of words) {
      if (sName.includes(word)) score += 5
    }
  }

  return score
}

// ── Main Handler ────────────────────────────────────────────────────────────

serve(async (req: Request) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const { image, mode, userId } = await req.json()

    if (!image) {
      return new Response(
        JSON.stringify({ success: false, error: 'Image is required' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } },
      )
    }

    // Get NVIDIA token from app_config
    const supabaseUrl = Deno.env.get('SUPABASE_URL')!
    const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
    const supabase = createClient(supabaseUrl, supabaseServiceKey)

    const { data: configData } = await supabase
      .from('app_config')
      .select('value')
      .eq('key', 'nvidia_token')
      .single()

    const nvidiaToken = configData?.value
    if (!nvidiaToken) {
      return new Response(
        JSON.stringify({ success: false, error: 'AI service not configured' }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } },
      )
    }

    const imageDataUrl = image.startsWith('data:') ? image : `data:image/jpeg;base64,${image}`

    // ── Step 1: Analyze the outfit photo ──
    const analysisResponseText = await callNvidiaVision(
      nvidiaToken,
      imageDataUrl,
      buildOutfitAnalysisPrompt(),
    )
    const analysisContent = extractContentFromResponse(analysisResponseText)
    const detectedItems = parseJsonArrayFromContent(analysisContent)

    if (detectedItems.length === 0) {
      return new Response(
        JSON.stringify({
          success: false,
          error: 'Could not detect clothing items in the image',
          detectedItems: [],
          recommendations: [],
        }),
        { status: 200, headers: { ...corsHeaders, 'Content-Type': 'application/json' } },
      )
    }

    // ── Step 2: Find similar items from wardrobe + shop ──
    let wardrobeItems: any[] = []
    let shopItems: any[] = []

    if (userId) {
      // Fetch user's wardrobe items
      const { data: wData } = await supabase
        .from('wardrobe_items')
        .select('id, category, type, item_type, color, primary_color, style, pattern, material, image_url, description, sub_category')
        .eq('user_id', userId)
        .limit(200)

      wardrobeItems = wData || []
    }

    // Fetch shop catalog items (active only)
    const { data: sData } = await supabase
      .from('shop_catalog')
      .select('id, name, brand, price, currency, image_url, garment_type, category, description')
      .eq('is_active', true)
      .limit(500)

    shopItems = sData || []

    // ── Step 3: Match and recommend ──
    const recommendations = detectedItems.map((detected: any) => {
      // Score wardrobe items
      const wardrobeMatches = wardrobeItems
        .map((w: any) => ({
          ...w,
          imageUrl: w.image_url,
          matchScore: itemSimilarityScore(detected, w),
        }))
        .filter((w: any) => w.matchScore >= 20)
        .sort((a: any, b: any) => b.matchScore - a.matchScore)
        .slice(0, 4)

      // Score shop items
      const shopMatches = shopItems
        .map((s: any) => ({
          ...s,
          imageUrl: s.image_url,
          isShopItem: true,
          matchScore: shopItemSimilarityScore(detected, s),
        }))
        .filter((s: any) => s.matchScore >= 20)
        .sort((a: any, b: any) => b.matchScore - a.matchScore)
        .slice(0, 4)

      return {
        detectedItem: {
          category: detected.category || 'clothing',
          section: detected.section || 'other',
          specificType: detected.specificType || detected.category,
          primaryColor: detected.primaryColor || 'unknown',
          colorHex: detected.colorHex || '#808080',
          style: detected.style || 'casual',
          material: detected.material || null,
          pattern: detected.pattern || 'solid',
          fit: detected.fit || 'regular',
          description: detected.description || '',
        },
        similarFromWardrobe: wardrobeMatches,
        similarFromShop: shopMatches,
      }
    })

    // ── Step 4: Generate overall outfit description ──
    const outfitDescription = detectedItems
      .map((item: any) => item.specificType || item.category)
      .join(' + ')

    return new Response(
      JSON.stringify({
        success: true,
        detectedItems: recommendations.map((r: any) => r.detectedItem),
        recommendations,
        outfitDescription: `Detected outfit: ${outfitDescription}`,
        itemCount: detectedItems.length,
      }),
      { status: 200, headers: { ...corsHeaders, 'Content-Type': 'application/json' } },
    )
  } catch (error) {
    console.error('analyze-outfit error:', error)
    return new Response(
      JSON.stringify({
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } },
    )
  }
})
