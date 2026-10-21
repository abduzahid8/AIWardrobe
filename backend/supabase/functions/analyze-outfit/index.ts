import { enforceRateLimit } from '../_shared/rateLimit.ts'
import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'X-Analysis-Contract': '20261019-patterns-v4',
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

const NVIDIA_API_URL = 'https://integrate.api.nvidia.com/v1/chat/completions'

function stripDataUri(value: string): string {
  return value.startsWith('data:') ? (value.split(',')[1] ?? value) : value
}

// ── Outfit Analysis Prompt ──────────────────────────────────────────────────

function buildOutfitAnalysisPrompt(mode: 'outfit' | 'single' = 'outfit'): string {
  return `${mode === 'single' ? 'Analyze exactly one main clothing item in this image. Ignore other garments.' : 'Analyze the outfit in this image. Identify each distinct clothing item the person is wearing.'}
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
    "pattern": "solid|striped|plaid|graphic|other or null when unknown",
    "fit": "slim|regular|oversized|fitted|loose or null when unknown or not applicable",
    "description": "1-2 sentence description of the item"
  }
]
Sections must follow category: t-shirt/shirt/blouse/sweater/hoodie => tops; jacket/coat/cardigan => outerwear; dress => dresses; skirt/pants/jeans/shorts => bottoms; sneakers/boots/sandals => shoes; bag/hat/scarf/belt => accessories; clothing => other. Include every field shown; use JSON null for unknown material, pattern or fit, never the quoted string \"null\".
${mode === 'single' ? 'Return valid JSON only. The array must contain exactly one main garment.' : 'Return valid JSON only. Identify ALL visible clothing items (typically 3-6 items).'}`
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
    signal: AbortSignal.timeout(45000),
    method: 'POST',
    headers: {
      Authorization: `Bearer ${nvidiaToken}`,
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify({
      // NOTE: nvidia/llama-3.1-nemotron-nano-vl-8b-v1 was retired (410 Gone).
      // meta/llama-3.2-11b-vision-instruct is the current free vision model.
      model: 'meta/llama-3.2-11b-vision-instruct',
      messages: [
        { role: 'user', content: 'Analyze an outfit containing one black leather belt. Return only a JSON array of clothing items, without prose or Markdown.' },
        { role: 'assistant', content: '[{"category":"belt","section":"accessories","specificType":"black leather belt","primaryColor":"black","colorHex":"#000000","style":"casual","material":"leather","pattern":"solid","fit":null,"description":"A black leather belt."}]' },
        {
          role: 'user',
          content: [
            { type: 'image_url', image_url: { url: imageDataUrl } },
            { type: 'text', text: prompt },
          ],
        },
      ],
      // 6 items x ~150 tokens needs headroom — a tight cap truncates
      // mid-array; incomplete output is rejected rather than salvaged.
      max_tokens: 2000,
      temperature: 0,
    }),
  })

  if (!response.ok) {
    const errorText = await response.text()
    console.error('NVIDIA API error:', response.status)
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

function parseJsonArrayFromContent(content: string): any[] | null {
  if (typeof content !== 'string') return null
  const arrayMatch = content.match(/\[[\s\S]*\]/)
  if (!arrayMatch) return null
  try {
    const value = JSON.parse(arrayMatch[0])
    return Array.isArray(value) && value.length <= 20 ? value : null
  } catch { return null }
}

function detectedSection(category: string | null): string | null {
  const sections: Record<string, string> = {
    't-shirt':'tops',shirt:'tops',blouse:'tops',sweater:'tops',hoodie:'tops',
    jacket:'outerwear',coat:'outerwear',cardigan:'outerwear',dress:'dresses',
    skirt:'bottoms',pants:'bottoms',jeans:'bottoms',shorts:'bottoms',
    sneakers:'shoes',boots:'shoes',sandals:'shoes',loafer:'shoes',oxfords:'shoes',derbies:'shoes',
    bag:'accessories',hat:'accessories',scarf:'accessories',belt:'accessories',
  }
  return category ? sections[category] || null : null
}

function normalizePattern(value: unknown): unknown {
  if (typeof value !== 'string') return value
  const key=value.trim().toLowerCase().replace(/[_-]+/g,' ').replace(/\s+/g,' ')
  const aliases: Record<string,string|null> = {
    solid:'solid',plain:'solid',unpatterned:'solid','no pattern':'solid',none:'solid',
    'solid color':'solid','solid colored':'solid','solid coloured':'solid',
    striped:'striped',stripe:'striped',stripes:'striped',pinstriped:'striped',
    plaid:'plaid',checked:'plaid',checkered:'plaid',tartan:'plaid',gingham:'plaid',
    graphic:'graphic','graphic print':'graphic',other:'other',
    floral:'other','polka dot':'other','polka dots':'other','animal print':'other',
    geometric:'other',abstract:'other',textured:'other',ribbed:'other','cable knit':'other',houndstooth:'other',
    'null':null,unknown:null,'n/a':null,'not applicable':null,'not specified':null,
  }
  return Object.hasOwn(aliases,key)?aliases[key]:value
}

function normalizeDetectedItem(item: any): any {
  if (!item || typeof item !== 'object' || Array.isArray(item)) return item
  const token=(value:unknown)=>typeof value==='string'?value.trim().toLowerCase():value
  const normalized={...item,category:token(item.category),section:token(item.section),style:token(item.style),fit:token(item.fit),pattern:normalizePattern(item.pattern)}
  // Broad labels resolve only through an explicit noun, never color/style.
  if (!['tops','outerwear','bottoms','dresses','shoes','accessories'].includes(normalized.category) || normalized.section !== normalized.category) return normalized
  const category = namedGarmentCategory(normalized.specificType)
  const section = detectedSection(category)
  if (!section || (section !== normalized.section && !(normalized.section === 'tops' && section === 'outerwear'))) return normalized
  return {...normalized, category, section}
}

function detectedItemIssues(item: any): string[] {
  if (!item || typeof item !== 'object' || Array.isArray(item)) return ['item_shape']
  const category = garmentCategory(item.category)
  const expectedSection = detectedSection(category) || (item.category === 'clothing' ? 'other' : null)
  const text = (value: unknown, max = 2000) => typeof value === 'string' && value.trim().length > 0 && value.length <= max
  const checks: Record<string, boolean> = {
    category: !!expectedSection, section: !!expectedSection && item.section === expectedSection,
    specificType: text(item.specificType, 300), primaryColor: text(item.primaryColor, 80),
    colorHex: typeof item.colorHex === 'string' && /^#[0-9a-f]{6}$/i.test(item.colorHex),
    style: ['casual','formal','sport','semi_classic','elegant','minimalist','other'].includes(item.style),
    material: item.material === null || text(item.material, 100),
    pattern: item.pattern === null || ['solid','striped','plaid','graphic','other'].includes(item.pattern),
    fit: item.fit === null || ['slim','regular','oversized','fitted','loose'].includes(item.fit), description: text(item.description),
  }
  return Object.keys(checks).filter(field => !checks[field])
}

function validDetectedItem(item: any): boolean {
  return detectedItemIssues(item).length === 0
}

function patternFailureForm(value: unknown): string {
  if (value === undefined) return 'missing'
  if (value === null) return 'null'
  if (Array.isArray(value)) return 'array'
  if (typeof value !== 'string') return typeof value
  const token=value.trim().toLowerCase()
  if (!token) return 'empty_string'
  if (token === 'null') return 'quoted_null'
  if (token.includes('|')) return 'copied_option_list'
  if (/^[{[]/.test(token)) return 'stringified_structure'
  return 'unrecognized_string'
}

const analysisUnavailable = (reason: string, details: Record<string, unknown> = {}) => {
  // Only static reason/field names and counts; never log images or model text.
  console.warn('analysis_rejected', JSON.stringify({ reason, ...details }))
  const fields = Array.isArray(details.items) ? [...new Set(details.items.flatMap((item:any)=>Array.isArray(item.fields)?item.fields:[]))].filter(field=>typeof field==='string'&&/^[a-zA-Z_]+$/.test(field)).join(',') : ''
  return new Response(JSON.stringify({success:false,error:'ai_analysis_unavailable',detectedItems:[],recommendations:[]}), {status:503,headers:{...corsHeaders,'Content-Type':'application/json','X-Analysis-Rejection':reason,'X-Analysis-Fields':fields,'X-Analysis-Pattern-Form':Array.isArray(details.patternForms)?details.patternForms.join(','):'',...(typeof details.providerStatus==='number'?{'X-Analysis-Provider-Status':String(details.providerStatus)}:{})}})
}

// ── Wardrobe Similarity Search ──────────────────────────────────────────────

// A shared color/style cannot establish that two products are the same kind.
// Unknown/generic sections carry no type authority; inspect specific type fields.
function garmentCategory(...values: unknown[]): string | null {
  const aliases: Record<string, string> = {
    'tshirt': 't-shirt', 'tshirts': 't-shirt', 't-shirts': 't-shirt', 'tee': 't-shirt',
    shirts: 'shirt', blouses: 'blouse', sweaters: 'sweater', hoodies: 'hoodie',
    jackets: 'jacket', coats: 'coat', cardigans: 'cardigan', dresses: 'dress', skirts: 'skirt',
    trousers: 'pants', trouser: 'pants', pant: 'pants', jean: 'jeans', short: 'shorts',
    sneaker: 'sneakers', boot: 'boots', sandal: 'sandals', bags: 'bag', hats: 'hat',
    scarves: 'scarf', belts: 'belt', loafers: 'loafer', oxford: 'oxfords', derby: 'derbies',
  }
  const known = new Set(['t-shirt','shirt','blouse','sweater','hoodie','jacket','coat','cardigan',
    'dress','skirt','pants','jeans','shorts','sneakers','boots','sandals','bag','hat','scarf','belt',
    'loafer','oxfords','derbies'])
  for (const value of values) {
    if (typeof value !== 'string') continue
    const key = value.trim().toLowerCase().replace(/[ _]+/g, '-')
    const category = aliases[key] || key
    if (known.has(category)) return category
  }
  return null
}

function itemSimilarityScore(detected: any, wardrobe: any): number {
  const dCat = garmentCategory(detected.category)
  const wCat = garmentCategory(wardrobe.type, wardrobe.sub_category, wardrobe.category, wardrobe.itemType)
  if (!dCat || dCat !== wCat) return 0
  let score = 40

  // Color similarity
  const dColor = (detected.primaryColor || '').toLowerCase()
  const wColor = (wardrobe.primary_color || wardrobe.primaryColor || wardrobe.color || '').toLowerCase()
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
  const dPattern = (detected.pattern || '').toLowerCase()
  const wPattern = (wardrobe.pattern || '').toLowerCase()
  if (dPattern && dPattern === wPattern) score += 10

  // Material match
  const dMaterial = (detected.material || '').toLowerCase()
  const wMaterial = (wardrobe.material || '').toLowerCase()
  if (dMaterial && wMaterial && dMaterial === wMaterial) score += 10

  return score
}

function namedGarmentCategory(value: unknown): string | null {
  if (typeof value !== 'string') return null
  const name = value.toLowerCase().replace(/\bt[ -]*shirts?\b/g, 'tshirt')
    .replace(/\bdress (?=shirts?|pants?|trousers?|shoes?|boots?)/g, '')
  const rules: Array<[string, RegExp]> = [
    ['t-shirt', /\b(?:tshirts?|tees?)\b|футболк/], ['shirt', /\bshirts?\b|рубаш/],
    ['blouse', /\bblouses?\b|блуз/], ['sweater', /\b(?:sweaters?|jumpers?|pullovers?)\b|свитер|джемпер|пуловер/],
    ['hoodie', /\bhoodies?\b|худи/], ['jacket', /\bjackets?\b|куртк/],
    ['coat', /\bcoats?\b|пальто/], ['cardigan', /\bcardigans?\b|кардиган/],
    ['dress', /\bdress(?:es)?\b|плать/], ['skirt', /\bskirts?\b|юбк/],
    ['pants', /\b(?:pants?|trousers?)\b|брюк/], ['jeans', /\bjeans?\b|джинс/],
    ['shorts', /\bshorts\b|шорт/], ['sneakers', /\b(?:sneakers?|trainers?)\b|кроссов|кеды/],
    ['boots', /\bboots?\b|ботин|сапог/], ['sandals', /\bsandals?\b|сандал/],
    ['loafer', /\b(?:loafers?|moccasins?)\b|лофер|мокасин/],
    ['bag', /\bbags?\b|сумк/], ['hat', /\bhats?\b|шляп/],
    ['scarf', /\bscar(?:f|ves)\b|шарф/], ['belt', /\bbelts?\b|ремень|ремни/],
  ]
  const found = rules.filter(([, pattern]) => pattern.test(name)).map(([category]) => category)
  // Ambiguous titles do not establish a product's type.
  return found.length === 1 ? found[0] : null
}

function shopItemSimilarityScore(detected: any, shop: any): number {
  const dCat = garmentCategory(detected.category)
  const structured = garmentCategory(shop.garment_type, shop.category)
  const named = namedGarmentCategory(shop.name)
  if (structured && named && structured !== named) return 0
  const sCat = structured || named
  if (!dCat || dCat !== sCat) return 0
  let score = 40

  // Color
  const dColor = (detected.primaryColor || '').toLowerCase()
  const sName = (shop.name || shop.description || '').toLowerCase()
  if (dColor && sName.includes(dColor)) score += 20

  // Style
  const dStyle = (detected.style || '').toLowerCase()
  if (dStyle && sName.includes(dStyle)) score += 15

  // Description overlap
  const dDesc = String(detected.specificType || '').toLowerCase()
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

    const rateLimitResponse = await enforceRateLimit(req, { feature: 'analyze-outfit', limit: 10, headers: corsHeaders })
    if (rateLimitResponse) return rateLimitResponse


  try {
    const { image, mode = 'outfit' } = await req.json()
    if (mode !== 'outfit' && mode !== 'single') {
      return new Response(JSON.stringify({success:false,error:'Invalid analysis mode'}), {status:400,headers:{...corsHeaders,'Content-Type':'application/json'}})
    }

    if (typeof image !== 'string' || !image.trim() || image.length > 14 * 1024 * 1024) {
      return new Response(
        JSON.stringify({ success: false, error: 'A nonempty image within the size limit is required' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } },
      )
    }

    // Get NVIDIA token from app_config
    const supabaseUrl = Deno.env.get('SUPABASE_URL')!
    const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
    const supabase = createClient(supabaseUrl, supabaseServiceKey)

    // Never accept an owner ID from the payload for a service-role query.
    const caller = createClient(supabaseUrl, Deno.env.get('SUPABASE_ANON_KEY')!, {
      global: { headers: { Authorization: req.headers.get('Authorization') || '' } },
    })
    const { data: identity, error: identityError } = await caller.auth.getUser()
    if (identityError || !identity?.user) {
      return new Response(JSON.stringify({ success: false, error: 'Unauthorized' }), {
        status: 401, headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      })
    }
    const userId = identity.user.id

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
    let analysisResponseText: string
    try {
      analysisResponseText = await callNvidiaVision(nvidiaToken, imageDataUrl, buildOutfitAnalysisPrompt(mode))
    } catch (error) {
      const status = error instanceof Error ? /^NVIDIA API error: (\d{3})$/.exec(error.message)?.[1] : undefined
      return analysisUnavailable('provider_failure', {kind:error instanceof Error&&['AbortError','TimeoutError'].includes(error.name)?error.name:'Error',...(status?{providerStatus:Number(status)}:{})})
    }
    const envelope = parseNvidiaResponseText(analysisResponseText)
    if (envelope.choices?.[0]?.finish_reason === 'length') return analysisUnavailable('truncated_completion')
    const analysisContent = extractContentFromResponse(analysisResponseText)
    const detectedItems = parseJsonArrayFromContent(analysisContent)?.map(normalizeDetectedItem)
    if (!detectedItems) return analysisUnavailable('invalid_json_array')
    const invalidItems = detectedItems.map((item,index)=>({index,fields:detectedItemIssues(item)})).filter(item=>item.fields.length)
    if (invalidItems.length) {
      const patternForms=invalidItems.filter(item=>item.fields.includes('pattern')).map(item=>patternFailureForm(detectedItems[item.index].pattern))
      return analysisUnavailable('invalid_item_fields',{items:invalidItems,...(patternForms.length?{patternForms}:{})})
    }
    if (mode === 'single' && detectedItems.length > 1) return analysisUnavailable('single_mode_multiple_items',{count:detectedItems.length})

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
      // NOTE: the wardrobe table is `clothing_items` (see 001_full_rebuild.sql)
      // — there is no `wardrobe_items` table. Columns mirror the canonical
      // select used by generate-outfits (no singular `occasion`).
      const { data: wData, error: wError } = await caller
        .from('clothing_items')
        .select('id, category, type, color, primary_color, style, pattern, material, image_url, description, sub_category, name, brand')
        .eq('user_id', userId)
        .limit(200)

      if (wError) console.error('analyze-outfit wardrobe fetch failed:', wError.message)
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
          category: garmentCategory(detected.category) || detected.category,
          section: detected.section,
          specificType: detected.specificType,
          primaryColor: detected.primaryColor,
          colorHex: detected.colorHex,
          style: detected.style,
          material: detected.material,
          pattern: detected.pattern,
          fit: detected.fit,
          description: detected.description,
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
        error: error instanceof SyntaxError ? 'Invalid JSON request' : 'Outfit analysis failed',
      }),
      { status: error instanceof SyntaxError ? 400 : 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } },
    )
  }
})
