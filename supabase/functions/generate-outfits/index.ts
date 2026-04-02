import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from "https://esm.sh/@supabase/supabase-js@2"

const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

function macroCategory(category: string, type: string): string {
    const t = `${category} ${type}`.toLowerCase()
    if (t.match(/jacket|coat|blazer|hoodie|cardigan|sweater|pullover|vest|puffer|outerwear/)) return 'outerwear'
    if (t.match(/shirt|t-shirt|tee|blouse|polo|tops|top/)) return 'top'
    if (t.match(/pant|trouser|jeans|bottom|shorts|skirt/)) return 'bottom'
    if (t.match(/shoe|sneaker|boot|loafer|sandal/)) return 'shoes'
    if (t.match(/dress/)) return 'top'
    return 'other'
}

function buildSystemPrompt(limit: number): string {
    return `You are a world-class AI fashion stylist. Create ${limit} complete, stylish outfits using ONLY the items in the user's wardrobe (reference by exact id).
Each outfit: 3-4 items covering top/outerwear + bottom (or dress) + shoes.
Consider color harmony, occasion, and any custom prompt from the user.
NEVER invent items not in the wardrobe.
Respond ONLY in pure JSON, no markdown fences:
{"outfits":[{"id":"outfit_1","style":"...","occasion":"...","description":"...","confidence":0.9,"items":[{"id":"EXACT_DB_ID","type":"top","color":"white","name":"...","brand":"...","imageUrl":"...","recommendation":"why it works"}],"stylingTips":["tip1","tip2"]}]}`
}

function buildUserPrompt(items: any[], prompt: string, style: string, occasion: string, weather: any, limit: number): string {
    return `${prompt ? `User request: "${prompt}"\n` : ''}${style ? `Style: ${style}\n` : ''}${occasion ? `Occasion: ${occasion}\n` : ''}${weather ? `Weather: ${weather.temp}°C ${weather.condition}\n` : ''}
Wardrobe (use ONLY these, exact ids):
${JSON.stringify(items)}

Create ${limit} distinct outfits. Use exact "id" from wardrobe.`
}

function localFallback(items: any[], style: string, occasion: string, limit: number): any[] {
    const tops = items.filter(i => ['top','outerwear'].includes(i.macroCategory))
    const bottoms = items.filter(i => i.macroCategory === 'bottom')
    const shoes = items.filter(i => i.macroCategory === 'shoes')
    const outfits = []
    for (let i = 0; i < Math.min(limit, Math.max(tops.length, 1)); i++) {
        const parts = [
            tops[i % Math.max(tops.length,1)] && { ...tops[i % tops.length], recommendation: 'Key piece' },
            bottoms[i % Math.max(bottoms.length,1)] && { ...bottoms[i % Math.max(bottoms.length,1)], recommendation: 'Pairs well' },
            shoes[i % Math.max(shoes.length,1)] && { ...shoes[i % Math.max(shoes.length,1)], recommendation: 'Completes the look' },
        ].filter(Boolean)
        if (!parts.length) continue
        outfits.push({ id: `local_${i}_${Date.now()}`, style: style||'Casual', occasion: occasion||'Everyday', description: `A ${style||'casual'} look from your wardrobe.`, confidence: 0.75, items: parts, stylingTips: ['Add accessories to personalize','Layer for depth'] })
    }
    return outfits.length ? outfits : [{ id:`local_0_${Date.now()}`, style: style||'Casual', occasion: occasion||'Everyday', description:'Add more items to your wardrobe for better suggestions.', confidence:0.5, items: items.slice(0,3).map(i=>({...i,recommendation:'From your wardrobe'})), stylingTips:['Scan more items'] }]
}

async function callNvidia(key: string, sys: string, usr: string): Promise<any[]|null> {
    try {
        const r = await fetch('https://integrate.api.nvidia.com/v1/chat/completions', {
            method:'POST', headers:{'Content-Type':'application/json','Authorization':`Bearer ${key}`},
            body: JSON.stringify({ model:'meta/llama-3.1-70b-instruct', messages:[{role:'system',content:sys},{role:'user',content:usr}], temperature:0.7, max_tokens:3000, response_format:{type:'json_object'} })
        })
        if (!r.ok) { console.error('NVIDIA',r.status); return null }
        const d = await r.json()
        return JSON.parse(d.choices?.[0]?.message?.content||'{}').outfits || null
    } catch(e) { console.error('NVIDIA call',e); return null }
}

async function callGemini(key: string, sys: string, usr: string): Promise<any[]|null> {
    try {
        const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${key}`, {
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({ contents:[{parts:[{text:`${sys}\n\n${usr}`}]}], generationConfig:{responseMimeType:'application/json',temperature:0.7,maxOutputTokens:3000} })
        })
        if (!r.ok) { console.error('Gemini',r.status); return null }
        const d = await r.json()
        return JSON.parse(d.candidates?.[0]?.content?.parts?.[0]?.text||'{}').outfits || null
    } catch(e) { console.error('Gemini call',e); return null }
}

serve(async (req) => {
    if (req.method === 'OPTIONS') {
        return new Response('ok', { headers: corsHeaders })
    }

    const SUPABASE_URL = Deno.env.get('SUPABASE_URL')!
    const SUPABASE_ANON_KEY = Deno.env.get('SUPABASE_ANON_KEY')!
    const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!

    let body: any = {}
    try { body = await req.json() } catch (_) { body = {} }

    const {
        prompt = '',
        stylePreferences = 'Casual',
        occasion = 'Everyday',
        weather,
        limit = 3,
        selectedItemIds = [],
        wardrobeItems: legacyWardrobe = [],
    } = body

    try {
        // ── 1. Authenticate user via JWT ──────────────────────────────────
        const authHeader = req.headers.get('Authorization') || ''
        let dbClothingItems: any[] = []

        if (authHeader) {
            const userSupabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
                global: { headers: { Authorization: authHeader } }
            })
            const { data: { user } } = await userSupabase.auth.getUser()

            if (user) {
                // ── 2. Fetch clothing items from DB ───────────────────────
                let q = userSupabase
                    .from('clothing_items')
                    .select('id, type, category, color, style, brand, season, occasion, pattern, material, image_url, description, primary_color, sub_category, wear_count')
                    .eq('is_archived', false)
                    .order('wear_count', { ascending: false })
                    .limit(60)

                if (selectedItemIds && selectedItemIds.length > 0) {
                    q = q.in('id', selectedItemIds)
                }

                const { data: rows } = await q
                if (rows && rows.length > 0) {
                    dbClothingItems = rows.map((row: any) => ({
                        id: row.id,
                        type: row.type || row.sub_category || row.category || 'clothing',
                        category: row.category || 'Other',
                        macroCategory: macroCategory(row.category || '', row.type || row.sub_category || ''),
                        color: Array.isArray(row.color) ? row.color.join(', ') : (row.primary_color || row.color || 'neutral'),
                        style: row.style || 'Casual',
                        brand: row.brand || '',
                        material: row.material || '',
                        pattern: row.pattern || '',
                        season: Array.isArray(row.season) ? row.season.join(', ') : (row.season || 'All Seasons'),
                        occasion: Array.isArray(row.occasion) ? row.occasion.join(', ') : (row.occasion || ''),
                        description: row.description || '',
                        imageUrl: row.image_url || '',
                        wearCount: row.wear_count || 0,
                    }))
                }
            }
        }

        // ── 3. Fall back to legacy client-sent items if DB fetch empty ────
        const wardrobeItems = dbClothingItems.length > 0
            ? dbClothingItems
            : (legacyWardrobe || []).map((item: any) => ({
                ...item,
                macroCategory: macroCategory(item.category || '', item.type || ''),
            }))

        console.log(`Outfit gen: ${wardrobeItems.length} items, prompt="${prompt}", style=${stylePreferences}`)

        // ── 4. Get API keys ───────────────────────────────────────────────
        let nvidiaKey = Deno.env.get('NVIDIA_API_KEY') || ''
        let geminiKey = Deno.env.get('GEMINI_API_KEY') || ''

        if (SUPABASE_SERVICE_ROLE_KEY && (!nvidiaKey || !geminiKey)) {
            const svc = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
            const { data: cfg } = await svc
                .from('app_config')
                .select('key, value')
                .in('key', ['nvidia_token', 'gemini_token'])
            if (cfg) {
                for (const row of cfg) {
                    if (row.key === 'nvidia_token' && !nvidiaKey) nvidiaKey = row.value
                    if (row.key === 'gemini_token' && !geminiKey) geminiKey = row.value
                }
            }
        }

        // ── 5. No AI key → local fallback ─────────────────────────────────
        if (!nvidiaKey && !geminiKey) {
            const outfits = localFallback(wardrobeItems, stylePreferences, occasion, limit)
            return new Response(JSON.stringify({ success: true, outfits, source: 'local' }), {
                headers: { ...corsHeaders, 'Content-Type': 'application/json' }
            })
        }

        // ── 6. Build prompts ───────────────────────────────────────────────
        const sys = buildSystemPrompt(limit)
        const usr = buildUserPrompt(wardrobeItems, prompt, stylePreferences, occasion, weather, limit)

        // ── 7. Call AI ─────────────────────────────────────────────────────
        let aiOutfits: any[] | null = null
        if (nvidiaKey) aiOutfits = await callNvidia(nvidiaKey, sys, usr)
        if (!aiOutfits && geminiKey) aiOutfits = await callGemini(geminiKey, sys, usr)
        if (!aiOutfits || aiOutfits.length === 0) {
            aiOutfits = localFallback(wardrobeItems, stylePreferences, occasion, limit)
        }

        // ── 8. Enrich items with imageUrl from DB ──────────────────────────
        const itemMap = new Map(wardrobeItems.map((i: any) => [i.id, i]))
        const enriched = aiOutfits.map((outfit: any) => ({
            ...outfit,
            items: (outfit.items || []).map((item: any) => {
                const src: any = itemMap.get(item.id) || {}
                return {
                    ...item,
                    imageUrl: item.imageUrl || src.imageUrl || '',
                    color: item.color || src.color || 'neutral',
                    type: item.type || src.type || 'clothing',
                    name: item.name || src.type || src.category || 'Item',
                    brand: item.brand || src.brand || '',
                    macroCategory: src.macroCategory || item.macroCategory || '',
                }
            }),
        }))

        return new Response(JSON.stringify({ success: true, outfits: enriched, source: 'ai' }), {
            headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        })

    } catch (error: any) {
        console.error('generate-outfits error:', error)
        return new Response(JSON.stringify({ success: false, error: error.message, outfits: [] }), {
            headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 500
        })
    }
})
