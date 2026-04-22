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

function isDressItem(item: any): boolean {
    const blob = `${item?.type || ''} ${item?.name || ''} ${item?.category || ''} ${item?.sub_category || ''}`.toLowerCase()
    return /\bdress(es)?\b/.test(blob)
}

// Decide whether an outfit needs a two-top layered composition
// (base top + outerwear + bottom + shoes). Mirrors the client-side
// `needsLayering` in features/outfit-generator/utils/styleInference.ts.
function needsLayering(style: string | null | undefined, weather: any, prompt: string | null | undefined): boolean {
    const normalized = (style || '').toLowerCase().replace(/[\s-]+/g, '_')
    const promptBlob = (prompt || '').toLowerCase()
    if (/\b(summer|hot|heatwave|tee[-\s]?only|no jacket|no outerwear|beach)\b/.test(promptBlob)) return false

    const condition = (weather?.condition || '').toString().toLowerCase()
    const temp = typeof weather?.temp === 'number' ? weather.temp : null
    const coldTemp = temp != null && temp < 18
    const coldCondition = /\b(cold|chilly|freezing|snow|rain|drizzle|wind|storm)\b/.test(condition)
    if (coldTemp || coldCondition) return true

    if (normalized === 'old_money' || normalized === 'business_casual') return true
    if (normalized === 'streetwear') return true
    if (normalized === 'y2k') return false
    return false
}

// ── Lightweight style inference (mirrors features/outfit-generator/utils/styleInference.ts)
// Deno can't import the client TS file, so we re-implement a subset here so
// items arriving from the DB / legacy payload still get styleTags, materials,
// and formality attached before prompt-building.
const MATERIAL_DICT = ['linen','wool','cashmere','merino','silk','satin','cotton','poplin','oxford','denim','leather','suede','corduroy','tweed','velvet','canvas','nylon','polyester','mesh','fleece','terry','jersey','knit']
const PATTERN_DICT = ['striped','stripes','checked','check','plaid','houndstooth','pinstripe','graphic','logo','print','printed','tie-dye','washed','distressed','ripped','embroidered','floral','camo','monogram','colorblock']
const STYLE_SIGNALS_EDGE: Record<string, Array<[RegExp, number]>> = {
    old_money: [
        [/\bcashmere\b/i, 4], [/\bmerino\b/i, 3], [/\bwool\b/i, 3], [/\blinen\b/i, 3],
        [/\bsilk\b/i, 2], [/\bblazer\b/i, 4], [/\bloafer(s)?\b/i, 5], [/\boxford\b/i, 3],
        [/\bpolo\b/i, 3], [/\bcardigan\b/i, 3], [/\bchino(s)?\b/i, 3], [/\btrouser(s)?\b/i, 3],
        [/\btailored\b/i, 3], [/\bpleated\b/i, 3], [/\bpinstripe\b/i, 2], [/\bhoundstooth\b/i, 3],
        [/\btweed\b/i, 3], [/\bsuit\b/i, 3], [/\bknit sweater\b/i, 3],
        [/\bcream\b/i, 2], [/\bcamel\b/i, 3], [/\bnavy\b/i, 2], [/\bbeige\b/i, 2],
        [/\bralph lauren\b/i, 5], [/\bbrunello\b/i, 5], [/\bloro piana\b/i, 5], [/\bmassimo dutti\b/i, 3],
        // Shoes-specific positives
        [/\bpenny loafer(s)?\b/i, 5], [/\bbit loafer(s)?\b/i, 5], [/\bmetal bit\b/i, 4],
        [/\bboat shoe(s)?\b/i, 4], [/\bderby\b/i, 4], [/\bdress shoe(s)?\b/i, 4],
        [/\bleather shoe(s)?\b/i, 3], [/\bleather loafer(s)?\b/i, 5],
        [/\bmoc toe\b/i, 2], [/\bdouble buckle\b/i, 3],
        // Negatives
        [/\bhoodie\b/i, -4], [/\bgraphic\b/i, -3], [/\blogo\b/i, -2], [/\boversized\b/i, -2],
        [/\bcargo\b/i, -3], [/\btrack pants\b/i, -4], [/\bsweatpants\b/i, -3],
        [/\bripped\b/i, -3], [/\bdistressed\b/i, -3], [/\bneon\b/i, -4],
        [/\bsequin\b/i, -4], [/\brhinestone\b/i, -4],
        // Shoes to avoid
        [/\bchunky sneaker(s)?\b/i, -4], [/\bbasketball\b/i, -5],
        [/\bskate sneaker(s)?\b/i, -4], [/\bthick[-\s]?soled\b/i, -3],
        [/\bretro sneaker(s)?\b/i, -2], [/\brope lace\b/i, -2],
        [/\bsneaker(s)?\b/i, -1], [/\bchunky\b/i, -3],
        [/\bheavyweight\s+tee\b/i, -2], [/\b3[-\s]?pack\b/i, -2],
    ],
    streetwear: [
        [/\bhoodie\b/i, 5], [/\boversized\b/i, 4], [/\bbaggy\b/i, 4], [/\bcargo\b/i, 5],
        [/\bgraphic\b/i, 4], [/\blogo\b/i, 3], [/\bprint(ed)?\b/i, 3], [/\bsneaker\b/i, 3],
        [/\bpuffer\b/i, 4], [/\bbomber\b/i, 3], [/\btrack\b/i, 3], [/\bsweatpants\b/i, 3],
        [/\bjoggers\b/i, 4], [/\butility\b/i, 3], [/\bparka\b/i, 3],
        [/\bblazer\b/i, -3], [/\bloafer\b/i, -3], [/\btailored\b/i, -3],
    ],
    minimalist: [
        [/\bminimal\b/i, 4], [/\bessential\b/i, 3], [/\bbasic\b/i, 2], [/\bplain\b/i, 3],
        [/\bmerino\b/i, 3], [/\bwool\b/i, 2], [/\bcos\b/i, 5], [/\buniqlo\b/i, 4],
        [/\bcrew neck\b/i, 2], [/\bturtleneck\b/i, 2], [/\bblack\b/i, 2], [/\bwhite\b/i, 2],
        [/\bgrey\b/i, 2], [/\bgray\b/i, 2], [/\bbeige\b/i, 2],
        [/\bgraphic\b/i, -4], [/\blogo\b/i, -3], [/\bneon\b/i, -5], [/\bsequin\b/i, -5],
    ],
    y2k: [
        [/\by2k\b/i, 5], [/\blow[-\s]?rise\b/i, 4], [/\bcrop(ped)?\b/i, 3], [/\brhinestone\b/i, 5],
        [/\bsequin\b/i, 4], [/\bmetallic\b/i, 4], [/\bvelour\b/i, 4], [/\bbaby tee\b/i, 4],
        [/\bplatform\b/i, 4], [/\biridescent\b/i, 4], [/\bholographic\b/i, 4],
        [/\bpink\b/i, 2], [/\bfuchsia\b/i, 3], [/\bneon\b/i, 3],
        [/\bblazer\b/i, -3], [/\btailored\b/i, -3], [/\bwool\b/i, -2],
    ],
    business_casual: [
        [/\bblazer\b/i, 4], [/\bchino(s)?\b/i, 4], [/\btrouser(s)?\b/i, 4], [/\boxford\b/i, 4],
        [/\bbutton[-\s]?down\b/i, 3], [/\bloafer(s)?\b/i, 4], [/\bdress shirt\b/i, 4],
        [/\bdress shoe(s)?\b/i, 4], [/\bderby\b/i, 4], [/\bpenny loafer(s)?\b/i, 4],
        [/\bbit loafer(s)?\b/i, 4], [/\bleather loafer(s)?\b/i, 4], [/\bleather shoe(s)?\b/i, 3],
        [/\bpoplin\b/i, 3], [/\btailored\b/i, 4], [/\bslim[-\s]?fit\b/i, 2],
        [/\bnavy\b/i, 2], [/\bcharcoal\b/i, 2], [/\btan\b/i, 2],
        [/\bhoodie\b/i, -4], [/\bgraphic\b/i, -3], [/\bcargo\b/i, -4], [/\bripped\b/i, -4],
        [/\bchunky\b/i, -3], [/\bbasketball\b/i, -5], [/\bskate\b/i, -3],
    ],
    casual: [
        [/\bt[-\s]?shirt\b/i, 2], [/\btee\b/i, 2], [/\bjeans\b/i, 2], [/\bsweater\b/i, 2],
        [/\bsneaker(s)?\b/i, 2], [/\bpolo\b/i, 1], [/\bdenim\b/i, 2],
    ],
}

function inferAttrsEdge(item: any): { styleTags: string[]; materials: string; patterns: string; formality: number } {
    const blob = [item.name, item.description, item.brand, item.color, item.type, item.category, item.macroCategory, item.style]
        .filter(Boolean).join(' ').toLowerCase()

    const materials = MATERIAL_DICT.filter(w => blob.includes(w)).join(', ')
    const patterns = PATTERN_DICT.filter(w => blob.includes(w)).join(', ')

    const scores: Array<[string, number]> = []
    for (const [style, signals] of Object.entries(STYLE_SIGNALS_EDGE)) {
        let score = 0
        for (const [re, w] of signals) {
            if (re.test(blob)) score += w
        }
        scores.push([style, score])
    }
    scores.sort((a, b) => b[1] - a[1])
    const styleTags = scores.filter(s => s[1] > 0).map(s => s[0])
    if (styleTags.length === 0) styleTags.push('casual')

    let formality = 0.5
    if (/\b(suit|blazer|tailored|trouser|oxford|loafer|pinstripe|tweed|cashmere|wool|silk)\b/i.test(blob)) formality += 0.2
    if (/\b(chino|polo|cardigan|linen|merino)\b/i.test(blob)) formality += 0.1
    if (/\b(jeans|t-shirt|tee|sneaker|hoodie|sweatpants|cargo|graphic)\b/i.test(blob)) formality -= 0.15
    if (/\b(baggy|oversized|distressed|athletic)\b/i.test(blob)) formality -= 0.08
    formality = Math.max(0, Math.min(1, formality))

    return { styleTags, materials, patterns, formality }
}

function scoreItemForStyleEdge(item: any, style: string): number {
    const blob = [item.name, item.description, item.brand, item.color, item.type, item.category, item.macroCategory, item.style]
        .filter(Boolean).join(' ').toLowerCase()
    const signals = STYLE_SIGNALS_EDGE[style] || []
    let raw = 0
    for (const [re, w] of signals) if (re.test(blob)) raw += w
    return raw
}

function rankItemsForStyleEdge<T extends { macroCategory?: string }>(
    items: T[],
    style: string,
    opts: { minKeep?: number; dropThreshold?: number; perCategoryFloor?: number } = {},
): T[] {
    const { minKeep = 12, dropThreshold = -2, perCategoryFloor = 3 } = opts
    const scored = items.map(i => ({
        item: i,
        raw: scoreItemForStyleEdge(i, style),
        macro: (i.macroCategory || '').toLowerCase(),
    }))
    scored.sort((a, b) => b.raw - a.raw)

    const keep = new Set<number>()
    scored.forEach((s, idx) => { if (s.raw > dropThreshold) keep.add(idx) })

    // Guarantee every macroCategory contributes some items, so shoes/outerwear
    // never get stripped out even when the global threshold rejects them all.
    if (perCategoryFloor > 0) {
        const perCat = new Map<string, number>()
        scored.forEach((s, idx) => {
            if (!s.macro) return
            const c = perCat.get(s.macro) || 0
            if (c < perCategoryFloor) {
                keep.add(idx)
                perCat.set(s.macro, c + 1)
            }
        })
    }

    const filtered = scored.filter((_, idx) => keep.has(idx))
    const final = filtered.length >= minKeep ? filtered : scored
    return final.map(s => s.item)
}

// ── Style-specific fashion context ───────────────────────────────────────
// Each block has: vibe paragraph + MUST / REJECT rules that the model must
// honor. The REJECT list is what forced the model to stop picking graphic
// tees, cargo shorts, and hoodies when the user asked for "Old Money".
const STYLE_CONTEXT: Record<string, { vibe: string; must: string[]; reject: string[]; palette: string[]; fabrics: string[] }> = {
    old_money: {
        vibe: `Old Money / Quiet Luxury — the wardrobe of someone who summers in the Hamptons, sails in Capri, and owns a Brunello Cucinelli cardigan. Think Ralph Lauren Purple Label, Loro Piana, Brooks Brothers. Nothing flashy, everything expensive. The outfit should feel understated, tonal, and timeless.`,
        must: [
            'Use only tonal / monochromatic / analogous palettes (e.g. camel + cream + white; navy + cream + brown).',
            'Prefer tailored silhouettes: blazers, cardigans, oxford shirts, polo shirts, chinos, trousers, pleated pants.',
            'Footwear must be loafers, oxfords, boat shoes, or minimal clean white/cream sneakers.',
            'Every piece should look like it could be from Ralph Lauren, Brunello Cucinelli, Loro Piana, Massimo Dutti, or Brooks Brothers.',
        ],
        reject: [
            'NEVER include: graphic tees, logo tees, 3-pack basic tees, printed tees, baby tees, tank tops, crop tops.',
            'NEVER include: cargo shorts, athletic shorts, basketball shorts, track pants, sweatpants, joggers.',
            'NEVER include: hoodies, zip-ups, puffers, bomber jackets, denim jackets, graphic sweatshirts.',
            'NEVER include: chunky sneakers, high-tops, platform shoes, athletic trainers.',
            'NEVER include: neon, rhinestone, sequin, metallic, or tie-dye pieces.',
        ],
        palette: ['navy', 'cream', 'ivory', 'beige', 'camel', 'chocolate', 'forest green', 'burgundy', 'white', 'charcoal'],
        fabrics: ['cashmere', 'wool', 'merino', 'linen', 'silk', 'cotton', 'poplin', 'tweed'],
    },
    streetwear: {
        vibe: `Streetwear — oversized silhouettes, bold graphics, sneaker culture. Think Stüssy, Off-White, Supreme, early Virgil, Travis Scott energy. Layering is key; contrast is king.`,
        must: [
            'Favor oversized fits, baggy cuts, and contrast layering.',
            'Graphic tees, hoodies, bombers, and cargo pants are the backbone.',
            'Footwear must be sneakers (chunky, runners, skate), chunky boots, or Y2K-adjacent trainers.',
            'Color-blocking and statement pieces are welcome.',
        ],
        reject: [
            'Avoid: tailored blazers, oxford shirts, wool trousers, loafers, boat shoes.',
            'Avoid: cashmere sweaters, pinstripe suits, herringbone tweed.',
        ],
        palette: ['black', 'white', 'neon', 'bright red', 'bright blue', 'earth tones as contrast'],
        fabrics: ['cotton', 'fleece', 'denim', 'nylon', 'technical'],
    },
    minimalist: {
        vibe: `Minimalist — quiet, clean, intentional. Think COS, Uniqlo U, Acne Studios, The Row, Jil Sander. Every piece should feel essential.`,
        must: [
            'Stick to a tight palette: black, white, grey, beige, navy. Monochromatic outfits are ideal.',
            'Silhouettes must be simple and clean; no busy patterns.',
            'Prefer structured, tailored fits over trendy oversized looks.',
        ],
        reject: [
            'NEVER include: graphic prints, logos, tie-dye, floral prints, neon colors, rhinestones, sequins.',
            'Avoid: color-blocking, chunky sneakers, puffers with branding.',
        ],
        palette: ['black', 'white', 'grey', 'charcoal', 'beige', 'navy', 'stone'],
        fabrics: ['merino', 'wool', 'cotton', 'linen', 'cashmere'],
    },
    y2k: {
        vibe: `Y2K — nostalgic 2000s maximalism. Think Paris Hilton, early Britney, Juicy Couture. Low-rise, cropped, shiny, playful.`,
        must: [
            'Low-rise bottoms, cropped tops, baby tees, tube tops, mini skirts, velour tracksuits.',
            'Metallics, rhinestones, sequins, butterflies, iridescent finishes are all welcome.',
            'Platform shoes, chunky sneakers, or pointy-toe heels.',
        ],
        reject: [
            'Avoid: tailored blazers, oxford shirts, wool trousers, loafers, anything understated.',
        ],
        palette: ['hot pink', 'baby blue', 'metallic silver', 'lime green', 'fuchsia', 'white'],
        fabrics: ['satin', 'velour', 'denim', 'nylon', 'metallic'],
    },
    business_casual: {
        vibe: `Modern Professional — sharp tailoring that still feels comfortable. Hugo Boss meets Everlane meets Theory.`,
        must: [
            'Blazers paired with chinos or tailored trousers. Oxford or poplin button-downs.',
            'Polished shoes: loafers, oxfords, or minimal clean white sneakers.',
            'Palette: navy, charcoal, tan, white, light blue.',
        ],
        reject: [
            'Avoid: hoodies, graphic tees, cargo pants, ripped jeans, sweatpants.',
            'Avoid: sequins, rhinestones, neon colors, crop tops.',
        ],
        palette: ['navy', 'charcoal', 'tan', 'white', 'light blue', 'grey'],
        fabrics: ['wool', 'cotton', 'poplin', 'oxford cloth', 'linen'],
    },
    casual: {
        vibe: `Smart Casual — relaxed but intentional. Well-fitted basics, clean color combinations.`,
        must: [
            'Well-fitted tees or polos with chinos or well-cut jeans.',
            'Clean sneakers, loafers, or desert boots.',
        ],
        reject: [
            'Avoid: sweatpants, basketball shorts, graphic hoodies.',
        ],
        palette: ['navy', 'white', 'grey', 'beige', 'olive'],
        fabrics: ['cotton', 'linen', 'denim', 'wool'],
    },
}

function getStyleContext(style: string): typeof STYLE_CONTEXT[string] {
    const key = style.toLowerCase().replace(/[\s-]+/g, '_')
    return STYLE_CONTEXT[key] || STYLE_CONTEXT.casual
}

function buildSystemPrompt(limit: number, style: string, layered: boolean): string {
    const ctx = getStyleContext(style)
    const slotRules = layered
        ? `- EVERY outfit MUST contain exactly these FOUR macroCategory slots, no exceptions:
    (1) a BASE TOP (macroCategory="top" — t-shirt, shirt, polo, blouse, tank worn next-to-skin)
    (2) a MAIN TOP / OUTERWEAR LAYER (macroCategory="outerwear" — sweater, cardigan, blazer, hoodie, jacket, coat, vest)
    (3) a BOTTOM (macroCategory="bottom" — pants, trousers, jeans, shorts, skirt)
    (4) SHOES (macroCategory="shoes")
  A 3-item outfit is INVALID unless the base top is a dress (in which case skip the bottom slot but STILL include outerwear + shoes). Never return fewer than 4 items for a layered look.`
        : `- EVERY outfit MUST contain exactly these three macroCategory slots, no exceptions: (1) a TOP (or a dress), (2) a BOTTOM (skip this slot only if the top is a dress), and (3) SHOES. Optionally add one OUTERWEAR layer for a 4-item look.`

    return `You are a world-class fashion stylist curating outfits for a specific aesthetic. You must obey the STYLE DIRECTION below as if your reputation depends on it.

TASK: Create ${limit} complete outfits using ONLY items from the provided wardrobe, referenced by their exact id.

GENERAL STYLING RULES:
${slotRules}
- If no shoe item obviously matches the style, pick the LEAST off-aesthetic shoe available. An outfit WITHOUT shoes is invalid — never return one.
- NEVER invent items. Use ONLY the exact ids listed in "Available wardrobe items".
- Pick items whose styleTags, name, description, or material indicate they FIT the requested style. If an item's tags or description clash with the style, do NOT use it (unless it is the only option for that slot — see shoes rule above).
- Favor color harmony (tonal, monochromatic, or analogous palettes from the style's preferred palette). When layering, the base top must color-coordinate with the outerwear (tonal or complementary — never clashing).
- Do not reuse the same item across outfits unless unavoidable.
- Each outfit needs a vivid 1-2 sentence description and 2 actionable styling tips.

STYLE DIRECTION — ${style.toUpperCase()}:
${ctx.vibe}

MUST:
${ctx.must.map(r => `  • ${r}`).join('\n')}

MUST NOT:
${ctx.reject.map(r => `  • ${r}`).join('\n')}

PREFERRED PALETTE: ${ctx.palette.join(', ')}
PREFERRED FABRICS: ${ctx.fabrics.join(', ')}

Self-check BEFORE responding:
  1. Did I pick only items whose description/material/styleTags match ${style}?
  2. Is the color palette coherent with ${style}?
  3. Did I avoid every REJECT category?
If the answer to any of these is "no", rebuild the outfit.

Respond ONLY in pure JSON (no markdown, no code fences):
{"outfits":[{"id":"outfit_1","style":"${style}","occasion":"...","description":"...","confidence":0.9,"items":[{"id":"EXACT_DB_ID","type":"top","color":"cream","name":"...","brand":"...","imageUrl":"...","macroCategory":"top","recommendation":"why this piece works for ${style}"}],"stylingTips":["tip1","tip2"]}]}`
}

function buildUserPrompt(items: any[], prompt: string, style: string, occasion: string, weather: any, limit: number): string {
    // Richer bullets: name + description + inferred styleTags give the model
    // real signal instead of only color + type.
    const itemSummary = items.map(i => {
        const name = i.name || i.type || 'item'
        const desc = (i.description || '').toString().slice(0, 120)
        const tags = Array.isArray(i.styleTags) && i.styleTags.length > 0
            ? i.styleTags.join(',')
            : ''
        const material = i.material || ''
        const pattern = i.pattern || ''
        const formality = typeof i.formality === 'number' ? ` formality:${i.formality.toFixed(2)}` : ''
        const parts = [
            `[${i.id}]`,
            `${name}`,
            `color:${i.color || 'neutral'}`,
            `cat:${i.macroCategory || 'other'}`,
            i.brand ? `brand:${i.brand}` : '',
            material ? `material:${material}` : '',
            pattern ? `pattern:${pattern}` : '',
            tags ? `styleTags:${tags}` : '',
        ].filter(Boolean).join(' | ')
        return `• ${parts}${formality}${desc ? `\n    desc: "${desc}"` : ''}`
    }).join('\n')

    return `${prompt ? `User's specific request: "${prompt}"\n\n` : ''}Requested style: ${style || 'Casual'}
${occasion ? `Occasion: ${occasion}\n` : ''}${weather ? `Weather: ${weather.temp}°C, ${weather.condition}\n` : ''}
Available wardrobe items (use ONLY these exact ids; every id is pre-filtered to be at least plausible for ${style}, but you MUST still reject any item whose name/description/styleTags clash with the style):
${itemSummary}

Create ${limit} distinct, on-aesthetic outfits. Each outfit must feel like it was hand-picked by a ${style} stylist, not a random color-matching algorithm.`
}

// ── Validator: reject outfits that violate the style's REJECT list ───────
// Runs AFTER the model responds. If too many outfits clash with the style,
// we can trigger a retry. This catches the failure mode where the model
// obeys color rules but picks, say, a graphic tee for Old Money.
function validateOutfitAgainstStyle(outfit: any, style: string, itemMap: Map<string, any>, layered: boolean): { ok: boolean; reason?: string } {
    const ctx = getStyleContext(style)
    const items = Array.isArray(outfit.items) ? outfit.items : []
    const minItems = layered ? 4 : 3
    if (items.length < minItems) return { ok: false, reason: `too few items (${items.length}, need ${minItems})` }

    // Require coverage per slot model.
    const macros = new Set<string>()
    let hasDress = false
    for (const it of items) {
        const src = itemMap.get(it.id) || {}
        const macro = (src.macroCategory || it.macroCategory || '').toLowerCase()
        if (macro) macros.add(macro)
        if (isDressItem(src) || isDressItem(it)) hasDress = true
    }
    const hasBaseTop = macros.has('top')
    const hasOuter = macros.has('outerwear')
    const hasBottom = macros.has('bottom') || hasDress
    const hasShoes = macros.has('shoes')
    if (!hasShoes) return { ok: false, reason: `missing shoes slot (macros: ${Array.from(macros).join(',')})` }
    if (!hasBottom) return { ok: false, reason: `missing bottom slot (macros: ${Array.from(macros).join(',')})` }
    if (layered) {
        // For layered looks we need both a base top AND outerwear, unless the
        // base top is a dress (which replaces base-top + bottom in one piece).
        if (!hasOuter) return { ok: false, reason: `missing outerwear/main-top slot (macros: ${Array.from(macros).join(',')})` }
        if (!hasBaseTop && !hasDress) return { ok: false, reason: `missing base-top slot (macros: ${Array.from(macros).join(',')})` }
    } else {
        if (!hasBaseTop && !hasOuter) return { ok: false, reason: `missing top/outerwear slot (macros: ${Array.from(macros).join(',')})` }
    }

    // Check the style's explicit reject keywords on every item.
    const rejectKeywords = extractRejectKeywords(ctx.reject)
    for (const it of items) {
        const src = itemMap.get(it.id) || {}
        const blob = `${it.name || ''} ${src.name || ''} ${src.description || ''} ${src.type || ''} ${it.type || ''}`.toLowerCase()
        for (const kw of rejectKeywords) {
            if (blob.includes(kw)) {
                return { ok: false, reason: `item "${it.name || it.id}" contains rejected keyword "${kw}" for style ${style}` }
            }
        }
    }
    return { ok: true }
}

function extractRejectKeywords(rejectRules: string[]): string[] {
    // Pull out the comma-separated nouns after "NEVER include:" / "Avoid:" so we
    // can substring-match item names and descriptions. Lowercased.
    const out: string[] = []
    for (const rule of rejectRules) {
        const m = rule.match(/(?:NEVER include|Avoid):\s*(.+?)\.?$/i)
        if (!m) continue
        const parts = m[1].split(/,|;/).map(s => s.trim().toLowerCase()).filter(Boolean)
        for (const p of parts) {
            // Keep multi-word phrases like "graphic tees" intact; also seed the
            // first word as a fallback (e.g. "graphic").
            out.push(p)
            const firstWord = p.split(/\s+/)[0]
            if (firstWord && firstWord !== p && firstWord.length > 3) out.push(firstWord)
        }
    }
    return Array.from(new Set(out))
}

function filterValidOutfits(outfits: any[], style: string, itemMap: Map<string, any>, layered: boolean): any[] {
    const kept: any[] = []
    for (const o of outfits) {
        const v = validateOutfitAgainstStyle(o, style, itemMap, layered)
        if (v.ok) kept.push(o)
        else console.log(`[validate] rejected outfit for ${style} (layered=${layered}): ${v.reason}`)
    }
    return kept
}

function localFallback(items: any[], style: string, occasion: string, limit: number, layered: boolean): any[] {
    const baseTops = items.filter(i => i.macroCategory === 'top')
    const outerwear = items.filter(i => i.macroCategory === 'outerwear')
    const legacyTops = items.filter(i => ['top','outerwear'].includes(i.macroCategory))
    const bottoms = items.filter(i => i.macroCategory === 'bottom')
    const shoes = items.filter(i => i.macroCategory === 'shoes')
    const outfits: any[] = []
    const seed = layered ? Math.max(baseTops.length, outerwear.length, 1) : Math.max(legacyTops.length, 1)

    for (let i = 0; i < Math.min(limit, seed); i++) {
        const parts: any[] = []
        if (layered) {
            // Always include outerwear slot (use any top if no outerwear available)
            const outer = outerwear[i % Math.max(outerwear.length, 1)]
            const base = baseTops[i % Math.max(baseTops.length, 1)]
            // Use outerwear if available, otherwise fallback to any top item
            const mainTop = outer || legacyTops[i % Math.max(legacyTops.length, 1)]
            if (mainTop) parts.push({ ...mainTop, recommendation: 'Main top / outerwear layer' })
            // Always include base top (use baseTop if available, otherwise reuse mainTop or any top)
            const baseTopItem = base || outer || legacyTops[(i + 1) % Math.max(legacyTops.length, 1)]
            if (baseTopItem) parts.push({ ...baseTopItem, recommendation: 'Base top worn underneath' })
        } else {
            const top = legacyTops[i % Math.max(legacyTops.length, 1)]
            if (top) parts.push({ ...top, recommendation: 'Key piece' })
        }
        // Always include bottom and shoes - reuse items if necessary
        const bottom = bottoms[i % Math.max(bottoms.length, 1)] || bottoms[0]
        const shoe = shoes[i % Math.max(shoes.length, 1)] || shoes[0]
        if (bottom) parts.push({ ...bottom, recommendation: 'Pairs well' })
        if (shoe) parts.push({ ...shoe, recommendation: 'Completes the look' })
        // Ensure minimum items for valid outfit (3 for non-layered, 4 for layered)
        const minItems = layered ? 4 : 3
        if (parts.length < minItems && items.length > 0) {
            // Fill remaining slots with available items
            while (parts.length < minItems && parts.length < items.length) {
                const fillItem = items[parts.length % items.length]
                parts.push({ ...fillItem, recommendation: 'Complementary piece' })
            }
        }
        if (!parts.length) continue
        outfits.push({
            id: `local_${i}_${Date.now()}`,
            style: style || 'Casual',
            occasion: occasion || 'Everyday',
            description: `A ${style || 'casual'} look from your wardrobe.`,
            confidence: 0.75,
            items: parts,
            stylingTips: layered
                ? ['Layer the base top under the outerwear for depth', 'Keep palette tonal for a refined finish']
                : ['Add accessories to personalize', 'Layer for depth'],
        })
    }
    // If no outfits generated, create at least one with available items
    if (outfits.length === 0 && items.length > 0) {
        const parts: any[] = []
        if (layered) {
            // For layered: try to get 4 items (outerwear, base, bottom, shoes)
            const outer = outerwear[0] || legacyTops[0]
            const base = baseTops[0] || legacyTops[0] || outer
            const bottom = bottoms[0] || items[0]
            const shoe = shoes[0] || items[1] || items[0]
            if (outer) parts.push({ ...outer, recommendation: 'Main top / outerwear layer' })
            if (base && base.id !== outer?.id) parts.push({ ...base, recommendation: 'Base top worn underneath' })
            if (bottom) parts.push({ ...bottom, recommendation: 'Pairs well' })
            if (shoe && shoe.id !== bottom?.id) parts.push({ ...shoe, recommendation: 'Completes the look' })
        } else {
            // For non-layered: try to get 3 items (top, bottom, shoes)
            const top = legacyTops[0] || items[0]
            const bottom = bottoms[0] || items[1] || items[0]
            const shoe = shoes[0] || items[2] || items[0]
            if (top) parts.push({ ...top, recommendation: 'Key piece' })
            if (bottom && bottom.id !== top?.id) parts.push({ ...bottom, recommendation: 'Pairs well' })
            if (shoe && shoe.id !== bottom?.id) parts.push({ ...shoe, recommendation: 'Completes the look' })
        }
        if (parts.length > 0) {
            outfits.push({
                id: `local_0_${Date.now()}`,
                style: style || 'Casual',
                occasion: occasion || 'Everyday',
                description: `A ${style || 'casual'} look from your wardrobe.`,
                confidence: 0.75,
                items: parts,
                stylingTips: layered
                    ? ['Layer the base top under the outerwear for depth', 'Keep palette tonal for a refined finish']
                    : ['Add accessories to personalize', 'Layer for depth'],
            })
        }
    }
    return outfits.length ? outfits : [{
        id: `local_0_${Date.now()}`,
        style: style || 'Casual',
        occasion: occasion || 'Everyday',
        description: 'Add more items to your wardrobe for better suggestions.',
        confidence: 0.5,
        items: items.slice(0, layered ? 4 : 3).map(i => ({ ...i, recommendation: 'From your wardrobe' })),
        stylingTips: ['Scan more items'],
    }]
}

// ── Safe JSON parse with regex fallback ──────────────────────────────────
function safeParseOutfits(raw: string): any[] | null {
    try {
        const parsed = JSON.parse(raw)
        if (Array.isArray(parsed?.outfits) && parsed.outfits.length > 0) return parsed.outfits
        if (Array.isArray(parsed) && parsed.length > 0) return parsed
    } catch (_) { /* try regex fallback */ }
    // Regex fallback: extract JSON object containing "outfits"
    const match = raw.match(/\{[\s\S]*"outfits"\s*:\s*\[[\s\S]*\][\s\S]*\}/)
    if (match) {
        try {
            const parsed = JSON.parse(match[0])
            if (Array.isArray(parsed.outfits)) return parsed.outfits
        } catch (_) { /* give up */ }
    }
    // Try extracting just the array
    const arrMatch = raw.match(/\[[\s\S]*\]/)
    if (arrMatch) {
        try {
            const arr = JSON.parse(arrMatch[0])
            if (Array.isArray(arr) && arr.length > 0 && arr[0].items) return arr
        } catch (_) { /* give up */ }
    }
    return null
}

// ── NVIDIA call with retry ───────────────────────────────────────────────
// Default to a faster free model. You can override from env/app_config:
// `NVIDIA_TEXT_MODEL` / `NVIDIA_MODEL` / app_config key `nvidia_model`.
const DEFAULT_NVIDIA_MODEL = 'meta/llama-3.1-8b-instruct'
function resolveNvidiaModel(input: string | null | undefined): string {
    const model = (input || '').trim()
    return model.length > 0 ? model : DEFAULT_NVIDIA_MODEL
}
async function callNvidia(key: string, model: string, sys: string, usr: string, retries = 0): Promise<any[]|null> {
    for (let attempt = 0; attempt <= retries; attempt++) {
        const controller = new AbortController()
        const timer = setTimeout(() => controller.abort(), 30_000)
        try {
            console.log(`[NVIDIA] attempt ${attempt + 1}, model: ${model}`)
            const r = await fetch('https://integrate.api.nvidia.com/v1/chat/completions', {
                method:'POST', headers:{'Content-Type':'application/json','Authorization':`Bearer ${key}`},
                body: JSON.stringify({ model, messages:[{role:'system',content:sys},{role:'user',content:usr}], temperature:0.4, max_tokens:1400, response_format:{type:'json_object'} }),
                signal: controller.signal,
            })
            clearTimeout(timer)
            if (r.status === 429 || r.status >= 500) {
                console.warn(`[NVIDIA] ${r.status}, retrying in 2s...`)
                if (attempt < retries) { await new Promise(r => setTimeout(r, 2000)); continue }
                return null
            }
            if (!r.ok) { console.error('[NVIDIA] error:', r.status); return null }
            const d = await r.json()
            const content = d.choices?.[0]?.message?.content || '{}'
            console.log(`[NVIDIA] response length: ${content.length} chars`)
            const outfits = safeParseOutfits(content)
            if (outfits && outfits.length > 0) {
                console.log(`[NVIDIA] parsed ${outfits.length} outfits`)
                return outfits
            }
            console.warn('[NVIDIA] returned unparseable content')
            return null
        } catch(e) {
            clearTimeout(timer)
            console.error('[NVIDIA] call error:', e)
            if (attempt < retries) { await new Promise(r => setTimeout(r, 1500)); continue }
            return null
        }
    }
    return null
}

// ── Gemini fallback call with retry ──────────────────────────────────────
async function callGemini(key: string, sys: string, usr: string, retries = 0): Promise<any[]|null> {
    for (let attempt = 0; attempt <= retries; attempt++) {
        const controller = new AbortController()
        const timer = setTimeout(() => controller.abort(), 25_000)
        try {
            console.log(`[Gemini] Attempt ${attempt + 1}`)
            const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${key}`, {
                method:'POST', headers:{'Content-Type':'application/json'},
                body: JSON.stringify({ contents:[{parts:[{text:`${sys}\n\n${usr}`}]}], generationConfig:{responseMimeType:'application/json',temperature:0.7,maxOutputTokens:2000} }),
                signal: controller.signal,
            })
            clearTimeout(timer)
            if (r.status === 429 || r.status >= 500) {
                console.warn(`[Gemini] ${r.status}, retrying...`)
                if (attempt < retries) { await new Promise(r => setTimeout(r, 2000)); continue }
                return null
            }
            if (!r.ok) { console.error('[Gemini] error:', r.status); return null }
            const d = await r.json()
            const content = d.candidates?.[0]?.content?.parts?.[0]?.text || '{}'
            console.log(`[Gemini] Response length: ${content.length} chars`)
            const outfits = safeParseOutfits(content)
            if (outfits && outfits.length > 0) {
                console.log(`[Gemini] Parsed ${outfits.length} outfits`)
                return outfits
            }
            console.warn('[Gemini] Unparseable content')
            return null
        } catch(e) {
            clearTimeout(timer)
            console.error('[Gemini] call error:', e)
            if (attempt < retries) { await new Promise(r => setTimeout(r, 1500)); continue }
            return null
        }
    }
    return null
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
        useProvidedWardrobeOnly = false,
    } = body

    try {
        // ── 1. Authenticate user via JWT ──────────────────────────────────
        const authHeader = req.headers.get('Authorization') || ''
        let dbClothingItems: any[] = []

        if (authHeader && !useProvidedWardrobeOnly) {
            const userSupabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
                global: { headers: { Authorization: authHeader } }
            })
            const { data: { user } } = await userSupabase.auth.getUser()

            if (user) {
                // ── 2. Fetch clothing items from DB ───────────────────────
                let q = userSupabase
                    .from('clothing_items')
                    .select('id, type, category, color, style, brand, season, occasion, pattern, material, image_url, description, primary_color, sub_category, wear_count')
                    .order('wear_count', { ascending: false })
                    .limit(30)

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
        // IMPORTANT: the legacy payload now includes name/description/styleTags
        // from the shop catalog — we preserve those instead of blanking them.
        let wardrobeItems = (dbClothingItems.length > 0 && !useProvidedWardrobeOnly)
            ? dbClothingItems
            : (legacyWardrobe || []).map((item: any) => ({
                ...item,
                macroCategory: item.macroCategory || macroCategory(item.category || '', item.type || ''),
            }))

        // ── 3b. Attach inferred style metadata to every item, regardless of
        // source. If the client already shipped styleTags (shop payload does),
        // we keep theirs; otherwise we compute from name/description/brand.
        wardrobeItems = wardrobeItems.map((item: any) => {
            const hasTags = Array.isArray(item.styleTags) && item.styleTags.length > 0
            if (hasTags && item.material !== undefined) return item
            const inferred = inferAttrsEdge(item)
            return {
                ...item,
                styleTags: hasTags ? item.styleTags : inferred.styleTags,
                material: item.material || inferred.materials,
                pattern: item.pattern || inferred.patterns,
                formality: typeof item.formality === 'number' ? item.formality : inferred.formality,
            }
        })

        // ── 3c. Rank + trim the pool for the requested style so the LLM only
        // sees plausible candidates. In manual mode (selectedItemIds) we keep
        // the user's picks as-is. The client already pre-filters shop items,
        // so this mainly helps the wardrobe + legacy branches.
        const styleKey = (stylePreferences || 'casual').toLowerCase().replace(/[\s-]+/g, '_')
        if (!selectedItemIds || selectedItemIds.length === 0) {
            const before = wardrobeItems.length
            wardrobeItems = rankItemsForStyleEdge(wardrobeItems, styleKey, { minKeep: 16, dropThreshold: -2, perCategoryFloor: 4 }).slice(0, 40)
            console.log(`[rank] ${before} -> ${wardrobeItems.length} items after style filter for ${styleKey}`)
        }

        console.log(`Outfit gen: ${wardrobeItems.length} items, prompt="${prompt}", style=${stylePreferences}`)

        if (wardrobeItems.length === 0) {
            return new Response(JSON.stringify({ success: false, outfits: [], error: 'No clothing items found in your wardrobe. Add items first!' }), {
                headers: { ...corsHeaders, 'Content-Type': 'application/json' }
            })
        }

        // ── 4. Get API keys ───────────────────────────────────────────────
        let nvidiaKey = Deno.env.get('NVIDIA_API_KEY') || ''
        let nvidiaModel = resolveNvidiaModel(Deno.env.get('NVIDIA_TEXT_MODEL') || Deno.env.get('NVIDIA_MODEL'))
        let geminiKey = Deno.env.get('GEMINI_API_KEY') || ''

        if (SUPABASE_SERVICE_ROLE_KEY && (!nvidiaKey || !geminiKey)) {
            const svc = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
            const { data: cfg } = await svc
                .from('app_config')
                .select('key, value')
                .in('key', ['nvidia_token', 'gemma', 'gemini_token', 'nvidia_model', 'nvidia_text_model'])
            if (cfg) {
                for (const row of cfg) {
                    if ((row.key === 'nvidia_token' || row.key === 'gemma') && !nvidiaKey) nvidiaKey = row.value
                    if (row.key === 'gemini_token' && !geminiKey) geminiKey = row.value
                    if ((row.key === 'nvidia_model' || row.key === 'nvidia_text_model') && (!nvidiaModel || nvidiaModel === DEFAULT_NVIDIA_MODEL)) {
                        nvidiaModel = resolveNvidiaModel(row.value)
                    }
                }
            }
        }

        // ── 5. Decide layering policy (2-top layered composition vs classic 3-slot) ──
        const layered = needsLayering(stylePreferences, weather, prompt)
        console.log(`[layer] style=${styleKey} weather=${weather?.temp ?? '?'}°C layered=${layered}`)

        // ── 5b. No AI key → local fallback ────────────────────────────────
        if (!nvidiaKey && !geminiKey) {
            const outfits = localFallback(wardrobeItems, stylePreferences, occasion, limit, layered)
            return new Response(JSON.stringify({ success: true, outfits, source: 'local', layered }), {
                headers: { ...corsHeaders, 'Content-Type': 'application/json' }
            })
        }

        // ── 6. Build prompts ───────────────────────────────────────────────
        const sys = buildSystemPrompt(limit, stylePreferences, layered)
        const usr = buildUserPrompt(wardrobeItems, prompt, stylePreferences, occasion, weather, limit)

        // ── 7. Call AI with validator-retry ─────────────────────────────────
        // After generation, run every outfit through validateOutfitAgainstStyle;
        // if >50% fail, call the model once more with a stricter correction
        // prompt. This is what turns "mostly on-style" into "reliably on-style".
        const itemMap = new Map<string, any>(wardrobeItems.map((i: any) => [String(i.id), i] as [string, any]))
        let aiOutfits: any[] | null = null
        if (nvidiaKey) aiOutfits = await callNvidia(nvidiaKey, nvidiaModel, sys, usr)
        if (!aiOutfits && geminiKey) aiOutfits = await callGemini(geminiKey, sys, usr)

        if (aiOutfits && aiOutfits.length > 0) {
            const valid = filterValidOutfits(aiOutfits, styleKey, itemMap, layered)
            const passRate = valid.length / aiOutfits.length
            console.log(`[validate] ${valid.length}/${aiOutfits.length} outfits passed style check for ${styleKey} (layered=${layered})`)

            if (valid.length >= Math.max(1, Math.min(limit, aiOutfits.length) - 1)) {
                aiOutfits = valid
            } else if (passRate < 0.5 && (nvidiaKey || geminiKey)) {
                console.log(`[validate] pass rate ${passRate.toFixed(2)} too low — retrying with correction`)
                const layeredReqs = layered
                    ? `  1. EVERY outfit MUST be a FOUR-slot layered look with ALL of these macroCategories present: "top" (base layer — t-shirt, shirt, polo, blouse), "outerwear" (main top — sweater, blazer, cardigan, hoodie, jacket), "bottom" (pants/trousers/jeans/shorts/skirt), "shoes". A 3-item outfit is INVALID.\n  2. The base top must sit UNDER the outerwear and color-coordinate with it. Think: white oxford shirt under a navy blazer, or cream tee under a camel cardigan.\n  3. Do NOT return two base tops or two outerwear pieces — exactly one of each.`
                    : `  1. EVERY outfit MUST include an item with macroCategory=shoes. No outfit without shoes is acceptable.\n  2. EVERY outfit MUST include a top (or outerwear) AND a bottom.`
                const correctionUsr = `${usr}\n\nIMPORTANT CORRECTION: Your previous attempt failed because (a) some items violated the ${styleKey} aesthetic, OR (b) outfits were missing required slots. REQUIREMENTS:\n${layeredReqs}\n  ${layered ? '4' : '3'}. Reject any item whose name/description contains: hoodie (unless streetwear), graphic tee, logo tee, cargo, ripped, sweatpants, neon, sequin, rhinestone, chunky sneaker, basketball, skate sneaker (for old_money / business_casual / minimalist).\n  ${layered ? '5' : '4'}. For ${styleKey} shoes, prefer: loafers, penny loafers, bit loafers, dress shoes, boat shoes, derbies, leather shoes, minimal leather sneakers. Avoid chunky, basketball, or skate sneakers.\nTry again and return ${limit} outfits that each have ${layered ? 'outerwear+base-top+bottom+shoes' : 'top+bottom+shoes'}.`
                let retry: any[] | null = null
                if (nvidiaKey) retry = await callNvidia(nvidiaKey, nvidiaModel, sys, correctionUsr)
                if (!retry && geminiKey) retry = await callGemini(geminiKey, sys, correctionUsr)
                if (retry && retry.length > 0) {
                    const retryValid = filterValidOutfits(retry, styleKey, itemMap, layered)
                    aiOutfits = retryValid.length > valid.length ? retryValid : (valid.length > 0 ? valid : retry)
                } else if (valid.length > 0) {
                    aiOutfits = valid
                }
            } else if (valid.length > 0) {
                aiOutfits = valid
            }
        }

        if (!aiOutfits || aiOutfits.length === 0) {
            aiOutfits = localFallback(wardrobeItems, stylePreferences, occasion, limit, layered)
        }

        // ── 8. Enrich items with imageUrl from DB ──────────────────────────
        const enriched = aiOutfits.map((outfit: any) => ({
            ...outfit,
            items: (outfit.items || []).map((item: any) => {
                const src: any = itemMap.get(item.id) || {}
                // Prioritize DB URL (src.imageUrl) over AI's potentially empty imageUrl
                // The DB URL is the known-good source; AI may return items without image URLs
                const imageUrl = src.imageUrl || item.imageUrl || item.image_url || ''
                return {
                    ...item,
                    imageUrl,
                    color: item.color || src.color || 'neutral',
                    type: item.type || src.type || 'clothing',
                    name: src.name || item.name || src.type || src.category || 'Item',
                    brand: item.brand || src.brand || '',
                    macroCategory: src.macroCategory || item.macroCategory || '',
                }
            }),
        }))

        return new Response(JSON.stringify({ success: true, outfits: enriched, source: 'ai', layered }), {
            headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        })

    } catch (error: any) {
        console.error('generate-outfits error:', error)
        return new Response(JSON.stringify({ success: false, error: error.message, outfits: [] }), {
            headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 500
        })
    }
})
