import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from "https://esm.sh/@supabase/supabase-js@2"

const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

function macroCategory(category: string, type: string, name?: string): string {
    const t = `${category} ${type} ${name || ''}`.toLowerCase()
    if (t.match(/jacket|coat|blazer|hoodie|cardigan|sweater|pullover|vest|puffer|parka|trench|outerwear/)) return 'outerwear'
    if (t.match(/shirt|t-shirt|tshirt|tee|blouse|polo|tops|top|upper[_\s-]?body/)) return 'top'
    if (t.match(/pant|trouser|jeans|bottom|shorts|skirt|lower[_\s-]?body/)) return 'bottom'
    if (t.match(/shoe|sneaker|boot|loafer|sandal|heel|footwear|trainer/)) return 'shoes'
    if (t.match(/dress/)) return 'top'
    return 'other'
}

function isDressItem(item: any): boolean {
    const blob = `${item?.type || ''} ${item?.name || ''} ${item?.category || ''} ${item?.sub_category || ''}`.toLowerCase()
    return /\bdress(es)?\b/.test(blob)
}

// Decide whether an outfit needs a layered composition (outerwear + top +
// bottom + shoes). Layering now follows WEATHER ONLY — the user confirmed
// the composition rule is "Top + Bottom + Shoes always; Layer only when
// cold". Style alone no longer forces a jacket in summer.
function needsLayering(_style: string | null | undefined, weather: any, prompt: string | null | undefined): boolean {
    const promptBlob = (prompt || '').toLowerCase()
    // Explicit "hot" / "summer" prompt keywords still opt the user OUT of
    // layered composition, even when weather is missing.
    if (/\b(summer|hot|heatwave|tee[-\s]?only|no jacket|no outerwear|beach)\b/.test(promptBlob)) return false

    const condition = (weather?.condition || '').toString().toLowerCase()
    const temp = typeof weather?.temp === 'number' ? weather.temp : null

    // When we have NEITHER a temperature nor a weather description, default
    // to layered=true. This matches the client's `needsOuterwear` default
    // and keeps outfits from degrading to 3-item looks that then render as
    // "layer + pants + shoes with no base top" on the client. A warm-weather
    // user will only hit this branch if Location permission is denied, in
    // which case a safe-layered look is the least-broken fallback.
    if (temp == null && !condition) return true

    const coldTemp = temp != null && temp < 18
    const coldCondition = /\b(cold|chilly|freezing|snow|rain|drizzle|wind|storm|cool)\b/.test(condition)
    return coldTemp || coldCondition
}

// Formal outerwear = blazer/suit jacket/overcoat/topcoat/trench/peacoat/sport
// coat/tuxedo. Never paired with shorts per the user's "never with formal
// layers" rule. Casual layers (denim jacket, cardigan, hoodie, bomber,
// puffer, windbreaker, fleece) intentionally fall through.
function isFormalLayerItem(item: any): boolean {
    const blob = `${item?.type || ''} ${item?.name || ''} ${item?.sub_category || ''} ${item?.subCategory || ''} ${item?.description || ''}`.toLowerCase()
    const macro = String(item?.macroCategory || '').toLowerCase()
    const isOuter = macro === 'outerwear' || /jacket|coat|blazer|vest|outerwear/.test(blob)
    if (!isOuter) return false
    return /\b(blazer|suit\s*jacket|sport\s*coat|sports\s*coat|overcoat|top\s*coat|topcoat|trench|peacoat|pea\s*coat|tuxedo)\b/.test(blob)
}

function isShortsItem(item: any): boolean {
    const blob = `${item?.type || ''} ${item?.name || ''} ${item?.sub_category || ''} ${item?.subCategory || ''} ${item?.description || ''}`.toLowerCase()
    const macro = String(item?.macroCategory || '').toLowerCase()
    const isBottom = macro === 'bottom' || /pant|trouser|jeans|bottom|shorts?|skirt|lower[_\s-]?body/.test(blob)
    if (!isBottom) return false
    return /\b(shorts?|bermudas?)\b/.test(blob)
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
        [/\bknit polo\b/i, 4], [/\bpolo\b/i, 2], [/\bcardigan\b/i, 3], [/\bchino(s)?\b/i, 3], [/\btrouser(s)?\b/i, 3],
        [/\btailored\b/i, 3], [/\bpleated\b/i, 3], [/\bpinstripe\b/i, 2], [/\bhoundstooth\b/i, 3],
        [/\btweed\b/i, 3], [/\bherringbone\b/i, 3], [/\bflannel\b/i, 3],
        [/\bsuit\b/i, 3], [/\bknit sweater\b/i, 3], [/\bturtleneck\b/i, 3],
        [/\bcream\b/i, 2], [/\bcamel\b/i, 3], [/\bnavy\b/i, 2], [/\bbeige\b/i, 2],
        [/\bburgundy\b/i, 2], [/\bforest\b/i, 2], [/\btan\b/i, 2],
        [/\bralph lauren\b/i, 5], [/\bbrunello\b/i, 5], [/\bloro piana\b/i, 5], [/\bmassimo dutti\b/i, 3],
        // Shoes-specific positives
        [/\bpenny loafer(s)?\b/i, 5], [/\bbit loafer(s)?\b/i, 5], [/\bmetal bit\b/i, 4],
        [/\bboat shoe(s)?\b/i, 4], [/\bderby\b/i, 4], [/\bdress shoe(s)?\b/i, 4],
        [/\bleather shoe(s)?\b/i, 3], [/\bleather loafer(s)?\b/i, 5],
        [/\bsuede\b/i, 3], [/\bmoc toe\b/i, 2], [/\bdouble buckle\b/i, 3],
        // Negatives
        [/\bhoodie\b/i, -4], [/\bgraphic\b/i, -3], [/\blogo\b/i, -2], [/\boversized\b/i, -2],
        [/\bcargo\b/i, -3], [/\btrack pants\b/i, -4], [/\bsweatpants\b/i, -3],
        [/\bripped\b/i, -3], [/\bdistressed\b/i, -3], [/\bneon\b/i, -4],
        [/\bsequin\b/i, -4], [/\brhinestone\b/i, -4],
        [/\bsquare[-\s]?toe\b/i, -4], [/\blow[-\s]?rise\b/i, -3], [/\bskinny\b/i, -2],
        [/\bbackpack\b/i, -2],
        // Shoes to avoid
        [/\bchunky sneaker(s)?\b/i, -4], [/\bbasketball\b/i, -5],
        [/\bskate sneaker(s)?\b/i, -4], [/\bthick[-\s]?soled\b/i, -3],
        [/\bretro sneaker(s)?\b/i, -2], [/\brope lace\b/i, -2],
        [/\bsneaker(s)?\b/i, -1], [/\bchunky\b/i, -3],
        [/\bheavyweight\s+tee\b/i, -2], [/\b3[-\s]?pack\b/i, -2],
    ],
    semi_classic: [
        [/\bblazer\b/i, 3], [/\bcardigan\b/i, 3], [/\bknit polo\b/i, 3], [/\bturtleneck\b/i, 2],
        [/\bchino(s)?\b/i, 3], [/\bslacks?\b/i, 2], [/\bloafer(s)?\b/i, 3], [/\bsuede\b/i, 3],
        [/\bdesert boot\b/i, 3], [/\bmerino\b/i, 2], [/\blinen\b/i, 2], [/\bpoplin\b/i, 2],
        [/\btailored\b/i, 2], [/\bstructured\b/i, 2], [/\bregular[-\s]?fit\b/i, 2], [/\brelaxed[-\s]?fit\b/i, 1],
        [/\bnavy\b/i, 2], [/\bcream\b/i, 2], [/\bbeige\b/i, 2], [/\bcamel\b/i, 2],
        [/\bolive\b/i, 2], [/\btan\b/i, 2], [/\bburgundy\b/i, 2], [/\bwhite\b/i, 1],
        [/\bsneaker(s)?\b/i, 1], [/\bjeans\b/i, 1], [/\bdenim\b/i, 1],
        [/\bsweater\b/i, 2], [/\bpullover\b/i, 1], [/\bvest\b/i, 1],
        [/\bhoodie\b/i, -2], [/\bgraphic\b/i, -2], [/\bcargo\b/i, -3],
        [/\bsweatpants\b/i, -3], [/\bjoggers\b/i, -3], [/\btrack\b/i, -2],
        [/\bneon\b/i, -4], [/\bsequin\b/i, -4], [/\brhinestone\b/i, -4],
        [/\boversized\b/i, -2], [/\bbaggy\b/i, -3], [/\bripped\b/i, -2],
    ],
    minimalist: [
        [/\bminimal\b/i, 4], [/\bessential\b/i, 3], [/\bbasic\b/i, 2], [/\bplain\b/i, 3],
        [/\bmerino\b/i, 3], [/\bwool\b/i, 2], [/\bcos\b/i, 5], [/\buniqlo\b/i, 4],
        [/\bcrew neck\b/i, 2], [/\bturtleneck\b/i, 2], [/\bknit polo\b/i, 3],
        [/\bblack\b/i, 2], [/\bwhite\b/i, 2], [/\bgrey\b/i, 2], [/\bgray\b/i, 2],
        [/\bbeige\b/i, 2], [/\bstone\b/i, 2], [/\bcream\b/i, 2], [/\bsand\b/i, 2],
        [/\bsuede\b/i, 2], [/\bpoplin\b/i, 2], [/\blinen\b/i, 2],
        [/\bgraphic\b/i, -4], [/\blogo\b/i, -3], [/\bneon\b/i, -5], [/\bsequin\b/i, -5],
        [/\bcolor[-\s]?block\b/i, -3],
    ],
    business_casual: [
        [/\bblazer\b/i, 4], [/\bchino(s)?\b/i, 4], [/\btrouser(s)?\b/i, 4], [/\boxford\b/i, 4],
        [/\bbutton[-\s]?down\b/i, 3], [/\bloafer(s)?\b/i, 4], [/\bdress shirt\b/i, 4],
        [/\bdress shoe(s)?\b/i, 4], [/\bderby\b/i, 4], [/\bpenny loafer(s)?\b/i, 4],
        [/\bbit loafer(s)?\b/i, 4], [/\bleather loafer(s)?\b/i, 4], [/\bleather shoe(s)?\b/i, 3],
        [/\bpoplin\b/i, 3], [/\btailored\b/i, 4], [/\bknit polo\b/i, 3],
        [/\bturtleneck\b/i, 3], [/\bcardigan\b/i, 3],
        [/\bflannel\b/i, 3], [/\bherringbone\b/i, 3], [/\btweed\b/i, 3], [/\bsuede\b/i, 3],
        [/\bnavy\b/i, 2], [/\bcharcoal\b/i, 2], [/\btan\b/i, 2], [/\bcream\b/i, 2],
        [/\bbrown\b/i, 2], [/\bburgundy\b/i, 2], [/\blight blue\b/i, 2],
        [/\bhoodie\b/i, -4], [/\bgraphic\b/i, -3], [/\bcargo\b/i, -4], [/\bripped\b/i, -4],
        [/\bchunky\b/i, -3], [/\bbasketball\b/i, -5], [/\bskate\b/i, -3],
        [/\blow[-\s]?rise\b/i, -3], [/\bskinny\b/i, -2], [/\bsquare[-\s]?toe\b/i, -4],
    ],
    casual: [
        [/\bt[-\s]?shirt\b/i, 2], [/\btee\b/i, 2], [/\bjeans\b/i, 2], [/\bsweater\b/i, 2],
        [/\bsneaker(s)?\b/i, 2], [/\bpolo\b/i, 1], [/\bdenim\b/i, 2],
        [/\bknit\b/i, 2], [/\bslacks?\b/i, 2], [/\bcardigan\b/i, 2], [/\bloafer(s)?\b/i, 2],
        [/\bchino(s)?\b/i, 2], [/\bcanvas\b/i, 1], [/\bsuede\b/i, 2],
        [/\bbasketball\b/i, -3], [/\bgraphic hoodie\b/i, -3], [/\bsweatpants\b/i, -2],
    ],
    classic: [
        // Core tailored pieces
        [/\bsuit\b/i, 5], [/\bblazer\b/i, 4], [/\bsport coat\b/i, 4], [/\bsport jacket\b/i, 4],
        [/\btailored\b/i, 4], [/\btrouser(s)?\b/i, 4], [/\bpleated\b/i, 3], [/\bslacks?\b/i, 3],
        [/\boxford\b/i, 4], [/\bbutton[-\s]?down\b/i, 3], [/\bdress shirt\b/i, 5],
        [/\bcardigan\b/i, 3], [/\bturtleneck\b/i, 4], [/\bknit polo\b/i, 4],
        // Fabrics
        [/\bflannel\b/i, 4], [/\btweed\b/i, 4], [/\bherringbone\b/i, 4], [/\bcashmere\b/i, 4],
        [/\bmerino\b/i, 3], [/\bwool\b/i, 3], [/\blinen\b/i, 2], [/\bsilk\b/i, 2],
        [/\bpoplin\b/i, 3], [/\boxford cloth\b/i, 3], [/\bsuede\b/i, 3],
        // Colors
        [/\bnavy\b/i, 3], [/\bcharcoal\b/i, 3], [/\bcream\b/i, 2], [/\bburgundy\b/i, 3],
        [/\bcamel\b/i, 3], [/\bbeige\b/i, 2], [/\btan\b/i, 2], [/\bforest\b/i, 2],
        // Shoes
        [/\bloafer(s)?\b/i, 4], [/\boxford shoe(s)?\b/i, 5], [/\bderby\b/i, 4],
        [/\bdress shoe(s)?\b/i, 5], [/\bleather shoe(s)?\b/i, 3], [/\bsuede shoe(s)?\b/i, 3],
        [/\bpenny loafer(s)?\b/i, 5], [/\bbit loafer(s)?\b/i, 5], [/\bbrogue(s)?\b/i, 4],
        [/\bmonk strap\b/i, 4], [/\bchelsea boot(s)?\b/i, 3],
        // Brands
        [/\bbrooks brothers\b/i, 5], [/\bralph lauren\b/i, 4], [/\bbrunello\b/i, 5],
        [/\bloro piana\b/i, 5], [/\bmassimo dutti\b/i, 3], [/\barket\b/i, 2],
        // Accessories
        [/\btie\b/i, 3], [/\bpocket square\b/i, 3], [/\bscarf\b/i, 2],
        // Negatives
        [/\bhoodie\b/i, -4], [/\bgraphic\b/i, -3], [/\blogo\b/i, -2],
        [/\bcargo\b/i, -4], [/\bsweatpants\b/i, -4], [/\bjoggers\b/i, -4],
        [/\btrack pants\b/i, -4], [/\bshorts\b/i, -3], [/\bbermuda\b/i, -3],
        [/\bpuffer\b/i, -3], [/\bbomber\b/i, -3], [/\bdenim jacket\b/i, -2],
        [/\bchunky sneaker(s)?\b/i, -4], [/\bathletic\b/i, -3], [/\bsquare[-\s]?toe\b/i, -4],
        [/\bneon\b/i, -4], [/\bsequin\b/i, -4], [/\bripped\b/i, -3],
        [/\blow[-\s]?rise\b/i, -3], [/\bskinny\b/i, -2], [/\boversized\b/i, -2],
        [/\bbackpack\b/i, -2], [/\bbi[kc]ini\b/i, -2],
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

// ── Universal style knowledge distilled from 18 expert style-guide transcripts ──
// These rules apply across ALL styles and are injected into every system prompt.
const STYLE_KNOWLEDGE = `
UNIVERSAL STYLE PRINCIPLES (apply to every outfit regardless of style):
1. FIT IS KING: Medium-tapered silhouettes always beat skinny/oversized for classic looks. No pulling buttons, no excess fabric. Shoulders must align.
2. HIGH/MID-RISE TROUSERS: Always prefer mid-to-high rise. Low-rise breaks body proportions and looks sloppy. Bottom hem width ~19-20cm for classic; slightly narrower for smart casual.
3. NEVER MIX STYLES: Do not pair sportswear with tailored pieces in one outfit. A suit jacket over a sports tee or sneakers with a blazer violates formality coherence.
4. BLACK IS NOT UNIVERSAL: Black only pairs with white. For classic/smart-casual wardrobes, replace black with navy, charcoal, dark brown, or burgundy — all far more versatile.
5. GOLDEN 8 COLORS: white, blue, light blue, brown, green, cream/milk, burgundy, grey. These 8 intermix freely: blue+brown, blue+green, green+brown always work.
6. TEXTURE ELEVATES: Same color in cheap cotton looks like underwear; in textured linen, cashmere, or suede it reads as luxury. Favor texture over trend.
7. SEASONAL FABRICS: Linen/tropical wool/silk-cotton for summer; flannel/tweed/heavy wool for winter. Accessories follow seasons too (linen ties in summer, wool in winter).
8. KNIT OVER WOVEN: For polos and t-shirts, knit (fine-gauge) reads as elegant and versatile; piqué reads as sporty; basic woven cotton reads as cheap.
9. COLLAR SIZE MATTERS: Large shirt collars frame the face and tuck under jacket lapels cleanly. Small collars, mandarin/stand collars cheapen any look.
10. LAPEL WIDTH: Minimum 8.5cm. Narrow lapels cheapen the look. Lapel should cover ~half the distance from neck to shoulder seam.
11. FOOTWEAR RULES: Loafers/oxfords/derbies for smart and classic; clean minimal sneakers only for casual; never athletic trainers with tailored clothing. Suede is more versatile than leather for smart-casual.
12. ACCESSORIES COMPLETE: Pocket squares, ties, seasonal scarves, watches elevate any outfit from ordinary to polished.
13. FORMALITY COHERENCE: Every item in one outfit should sit within 1-2 formality tiers of each other. A formal blazer + gym shorts = hard clash.
14. QUALITY OVER QUANTITY: One well-made piece in a good fabric outperforms three cheap ones. Natural fibers (wool, linen, cotton, silk) always look more expensive than synthetics.
15. AVOID: puffer jackets with suits, square-toe shoes, backpacks with tailored clothing, sports watches with formal outfits, graphic tees in any non-casual look.
16. SOCKS MATCH TROUSERS, NOT SHOES: Grey, navy, and burgundy socks are the foundation. Black socks are disallowed — they show wear faster and pair with nothing.
17. CONTRAST & COLOR BALANCE: Dark top + light bottom (or vice versa) creates more masculine, harmonious outfits. Limit to 3 colors per outfit; white doesn't count and can be a 4th. Monochromatic looks need different textures/shades to avoid looking flat.
18. JACKET FIT DETAILS: Jacket must cover the buttocks. 1-2cm of shirt cuff must show beyond the jacket sleeve. Never button the bottom vest button. Tie width should be 7.5-9.5cm.
19. CITY BANS: No shorts, sandals/mules, mesh/see-through tees, or mandarin/stand collars in urban settings. No popped polo collars. Always button polo buttons (all or leave top one open). No exotic leather shoes (crocodile, python) — quality calf or suede only.
20. PATTERNED SPORT JACKETS: Glen check, Prince of Wales, Gun Club are the most versatile patterns — they contain multiple colors for easy coordination. Avoid smooth solid-color sport jackets (look cheap); choose textured or patterned fabrics. Mixed fabric compositions (wool-linen-silk blends) are more textured and practical than pure fibers.`

// ── Style-specific fashion context ───────────────────────────────────────
// Each block has: vibe paragraph + MUST / REJECT rules that the model must
// honor. The REJECT list is what forced the model to stop picking graphic
// tees, cargo shorts, and hoodies when the user asked for "Old Money".
const STYLE_CONTEXT: Record<string, { vibe: string; must: string[]; reject: string[]; palette: string[]; fabrics: string[] }> = {
    old_money: {
        vibe: `Old Money / Quiet Luxury — the wardrobe of someone who summers in the Hamptons, sails in Capri, and owns a Brunello Cucinelli cardigan. Think Ralph Lauren Purple Label, Loro Piana, Brooks Brothers. Nothing flashy, everything expensive. The outfit should feel understated, tonal, and timeless. Medium-tapered fit, never skinny. High-rise trousers. Texture over trend.`,
        must: [
            'Use only tonal / monochromatic / analogous palettes (e.g. camel + cream + white; navy + cream + brown; blue + green + brown).',
            'Prefer tailored silhouettes with medium taper: blazers, cardigans, oxford shirts, knit polos, chinos, pleated trousers, slacks. NEVER skinny/super-slim fit.',
            'Trousers must be mid-to-high rise. Low-rise is disallowed for this style.',
            'Footwear must be loafers (penny, bit, tassel), oxfords, derbies, boat shoes, or minimal clean white/cream leather sneakers.',
            'Prefer knit polos and knit tees over piqué or basic cotton — they read as more elegant and versatile.',
            'Shirts must have large collars that tuck cleanly under jacket lapels. Small collars or mandarin collars are disallowed.',
            'Blazer lapels must be ≥8.5cm wide. Narrow lapels cheapen the look.',
            'Favor textured fabrics: flannel, tweed, herringbone, cashmere, merino, linen blends. Smooth flat fabrics look cheaper.',
            'Every piece should look like it could be from Ralph Lauren, Brunello Cucinelli, Loro Piana, Massimo Dutti, or Brooks Brothers.',
            'Pocket square is mandatory with any blazer or suit outfit — its absence is a major style error.',
        ],
        reject: [
            'NEVER include: graphic tees, logo tees, 3-pack basic tees, printed tees, baby tees, tank tops, crop tops.',
            'NEVER include: cargo shorts, athletic shorts, basketball shorts, track pants, sweatpants, joggers, shorts/bermudas.',
            'NEVER include: hoodies, zip-ups, puffers, bomber jackets, denim jackets, graphic sweatshirts.',
            'NEVER include: chunky sneakers, high-tops, platform shoes, athletic trainers, square-toe shoes.',
            'NEVER include: neon, rhinestone, sequin, metallic, or tie-dye pieces.',
            'NEVER include: black as a primary color — replace with navy, charcoal, or dark brown. Black only for formal eveningwear.',
            'NEVER include: exotic leather shoes (crocodile, python, stingray) — they look vulgar. Quality calf leather or suede only.',
            'NEVER pair: formal outerwear with shorts, backpacks with tailored clothing, sports watches with formal outfits.',
        ],
        palette: ['navy', 'cream', 'ivory', 'beige', 'camel', 'chocolate', 'forest green', 'burgundy', 'white', 'charcoal', 'midnight blue', 'tan', 'olive'],
        fabrics: ['cashmere', 'wool', 'merino', 'linen', 'silk', 'cotton', 'poplin', 'tweed', 'flannel', 'herringbone', 'suede', 'tropical wool'],
    },
    semi_classic: {
        vibe: `Semi-Classic — refined everyday elegance that bridges the gap between casual and formal. Think Massimo Dutti, Arket, COS. Tailored touches with relaxed comfort — structured but not stiff, polished but not overdressed. Cardigans over tees, chinos with loafers, knit polos with slacks.`,
        must: [
            'Blazers and structured cardigans are key layering pieces — more relaxed than formal but more polished than hoodies.',
            'Knit polos, fine-gauge tees, and turtlenecks as base layers. Piqué polos and basic cotton tees are too sporty.',
            'Chinos, slacks, and well-cut jeans. Mid-rise, medium taper. No skinny, no baggy.',
            'Loafers (suede or leather), desert boots, clean minimal sneakers, or Chelsea boots.',
            'Stick to the golden 8 colors: white, blue, light blue, brown, green, cream, burgundy, grey — they all intermix.',
            'Favor textured fabrics: merino, cotton blends, linen, poplin, suede. Texture elevates simple outfits.',
            'Lightweight cardigans (cotton/merino) are the signature semi-classic layer — more relaxed than a blazer but more polished than a hoodie. Can be tied around the neck as a style accent when not worn.',
            'Socks must match trousers, not shoes. Grey, navy, and burgundy socks are the foundation.',
        ],
        reject: [
            'Avoid: hoodies, graphic tees, cargo pants, sweatpants, joggers.',
            'Avoid: chunky sneakers, athletic trainers, basketball shoes.',
            'Avoid: neon, rhinestone, sequin, metallic, or tie-dye pieces.',
            'Avoid: oversized/baggy fits, ripped/distressed denim.',
            'Avoid: low-rise trousers — they break the refined proportions.',
        ],
        palette: ['navy', 'cream', 'beige', 'camel', 'olive', 'tan', 'burgundy', 'white', 'charcoal', 'brown', 'light blue'],
        fabrics: ['merino', 'cotton', 'linen', 'poplin', 'suede', 'knit', 'denim', 'wool', 'tropical wool'],
    },
    minimalist: {
        vibe: `Minimalist — quiet, clean, intentional. Think COS, Uniqlo U, Acne Studios, The Row, Jil Sander. Every piece should feel essential. Texture and cut do the talking; color stays restrained.`,
        must: [
            'Stick to a tight palette: black, white, grey, beige, navy, stone. Monochromatic outfits are ideal.',
            'Silhouettes must be simple and clean; no busy patterns. Medium taper, never skinny.',
            'Prefer structured, tailored fits over trendy oversized looks.',
            'Use texture to add depth within monochromatic looks: ribbed knits, brushed wool, matte leather, raw linen.',
            'Knit polos and fine-gauge tees over piqué or basic cotton for a more polished minimal look.',
        ],
        reject: [
            'NEVER include: graphic prints, logos, tie-dye, floral prints, neon colors, rhinestones, sequins.',
            'Avoid: color-blocking, chunky sneakers, puffers with branding.',
        ],
        palette: ['black', 'white', 'grey', 'charcoal', 'beige', 'navy', 'stone', 'cream', 'sand'],
        fabrics: ['merino', 'wool', 'cotton', 'linen', 'cashmere', 'poplin', 'suede'],
    },
    business_casual: {
        vibe: `Modern Professional — sharp tailoring that still feels comfortable. Hugo Boss meets Everlane meets Theory. Smart Casual with a corporate edge. Blazers with texture, knit polos, well-cut trousers.`,
        must: [
            'Blazers (textured: flannel, tweed, herringbone) paired with chinos or tailored trousers. Oxford or poplin button-downs with large collars.',
            'Knit polos and fine-gauge tees are excellent under blazers — more elegant than piqué or basic cotton.',
            'Polished shoes: leather loafers, oxfords, derbies, or minimal clean white leather sneakers. Dark brown shoes are as formal as black but more versatile.',
            'Trousers must be mid-to-high rise. Low-rise is disallowed.',
            'Blazer lapels ≥8.5cm. Narrow lapels cheapen the professional look.',
            'Palette: navy, charcoal, tan, white, light blue, cream, brown, burgundy.',
            'Favor tonal/analogous combinations: navy + cream + brown; charcoal + light blue + tan.',
            'Navy blazer with metal buttons is the most versatile starter jacket — pairs with nearly any trouser color.',
            'Pocket square (white or tonal) is expected with any blazer — its absence is a noticeable gap.',
            'Socks must match trousers, not shoes. Grey, navy, and burgundy are the only acceptable sock colors.',
        ],
        reject: [
            'Avoid: hoodies, graphic tees, cargo pants, ripped jeans, sweatpants, shorts.',
            'Avoid: sequins, rhinestones, neon colors, crop tops.',
            'Avoid: black as primary color — use navy or charcoal instead.',
            'Avoid: skinny/super-slim fit, low-rise trousers, small shirt collars, narrow lapels.',
        ],
        palette: ['navy', 'charcoal', 'tan', 'white', 'light blue', 'grey', 'cream', 'brown', 'burgundy', 'beige'],
        fabrics: ['wool', 'cotton', 'poplin', 'oxford cloth', 'linen', 'flannel', 'tweed', 'merino', 'cashmere', 'tropical wool', 'glen check', 'prince of wales'],
    },
    casual: {
        vibe: `Smart Casual — relaxed but intentional. Well-fitted basics, clean color combinations. The bridge between sport and classic. Knit tees, slacks, loafers or clean sneakers.`,
        must: [
            'Well-fitted knit tees or knit polos with chinos, slacks, or well-cut jeans. Medium taper, not skinny.',
            'Clean minimal sneakers, loafers (suede or leather), or desert boots.',
            'Trousers at mid-rise. Low-rise breaks proportions.',
            'Stick to the golden 8 colors: white, blue, light blue, brown, green, cream, burgundy, grey — they all intermix.',
            'Add texture: ribbed knits, brushed cotton, linen blends elevate simple outfits.',
            'Lightweight cardigans (cotton/silk) are excellent smart-casual layers — more relaxed than a blazer but more polished than a hoodie. Can be tied around the neck as a style accent.',
            'Socks should match trousers or another outfit item, not shoes.',
        ],
        reject: [
            'Avoid: sweatpants, basketball shorts, graphic hoodies.',
            'Avoid: athletic sneakers with trousers, piqué polos with dressy bottoms.',
            'Avoid: mesh/see-through tees, popped polo collars, unbuttoned polo collars.',
        ],
        palette: ['navy', 'white', 'grey', 'beige', 'olive', 'cream', 'brown', 'light blue', 'burgundy', 'tan'],
        fabrics: ['cotton', 'linen', 'denim', 'wool', 'merino', 'knit', 'suede', 'canvas', 'tropical wool'],
    },
    classic: {
        vibe: `Classic Menswear — timeless elegance rooted in traditional tailoring. Think Savile Row, Italian sartoria, Alexander from Strokanor. Suits, sport jackets, high-rise trousers, knitwear, and proper shoes. Formality levels from business suit to smart-casual separates. Never trendy, always refined.`,
        must: [
            'Suits (same-fabric top+bottom) or sport-jacket separates with complementary trousers. Striped suits must only be worn as matching sets — never mix a striped jacket with different trousers.',
            'Trousers must be mid-to-high rise with a hem width of ~19-20cm for classic, slightly narrower for smart-casual. NEVER skinny/super-slim.',
            'Shirts must have large collars that tuck under lapels. Small collars, mandarin/stand collars are disallowed.',
            'Blazer/suit lapels must be ≥8.5cm wide. Narrow lapels are a style error.',
            'Knit polos, turtlenecks, and fine-gauge tees are the correct base layers under jackets. Piqué polos and basic cotton tees are too sporty for classic looks.',
            'Footwear: oxfords and derbies with suits; loafers (leather for dressy, suede for smart-casual); clean minimal leather sneakers only with casual trousers. Dark brown shoes are as formal as black but more versatile.',
            'Suits with smooth/office fabric require a shirt + tie. To go tieless, the suit must be in a seasonal/textured fabric (flannel, tweed, linen).',
            'Favor the golden 8 colors: white, blue, light blue, brown, green, cream, burgundy, grey. Blue+brown, blue+green, green+brown are foolproof formulas.',
            'Favor textured fabrics: flannel, tweed, herringbone, cashmere, merino, linen blends. Texture = luxury. Mixed fabric compositions (wool-linen-silk blends) are more textured and practical than pure fibers.',
            'Accessories complete the look: pocket squares, ties, seasonal scarves, watches. Tie and pocket square must coordinate but NOT be identical fabric. Pocket square is mandatory with any jacket — its absence is one of the biggest classic style errors.',
            'Socks must match trousers, not shoes. Grey, navy, and burgundy socks are the foundation. Black socks are disallowed.',
            'Navy blazer with metal buttons is the most versatile sport jacket — pairs with almost any trousers except very dark navy or graphite. Next additions: Glen check, Prince of Wales, Gun Club patterns for multi-color coordination.',
        ],
        reject: [
            'NEVER include: graphic tees, logo tees, printed tees, tank tops, crop tops, shorts/bermudas in city settings.',
            'NEVER include: cargo pants, athletic shorts, track pants, sweatpants, joggers.',
            'NEVER include: hoodies, puffer jackets, bomber jackets, denim jackets worn with suits.',
            'NEVER include: chunky sneakers, athletic trainers, square-toe shoes, high-tops with tailored clothing.',
            'NEVER include: black as a primary everyday color — navy, charcoal, dark brown are more versatile and elegant.',
            'NEVER include: exotic leather shoes (crocodile, python, stingray) — they look vulgar. Quality calf leather or suede only.',
            'NEVER pair: formal outerwear with shorts, sportswear with tailored pieces, backpacks with suits.',
            'NEVER: mix a striped suit jacket with non-matching trousers. Striped jackets only go with their matching pants.',
            'NEVER: wear a bow tie with a business suit — bow ties are for tuxedos/evening jackets only.',
            'NEVER: identical tie + pocket square fabric. They must coordinate, not match exactly.',
            'NEVER: smooth solid-color sport jackets without texture or pattern — they look cheap. Choose textured or patterned fabrics.',
        ],
        palette: ['navy', 'charcoal', 'cream', 'white', 'beige', 'camel', 'brown', 'burgundy', 'forest green', 'light blue', 'grey', 'olive', 'tan'],
        fabrics: ['wool', 'flannel', 'tweed', 'herringbone', 'cashmere', 'merino', 'linen', 'silk', 'cotton', 'poplin', 'oxford cloth', 'suede', 'tropical wool', 'glen check', 'prince of wales', 'gun club'],
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
        : `- EVERY outfit MUST contain exactly THREE items and exactly these three macroCategory slots, no exceptions: (1) a TOP (or a dress), (2) a BOTTOM (skip this slot only if the top is a dress — then the outfit has 2 items: dress + shoes), and (3) SHOES. Do NOT add a fourth outerwear/layer item — the weather does not require layering. A non-layered outfit with 4+ items is INVALID.`

    return `You are a world-class fashion stylist curating outfits for a specific aesthetic. You must obey the STYLE DIRECTION below as if your reputation depends on it.

TASK: Create ${limit} complete outfits using ONLY items from the provided wardrobe, referenced by their exact id.

GENERAL STYLING RULES:
${slotRules}
- If no shoe item obviously matches the style, pick the LEAST off-aesthetic shoe available. An outfit WITHOUT shoes is invalid — never return one.
- NEVER invent items. Use ONLY the exact ids listed in "Available wardrobe items".
- Pick items whose styleTags, name, description, or material indicate they FIT the requested style. If an item's tags or description clash with the style, do NOT use it (unless it is the only option for that slot — see shoes rule above).
- Favor color harmony (tonal, monochromatic, or analogous palettes from the style's preferred palette). When layering, the base top must color-coordinate with the outerwear (tonal or complementary — never clashing).
- Do not reuse the same item across outfits unless unavoidable.
- NEVER pair formal outerwear (blazer, suit jacket, sport coat, overcoat, topcoat, trench, peacoat, tuxedo) with shorts or bermudas. If the only available bottom is shorts, drop the formal outerwear and return a non-layered 3-item look instead.
- Each outfit needs a vivid 1-2 sentence description and 2 actionable styling tips.

${STYLE_KNOWLEDGE}

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

// Reject outfits that mix formal outerwear (blazer, suit jacket, overcoat,
// trench, peacoat, tuxedo, sport coat) with shorts/bermudas. This is a hard
// styling rule the model occasionally violates even when on-palette.
function validateOutfitCompatibility(outfit: any, itemMap: Map<string, any>): { ok: boolean; reason?: string } {
    const items = Array.isArray(outfit.items) ? outfit.items : []
    let hasFormalLayer = false
    let hasShorts = false
    for (const it of items) {
        const src = itemMap.get(it.id) || {}
        const merged = { ...src, ...it }
        if (isFormalLayerItem(merged)) hasFormalLayer = true
        if (isShortsItem(merged)) hasShorts = true
    }
    if (hasFormalLayer && hasShorts) {
        return { ok: false, reason: 'formal outerwear paired with shorts (disallowed)' }
    }
    return { ok: true }
}

function filterValidOutfits(outfits: any[], style: string, itemMap: Map<string, any>, layered: boolean): any[] {
    const kept: any[] = []
    for (const o of outfits) {
        const v = validateOutfitAgainstStyle(o, style, itemMap, layered)
        if (!v.ok) {
            console.log(`[validate] rejected outfit for ${style} (layered=${layered}): ${v.reason}`)
            continue
        }
        const c = validateOutfitCompatibility(o, itemMap)
        if (!c.ok) {
            console.log(`[validate] rejected outfit for ${style} (layered=${layered}): ${c.reason}`)
            continue
        }
        kept.push(o)
    }
    return kept
}

// Placeholder shoes item injected when the wardrobe has no shoes at all.
const PLACEHOLDER_SHOES: Record<string, any> = {
    id: 'placeholder_shoes',
    type: 'shoes',
    macroCategory: 'shoes',
    color: 'neutral',
    name: 'Shoes',
    imageUrl: '',
    image: '',
    recommendation: 'Add shoes to your wardrobe for better outfits',
    isShopItem: false,
}

// ── Shop-catalog fill for missing slots (shoes, outerwear, etc.) ───────
// Mirrors the client-side fillMissingSlots from shoppingService.ts.
// Queries shop_catalog for 1 style-matching item per missing macro slot.
async function fillMissingSlotsEdge(
    supabaseClient: any,
    missingSlots: string[],
    style: string,
): Promise<any[]> {
    const picks: any[] = []
    for (const slot of missingSlots) {
        try {
            let q = supabaseClient
                .from('shop_catalog')
                .select('id, brand, name, price, image_url, garment_type, category, description, primary_color, source')
                .eq('is_active', true)
                .limit(30)
            if (slot === 'shoes') q = q.or('category.eq.shoes,garment_type.eq.shoes')
            else if (slot === 'outerwear') q = q.or('category.eq.outerwear,garment_type.eq.outerwear,name.ilike.%jacket%,name.ilike.%coat%,name.ilike.%blazer%,name.ilike.%cardigan%,name.ilike.%sweater%,name.ilike.%hoodie%,name.ilike.%puffer%,name.ilike.%vest%')
            else if (slot === 'top') q = q.or('category.eq.tops,garment_type.eq.upper_body')
            else if (slot === 'bottom') q = q.or('category.eq.bottoms,garment_type.eq.lower_body')
            const { data, error } = await q
            if (error || !data || data.length === 0) continue
            // Pick the first active row (style scoring is client-side only).
            const row = data[0]
            if (row && row.image_url) {
                picks.push({
                    id: `shop_${row.id}`,
                    type: row.garment_type || row.category || slot,
                    category: row.category || slot,
                    macroCategory: slot,
                    name: row.name || row.brand || 'Shop pick',
                    brand: row.brand || '',
                    color: row.primary_color || 'neutral',
                    imageUrl: row.image_url,
                    image: row.image_url,
                    style: style || 'Casual',
                    isShopItem: true,
                    price: row.price || undefined,
                    recommendation: `Suggested from shop to complete your ${slot}`,
                })
            }
        } catch (_) {
            // Shop catalog unreachable — skip.
        }
    }
    return picks
}

async function localFallback(items: any[], style: string, occasion: string, limit: number, layered: boolean, supabaseClient?: any): Promise<any[]> {
    const baseTops = items.filter(i => i.macroCategory === 'top')
    const outerwear = items.filter(i => i.macroCategory === 'outerwear')
    const legacyTops = items.filter(i => ['top','outerwear'].includes(i.macroCategory))
    const bottoms = items.filter(i => i.macroCategory === 'bottom')
    const nonShortsBottoms = bottoms.filter(b => !isShortsItem(b))
    const casualOuterwear = outerwear.filter(o => !isFormalLayerItem(o))
    const shoes = items.filter(i => i.macroCategory === 'shoes')

    // ── Fill missing slots from shop_catalog ────────────────────────────
    // If the wardrobe has no shoes (or no outerwear when layered), query
    // the shop catalog so the AI / local builder can still produce a
    // complete outfit.
    const macros = new Set(items.map((i: any) => (i.macroCategory || '').toLowerCase()))
    const missingSlots: string[] = []
    if (!macros.has('shoes')) missingSlots.push('shoes')
    if (!macros.has('bottom')) missingSlots.push('bottom')
    if (!macros.has('top')) missingSlots.push('top')
    if (layered && !macros.has('outerwear')) missingSlots.push('outerwear')
    let shopFills: any[] = []
    if (missingSlots.length > 0 && supabaseClient) {
        shopFills = await fillMissingSlotsEdge(supabaseClient, missingSlots, style)
    }
    // Merge shop items into the pool so the builder picks them naturally.
    const allItems = [...items, ...shopFills]
    // Re-derive category buckets from the merged pool.
    const allBaseTops = allItems.filter((i: any) => i.macroCategory === 'top')
    const allOuterwear = allItems.filter((i: any) => i.macroCategory === 'outerwear')
    const allLegacyTops = allItems.filter((i: any) => ['top','outerwear'].includes(i.macroCategory))
    const allBottoms = allItems.filter((i: any) => i.macroCategory === 'bottom')
    const allNonShortsBottoms = allBottoms.filter((b: any) => !isShortsItem(b))
    const allCasualOuterwear = allOuterwear.filter((o: any) => !isFormalLayerItem(o))
    const allShoes = allItems.filter((i: any) => i.macroCategory === 'shoes')

    // For old_money, classic, and business_casual styles, completely exclude shorts
    // when formal outerwear is available. This prevents the illogical coat + shorts combination.
    const isFormalStyle = ['old_money', 'classic', 'business_casual'].includes(style.toLowerCase())
    if (isFormalStyle && allOuterwear.some((o: any) => isFormalLayerItem(o))) {
        // Replace allBottoms with only non-shorts options
        allBottoms.length = 0
        allBottoms.push(...allNonShortsBottoms)
    }

    const outfits: any[] = []
    const seed = layered ? Math.max(allBaseTops.length, allOuterwear.length, 1) : Math.max(allLegacyTops.length, 1)

    for (let i = 0; i < Math.min(limit, seed); i++) {
        const parts: any[] = []
        // Pre-pick the bottom so we can decide whether a formal layer is safe.
        const bottom = allBottoms[i % Math.max(allBottoms.length, 1)] || allBottoms[0]
        const bottomIsShorts = !!bottom && isShortsItem(bottom)
        if (layered) {
            // If the bottom is shorts, never pair with a formal outerwear piece;
            // prefer casual outerwear, else fall back to a safe non-shorts bottom.
            let outer = allOuterwear[i % Math.max(allOuterwear.length, 1)]
            if (bottomIsShorts && outer && isFormalLayerItem(outer)) {
                outer = allCasualOuterwear[i % Math.max(allCasualOuterwear.length, 1)] || undefined
            }
            const base = allBaseTops[i % Math.max(allBaseTops.length, 1)]
            // Use outerwear if available, otherwise fallback to any top item
            const mainTop = outer || allLegacyTops[i % Math.max(allLegacyTops.length, 1)]
            if (mainTop) parts.push({ ...mainTop, recommendation: 'Main top / outerwear layer' })
            // Always include base top (use baseTop if available, otherwise reuse mainTop or any top)
            const baseTopItem = base || outer || allLegacyTops[(i + 1) % Math.max(allLegacyTops.length, 1)]
            if (baseTopItem) parts.push({ ...baseTopItem, recommendation: 'Base top worn underneath' })
        } else {
            const top = allLegacyTops[i % Math.max(allLegacyTops.length, 1)]
            if (top) parts.push({ ...top, recommendation: 'Key piece' })
        }
        // Always include bottom and shoes - reuse items if necessary.
        // For layered looks we keep the pre-picked bottom; if there's a formal
        // layer + shorts conflict we still couldn't resolve, swap to a
        // non-shorts bottom when one exists.
        let finalBottom = bottom
        if (layered && bottomIsShorts) {
            const mainTopItem = parts[0]
            if (mainTopItem && isFormalLayerItem(mainTopItem) && allNonShortsBottoms.length > 0) {
                finalBottom = allNonShortsBottoms[i % allNonShortsBottoms.length]
            }
        }
        const shoe = allShoes[i % Math.max(allShoes.length, 1)] || allShoes[0]
        if (finalBottom) parts.push({ ...finalBottom, recommendation: 'Pairs well' })
        if (shoe) parts.push({ ...shoe, recommendation: 'Completes the look' })
        else parts.push({ ...PLACEHOLDER_SHOES })
        // Exact item contract: 3 for non-layered, 4 for layered.
        const targetItems = layered ? 4 : 3
        if (layered && parts.length < targetItems && items.length > 0) {
            // Only pad layered outfits — non-layered must stay at exactly 3.
            while (parts.length < targetItems && parts.length < allItems.length) {
                const fillItem = allItems[parts.length % allItems.length]
                parts.push({ ...fillItem, recommendation: 'Complementary piece' })
            }
        }
        // For non-layered, trim any accidental extras down to 3.
        if (!layered && parts.length > targetItems) parts.length = targetItems
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
    if (outfits.length === 0 && allItems.length > 0) {
        const parts: any[] = []
        if (layered) {
            // For layered: try to get 4 items (outerwear, base, bottom, shoes).
            // If only shorts are available, avoid formal outerwear.
            const candidateBottom = allBottoms[0] || allItems[0]
            const bottomIsShorts = !!candidateBottom && isShortsItem(candidateBottom)
            let outer = allOuterwear[0] || allLegacyTops[0]
            if (bottomIsShorts && outer && isFormalLayerItem(outer)) {
                outer = allCasualOuterwear[0] || allLegacyTops.find((t: any) => !isFormalLayerItem(t)) || outer
            }
            const base = allBaseTops[0] || allLegacyTops[0] || outer
            const finalBottom = (bottomIsShorts && outer && isFormalLayerItem(outer) && allNonShortsBottoms[0])
                ? allNonShortsBottoms[0]
                : candidateBottom
            const shoe = allShoes[0]
            if (outer) parts.push({ ...outer, recommendation: 'Main top / outerwear layer' })
            if (base && base.id !== outer?.id) parts.push({ ...base, recommendation: 'Base top worn underneath' })
            if (finalBottom) parts.push({ ...finalBottom, recommendation: 'Pairs well' })
            if (shoe && shoe.id !== finalBottom?.id) parts.push({ ...shoe, recommendation: 'Completes the look' })
            else parts.push({ ...PLACEHOLDER_SHOES })
        } else {
            // For non-layered: exactly 3 items (top, bottom, shoes)
            const top = allLegacyTops[0] || allItems[0]
            const bottom = allBottoms[0] || allItems[1] || allItems[0]
            const shoe = allShoes[0]
            if (top) parts.push({ ...top, recommendation: 'Key piece' })
            if (bottom && bottom.id !== top?.id) parts.push({ ...bottom, recommendation: 'Pairs well' })
            if (shoe && shoe.id !== bottom?.id) parts.push({ ...shoe, recommendation: 'Completes the look' })
            else parts.push({ ...PLACEHOLDER_SHOES })
            if (parts.length > 3) parts.length = 3
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
        items: items.slice(0, layered ? 4 : 3).map((i: any) => ({ ...i, recommendation: 'From your wardrobe' })),
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

        // ── 3c. Fill missing macro-category slots from shop_catalog ────────
        // If the user's wardrobe has no shoes (or no outerwear/top/bottom),
        // pull matching items from the shop catalog so the AI prompt includes
        // them and can produce a complete outfit.
        const wardrobeMacros = new Set(wardrobeItems.map((i: any) => (i.macroCategory || '').toLowerCase()))
        const requiredSlots = ['top', 'bottom', 'shoes']
        if (needsLayering(stylePreferences, weather, prompt)) requiredSlots.push('outerwear')
        const missingWardrobeSlots = requiredSlots.filter(s => !wardrobeMacros.has(s))
        if (missingWardrobeSlots.length > 0) {
            const svcClient = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
            const shopFills = await fillMissingSlotsEdge(svcClient, missingWardrobeSlots, stylePreferences)
            if (shopFills.length > 0) {
                console.log(`[shop-fill] Added ${shopFills.length} shop item(s) for missing slots: ${missingWardrobeSlots.join(', ')}`)
                wardrobeItems = [...wardrobeItems, ...shopFills]
            }
        }

        // ── 3d. Rank + trim the pool for the requested style so the LLM only
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
            const svcClient = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
            const outfits = await localFallback(wardrobeItems, stylePreferences, occasion, limit, layered, svcClient)
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
                const correctionUsr = `${usr}\n\nIMPORTANT CORRECTION: Your previous attempt failed because (a) some items violated the ${styleKey} aesthetic, OR (b) outfits were missing required slots. REQUIREMENTS:\n${layeredReqs}\n  ${layered ? '4' : '3'}. Reject any item whose name/description contains: hoodie (unless casual), graphic tee, logo tee, cargo, ripped, sweatpants, neon, sequin, rhinestone, chunky sneaker, basketball, skate sneaker (for old_money / business_casual / minimalist / semi_classic).\n  ${layered ? '5' : '4'}. For ${styleKey} shoes, prefer: loafers, penny loafers, bit loafers, dress shoes, boat shoes, derbies, leather shoes, minimal leather sneakers. Avoid chunky, basketball, or skate sneakers.\nTry again and return ${limit} outfits that each have ${layered ? 'outerwear+base-top+bottom+shoes' : 'top+bottom+shoes'}.`
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
            return new Response(JSON.stringify({ success: false, outfits: [], error: 'AI failed to generate valid outfits' }), {
                headers: { ...corsHeaders, 'Content-Type': 'application/json' }
            })
        }

        // ── 8. Enrich items with imageUrl from DB ──────────────────────────
        // Also replace placeholder items (without valid images) with shop catalog items
        const svcClient = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        const { data: shopItems, error: shopError } = await svcClient
            .from('shop_catalog')
            .select('id, brand, name, price, image_url, garment_type, category, description')
            .eq('is_active', true)
            .limit(50)
        
        console.log('[enrich] Shop catalog query result:', { count: shopItems?.length || 0, error: shopError })
        
        const shopByMacro: Record<string, any[]> = {}
        if (shopItems) {
            for (const item of shopItems) {
                // Use name-based keyword matching to distinguish outerwear from
                // base tops — both share garment_type='upper_body' in shop_catalog.
                const nameStr = (item.name || '').toLowerCase()
                const descStr = (item.description || '').toLowerCase()
                const blob = `${nameStr} ${descStr}`
                let macro: string
                if (/\b(jacket|coat|blazer|cardigan|sweater|hoodie|puffer|bomber|vest|outerwear|trench|peacoat)\b/.test(blob)) {
                    macro = 'outerwear'
                } else if (item.garment_type === 'upper_body') {
                    macro = 'top'
                } else if (item.garment_type === 'lower_body') {
                    macro = 'bottom'
                } else if (item.garment_type === 'shoes') {
                    macro = 'shoes'
                } else {
                    macro = 'other'
                }
                if (!shopByMacro[macro]) shopByMacro[macro] = []
                shopByMacro[macro].push(item)
            }
            console.log('[enrich] Shop items by macro:', Object.keys(shopByMacro).map(k => `${k}: ${shopByMacro[k].length}`))
        }
        
        const enriched = aiOutfits.map((outfit: any) => ({
            ...outfit,
            items: (outfit.items || []).map((item: any) => {
                const src: any = itemMap.get(item.id) || {};
                // Prioritize DB URL (src.imageUrl) over AI's potentially empty imageUrl
                const imageUrl = src.imageUrl || item.imageUrl || item.image_url || '';
                
                // If no valid image URL, replace with shop catalog item
                let finalItem = { ...item, imageUrl };
                if (!imageUrl || imageUrl.length === 0) {
                    const macro = (src.macroCategory || item.macroCategory || '').toLowerCase();
                    // Map 'upper_body' alias to the correct slot: if the item name
                    // contains outerwear keywords, it's outerwear; otherwise top.
                    let macroNormalized = macro === 'upper_body' ? 'top' : macro;
                    if (macroNormalized === 'top' || macro === 'upper_body') {
                        const nameBlob = `${finalItem.name || ''} ${item.name || ''} ${src.name || ''} ${item.description || ''} ${src.description || ''}`.toLowerCase();
                        if (/\b(jacket|coat|blazer|cardigan|sweater|hoodie|puffer|bomber|vest|outerwear|trench|peacoat)\b/.test(nameBlob)) {
                            macroNormalized = 'outerwear';
                        }
                    }
                    if (shopByMacro[macroNormalized] && shopByMacro[macroNormalized].length > 0) {
                        // Pick a random shop catalog item for this macro
                        const shopItem = shopByMacro[macroNormalized][Math.floor(Math.random() * shopByMacro[macroNormalized].length)]
                        console.log(`[enrich] Replacing placeholder ${item.id} (${macro}) with shop item ${shopItem.id}`)
                        finalItem = {
                            ...item,
                            id: shopItem.id,
                            imageUrl: shopItem.image_url,
                            image: shopItem.image_url,
                            name: shopItem.name,
                            brand: shopItem.brand,
                            isShopItem: true,
                        }
                    }
                }
                
                // Canonicalize macroCategory so aliases like 'upper_body' / 'lower_body'
                // / 'tops' / 'footwear' never reach the client.
                const rawMacro = (src.macroCategory || item.macroCategory || '').toLowerCase()
                const itemName = finalItem.name || src.name || item.name || ''
                const canonicalMacro = macroCategory(rawMacro, '', itemName) || rawMacro
                return {
                    ...finalItem,
                    color: item.color || src.color || 'neutral',
                    type: item.type || src.type || 'clothing',
                    name: finalItem.name || src.name || item.name || src.type || src.category || 'Item',
                    brand: finalItem.brand || item.brand || src.brand || '',
                    macroCategory: canonicalMacro || rawMacro || '',
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
