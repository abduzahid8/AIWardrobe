import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from "https://esm.sh/@supabase/supabase-js@2"

const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

// Team Collaboration is always 100% shop — no wardrobe anchor.
// Matches by style (`business_casual`) OR occasion label (`Team Collaboration`).
function parseBudgetFromText(text?: string | null): { amount: number; currency: string } | null {
    if (!text) return null
    const src = String(text)
    const CURRENCY: Record<string, string> = {
        $: 'USD', usd: 'USD', dollar: 'USD', dollars: 'USD',
        '€': 'EUR', eur: 'EUR', euro: 'EUR', euros: 'EUR',
        '£': 'GBP', gbp: 'GBP', pound: 'GBP', pounds: 'GBP',
    }
    const unitFollow = /^(?:\s*(?:°\s*)?(?:c|f|cm|kg|lbs?|items?|pieces?|degrees?)\b)/i
    const tokenCurr = (t?: string) => {
        if (!t) return 'USD'
        return CURRENCY[t.trim().toLowerCase()] || CURRENCY[t.trim()] || 'USD'
    }
    const parseAmt = (raw: string) => {
        const n = Number(String(raw).replace(',', '.'))
        if (!Number.isFinite(n) || n < 1 || n > 100000) return null
        return Math.round(n * 100) / 100
    }
    const found: Array<{ amount: number; currency: string }> = []
    const consider = (amountRaw: string, currRaw: string, after: string) => {
        if (unitFollow.test(after)) return
        const amount = parseAmt(amountRaw)
        if (amount == null) return
        found.push({ amount, currency: tokenCurr(currRaw) })
    }
    let m: RegExpExecArray | null
    const prefix = /([$€£])\s*(\d+(?:[.,]\d{1,2})?)/g
    while ((m = prefix.exec(src)) !== null) consider(m[2], m[1], src.slice(m.index + m[0].length))
    const suffix = /(\d+(?:[.,]\d{1,2})?)\s*([$€£]|usd|eur|gbp|dollars?|euros?|pounds?)/gi
    while ((m = suffix.exec(src)) !== null) consider(m[1], m[2], src.slice(m.index + m[0].length))
    const keyword = /\b(?:budget(?:\s*(?:of|is|:))?|under|below|within|up\s*to|at\s*most|no\s+more\s+than|less\s+than|max(?:imum)?(?:\s+of)?)\s*([$€£]|usd|eur|gbp)?\s*(\d+(?:[.,]\d{1,2})?)/gi
    while ((m = keyword.exec(src)) !== null) consider(m[2], m[1] || '', src.slice(m.index + m[0].length))
    if (found.length === 0) return null
    return found.reduce((best, cur) => (cur.amount < best.amount ? cur : best))
}

function isShopPricedItem(item: any): boolean {
    if (!item) return false
    if (item.isWardrobe === true && !item.isShopItem) return false
    const id = String(item.id || '')
    return Boolean(item.isShopItem || item.isWardrobe === false || /^(shop_|md-|zara-|classic-m-)/i.test(id))
}

function shopPriceOf(item: any): number {
    if (!isShopPricedItem(item)) return 0
    const n = Number(item?.price)
    return Number.isFinite(n) && n > 0 ? n : 0
}

function outfitShopSpend(items: any[], itemMap?: Map<string, any>): number {
    return (items || []).reduce((sum, it) => {
        const src = itemMap?.get(String(it?.id)) || it
        return sum + shopPriceOf({ ...src, ...it, price: src?.price ?? it?.price, isShopItem: src?.isShopItem ?? it?.isShopItem, id: src?.id ?? it?.id })
    }, 0)
}

function fitOutfitToBudget(parts: any[], pool: any[], budget: { amount: number } | null | undefined): any[] {
    if (!budget || !Array.isArray(parts) || parts.length === 0) return parts
    const cap = budget.amount
    const slotOf = (p: any) => String(p?.macroCategory || '').toLowerCase()
    const cheaper = (slot: string, exclude: Set<string>, maxPrice: number) => {
        return pool
            .filter((p: any) => isShopPricedItem(p) && slotOf(p) === slot && !exclude.has(String(p.id)))
            .filter((p: any) => {
                const pr = shopPriceOf(p)
                return pr > 0 && pr <= maxPrice
            })
            .sort((a: any, b: any) => shopPriceOf(a) - shopPriceOf(b))[0]
    }
    const next = parts.map((p: any) => ({ ...p }))
    for (let guard = 0; guard < 8 && outfitShopSpend(next) > cap; guard++) {
        let worstIdx = -1
        let worstPrice = -1
        next.forEach((p: any, i: number) => {
            const pr = shopPriceOf(p)
            if (pr > worstPrice) { worstPrice = pr; worstIdx = i }
        })
        if (worstIdx < 0) break
        const used = new Set(next.map((p: any) => String(p.id)))
        const room = cap - (outfitShopSpend(next) - shopPriceOf(next[worstIdx]))
        const repl = cheaper(slotOf(next[worstIdx]), used, room)
        if (repl) {
            next[worstIdx] = { ...repl, recommendation: next[worstIdx].recommendation || 'Suggested from shop' }
        } else if (slotOf(next[worstIdx]) === 'outerwear' && next.length > 3) {
            next.splice(worstIdx, 1)
        } else {
            break
        }
    }
    return next
}

function isShopOnlyDeck(style: string | null | undefined, occasion: string | null | undefined): boolean {
    const s = (style || '').toLowerCase().replace(/[\s-]+/g, '_')
    if (s === 'business_casual') return true
    const o = (occasion || '').toLowerCase()
    if (o.includes('team') && o.includes('collab')) return true
    if (o.trim() === 'team collaboration') return true
    return false
}

function macroCategory(category: string, type: string, name?: string): string {
    const t = `${category} ${type} ${name || ''}`.toLowerCase()
    
    let result = 'other'
    if (t.match(/\b(blazer|overcoat|topcoat|peacoat|trench(?:es)?|parka|puffer|windbreaker|bomber)s?\b/) ||
        t.match(/\b(coat|jacket|cardigan|sweater|hoodie|vest|pullover|fleece)s?\b/) ||
        t.match(/jacket|coat|blazer|hoodie|cardigan|sweater|pullover|vest|puffer|parka|trench|outerwear/)) {
        result = 'outerwear'
    } else if (t.match(/\b(pants?|trousers?|jeans|shorts?|skirts?|chinos?|slacks?|joggers?|sweatpants?)\b/) ||
        t.match(/pant|trouser|jeans|bottom|shorts|skirt|lower[_\s-]?body/)) {
        result = 'bottom'
    } else if (t.match(/\b(shoes?|sneakers?|boots?|loafers?|sandals?|heels?|trainers?|derbys?|derbies|mules?)\b/) ||
        (t.match(/\boxfords?\b/) && !t.match(/\bshirts?\b/)) ||
        t.match(/\boxfords?\s*(shoes?|flats?|lace|brogues?|derbys?|derbies)\b/) ||
        t.match(/shoe|sneaker|boot|loafer|sandal|heel|footwear|trainer/)) {
        result = 'shoes'
    } else if (t.match(/\b(t-shirt|tshirt|tee|polo|blouse|shirt|dress(?:es)?)s?\b/) ||
        t.match(/shirt|t-shirt|tshirt|tee|blouse|polo|tops|top|upper[_\s-]?body/) ||
        t.match(/dress/)) {
        result = 'top'
    }
    
    console.log(`[Deno macroCategory] Input -> category: "${category}", type: "${type}", name: "${name || ''}" => Mapped to: "${result}"`)
    return result
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
    // 'short-sleeved' describes SLEEVES on tops — strip it first, otherwise
    // every short-sleeve tee mislabeled as bottoms reads as shorts.
    const blob = `${item?.type || ''} ${item?.name || ''} ${item?.sub_category || ''} ${item?.subCategory || ''} ${item?.description || ''}`.toLowerCase().replace(/\bshort[\s-]?sleeve(?:s|d)?\b/g, '')
    const macro = String(item?.macroCategory || '').toLowerCase()
    const isBottom = macro === 'bottom' || /pant|trouser|jeans|bottom|shorts?|skirt|trunks?|swim|lower[_\s-]?body/.test(blob)
    if (!isBottom) return false
    // NOTE: must include swim/trunks — "Swimming trunks" contains neither
    // "shorts" nor "bermuda" and previously passed validation next to suits.
    return /\b(shorts?|bermudas?|trunks?|swim\w*|swimwear|board\s*shorts?|bathing\s*suit)\b/.test(blob)
}

// Swim bottoms (trunks, swim shorts) are never work/formal wear, even with a
// casual layer. Rejected for work-ish occasions regardless of the layer.
function isSwimBottomItem(item: any): boolean {
    const blob = `${item?.type || ''} ${item?.name || ''} ${item?.sub_category || ''} ${item?.subCategory || ''} ${item?.description || ''}`.toLowerCase()
    const macro = String(item?.macroCategory || '').toLowerCase()
    if (macro !== 'bottom' && !/lower[_\s-]?body/.test(blob)) return false
    return /\b(swim\w*|swimwear|trunks?|board\s*shorts?|bathing\s*suit)\b/.test(blob)
}

// Driving moccasins sit a tier below loafers (soft sole, tie laces) and are
// never work/formal wear — same workish-occasion rejection as swim bottoms.
// Penny/bit/tassel loafers, oxfords, and derbies are unaffected.
function isDriverShoeItem(item: any): boolean {
    const blob = `${item?.type || ''} ${item?.name || ''} ${item?.sub_category || ''} ${item?.subCategory || ''} ${item?.description || ''}`.toLowerCase()
    const macro = String(item?.macroCategory || '').toLowerCase()
    if (macro !== 'shoes' && !/shoe|sneaker|footwear|upper[_\s-]?body/.test(blob)) return false
    return /\b(moccasins?|drivers?|driving|gommino|mules?)\b/.test(blob)
}

function isWorkishOccasion(occasion: string | null | undefined): boolean {
    return /work|business|office|meeting|interview|wedding|funeral|formal|conference|corporate/.test((occasion || '').toLowerCase())
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
        [/\bsneaker(s)?\b/i, -3], [/\bchunky\b/i, -3],
        [/\bheavyweight\s+tee\b/i, -2], [/\b3[-\s]?pack\b/i, -2],
        [/\bt-shirt\b/i, -2], [/\btee\b/i, -2], [/\bwindbreaker\b/i, -3], [/\bzip[-\s]?up\b/i, -3],
        // Driving moccasins sit a tier below loafers — never with tailoring.
        [/\bmoccasins?\b/i, -3], [/\bdriving\b/i, -2], [/\bgommino\b/i, -3],
        [/\bmules?\b/i, -2],
        // Souvenir graphics and loud prints break quiet luxury.
        [/\bsouvenirs?\b/i, -4], [/\bfloral\b/i, -3], [/\bprints?\b/i, -2], [/\bprinted\b/i, -2],
        [/\bpaisley\b/i, -3],
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
        [/\bshorts\b/i, -5], [/\bbermuda\b/i, -5], [/\btrunks\b/i, -5], [/\bswim\w*\b/i, -5],
        [/\bmoccasins?\b/i, -3], [/\bdriving\b/i, -2], [/\bgommino\b/i, -3],
        [/\bmules?\b/i, -2],
        [/\bsouvenirs?\b/i, -4], [/\bfloral\b/i, -3], [/\bprints?\b/i, -2], [/\bprinted\b/i, -2],
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
        [/\bchunky sneaker(s)?\b/i, -4], [/\bsneaker(s)?\b/i, -3], [/\bt-shirt\b/i, -2], [/\btee\b/i, -2],
        [/\bathletic\b/i, -3], [/\bsquare[-\s]?toe\b/i, -4],
        [/\bneon\b/i, -4], [/\bsequin\b/i, -4], [/\bripped\b/i, -3],
        [/\blow[-\s]?rise\b/i, -3], [/\bskinny\b/i, -2], [/\boversized\b/i, -2],
        [/\bbackpack\b/i, -2], [/\bbi[kc]ini\b/i, -2], [/\bwindbreaker\b/i, -3], [/\bzip[-\s]?up\b/i, -3],
        [/\bmoccasins?\b/i, -3], [/\bdriving\b/i, -2], [/\bgommino\b/i, -3],
        [/\bmules?\b/i, -2],
        [/\bsouvenirs?\b/i, -4], [/\bfloral\b/i, -3], [/\bprints?\b/i, -2], [/\bprinted\b/i, -2],
        [/\bpaisley\b/i, -3],
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

// ── Product-family similarity (mirrors src/utils/shopCatalogOrder.ts) ──
// Deno can't import client TS, so this is a compact port: listings that look
// identical at card size ("Brown Shirt" ×3, eight penny-loafer variants)
// share a key. Rank ties cluster them adjacently, so rotation must spread
// them apart instead of walking through lookalikes.
const SIM_COLOR_WORDS = new Set('black white offwhite ecru cream beige stone sand taupe camel khaki olive green sage mint blue navy indigo brown chocolate grey gray silver charcoal red burgundy maroon pink purple lilac yellow mustard orange rust'.split(' '))
const SIM_LOW_WORDS = new Set('basic classic essential premium soft touch washed faded heavyweight mediumweight lightweight regularfit relaxedfit slimfit loosefit textured structured comfort interlock stretch cotton viscose polyester polyamide elastane lyocell modal jersey fit detail details edition men man zara suit shoe'.split(' '))
function simKeyEdge(name: string, slot: string): string {
    const toks: string[] = []
    const seen = new Set<string>()
    const words = String(name || '').toLowerCase().replace(/https?:\/\//g, '').replace(/[^a-z0-9]+/g, ' ').split(' ')
    for (let raw of words) {
        if (!raw || /^\d+$/.test(raw) || raw.length < 2) continue
        if (raw.endsWith('ies') && raw.length > 4) raw = raw.slice(0, -3) + 'y'
        else if (raw.endsWith('s') && raw.length > 3 && !raw.endsWith('ss')) raw = raw.slice(0, -1)
        if (raw === 'tshirt' || raw === 'tee' || raw === 'tees') raw = 'tee'
        if (raw === 'trainer' || raw === 'trainers') raw = 'sneaker'
        if (raw === 'sweater' || raw === 'jumper') raw = 'knit'
        if (SIM_COLOR_WORDS.has(raw) || SIM_LOW_WORDS.has(raw) || seen.has(raw)) continue
        seen.add(raw)
        toks.push(raw)
    }
    return `${slot}:${toks.join(' ') || slot}`
}
function spreadBySimEdge<T>(items: T[], keyFn: (x: T) => string): T[] {
    if (items.length <= 2) return items
    const remaining = items.map((item) => ({ item, key: keyFn(item) }))
    const ordered: typeof remaining = []
    const recent: string[] = []
    while (remaining.length > 0) {
        let idx = remaining.findIndex((c) => !recent.includes(c.key))
        if (idx === -1) {
            const last = recent[recent.length - 1]
            idx = remaining.findIndex((c) => c.key !== last)
        }
        if (idx === -1) idx = 0
        const [next] = remaining.splice(idx, 1)
        ordered.push(next)
        recent.push(next.key)
        if (recent.length > 2) recent.shift()
    }
    return ordered.map((o) => o.item)
}
/** Collapse same-product listings post-rank (rank is score-sorted, so the
 * first id per similarity key is the best one). Must run BEFORE any
 * top-N trim, otherwise lookalikes eat the elite budget. */
function collapseBySimEdge<T>(items: T[], keyFn: (x: T) => string): T[] {
    const seen = new Set<string>()
    const out: T[] = []
    for (const item of items) {
        const key = keyFn(item)
        if (seen.has(key)) continue
        seen.add(key)
        out.push(item)
    }
    return out
}
function deckHashEdge(style: string, occasion: string): number {
    const key = `${style}::${occasion}`.toLowerCase()
    let hash = 0
    for (let i = 0; i < key.length; i++) hash = (Math.imul(31, hash) + key.charCodeAt(i)) | 0
    return Math.abs(hash)
}
function deckSeedEdge(style: string, occasion: string): number {
    return deckHashEdge(style, occasion) % 32
}
function deckReverseEdge(style: string, occasion: string): boolean {
    return ((deckHashEdge(style, occasion) >> 5) & 1) === 1
}

// ── Rotating styling tips (transcript doctrine, user-facing) ──────────
// Fixed tips repeat on every card; a rotating pool surfaces the guide
// rules that never fit item data: socks match trousers (never shoes),
// the bottom jacket button stays undone, 1-2cm of cuff shows, three
// colors max (white doesn't count), dark/light contrast beats flat
// monochrome. Two distinct tips per outfit.
const STYLING_TIPS_LAYERED = [
    'Layer the base top under the outerwear for depth',
    'Keep the palette tonal for a refined finish',
    'Match your socks to your trousers, not your shoes',
    'Leave the bottom jacket button undone',
    'Show 1-2cm of shirt cuff beyond the jacket sleeve',
    'Texture elevates: linen, flannel or suede over flat cotton',
]
const STYLING_TIPS_FLAT = [
    'Add accessories to personalize',
    'Roll sleeves or cuffs for a lived-in feel',
    'Match your socks to your trousers, not your shoes',
    "Keep it to three colors — white doesn't count",
    'Dark top + light bottom (or the reverse) reads most masculine',
    'Trousers sit at mid-rise; low-rise breaks the proportions',
]
function pickStylingTips(layered: boolean): string[] {
    const pool = layered ? STYLING_TIPS_LAYERED : STYLING_TIPS_FLAT
    const a = Math.floor(Math.random() * pool.length)
    let b = Math.floor(Math.random() * pool.length)
    if (b === a) b = (b + 1) % pool.length
    return [pool[a], pool[b]]
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
11. FOOTWEAR RULES: Loafers (penny/bit/tassel), oxfords, and derbies for smart and classic — all pair with tailoring. Driving moccasins sit a full tier BELOW loafers (soft unstructured sole, tie laces): Riviera-casual with chinos or linen, NEVER with suits or at work. Clean minimal sneakers only for casual; never athletic trainers with tailored clothing. Suede is more versatile than leather for smart-casual.
12. ACCESSORIES COMPLETE: Pocket squares, ties, seasonal scarves, watches elevate any outfit from ordinary to polished.
13. FORMALITY COHERENCE: Every item in one outfit should sit within 1-2 formality tiers of each other. A formal blazer + gym shorts = hard clash.
14. QUALITY OVER QUANTITY: One well-made piece in a good fabric outperforms three cheap ones. Natural fibers (wool, linen, cotton, silk) always look more expensive than synthetics.
15. AVOID: puffer jackets with suits, square-toe shoes, backpacks with tailored clothing, sports watches with formal outfits, graphic tees in any non-casual look.
16. SOCKS MATCH TROUSERS, NOT SHOES: Grey, navy, and burgundy socks are the foundation. Black socks are disallowed — they show wear faster and pair with nothing.
17. CONTRAST & COLOR BALANCE: Dark top + light bottom (or vice versa) creates more masculine, harmonious outfits. Limit to 3 colors per outfit; white doesn't count and can be a 4th. Monochromatic looks need different textures/shades to avoid looking flat.
18. JACKET FIT DETAILS: Jacket must cover the buttocks. 1-2cm of shirt cuff must show beyond the jacket sleeve. Never button the bottom vest button. Tie width should be 7.5-9.5cm.
19. CITY BANS: No shorts, sandals/mules, mesh/see-through tees, or mandarin/stand collars in urban settings. No popped polo collars. Always button polo buttons (all or leave top one open). No exotic leather shoes (crocodile, python) — quality calf or suede only.
20. PATTERNED SPORT JACKETS: Glen check, Prince of Wales, Gun Club are the most versatile patterns — they contain multiple colors for easy coordination. Avoid smooth solid-color sport jackets (look cheap); choose textured or patterned fabrics. Mixed fabric compositions (wool-linen-silk blends) are more textured and practical than pure fibers.`

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
            'NEVER include: graphic, logo, print, printed, patterned, floral, paisley, camo, tie-dye, pattern, patterns, souvenir jackets, pajama, pyjama, sleepwear, loungewear, t-shirt.',
            'NEVER include: hoodie, hoodies, hooded, windbreaker, windbreakers, puffer, puffers, zip-up, zip-ups, bomber, bombers, track jacket, track jackets, graphic sweatshirts.',
            'NEVER include: cargo shorts, cargo pants, athletic shorts, basketball shorts, track pants, sweatpants, joggers, shorts/bermudas, drawstring pants, elastic-waist trousers.',
            'NEVER include: washed-effect jackets, bell-bottom or splatter pants, fashion denim, twisted-seam jeans, piqué polos for work.',
            'NEVER include: driving moccasins, loafers with tie laces, chunky sneakers, high-tops, platform shoes, athletic trainers, running shoes, basketball shoes, square-toe shoes.',
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
            'Avoid: hoodie, hoodies, hooded, graphic, logo, print, printed, cargo, sweatpants, joggers, pajama, pyjama, sleepwear, t-shirt.',
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
            'NEVER include: graphic prints, logos, tie-dye, floral prints, neon colors, rhinestones, sequins, pleated trousers, pajama, pyjama, sleepwear.',
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
            'Avoid: hoodie, hoodies, hooded, windbreaker, windbreakers, puffer, puffers, zip-up, zip-ups, bomber, bombers, track jacket, track jackets, souvenir jackets.',
            'Avoid: graphic, logo, print, printed, patterned, floral, paisley, camo, tie-dye, pattern, patterns, neon, rhinestones, sequins, pajama, pyjama, sleepwear, loungewear, t-shirt.',
            'Avoid: cargo pants, cargo shorts, ripped jeans, sweatpants, joggers, shorts, drawstring pants, elastic-waist trousers.',
            'Avoid: black as primary color — use navy or charcoal instead.',
            'Avoid: chunky sneakers, platform shoes, athletic trainers, running shoes, basketball shoes, square-toe shoes, driving moccasins, mules.',
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
            'Avoid: sweatpants, basketball shorts, graphic hoodies, hoodies, hooded layers, pajama, pyjama, sleepwear.',
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
            'NEVER include: hoodie, hoodies, hooded, windbreaker, windbreakers, puffer, puffers, zip-up, zip-ups, bomber, bombers, track jacket, track jackets.',
            'NEVER include: graphic, logo, print, printed, patterned, floral, paisley, camo, tie-dye, pattern, patterns, neon, rhinestones, sequins, pajama, pyjama, sleepwear, loungewear, t-shirt.',
            'NEVER include: cargo pants, cargo shorts, athletic shorts, basketball shorts, track pants, sweatpants, joggers, shorts/bermudas, drawstring pants, elastic-waist trousers.',
            'NEVER include: chunky sneakers, platform shoes, athletic trainers, running shoes, basketball shoes, square-toe shoes, driving moccasins, mules.',
            'NEVER include: souvenir jackets, floral/paisley/printed layers.',
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
};

function getStyleContext(style: string): typeof STYLE_CONTEXT[string] {
    const key = style.toLowerCase().replace(/[\s-]+/g, '_')
    return STYLE_CONTEXT[key] || STYLE_CONTEXT.casual
}

function buildSystemPrompt(limit: number, style: string, layered: boolean, preferences?: any, occasion?: string, budget?: { amount: number; currency: string } | null): string {
    const ctx = getStyleContext(style)
    const shopOnly = isShopOnlyDeck(style, occasion)
    const slotRules = layered
        ? `- EVERY outfit MUST contain exactly these FOUR macroCategory slots, no exceptions:
    (1) a BASE TOP (macroCategory="top" — t-shirt, shirt, polo, blouse, tank worn next-to-skin)
    (2) a MAIN TOP / OUTERWEAR LAYER (macroCategory="outerwear" — sweater, cardigan, blazer, hoodie, jacket, coat, vest)
    (3) a BOTTOM (macroCategory="bottom" — pants, trousers, jeans, shorts, skirt)
    (4) SHOES (macroCategory="shoes")
  A 3-item outfit is INVALID unless the base top is a dress (in which case skip the bottom slot but STILL include outerwear + shoes). Never return fewer than 4 items for a layered look.`
        : `- EVERY outfit MUST contain exactly THREE items and exactly these three macroCategory slots, no exceptions: (1) a TOP (or a dress), (2) a BOTTOM (skip this slot only if the top is a dress — then the outfit has 2 items: dress + shoes), and (3) SHOES. Do NOT add a fourth outerwear/layer item — the weather does not require layering. A non-layered outfit with 4+ items is INVALID.`

    const ownershipRule = shopOnly
        ? `- THIS IS A SHOP-ONLY DECK (Team Collaboration). EVERY item in EVERY outfit MUST have an id starting with "shop_". NEVER use user wardrobe items. Build all ${limit} looks purely from shop candidates.`
        : `- PRIORITIZE USER-OWNED ITEMS. Items whose id starts with "shop_" are optional catalog suggestions; use them only to complete a missing/sparse slot or add necessary variety. Every outfit should contain owned pieces whenever any compatible owned pieces exist.`
    const budgetRule = budget
        ? `- HARD BUDGET ${budget.amount} ${budget.currency}: for EACH outfit, the sum of shop-item prices MUST be <= ${budget.amount}. Owned wardrobe items (source:user wardrobe) cost 0. Never pick a shop item whose price alone exceeds remaining budget. Prefer a cheaper shop substitute over dropping a required slot.`
        : `- Shop items may include a price. If the user prompt names a budget, shop prices in each look MUST sum to that cap or less (owned = 0). Always cite prices for shop picks.`
    const uniquenessRule = shopOnly
        ? `- ALL ${limit} OUTFITS MUST BE VISIBLY DIFFERENT. Each outfit must differ from EVERY other outfit by AT LEAST 2 item ids (for 3-item looks) / AT LEAST 2 item ids (for 4-item layered looks). Enumerate the Cartesian product of shop slots — rotate tops, bottoms AND shoes so outfit_1 uses top A / bottom A / shoes A, outfit_2 uses top B / bottom B / shoes B, etc. Repeating the same set — or sharing all-but-one item — is INVALID.
- VARY THE PRODUCTS, not just the ids: candidates with the same product name are color variants of one garment. Do not feature the same product name in more than 3 of the ${limit} looks — spread same-name items across outfits instead of clustering them.`
        : `- ALL ${limit} OUTFITS MUST BE VISIBLY DIFFERENT. Each outfit must differ from EVERY other outfit by AT LEAST 2 item ids. Rotate slots: if outfit_1 uses top A, outfit_2 MUST use a different top B (if ≥${limit} tops exist), same rotation for bottoms and shoes. Never return the same set of item ids twice, and never return two outfits sharing all-but-one item. Reusing one anchor item across all outfits is allowed ONLY when an ANCHOR ITEM is locked; all other slots MUST still rotate.`

    return `You are a world-class fashion stylist curating outfits for a specific aesthetic. You must obey the STYLE DIRECTION below as if your reputation depends on it.

TASK: Create ${limit} complete outfits using ONLY items from the provided wardrobe, referenced by their exact id.

GENERAL STYLING RULES:
${slotRules}
${ownershipRule}
${budgetRule}
${uniquenessRule}
- If no shoe item obviously matches the style, pick the LEAST off-aesthetic shoe available. An outfit WITHOUT shoes is invalid — never return one.
- NEVER invent items. Use ONLY the exact ids listed in "Available wardrobe items".
- Pick items whose styleTags, name, description, or material indicate they FIT the requested style. If an item's tags or description clash with the style, do NOT use it (unless it is the only option for that slot — see shoes rule above).
- Favor color harmony (tonal, monochromatic, or analogous palettes from the style's preferred palette). When layering, the base top must color-coordinate with the outerwear (tonal or complementary — never clashing).
- DIVERSITY IS MANDATORY: spread usage evenly across the available pool. Do not put the same top, bottom, or shoes in all ${limit} outfits when alternatives exist. If a slot has ≥${limit} candidates, each outfit MUST use a different one.
- NEVER pair formal outerwear (blazer, suit jacket, sport coat, overcoat, topcoat, trench, peacoat, tuxedo) with shorts or bermudas. If the only available bottom is shorts, drop the formal outerwear and return a non-layered 3-item look instead.
- Each outfit needs a vivid 1-2 sentence description and 2 actionable styling tips.

USER'S SPECIFIC PERSONAL STYLE SETTINGS:
You MUST strictly satisfy the following style, color, and pattern settings chosen by the user in onboarding:
${preferences && preferences.favoriteColors && preferences.favoriteColors.length > 0 ? `- FAVORITE COLORS: ${preferences.favoriteColors.join(', ')}. Actively prioritize and suggest outfits featuring these colors.` : ''}
${preferences && preferences.avoidColors && preferences.avoidColors.length > 0 ? `- AVOID COLORS: ${preferences.avoidColors.join(', ')}. NEVER include any clothing item in these colors. This is a strict constraint.` : ''}
${preferences && preferences.fitPreference ? `- FIT PREFERENCE: ${preferences.fitPreference}. Choose garments that match this silhouette.` : ''}
${preferences && preferences.likedPatterns && preferences.likedPatterns.length > 0 ? `- LIKED PATTERNS: ${preferences.likedPatterns.join(', ')}. Prefer these patterns.` : ''}
${preferences && preferences.dislikedPatterns && preferences.dislikedPatterns.length > 0 ? `- DISLIKED PATTERNS: ${preferences.dislikedPatterns.join(', ')}. NEVER include any item featuring these patterns.` : ''}
${preferences && preferences.prefersSustainable ? `- SUSTAINABILITY: The user prefers sustainable materials when possible.` : ''}
${preferences && preferences.styleGoals && preferences.styleGoals.length > 0 ? `- STYLE GOALS: ${preferences.styleGoals.join(', ')}.` : ''}

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
  4. Do all ${limit} outfits differ from each other by AT LEAST 2 item ids? If any two share all-but-one item, rebuild the overlapping outfit with different slot picks.
If the answer to any of these is "no", rebuild the outfit.

Respond ONLY in pure JSON (no markdown, no code fences):
{"outfits":[{"id":"outfit_1","style":"${style}","occasion":"...","description":"...","confidence":0.9,"items":[{"id":"EXACT_DB_ID","type":"top","color":"cream","name":"...","brand":"...","imageUrl":"...","macroCategory":"top","recommendation":"why this piece works for ${style}"}],"stylingTips":["tip1","tip2"]}]}`
}

function buildUserPrompt(items: any[], prompt: string, style: string, occasion: string, weather: any, limit: number, preferences?: any): string {
    // Richer bullets: name + description + inferred styleTags give the model
    // real signal instead of only color + type.
    const itemSummary = items.map(i => {
        const name = i.name || i.type || 'item'
        // Descriptions carry fabric/fit in the first ~80 chars; the tail is
        // marketing filler billed per token across 30-40 items per request.
        const desc = (i.description || '').toString().slice(0, 80)
        const tags = Array.isArray(i.styleTags) && i.styleTags.length > 0
            ? i.styleTags.join(',')
            : ''
        const material = i.material || ''
        const pattern = i.pattern || ''
        const formality = typeof i.formality === 'number' ? ` formality:${i.formality.toFixed(2)}` : ''
        const parts = [
            `[${i.id}]`,
            `${name}`,
            i.type ? `type:${i.type}` : '',
            `color:${i.color || 'neutral'}`,
            `cat:${i.macroCategory || 'other'}`,
            i.brand ? `brand:${i.brand}` : '',
            i.isShopItem || String(i.id || '').startsWith('shop_') ? 'source:shop suggestion' : 'source:user wardrobe',
            i.isShopItem || String(i.id || '').startsWith('shop_')
                ? (typeof i.price === 'number' && i.price > 0
                    ? `price:${i.price}${i.currency ? ` ${i.currency}` : ''}`
                    : 'price:unknown')
                : 'price:owned(0)',
            material ? `material:${material}` : '',
            pattern ? `pattern:${pattern}` : '',
            tags ? `styleTags:${tags}` : '',
        ].filter(Boolean).join(' | ')
        return `• ${parts}${formality}${desc ? `\n    desc: "${desc}"` : ''}`
    }).join('\n')

    let prefText = ''
    if (preferences) {
        const lines = []
        if (preferences.favoriteColors?.length) lines.push(`- Highly favored colors: ${preferences.favoriteColors.join(', ')}`)
        if (preferences.avoidColors?.length) lines.push(`- strictly banned colors (Do not use): ${preferences.avoidColors.join(', ')}`)
        if (preferences.fitPreference) lines.push(`- Fit preference: ${preferences.fitPreference}`)
        if (preferences.likedPatterns?.length) lines.push(`- Patterns to favor: ${preferences.likedPatterns.join(', ')}`)
        if (preferences.dislikedPatterns?.length) lines.push(`- Patterns to strictly avoid: ${preferences.dislikedPatterns.join(', ')}`)
        if (preferences.maxShopBudget?.amount) {
            lines.push(`- HARD BUDGET: shop items in each outfit must sum to <= ${preferences.maxShopBudget.amount} ${preferences.maxShopBudget.currency || ''}. Owned items cost 0.`)
        }
        if (lines.length > 0) {
            prefText = `\n\nREMINDER OF USER PREFERENCES:\n${lines.join('\n')}`
        }
    }

    const temp = typeof weather?.temp === 'number' ? weather.temp : null
    let climateRules = ''
    if (temp != null && temp < 18) {
        climateRules = `CLIMATE HARD RULES: It is ${temp}°C (${weather?.condition || 'cool'}). Do NOT pick linen, seersucker, tropical wool, shorts, sandals, or espadrilles. Prefer wool, merino, cashmere, flannel, cotton oxfords, and leather shoes. EVERY outfit MUST include a warm layer (wool blazer, knit, coat, cardigan) — a summer shirt alone is invalid.\n`
    } else if (temp != null && temp >= 26) {
        climateRules = `CLIMATE HARD RULES: It is ${temp}°C. Do NOT pick puffer/parka/overcoat. Skip heavy tweed and flannel. Linen, cotton, and tropical wool are appropriate.\n`
    }

    return `${prompt ? `User's specific request: "${prompt}"\n\n` : ''}Requested style: ${style || 'Casual'}
${occasion ? `Occasion: ${occasion}\n` : ''}${weather ? `Weather: ${weather.temp}°C, ${weather.condition}\n` : ''}${climateRules}

Available wardrobe items (use ONLY these exact ids; every id is pre-filtered to be at least plausible for ${style}, but you MUST still reject any item whose name/description/styleTags clash with the style):
${itemSummary}${prefText}

Create ${limit} distinct, on-aesthetic outfits. Each outfit must feel like it was hand-picked by a ${style} stylist, not a random color-matching algorithm.`
}

// ── Validator: reject outfits that violate the style's REJECT list ───────
// Runs AFTER the model responds. If too many outfits clash with the style,
// we can trigger a retry. This catches the failure mode where the model
// obeys color rules but picks, say, a graphic tee for Old Money.
function validateOutfitAgainstStyle(outfit: any, style: string, itemMap: Map<string, any>, layered: boolean, preferences?: any): { ok: boolean; reason?: string } {
    const ctx = getStyleContext(style)
    const items = Array.isArray(outfit.items) ? outfit.items : []
    const minItems = layered ? 4 : 3
    if (items.length < minItems) return { ok: false, reason: `too few items (${items.length}, need ${minItems})` }

    // Identity integrity: every id must come from the candidate pool.
    // The model invents ids ("fake_999") and repeats one id across two slots
    // ([blazer, blazer, pants, shoes]) — both pass slot checks below but
    // render broken cards. Reject before anything else.
    const seenIds = new Set<string>()
    for (const it of items) {
        const key = String(it?.id)
        if (!itemMap.has(key)) return { ok: false, reason: `unknown item id "${it?.id}" (not in candidate pool)` }
        if (seenIds.has(key)) return { ok: false, reason: `duplicate item id "${it?.id}" inside one outfit` }
        seenIds.add(key)
    }

    // Slot integrity: feed rows are mistagged at scale (a T-shirt stored as
    // bottoms). Resolve the slot from the NAME only — category/type carry
    // the same mistag. A definitive mismatch rejects the outfit.
    for (const it of items) {
        const src = itemMap.get(String(it?.id)) || {}
        const macro = String(src.macroCategory || it.macroCategory || '').toLowerCase()
        if (macro !== 'top' && macro !== 'outerwear' && macro !== 'bottom' && macro !== 'shoes') continue
        const nameOnly = `${it?.name || ''} ${src.name || ''}`.toLowerCase()
        const nameIsTop = /\bt-?shirts?\b|\btees?\b|\bpolos?\b|\bblouses?\b|\btanks?\b|\bhenleys?\b|\bshirts?\b|\boxfords?\b|\bbutton[\s-]?downs?\b/.test(nameOnly)
        const nameIsOuter = /\bjackets?|coats?|blazers?|hoodies?|cardigans?|sweaters?|pullovers?|puffers?|bombers?|vests?|outerwears?|trenchs?|peacoats?|sweatshirts?|knitwears?|jumpers?|fleece\b/.test(nameOnly)
        const nameIsBottom = /\bpants?|trousers?|jeans?|bottoms?|shorts|bermudas?|skirts?|chinos?|slacks?|joggers?|trunks?|swim/.test(nameOnly)
        const nameIsShoe = /\bshoes?|sneakers?|boots?|loafers?|sandals?|heels?|trainers?|derbys?|derbies|moccasins?|drivers?|mules?\b/.test(nameOnly)
        const nameSlot = nameIsShoe ? 'shoes' : nameIsBottom ? 'bottom' : nameIsOuter ? 'outerwear' : nameIsTop ? 'top' : null
        if (nameSlot !== null && nameSlot !== macro) {
            return { ok: false, reason: `item "${it?.name || it?.id}" is a ${nameSlot} sitting in the ${macro} slot (mislabeled catalog row)` }
        }
    }

    // Underwear is never an outfit part — boxer briefs were nominated as
    // bottoms for work decks. ("Thong" deliberately excluded: thong sandals
    // are legitimate footwear.)
    const UNDERWEAR_RE = /\b(boxers?|briefs?|panties|lingerie|underwear|jockstrap|bra\b)/
    for (const it of items) {
        const src = itemMap.get(String(it.id)) || {}
        const blob = `${it.name || ''} ${src.name || ''} ${src.description || ''} ${it.type || ''} ${src.type || ''}`.toLowerCase()
        if (UNDERWEAR_RE.test(blob)) return { ok: false, reason: `underwear item "${it.name || it.id}" can never be an outfit part` }
    }

    // Require coverage per slot model.
    const macros = new Set<string>()
    let hasDress = false
    for (const it of items) {
        const src = itemMap.get(it.id) || {}
        const macro = (src.macroCategory || it.macroCategory || '').toLowerCase()
        if (macro) macros.add(macro)
        if (isDressItem(src) || isDressItem(it)) hasDress = true
    }
    // Non-slot macros (accessory/'other' leftovers) fill no slot — a dress
    // is the only exception (it replaces base-top + bottom in one piece).
    if (!hasDress) {
        for (const m of macros) {
            if (m !== 'top' && m !== 'outerwear' && m !== 'bottom' && m !== 'shoes') {
                return { ok: false, reason: `non-slot macro "${m}" fills no outfit slot` }
            }
        }
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

    // Check user's avoided colors if specified
    if (preferences && Array.isArray(preferences.avoidColors) && preferences.avoidColors.length > 0) {
        const avoid = preferences.avoidColors.map((c: string) => c.toLowerCase().trim()).filter(Boolean)
        for (const it of items) {
            const src = itemMap.get(it.id) || {}
            const itemColor = (it.color || src.color || '').toLowerCase()
            const itemName = (it.name || src.name || '').toLowerCase()
            const itemDesc = (it.description || src.description || '').toLowerCase()
            for (const color of avoid) {
                if (itemColor.includes(color) || itemName.includes(` ${color} `) || itemDesc.includes(` ${color} `)) {
                    return { ok: false, reason: `item "${it.name || it.id}" matches user's avoided color "${color}"` }
                }
            }
        }
    }

    // Check user's disliked patterns if specified
    if (preferences && Array.isArray(preferences.dislikedPatterns) && preferences.dislikedPatterns.length > 0) {
        const disliked = preferences.dislikedPatterns.map((p: string) => p.toLowerCase().trim()).filter(Boolean)
        for (const it of items) {
            const src = itemMap.get(it.id) || {}
            const itemPattern = (it.pattern || src.pattern || '').toLowerCase()
            const itemName = (it.name || src.name || '').toLowerCase()
            const itemDesc = (it.description || src.description || '').toLowerCase()
            for (const pattern of disliked) {
                if (itemPattern.includes(pattern) || itemName.includes(pattern) || itemDesc.includes(pattern)) {
                    return { ok: false, reason: `item "${it.name || it.id}" matches user's disliked pattern "${pattern}"` }
                }
            }
        }
    }

    // Check the style's explicit reject keywords on every item.
    const rejectKeywords = extractRejectKeywords(ctx.reject)
    for (const it of items) {
        const src = itemMap.get(it.id) || {}
        const blob = `${it.name || ''} ${src.name || ''} ${src.description || ''} ${src.type || ''} ${it.type || ''} ${it.pattern || ''} ${src.pattern || ''} ${it.material || ''} ${src.material || ''}`.toLowerCase()
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
            if (p.includes('sneaker')) out.push('sneaker')
            const firstWord = p.split(/\s+/)[0]
            if (firstWord && firstWord !== p && firstWord.length > 3) out.push(firstWord)
        }
    }
    return Array.from(new Set(out))
}

// Reject outfits that mix formal outerwear (blazer, suit jacket, overcoat,
// trench, peacoat, tuxedo, sport coat) with shorts/bermudas. This is a hard
// styling rule the model occasionally violates even when on-palette.
// Formal decks (business_casual / old_money / classic) reject shorts
// unconditionally — no shorts with suits, ever.
function validateOutfitCompatibility(outfit: any, itemMap: Map<string, any>, style?: string | null): { ok: boolean; reason?: string } {
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
    const s = (style || '').toLowerCase().replace(/[\s-]+/g, '_')
    if ((s === 'business_casual' || s === 'old_money' || s === 'classic' || s === 'formal') && hasShorts) {
        return { ok: false, reason: `shorts disallowed in formal deck ${s}` }
    }
    return { ok: true }
}

function filterValidOutfits(outfits: any[], style: string, itemMap: Map<string, any>, layered: boolean, preferences?: any, occasion?: string): any[] {
    const kept: any[] = []
    const signatures = new Set<string>()
    const keptIdSets: Array<Set<string>> = []
    // 3-item looks must differ by ≥2 items; 4-item layered looks by ≥2 items.
    // Sharing all-but-one piece (e.g. same top+bottom, different shoes) is
    // the "3 similar outfits" bug — reject it while rotation tries remain.
    const targetItems = layered ? 4 : 3
    const maxAllowedShared = targetItems - 2
    const shopOnly = isShopOnlyDeck(style, occasion)
    const hasOwnedCandidates = !shopOnly && Array.from(itemMap.values()).some(
        (item: any) => !item.isShopItem && !String(item.id || '').startsWith('shop_'),
    )
    for (const o of outfits) {
        const v = validateOutfitAgainstStyle(o, style, itemMap, layered, preferences)
        if (!v.ok) {
            console.log(`[validate] rejected outfit for ${style} (layered=${layered}): ${v.reason}`)
            continue
        }
        const c = validateOutfitCompatibility(o, itemMap, style)
        if (!c.ok) {
            console.log(`[validate] rejected outfit for ${style} (layered=${layered}): ${c.reason}`)
            continue
        }
        const budgetCap = preferences?.maxShopBudget
        if (budgetCap && typeof budgetCap.amount === 'number') {
            const spend = outfitShopSpend(o.items || [], itemMap)
            if (spend > budgetCap.amount + 0.009) {
                console.log(`[validate] rejected outfit over budget: shop spend ${spend} > ${budgetCap.amount}`)
                continue
            }
        }
        // Swim bottoms are never acceptable for work/formal occasions —
        // no "swimming trunks" in a Work outfit, with or without a blazer.
        if (isWorkishOccasion(occasion)) {
            let hasSwimBottom = false
            for (const it of (o.items || [])) {
                const src = itemMap.get(String(it.id)) || {}
                if (isSwimBottomItem({ ...src, ...it })) { hasSwimBottom = true; break }
            }
            if (hasSwimBottom) {
                console.log(`[validate] rejected swim bottom for workish occasion ${occasion}`)
                continue
            }
            // Driving moccasins are a tier below loafers — never work wear.
            let hasDriverShoe = false
            for (const it of (o.items || [])) {
                const src = itemMap.get(String(it.id)) || {}
                if (isDriverShoeItem({ ...src, ...it })) { hasDriverShoe = true; break }
            }
            if (hasDriverShoe) {
                console.log(`[validate] rejected driving moccasin for workish occasion ${occasion}`)
                continue
            }
        }
        const ids = (o.items || []).map((item: any) => String(item.id)).sort()
        const signature = ids.join('|')
        if (!signature || signatures.has(signature)) {
            console.log(`[validate] rejected duplicate outfit combination: ${signature}`)
            continue
        }
        // Near-duplicate check: reject outfits sharing more than allowed
        // items with any already-kept outfit.
        let maxShared = 0
        for (const keptSet of keptIdSets) {
            let shared = 0
            for (const id of ids) if (keptSet.has(id)) shared++
            if (shared > maxShared) maxShared = shared
        }
        if (keptIdSets.length > 0 && maxShared > maxAllowedShared) {
            console.log(`[validate] rejected near-duplicate outfit (shares ${maxShared}/${ids.length}, max allowed ${maxAllowedShared}): ${signature}`)
            continue
        }
        if (shopOnly) {
            const allShop = (o.items || []).length > 0 && (o.items || []).every((item: any) => {
                const source = itemMap.get(String(item.id)) || item
                return Boolean(source.isShopItem) || String(source.id || item.id || '').startsWith('shop_')
            })
            if (!allShop) {
                console.log(`[validate] rejected mixed/wardrobe outfit for shop-only deck ${style}/${occasion}`)
                continue
            }
        }
        if (hasOwnedCandidates) {
            const usesOwnedItem = (o.items || []).some((item: any) => {
                const source = itemMap.get(String(item.id)) || item
                return !source.isShopItem && !String(source.id || item.id || '').startsWith('shop_')
            })
            if (!usesOwnedItem) {
                console.log('[validate] rejected shop-only outfit while wardrobe items are available')
                continue
            }
        }
        signatures.add(signature)
        keptIdSets.push(new Set(ids))
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

// ── Shop-catalog color inference ──────────────────────────────────────
// shop_catalog has no color column — infer from name/description so style
// ranking and avoid-color filters still work.
// Photo-boilerplate ("white-bg product shot", "front view", ...) must be
// stripped first, otherwise every studio photo infers as white.
const SHOP_COLOR_WORDS = ['black','white','offwhite','ecru','cream','beige','stone','sand','taupe','camel','khaki','olive','green','sage','mint','blue','navy','indigo','brown','chocolate','grey','gray','silver','charcoal','red','burgundy','maroon','pink','purple','lilac','yellow','mustard','orange','rust','tan','burgundy','forest']
const SHOP_BOILERPLATE_RE = /white-bg|white background|product shots?|front views?|back views?|side views?|flat lays?|ghost mannequins?|on models?|studio shots?|model shot|e-commerce/gi
function inferShopColor(row: any): string {
    const raw = `${row?.name || ''} ${row?.description || ''}`.toLowerCase()
    const blob = raw.replace(SHOP_BOILERPLATE_RE, ' ')
    for (const c of SHOP_COLOR_WORDS) {
        if (new RegExp(`\\b${c}\\b`).test(blob)) return c
    }
    return 'neutral'
}

// ── Preference filter for SHOP candidates ─────────────────────────────
// Wardrobe items are filtered for avoidColors/dislikedPatterns before the
// shop supplement runs — but shop candidates appended afterwards used to
// skip that filter entirely, so a banned color could sneak back in via a
// shop suggestion. Apply this to every shop batch right after fetching.
function filterShopByPrefs(items: any[], preferences?: any): any[] {
    if (!preferences || !Array.isArray(items) || items.length === 0) return items
    let out = [...items]
    if (Array.isArray(preferences.avoidColors) && preferences.avoidColors.length > 0) {
        const avoid = preferences.avoidColors.map((c: string) => c.toLowerCase().trim()).filter(Boolean)
        out = out.filter((item: any) => {
            const color = (item.color || '').toLowerCase()
            const name = (item.name || '').toLowerCase()
            const desc = (item.description || '').toLowerCase()
            return !avoid.some((c: string) => color.includes(c) || name.includes(` ${c} `) || desc.includes(` ${c} `))
        })
    }
    if (Array.isArray(preferences.dislikedPatterns) && preferences.dislikedPatterns.length > 0) {
        const disliked = preferences.dislikedPatterns.map((p: string) => p.toLowerCase().trim()).filter(Boolean)
        out = out.filter((item: any) => {
            const pattern = (item.pattern || '').toLowerCase()
            const name = (item.name || '').toLowerCase()
            const desc = (item.description || '').toLowerCase()
            return !disliked.some((p: string) => pattern.includes(p) || name.includes(p) || desc.includes(p))
        })
    }
    return out
}

// ── Shop-catalog candidate fetch ──────────────────────────────────────
// The caller passes only wardrobe slots that are absent or too sparse.
// Candidates are style-ranked before they are offered to the outfit builder.
async function fetchShopItemsForAllSlotsEdge(
    supabaseClient: any,
    requiredSlots: string[],
    style: string,
    perSlot = 4,
): Promise<any[]> {
    const picks: any[] = []
    const styleKey = style.toLowerCase().replace(/[\s-]+/g, '_')
    const ctx = getStyleContext(style)
    const rejectKeywords = extractRejectKeywords(ctx.reject)

    for (const slot of requiredSlots) {
        try {
            // NOTE: shop_catalog has NO primary_color column (see 005/006
            // migrations). Selecting it makes PostgREST return an error and
            // silently disables the entire shop supplement. Color is inferred
            // from name/description via inferShopColor() below.
            let q = supabaseClient
                .from('shop_catalog')
                .select('id, brand, name, price, image_url, garment_type, category, description, source')
                .eq('is_active', true)
                .limit(100)
            if (slot === 'shoes') q = q.or('category.eq.shoes,garment_type.eq.shoes')
            else if (slot === 'outerwear') q = q.or('category.eq.outerwear,garment_type.eq.outerwear,name.ilike.%jacket%,name.ilike.%coat%,name.ilike.%blazer%,name.ilike.%cardigan%,name.ilike.%sweater%,name.ilike.%hoodie%,name.ilike.%puffer%,name.ilike.%vest%')
            else if (slot === 'top') q = q.or('category.eq.tops,garment_type.eq.upper_body')
            else if (slot === 'bottom') q = q.or('category.eq.bottoms,garment_type.eq.lower_body')
            const { data, error } = await q
            if (error || !data || data.length === 0) continue

            const scored = data
                .map((row: any) => ({
                    row,
                    score: scoreItemForStyleEdge(
                        { name: row.name, description: row.description, brand: row.brand, color: inferShopColor(row), type: row.garment_type, category: row.category, macroCategory: slot },
                        styleKey,
                    ),
                }))
                .filter(({ row }: { row: any }) => {
                    const blob = `${(row.name || '').toLowerCase()} ${(row.description || '').toLowerCase()} ${row.garment_type || ''} ${row.category || ''}`.toLowerCase()
                    for (const kw of rejectKeywords) { if (blob.includes(kw)) return false }
                    // Feed rows are mistagged at scale (sweaters stored as
                    // bottoms). Resolve the macro from the NAME and skip rows
                    // that are definitively another slot — otherwise a knit
                    // sweater lands in the pants slot. 'other' (ambiguous)
                    // is allowed through to avoid starving thin slots.
                    const resolved = macroCategory(row.category || '', row.garment_type || '', row.name || '')
                    if (resolved !== 'other' && resolved !== slot) return false
                    // Small leather goods carry no outfit slot (no bag/belt
                    // slot exists in 4-slot decks) — a card holder once
                    // rendered as a TOP. Skip them for every clothing slot.
                    const rowType = String(row.garment_type || '').toLowerCase()
                    if (rowType === 'accessory') return false
                    const nameBlob = String(row.name || '').toLowerCase()
                    // NOTE: no 'buckles?' — buckle loafers/monks are shoes.
                    if (/\b(wallets?|card(\s|-)holders?|purses?|handbags?|backpacks?|belts?|scarves?|scarf|hats?|beanies?|jewelry|jewellery|watches?|sunglasses|gloves?|keychains?|cufflinks?|pocket[\s-]?squares?)\b/.test(nameBlob)) return false
                    // Nameless rows render as raw ids ("63092782") on cards.
                    const itemName = String(row.name || '').trim()
                    if (!itemName || /^[0-9\s\-]+$/.test(itemName) || itemName === String(row.id)) return false
                    return true
                })
                .sort((a: any, b: any) => b.score - a.score)

            // Collapse same-product listings BEFORE the per-slot trim so N
            // picks mean N distinct products. Rank ties cluster color
            // variants adjacently (22 penny-loafer UUIDs), which used to eat
            // the whole budget and starve decks to ~7 distinct items.
            // Best-scored id per similarity key wins (rank is score-sorted).
            const seenProducts = new Set<string>()
            let added = 0
            for (const { row } of scored) {
                if (added >= perSlot) break
                if (!row || !row.image_url) continue
                const pkey = simKeyEdge(row.name || row.type || '', slot)
                if (seenProducts.has(pkey)) continue
                seenProducts.add(pkey)
                picks.push({
                    id: `shop_${row.id}`,
                    type: row.garment_type || row.category || slot,
                    category: row.category || slot,
                    macroCategory: slot,
                    name: row.name || row.brand || 'Shop pick',
                    brand: row.brand || '',
                    color: inferShopColor(row),
                    imageUrl: row.image_url,
                    image: row.image_url,
                    style: style || 'Casual',
                    isShopItem: true,
                    price: row.price || undefined,
                    description: row.description || '',
                    recommendation: `Suggested from shop to complete your ${slot}`,
                })
                added++
            }
        } catch (_) {
            // Shop catalog unreachable — skip.
        }
    }
    return picks
}

function requiredShopSupplementSlots(
    items: any[],
    requiredSlots: string[],
    limit: number,
    selectedItemIds: string[] = [],
): { slots: string[]; targetChoicesPerSlot: number } {
    const manualSelection = Array.isArray(selectedItemIds) && selectedItemIds.length > 0
    // Floor of 6/slot: tiny per-slot fetches plus same-product collapsing
    // starved decks to ~7 distinct items (Jaccard 1.0 across taps, measured
    // Oct 2026). Fetching deeper is cheap; rotation needs real choice.
    const targetChoicesPerSlot = manualSelection
        ? 1
        : (limit >= 10 ? 8 : 6)
    const countBySlot = new Map<string, number>()

    for (const item of items) {
        const slot = String(item.macroCategory || '').toLowerCase()
        if (!slot) continue
        countBySlot.set(slot, (countBySlot.get(slot) || 0) + 1)
    }

    return {
        targetChoicesPerSlot,
        slots: requiredSlots.filter((slot) => (countBySlot.get(slot) || 0) < targetChoicesPerSlot),
    }
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
                .select('id, brand, name, price, image_url, garment_type, category, description, source')
                .eq('is_active', true)
                .limit(30)
            if (slot === 'shoes') q = q.or('category.eq.shoes,garment_type.eq.shoes')
            else if (slot === 'outerwear') q = q.or('category.eq.outerwear,garment_type.eq.outerwear,name.ilike.%jacket%,name.ilike.%coat%,name.ilike.%blazer%,name.ilike.%cardigan%,name.ilike.%sweater%,name.ilike.%hoodie%,name.ilike.%puffer%,name.ilike.%vest%')
            else if (slot === 'top') q = q.or('category.eq.tops,garment_type.eq.upper_body')
            else if (slot === 'bottom') q = q.or('category.eq.bottoms,garment_type.eq.lower_body')
            const { data, error } = await q
            if (error || !data || data.length === 0) continue

            // Get style context and reject keywords
            const ctx = getStyleContext(style)
            const rejectKeywords = extractRejectKeywords(ctx.reject)

            // Find the first item that doesn't violate style reject rules
            const allowedItems = data.filter((row: any) => {
                const nameStr = (row.name || '').toLowerCase()
                const descStr = (row.description || '').toLowerCase()
                const blob = `${nameStr} ${descStr} ${row.garment_type || ''} ${row.category || ''}`.toLowerCase()
                for (const kw of rejectKeywords) {
                    if (blob.includes(kw)) return false
                }
                return true
            })

            const row = allowedItems.length > 0 ? allowedItems[0] : data[0]
            if (row && row.image_url) {
                picks.push({
                    id: `shop_${row.id}`,
                    type: row.garment_type || row.category || slot,
                    category: row.category || slot,
                    macroCategory: slot,
                    name: row.name || row.brand || 'Shop pick',
                    brand: row.brand || '',
                    color: inferShopColor(row),
                    imageUrl: row.image_url,
                    image: row.image_url,
                    style: style || 'Casual',
                    isShopItem: true,
                    price: row.price || undefined,
                    description: row.description || '',
                    recommendation: `Suggested from shop to complete your ${slot}`,
                })
            }
        } catch (_) {
            // Shop catalog unreachable — skip.
        }
    }
    return picks
}

async function localFallback(items: any[], style: string, occasion: string, limit: number, layered: boolean, supabaseClient?: any, preferences?: any, wardrobeOnly = false, deckSalt = -1): Promise<any[]> {
    // A non-positive limit means "no outfits" — bail before the last-resort
    // single-outfit block below, which otherwise emits 1 look for limit 0.
    if ((limit ?? 3) <= 0) return []
    const shopOnly = isShopOnlyDeck(style, occasion)
    let filteredItems = [...items]
    if (preferences) {
        if (Array.isArray(preferences.avoidColors) && preferences.avoidColors.length > 0) {
            const avoid = preferences.avoidColors.map((c: string) => c.toLowerCase().trim()).filter(Boolean)
            filteredItems = filteredItems.filter(item => {
                const color = (item.color || item.primary_color || '').toLowerCase()
                const name = (item.name || '').toLowerCase()
                const desc = (item.description || '').toLowerCase()
                return !avoid.some(c => color.includes(c) || name.includes(` ${c} `) || desc.includes(` ${c} `))
            })
        }
    }
    // Shop-only decks ignore wardrobe pieces from the start.
    if (shopOnly) {
        filteredItems = filteredItems.filter((item: any) => item.isShopItem || String(item.id || '').startsWith('shop_'))
    }

    // Style-guard: the AI path rejects outfits containing style-clashing
    // pieces (validateOutfitAgainstStyle) and the client local path excludes
    // them per slot (filterSlotByStyle tiers) — but this fallback used to
    // apply NEITHER, so e.g. a graphic hoodie surfaced in Old Money decks.
    // Floor-protected per slot: if filtering would empty a required slot,
    // restore that slot's unfiltered candidates instead of breaking outfits.
    {
        const styleCtx = getStyleContext(style)
        const rejKws = extractRejectKeywords(styleCtx.reject)
        if (rejKws.length > 0 && filteredItems.length > 0) {
            const blobOf = (it: any) =>
                `${it?.name || ''} ${it?.description || ''} ${it?.type || ''} ${it?.pattern || ''} ${it?.material || ''} ${it?.brand || ''}`.toLowerCase()
            const clean = filteredItems.filter((it: any) => !rejKws.some((kw) => blobOf(it).includes(kw)))
            const slotOf = (it: any) => String(it.macroCategory || '').toLowerCase()
            const needSlots = ['top', 'bottom', 'shoes']
            if (layered) needSlots.push('outerwear')
            const guarded = [...clean]
            for (const s of needSlots) {
                if (!guarded.some((it) => slotOf(it) === s)) {
                    for (const it of filteredItems) {
                        if (slotOf(it) === s && !guarded.includes(it)) guarded.push(it)
                    }
                }
            }
            if (guarded.length > 0 && guarded.length !== filteredItems.length) {
                console.log(`[style-guard] ${style}: ${filteredItems.length} -> ${guarded.length} items`)
            }
            if (guarded.length > 0) filteredItems = guarded
        }
    }

    const sortByFavorite = (pool: any[]) => {
        if (!preferences || !Array.isArray(preferences.favoriteColors) || preferences.favoriteColors.length === 0) {
            return pool
        }
        const favorites = preferences.favoriteColors.map((c: string) => c.toLowerCase().trim()).filter(Boolean)
        return [...pool].sort((a, b) => {
            const colorA = (a.color || a.primary_color || '').toLowerCase()
            const colorB = (b.color || b.primary_color || '').toLowerCase()
            const matchA = favorites.some(c => colorA.includes(c)) ? 1 : 0
            const matchB = favorites.some(c => colorB.includes(c)) ? 1 : 0
            return matchB - matchA // Favorite colors first
        })
    }

    // Shuffle each category to ensure variety when we have limited items
    let baseTops = shuffleArray(filteredItems.filter(i => i.macroCategory === 'top'))
    let outerwear = shuffleArray(filteredItems.filter(i => i.macroCategory === 'outerwear'))
    let legacyTops = shuffleArray(filteredItems.filter(i => ['top','outerwear'].includes(i.macroCategory)))
    let bottoms = shuffleArray(filteredItems.filter(i => i.macroCategory === 'bottom'))
    const nonShortsBottoms = shuffleArray(bottoms.filter(b => !isShortsItem(b)))
    const casualOuterwear = shuffleArray(outerwear.filter(o => !isFormalLayerItem(o)))
    let shoes = shuffleArray(filteredItems.filter(i => i.macroCategory === 'shoes'))

    if (preferences) {
        baseTops = sortByFavorite(baseTops)
        outerwear = sortByFavorite(outerwear)
        legacyTops = sortByFavorite(legacyTops)
        bottoms = sortByFavorite(bottoms)
        shoes = sortByFavorite(shoes)
    }

    // Wardrobe-first, but daily 15-card decks need enough choices per slot to
    // vary. A sparse wardrobe with one item per category should be supplemented
    // from the shop catalog instead of repeating the same outfit all day.
    // Shop-only decks (Team Collaboration) always fetch every required slot.
    const requiredSlots: string[] = ['top', 'bottom', 'shoes']
    if (layered) requiredSlots.push('outerwear')
    const shopSupplement = shopOnly
        ? { slots: requiredSlots, targetChoicesPerSlot: limit >= 10 ? 16 : 8 }
        : requiredShopSupplementSlots(filteredItems, requiredSlots, limit)
    const { slots: fillSlots, targetChoicesPerSlot } = shopSupplement
    let shopFills: any[] = []
    // wardrobeOnly: Supabase clothing only — never supplement from shop.
    if (wardrobeOnly) {
        console.log(`[wardrobeOnly] localFallback skipping shop supplement — ${filteredItems.length} wardrobe item(s) only`)
    } else if (supabaseClient) {
        try {
            shopFills = await fetchShopItemsForAllSlotsEdge(supabaseClient, fillSlots, style, targetChoicesPerSlot)
            // Shop candidates must respect color/pattern prefs too — the
            // wardrobe-side filter above ran before they were appended.
            shopFills = filterShopByPrefs(shopFills, preferences)
            if (shopFills.length > 0) {
                console.log(`[shop-fill] Local fallback added ${shopFills.length} shop item(s) for sparse slots: ${fillSlots.join(', ')} (target=${targetChoicesPerSlot})`)
            }
        } catch (_) { /* skip */ }
    }
    const allItems = [...filteredItems, ...shopFills]
    // Shop-only: build purely from shop candidates.
    const shopOnlyPool = shopOnly ? [...shopFills, ...filteredItems.filter((i: any) => i.isShopItem || String(i.id || '').startsWith('shop_'))] : null
    const poolItems = shopOnly && shopOnlyPool && shopOnlyPool.length > 0 ? shopOnlyPool : allItems
    // Shop-only decks ignore wardrobe pieces entirely — unless the shop pool
    // is empty, in which case owned items fill in rather than emitting 0
    // looks (a business_casual request must not hard-fail when offline).
    const deckShopOnly = shopOnly && poolItems.some((i: any) => i.isShopItem || String(i.id || '').startsWith('shop_'))
    // Exact item contract shared by buildParts + keepParts below (both used
    // to reference this from the wrong scope — keepParts threw
    // `targetItems is not defined` on diverse pools).
    const targetItems = layered ? 4 : 3
    // Human-readable style name for user-facing copy ("old_money" -> "Old Money").
    // Unknown/hostile ids MUST fall back to Casual — never echo raw input
    // (mirrors the client adversarial guarantee; previously a style like
    // "goth_ninja_../../etc" leaked verbatim into the description).
    const styleLabel = (() => {
        const key = String(style || 'casual').toLowerCase()
        const known: Record<string, string> = {
            old_money: 'Old Money', semi_classic: 'Semi-Classic', minimalist: 'Minimalist',
            casual: 'Casual', classic: 'Classic', business_casual: 'Modern Professional',
        }
        return known[key] || 'Casual'
    })()
    const stylePhrase = `${/^[aeiou]/i.test(styleLabel) ? 'An' : 'A'} ${styleLabel}`
    // Re-derive category buckets from the merged pool.
    const allBaseTops = poolItems.filter((i: any) => i.macroCategory === 'top')
    const allOuterwear = poolItems.filter((i: any) => i.macroCategory === 'outerwear')
    const allLegacyTops = poolItems.filter((i: any) => ['top','outerwear'].includes(i.macroCategory))
    const allBottoms = poolItems.filter((i: any) => i.macroCategory === 'bottom')
    const allNonShortsBottoms = allBottoms.filter((b: any) => !isShortsItem(b))
    const allCasualOuterwear = allOuterwear.filter((o: any) => !isFormalLayerItem(o))
    const allShoes = poolItems.filter((i: any) => i.macroCategory === 'shoes')
    const ownedItems = deckShopOnly ? [] : allItems.filter((i: any) => !i.isShopItem && !String(i.id || '').startsWith('shop_'))
    const shopItems = (shopOnly && shopOnlyPool && shopOnlyPool.length > 0 ? shopOnlyPool : allItems).filter((i: any) => i.isShopItem || String(i.id || '').startsWith('shop_'))
    const isOwnedPart = (p: any) => p && !p.isShopItem && !String(p.id || '').startsWith('shop_') && p.id !== PLACEHOLDER_SHOES.id
    const isShopPart = (p: any) => p && (p.isShopItem || String(p.id || '').startsWith('shop_'))
    const replacePart = (parts: any[], replacement: any, protectId?: string) => {
        const slot = String(replacement.macroCategory || '').toLowerCase()
        const sameSlotIndex = parts.findIndex((p: any) => String(p.macroCategory || '').toLowerCase() === slot && p.id !== replacement.id && p.id !== protectId)
        if (sameSlotIndex >= 0) {
            parts[sameSlotIndex] = replacement
            return
        }
        const shopIndex = parts.findIndex((p: any) => isShopPart(p) && p.id !== protectId)
        if (shopIndex >= 0) {
            parts[shopIndex] = replacement
            return
        }
        const replaceableIndex = parts.findIndex((p: any) => p.id !== protectId)
        if (replaceableIndex >= 0) parts[replaceableIndex] = replacement
    }

    // Formal decks never contain shorts — unconditionally. A suit + shorts
    // card must be impossible by construction, not just when a formal layer
    // happens to be in the pool.
    const isFormalStyle = ['old_money', 'classic', 'business_casual'].includes(style.toLowerCase())
    if (isFormalStyle && allNonShortsBottoms.length > 0) {
        // Replace allBottoms with only non-shorts options
        allBottoms.length = 0
        allBottoms.push(...allNonShortsBottoms)
    }

    // Spread same-looking listings apart so the rotation window covers
    // varied products instead of adjacent score-tied lookalikes.
    // NOTE: pools of <= 2 need no spreading — and MUST early-return:
    // spreadBySimEdge returns the SAME array reference for tiny pools, so
    // `pool.length = 0` would also clear `spread` and wipe the bucket.
    // That aliasing bug emptied every thin slot (1-2 candidates), forcing
    // placeholders, pads, and duplicate-id outfits downstream.
    const spreadPoolsEdge = (): void => {
        const spreadInPlace = (pool: any[], slot: string): void => {
            if (pool.length <= 2) return
            const spread = spreadBySimEdge(pool, (p: any) => simKeyEdge(p.name || p.type || '', slot))
            pool.length = 0
            pool.push(...spread)
        }
        spreadInPlace(allBaseTops, 'top')
        spreadInPlace(allOuterwear, 'outerwear')
        spreadInPlace(allBottoms, 'bottom')
        spreadInPlace(allShoes, 'shoes')
        spreadInPlace(allLegacyTops, 'top')
    }
    spreadPoolsEdge()

    const outfits: any[] = []
    const slotSize = (pool: any[]) => Math.max(pool.length, 1)
    const seenSigs = new Set<string>()
    // Diverse round-robin — every slot rotates with i so a 15-card
    // Team Collaboration deck doesn't freeze bottom/shoes on one item.
    // Steps are primes coprime to 3/4/5 so retries perturb every slot.
    const DIVERSE_STEPS = [7, 11, 13, 17]
    // Per-tap salt breaks the style+occasion determinism: without it every
    // tap rebuilds the identical deck from the same pool order.
    const salt = deckSalt < 0 ? Math.floor(Math.random() * 32) : (Number(deckSalt) || 0)
    const deckSeed = (deckSeedEdge(style, occasion) + salt) % 32
    const deckReverse = salt % 2 === 1 ? !deckReverseEdge(style, occasion) : deckReverseEdge(style, occasion)
    const diverseIdx = (outfitIdx: number, slotOffset: number, attempt: number, step: number, len: number): number => {
        if (len <= 0) return 0
        const base = (outfitIdx + deckSeed + slotOffset + attempt * step) % len
        return deckReverse ? (len - 1 - base) : base
    }

    // Build one candidate deck entry, or null when the pool cannot complete
    // an outfit for this (i, attempt). `attempt` perturbs every slot rotation
    // so a rejected combination is never rebuilt identically on retry.
    const buildParts = (i: number, attemptForPick: number): any[] | null => {
        const parts: any[] = []

        // Diverse round-robin: each slot advances with i plus a small
        // per-slot offset. The old mixed-radix floor(i / hugeDivisor) stayed
        // 0 for all i < poolSize, leaving 70%+ of each outfit identical.
        const topIndex = diverseIdx(i, 0, attemptForPick, DIVERSE_STEPS[0], slotSize(allOuterwear))
        const baseIndex = diverseIdx(i, 3, attemptForPick, DIVERSE_STEPS[1], slotSize(allBaseTops))
        const bottomIndex = diverseIdx(i, 6, attemptForPick, DIVERSE_STEPS[2], slotSize(allBottoms))
        const shoeIndex = diverseIdx(i, 9, attemptForPick, DIVERSE_STEPS[3], slotSize(allShoes))
        const legacyTopIndex = diverseIdx(i, 0, attemptForPick, DIVERSE_STEPS[0], slotSize(allLegacyTops))
        const legacyBottomIndex = diverseIdx(i, 3, attemptForPick, DIVERSE_STEPS[1], slotSize(allBottoms))

        // Pre-pick the bottom so we can decide whether a formal layer is safe.
        const bottom = allBottoms[(layered ? bottomIndex : legacyBottomIndex) % slotSize(allBottoms)] || allBottoms[0]
        const bottomIsShorts = !!bottom && isShortsItem(bottom)
        
        if (layered) {
            // If the bottom is shorts, never pair with a formal outerwear piece;
            // prefer casual outerwear, else fall back to a safe non-shorts bottom.
            let outer = allOuterwear[topIndex % slotSize(allOuterwear)]
            if (bottomIsShorts && outer && isFormalLayerItem(outer)) {
                outer = allCasualOuterwear[topIndex % slotSize(allCasualOuterwear)] || undefined
            }
            // Use outerwear if available, otherwise fallback to any top item
            const mainTop = outer || allLegacyTops[legacyTopIndex % slotSize(allLegacyTops)]
            if (mainTop) parts.push({ ...mainTop, recommendation: deckShopOnly ? 'Suggested from shop' : 'Main top / outerwear layer' })

            // For second top, use a different offset to ensure variety.
            // Never push the same item twice: when base tops are starved,
            // `outer` would otherwise land in both slots (mirrors the client
            // `base.id !== mainTop?.id` guard). The pad loop below fills the
            // gap instead of emitting a duplicate-id outfit.
            const base = allBaseTops[baseIndex % slotSize(allBaseTops)]
            const baseTopItem = base || outer || allLegacyTops[(baseIndex + 1) % slotSize(allLegacyTops)]
            if (baseTopItem && baseTopItem.id !== mainTop?.id) {
                parts.push({ ...baseTopItem, recommendation: deckShopOnly ? 'Suggested from shop' : 'Base top worn underneath' })
            }
        } else {
            const top = allLegacyTops[legacyTopIndex % slotSize(allLegacyTops)]
            if (top) parts.push({ ...top, recommendation: deckShopOnly ? 'Suggested from shop' : 'Key piece' })
        }
        
        // Always include bottom and shoes
        let finalBottom = bottom
        if (layered && bottomIsShorts) {
            const mainTopItem = parts[0]
            if (mainTopItem && isFormalLayerItem(mainTopItem) && allNonShortsBottoms.length > 0) {
                finalBottom = allNonShortsBottoms[bottomIndex % allNonShortsBottoms.length]
            }
        }
        const nonLayeredShoeIndex = diverseIdx(i, 6, attemptForPick, DIVERSE_STEPS[2], slotSize(allShoes))
        const shoe = allShoes[(layered ? shoeIndex : nonLayeredShoeIndex) % slotSize(allShoes)] || allShoes[0]
        if (finalBottom) parts.push({ ...finalBottom, recommendation: deckShopOnly ? 'Suggested from shop' : 'Pairs well' })
        if (shoe) parts.push({ ...shoe, recommendation: deckShopOnly ? 'Suggested from shop' : 'Completes the look' })
        else if (!deckShopOnly) parts.push({ ...PLACEHOLDER_SHOES })
        
        // Exact item contract: 3 for non-layered, 4 for layered.
        // (targetItems is hoisted to localFallback scope — shared with keepParts.)
        if (layered && parts.length < targetItems && poolItems.length > 0) {
            // Only pad layered outfits — non-layered must stay at exactly 3.
            let fillIndex = 0
            while (parts.length < targetItems && fillIndex < poolItems.length) {
                const fillItem = poolItems[(fillIndex + i) % poolItems.length]
                const alreadyAdded = parts.some(p => p.id === fillItem.id)
                if (!alreadyAdded) {
                    parts.push({ ...fillItem, recommendation: deckShopOnly ? 'Suggested from shop' : 'Complementary piece' })
                }
                fillIndex++
            }
        }
        // For non-layered, trim any accidental extras down to 3.
        if (!layered && parts.length > targetItems) parts.length = targetItems
        // Identity repair: one mislabeled catalog row can match several slot
        // queries (e.g. a sweater tagged `bottoms`), so two slots may hold
        // the SAME id. The client builder can't do this (its pick() tracks
        // used ids); repair here — swap the later occurrence for an unused
        // same-slot candidate, else drop it and re-pad below.
        {
            const seenIds = new Set<string>()
            for (let pi = 0; pi < parts.length; pi++) {
                const pid = String(parts[pi].id)
                if (!seenIds.has(pid)) { seenIds.add(pid); continue }
                const slot = String(parts[pi].macroCategory || '').toLowerCase()
                const slotPool = slot === 'top' ? allBaseTops
                    : slot === 'outerwear' ? allOuterwear
                    : slot === 'bottom' ? allBottoms
                    : slot === 'shoes' ? allShoes : poolItems
                const repl = slotPool.find((c: any) => !seenIds.has(String(c.id)))
                    || poolItems.find((c: any) => !seenIds.has(String(c.id)))
                if (repl) {
                    parts[pi] = { ...repl, recommendation: parts[pi].recommendation }
                    seenIds.add(String(repl.id))
                } else {
                    parts.splice(pi, 1)
                    pi--
                }
            }
            // Re-pad (either mode) after a drop so the outfit still meets
            // the item contract with distinct ids.
            if (parts.length > 0 && parts.length < targetItems && poolItems.length > 0) {
                for (const fill of poolItems) {
                    if (parts.length >= targetItems) break
                    if (!fill || parts.some((p: any) => p.id === fill.id)) continue
                    if (deckShopOnly && !isShopPart(fill)) continue
                    parts.push({ ...fill, recommendation: deckShopOnly ? 'Suggested from shop' : 'Complementary piece' })
                }
            }
        }
        if (!parts.length) return null
        // Shop-only: drop any wardrobe piece and incomplete looks.
        if (deckShopOnly) {
            const filtered = parts.filter((p: any) => isShopPart(p))
            if (filtered.length < targetItems) return null
            // Replace in place so slot coverage stays intact.
            parts.length = 0
            parts.push(...filtered.slice(0, targetItems))
        }

        let protectedOwnedId: string | undefined
        if (!deckShopOnly && ownedItems.length > 0 && !parts.some(isOwnedPart)) {
            const owned = { ...ownedItems[i % ownedItems.length], recommendation: 'From your wardrobe' }
            protectedOwnedId = owned.id
            replacePart(parts, owned)
        } else if (!deckShopOnly) {
            protectedOwnedId = parts.find(isOwnedPart)?.id
        }
        if (shopItems.length > 0 && !parts.some(isShopPart)) {
            replacePart(parts, { ...shopItems[i % shopItems.length], recommendation: 'Suggested from shop' }, protectedOwnedId)
        }
        if (deckShopOnly && !parts.every(isShopPart)) return null
        // Hard rule: suits never pair with shorts; formal decks never shorts.
        {
            const hasShorts = parts.some((p: any) => isShortsItem(p))
            const hasFormal = parts.some((p: any) => isFormalLayerItem(p))
            const sKey = (style || '').toLowerCase()
            if (hasShorts && (hasFormal || ['old_money', 'classic', 'business_casual'].includes(sKey))) return null
        }
        // Hard rule: work never takes beach/casual footwear (boat shoes,
        // espadrilles, mules, drivers, sandals). Mirrors the client formal
        // shoe veto so rotation skips them instead of relying on post-hoc
        // filtering. The last-resort single is exempt (something > nothing).
        if (isWorkishOccasion(occasion)) {
            const hasBeachShoe = parts.some((p: any) => {
                if (String(p.macroCategory || '').toLowerCase() !== 'shoes') return false
                const b = `${p.name || ''} ${p.type || ''}`.toLowerCase()
                return /\b(boat|espadrilles?|mules?|backless|slip[\s-]?ons?|moccasins?|drivers?|driving|gommino|sandals?|slides?)\b/.test(b)
            })
            if (hasBeachShoe) return null
        }
        const budgetCap = preferences?.maxShopBudget
        if (budgetCap && typeof budgetCap.amount === 'number') {
            const fitted = fitOutfitToBudget(parts, poolItems, budgetCap)
            parts.length = 0
            parts.push(...fitted)
            if (outfitShopSpend(parts) > budgetCap.amount + 0.009) return null
        }
        return parts
    }

    const keepParts = (parts: any[], relaxed: boolean): boolean => {
        const sig = parts.map((p: any) => String(p.id)).sort().join('|')
        if (!sig || seenSigs.has(sig)) return false
        if (!relaxed) {
            // Skip near-duplicates sharing all-but-one piece while rotation
            // tries remain.
            const partIds = parts.map((p: any) => String(p.id))
            let maxShared = 0
            for (const built of outfits) {
                const builtSet = new Set((built.items || []).map((p: any) => String(p.id)))
                let shared = 0
                for (const id of partIds) if (builtSet.has(id)) shared++
                if (shared > maxShared) maxShared = shared
            }
            if (outfits.length > 0 && maxShared >= targetItems - 1) return false
        }
        seenSigs.add(sig)
        // Ensure each outfit has a unique ID
        outfits.push({
            id: `local_${Date.now()}_${Math.floor(Math.random() * 100000)}`,
            style: style || 'Casual',
            occasion: occasion || 'Everyday',
            description: deckShopOnly ? `${stylePhrase} look curated from the shop.` : `${stylePhrase} look built around your wardrobe.`,
            confidence: 0.75 + (Math.random() * 0.1), // Slight confidence variation
            items: parts,
            stylingTips: pickStylingTips(layered),
        })
        return true
    }

    // Strict pass: decks must differ by >= 2 items. Rejections perturb the
    // attempt so retries explore new slot combinations.
    let edgeAttempt = 0
    for (let i = 0; outfits.length < limit && i < limit * 12; i++) {
        const candidate = buildParts(i, edgeAttempt)
        if (!candidate) { edgeAttempt++; continue }
        if (keepParts(candidate, false)) edgeAttempt = 0
        else edgeAttempt++
    }

    // Relaxed pass: thin pools (a single shoes/bottom/outerwear choice) can
    // NEVER satisfy the strict rule — every second look shares 3 of 4 by
    // construction. Fill the remaining slots with exact-dupe-only rejection
    // so the user gets 3 looks instead of 1.
    if (outfits.length < limit) {
        for (let i = 0; outfits.length < limit && i < limit * 12; i++) {
            const candidate = buildParts(i + limit * 12, edgeAttempt)
            if (!candidate) { edgeAttempt++; continue }
            if (keepParts(candidate, true)) edgeAttempt = 0
            else edgeAttempt++
        }
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
            if (shoe && shoe.id !== finalBottom?.id) parts.push({ ...shoe, recommendation: deckShopOnly ? 'Suggested from shop' : 'Completes the look' })
            else if (!deckShopOnly) parts.push({ ...PLACEHOLDER_SHOES })
            // Pad a starved layered look up to the 4-item contract with any
            // unused pool items instead of emitting an incomplete outfit.
            if (parts.length > 0 && parts.length < 4) {
                for (const fill of poolItems) {
                    if (parts.length >= 4) break
                    if (!fill || parts.some((p: any) => p.id === fill.id)) continue
                    if (deckShopOnly && !isShopPart(fill)) continue
                    parts.push({ ...fill, recommendation: 'Complementary piece' })
                }
            }
        } else {
            // For non-layered: exactly 3 items (top, bottom, shoes)
            const top = allLegacyTops[0] || allItems[0]
            const bottom = allBottoms[0] || allItems[1] || allItems[0]
            const shoe = allShoes[0]
            if (top) parts.push({ ...top, recommendation: 'Key piece' })
            if (bottom && bottom.id !== top?.id) parts.push({ ...bottom, recommendation: 'Pairs well' })
            if (shoe && shoe.id !== bottom?.id) parts.push({ ...shoe, recommendation: deckShopOnly ? 'Suggested from shop' : 'Completes the look' })
            else if (!deckShopOnly) parts.push({ ...PLACEHOLDER_SHOES })
            if (parts.length > 3) parts.length = 3
        }
        if (parts.length > 0) {
            outfits.push({
                id: `local_0_${Date.now()}`,
                style: style || 'Casual',
                occasion: occasion || 'Everyday',
                description: `${stylePhrase} look built around your wardrobe.`,
                confidence: 0.75,
                items: parts,
                stylingTips: pickStylingTips(layered),
            })
        }
    }
    // Contract sanitizer: legacy client payloads may carry items without any
    // image field — default it so every emitted item satisfies the client
    // GeneratedOutfitItem shape (the UI tolerates '' but not undefined).
    for (const o of outfits) {
        for (const it of (o.items || [])) {
            if (typeof it.imageUrl !== 'string') it.imageUrl = (it as any).image_url || (it as any).image || ''
            if (typeof it.image !== 'string') it.image = it.imageUrl
            if (!it.macroCategory) it.macroCategory = 'other'
        }
    }
    if (outfits.length) return outfits
    // Shop-only decks must never fall back to wardrobe items — every outfit is
    // 100% shopping, and every combination is unique. (deckShopOnly, not
    // shopOnly: an empty shop pool already fell back to owned above.)
    if (deckShopOnly) return []
    return [{
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
// NOTE: meta/llama-3.1-8b-instruct was retired (410 Gone, Oct 2026), and
// most text-only models are account-gated (404). llama-3.2-11b-vision is
// verified working for BOTH text and vision on this account (Oct 2026).
const DEFAULT_NVIDIA_MODEL = 'meta/llama-3.2-11b-vision-instruct'

// ── Shuffle array for outfit variety ─────────────────────────────────────
function shuffleArray<T>(array: T[]): T[] {
    const arr = [...array]
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1))
        ;[arr[i], arr[j]] = [arr[j], arr[i]]
    }
    return arr
}
function resolveNvidiaModel(input: string | null | undefined): string {
    const model = (input || '').trim()
    return model.length > 0 ? model : DEFAULT_NVIDIA_MODEL
}
export interface ProviderAttempt {
    provider: string;
    model: string;
    status: string;
}

async function callNvidia(key: string, model: string, sys: string, usr: string, retries = 0, trace?: ProviderAttempt[]): Promise<any[]|null> {
    for (let attempt = 0; attempt <= retries; attempt++) {
        const controller = new AbortController()
        // NOTE (Oct 2026): measured ~12 tok/s on llama-3.2-11b-vision — a full
        // outfit JSON needs ~50s, unusable for interactive taps (client caps at
        // 25s). Bound each attempt so the edge always falls through to the
        // validated local builder quickly instead of burning a minute.
        const timer = setTimeout(() => controller.abort(), 20_000)
        try {
            console.log(`[NVIDIA] attempt ${attempt + 1}, model: ${model}`)
            const r = await fetch('https://integrate.api.nvidia.com/v1/chat/completions', {
                method:'POST', headers:{'Content-Type':'application/json','Authorization':`Bearer ${key}`},
                body: JSON.stringify({ model, messages:[{role:'system',content:sys},{role:'user',content:usr}], temperature:0.95, max_tokens:2000, frequency_penalty:0.6, presence_penalty:0.4, response_format:{type:'json_object'} }),
                signal: controller.signal,
            })
            clearTimeout(timer)
            if (r.status === 429 || r.status >= 500) {
                console.warn(`[NVIDIA] ${r.status}, retrying in 2s...`)
                trace?.push({ provider: 'nvidia', model, status: String(r.status) })
                if (attempt < retries) { await new Promise(r => setTimeout(r, 2000)); continue }
                return null
            }
            if (!r.ok) { console.error('[NVIDIA] error:', r.status); trace?.push({ provider: 'nvidia', model, status: String(r.status) }); return null }
            const d = await r.json()
            const content = d.choices?.[0]?.message?.content || '{}'
            console.log(`[NVIDIA] response length: ${content.length} chars`)
            const outfits = safeParseOutfits(content)
            if (outfits && outfits.length > 0) {
                console.log(`[NVIDIA] parsed ${outfits.length} outfits`)
                trace?.push({ provider: 'nvidia', model, status: 'ok' })
                return outfits
            }
            console.warn('[NVIDIA] returned unparseable content')
            trace?.push({ provider: 'nvidia', model, status: 'unparseable' })
            return null
        } catch(e) {
            clearTimeout(timer)
            console.error('[NVIDIA] call error:', e)
            trace?.push({ provider: 'nvidia', model, status: 'timeout' })
            if (attempt < retries) { await new Promise(r => setTimeout(r, 1500)); continue }
            return null
        }
    }
    return null
}

// ── Gemini call with model fallback chain ───────────────────────────────
// Tries each model in order (first = configured/preferred, last = known
// good). A missing model id fails fast (404, no retry) so misconfiguration
// costs one cheap roundtrip, not a timeout.
async function callGemini(key: string, model: string, sys: string, usr: string, retries = 0, trace?: ProviderAttempt[]): Promise<any[]|null> {
    for (let attempt = 0; attempt <= retries; attempt++) {
        const controller = new AbortController()
        const timer = setTimeout(() => controller.abort(), 25_000)
        try {
            console.log(`[Gemini] Attempt ${attempt + 1}, model: ${model}`)
            const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`, {
                method:'POST', headers:{'Content-Type':'application/json'},
                body: JSON.stringify({ contents:[{parts:[{text:`${sys}\n\n${usr}`}]}], generationConfig:{responseMimeType:'application/json',temperature:0.95,maxOutputTokens:3000} }),
                signal: controller.signal,
            })
            clearTimeout(timer)
            if (r.status === 429 || r.status >= 500) {
                console.warn(`[Gemini] ${r.status}, retrying...`)
                trace?.push({ provider: 'gemini', model, status: String(r.status) })
                if (attempt < retries) { await new Promise(r => setTimeout(r, 2000)); continue }
                return null
            }
            if (!r.ok) { console.error('[Gemini] error:', r.status); trace?.push({ provider: 'gemini', model, status: String(r.status) }); return null }
            const d = await r.json()
            const content = d.candidates?.[0]?.content?.parts?.[0]?.text || '{}'
            console.log(`[Gemini] Response length: ${content.length} chars`)
            const outfits = safeParseOutfits(content)
            if (outfits && outfits.length > 0) {
                console.log(`[Gemini] Parsed ${outfits.length} outfits`)
                trace?.push({ provider: 'gemini', model, status: 'ok' })
                return outfits
            }
            console.warn('[Gemini] Unparseable content')
            trace?.push({ provider: 'gemini', model, status: 'unparseable' })
            return null
        } catch(e) {
            clearTimeout(timer)
            console.error('[Gemini] call error:', e)
            trace?.push({ provider: 'gemini', model, status: 'timeout' })
            if (attempt < retries) { await new Promise(r => setTimeout(r, 1500)); continue }
            return null
        }
    }
    return null
}

// ── Anchor-item enforcement ───────────────────────────────────────────────
// After the AI (or local fallback) selects outfit items, lock the anchor item
// into its matching slot in every outfit. If no slot matches the anchor's
// macroCategory, the anchor is prepended to the items array.
function enforceAnchorInOutfits(outfits: any[], anchorDbItem: any): any[] {
    const anchorMacro = macroCategory(anchorDbItem.category || '', anchorDbItem.type || '')
    const anchorOutfitItem = {
        id: anchorDbItem.id,
        type: anchorDbItem.type || anchorDbItem.category || 'clothing',
        macroCategory: anchorMacro,
        color: Array.isArray(anchorDbItem.color)
            ? anchorDbItem.color.join(', ')
            : (anchorDbItem.primary_color || anchorDbItem.color || 'neutral'),
        name: anchorDbItem.name || anchorDbItem.type || anchorDbItem.category || 'Item',
        imageUrl: anchorDbItem.image_url || anchorDbItem.imageUrl || '',
        image: anchorDbItem.image_url || anchorDbItem.imageUrl || '',
        brand: anchorDbItem.brand || '',
        recommendation: 'Anchor item — user selected',
        isShopItem: false,
    }

    return outfits.map((outfit: any) => {
        const items: any[] = Array.isArray(outfit.items) ? [...outfit.items] : []
        const slotIdx = items.findIndex(
            (i: any) => macroCategory(i.category || i.macroCategory || '', i.type || '') === anchorMacro
                || (i.macroCategory || '').toLowerCase() === anchorMacro
        )
        if (slotIdx >= 0) {
            items[slotIdx] = anchorOutfitItem
        } else {
            items.unshift(anchorOutfitItem)
        }
        return { ...outfit, items }
    })
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
        anchorItemId,
        // Wardrobe-only (Supabase clothing) requests: build ONLY from
        // clothing_items + provided wardrobe items — no shop_catalog
        // supplement, no static fill.
        wardrobeOnly = false,
        // Per-tap salt from the client (generation nonce). The local-fallback
        // deck rotation is otherwise a pure function of style+occasion, so
        // repeat taps rebuild byte-identical decks. Absent (other callers) →
        // random salt: variety by default, determinism only when requested.
        deckSalt = -1,
    } = body

    const parsedBudget = parseBudgetFromText(`${prompt} ${occasion}`)
    const preferences = parsedBudget
        ? { ...(body.preferences || {}), maxShopBudget: parsedBudget }
        : body.preferences

    try {
        // ── 1. Authenticate user via JWT ──────────────────────────────────
        const authHeader = req.headers.get('Authorization') || ''
        let dbClothingItems: any[] = []
        let requestUserId: string | null = null

        if (authHeader && !useProvidedWardrobeOnly) {
            const userSupabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
                global: { headers: { Authorization: authHeader } }
            })
            const { data: { user } } = await userSupabase.auth.getUser()

            if (user) {
                requestUserId = user.id
                // ── 2. Fetch clothing items from DB ───────────────────────
                // NOTE: live schema uses `occasions` (JSONB) + `seasons`
                // (JSONB), not singular `occasion`. Selecting a missing column
                // makes PostgREST fail the whole query → empty wardrobe.
                let q = userSupabase
                    .from('clothing_items')
                    .select('id, type, category, color, style, brand, season, seasons, occasions, pattern, material, image_url, description, primary_color, sub_category, wear_count, name')
                    .order('wear_count', { ascending: false })
                    .limit(30)

                if (selectedItemIds && selectedItemIds.length > 0) {
                    q = q.in('id', selectedItemIds)
                }

                const { data: rows, error: wardrobeError } = await q
                if (wardrobeError) {
                    console.error('[wardrobe] clothing_items fetch failed:', wardrobeError.message)
                }
                if (rows && rows.length > 0) {
                    dbClothingItems = rows.map((row: any) => {
                        const occasionsRaw = row.occasions ?? (row as any).occasion
                        const seasonsRaw = row.seasons ?? row.season
                        const fmtList = (v: any, fallback: string): string => {
                            if (Array.isArray(v)) return v.join(', ')
                            if (v && typeof v === 'object') return Object.keys(v).join(', ')
                            if (typeof v === 'string' && v.length > 0) return v
                            return fallback
                        }
                        return {
                            id: row.id,
                            type: row.type || row.sub_category || row.category || 'clothing',
                            category: row.category || 'Other',
                            macroCategory: macroCategory(row.category || '', row.type || row.sub_category || ''),
                            color: Array.isArray(row.color) ? row.color.join(', ') : (row.primary_color || row.color || 'neutral'),
                            style: row.style || 'Casual',
                            brand: row.brand || '',
                            name: row.name || row.type || row.sub_category || row.category || 'Item',
                            material: row.material || '',
                            pattern: row.pattern || '',
                            season: fmtList(seasonsRaw, 'All Seasons'),
                            occasion: fmtList(occasionsRaw, ''),
                            description: row.description || '',
                            imageUrl: row.image_url || '',
                            wearCount: row.wear_count || 0,
                        }
                    })
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

        // Filter out avoided colors if requested
        if (preferences && Array.isArray(preferences.avoidColors) && preferences.avoidColors.length > 0) {
            const avoid = preferences.avoidColors.map((c: string) => c.toLowerCase().trim()).filter(Boolean)
            const beforeAvoid = wardrobeItems.length
            wardrobeItems = wardrobeItems.filter((item: any) => {
                const color = (item.color || item.primary_color || '').toLowerCase()
                const name = (item.name || '').toLowerCase()
                const desc = (item.description || '').toLowerCase()
                return !avoid.some(c => color.includes(c) || name.includes(` ${c} `) || desc.includes(` ${c} `))
            })
            console.log(`[preferences] Filtered out avoided colors: ${beforeAvoid} -> ${wardrobeItems.length} items`)
        }

        // ── 3c. Supplement sparse wardrobe slots from the shop ─────────────
        // Wardrobe pieces remain in the pool, but a 15-card daily deck needs
        // several choices per slot. If the user owns only a few garments, add
        // style-ranked shop catalog items so the combinations can actually vary.
        // Shop-only decks (Team Collaboration) fetch every slot from the shop.
        // wardrobeOnly requests skip this entirely — Supabase clothing only.
        if (!wardrobeOnly) {
            const requiredSlots = ['top', 'bottom', 'shoes']
            if (needsLayering(stylePreferences, weather, prompt)) requiredSlots.push('outerwear')
            const shopOnlyRequest = isShopOnlyDeck(stylePreferences, occasion)
            const { slots: missingSlots, targetChoicesPerSlot } = shopOnlyRequest
                ? { slots: requiredSlots, targetChoicesPerSlot: limit >= 10 ? 16 : 8 }
                : requiredShopSupplementSlots(
                    wardrobeItems,
                    requiredSlots,
                    limit,
                    selectedItemIds,
                )
            try {
                const svcClient = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
                const fetchedShopItems = await fetchShopItemsForAllSlotsEdge(svcClient, missingSlots, stylePreferences, targetChoicesPerSlot)
                // Shop candidates must respect color/pattern prefs too — the
                // wardrobe-side filter above ran before they were appended.
                const shopItems = filterShopByPrefs(fetchedShopItems, preferences)
                if (shopItems.length > 0) {
                    console.log(`[shop-fill] Added ${shopItems.length} shop item(s) for sparse slots: ${missingSlots.join(', ')} (target=${targetChoicesPerSlot})`)
                    wardrobeItems = [...wardrobeItems, ...shopItems]
                }
            } catch (_) {
                console.warn('[shop-fill] Failed to fetch shop catalog items')
            }
        } else {
            console.log(`[wardrobeOnly] Skipping shop supplement — ${wardrobeItems.length} Supabase wardrobe item(s) only`)
        }

        if (parsedBudget) {
            const beforeBudget = wardrobeItems.length
            wardrobeItems = wardrobeItems.filter((item: any) => {
                if (!isShopPricedItem(item)) return true
                const p = shopPriceOf(item)
                if (p <= 0) return true
                return p <= parsedBudget.amount
            })
            console.log(`[budget] shop items over ${parsedBudget.amount} ${parsedBudget.currency} dropped: ${beforeBudget} -> ${wardrobeItems.length}`)
        }

        // ── 3d. Rank + trim the pool for the requested style so the LLM only
        // sees plausible candidates. In manual mode (selectedItemIds) we keep
        // the user's picks as-is. The client already pre-filters shop items,
        // so this mainly helps the wardrobe + legacy branches.
        // The locked anchor is EXEMPT from trimming: dropping it here is what
        // caused "locked item missing / locked to wrong slot" downstream.
        const styleKey = (stylePreferences || 'casual').toLowerCase().replace(/[\s-]+/g, '_')
        const anchorPreRank = anchorItemId
            ? wardrobeItems.find((i: any) => String(i.id) === String(anchorItemId))
            : undefined
        if (!selectedItemIds || selectedItemIds.length === 0) {
            const before = wardrobeItems.length
            wardrobeItems = rankItemsForStyleEdge(wardrobeItems, styleKey, { minKeep: 16, dropThreshold: -2, perCategoryFloor: 4 })
            // Collapse same-product ingestions (keeps best-scored id per
            // similarity key) BEFORE the top-40 trim, so 22 penny-loafer
            // UUIDs can't occupy half the elite pool.
            wardrobeItems = collapseBySimEdge(wardrobeItems, (i: any) => simKeyEdge(i.name || i.type || '', i.macroCategory || ''))
            const occKey = String(occasion || '').toLowerCase()
            const formalOcc = /\b(work|office|date|formal|wedding)\b/.test(occKey)
            const classicStyle = /old_money|classic|business/.test(styleKey)
            if (formalOcc || classicStyle) {
                const beforeTaste = wardrobeItems.length
                wardrobeItems = wardrobeItems.filter((it: any) => {
                    const blob = `${it.name || ''} ${it.type || ''} ${it.description || ''}`.toLowerCase()
                    if (/\bloafer[s]?\s+with\s+(?:tie|laces)|driving|moccasin/.test(blob)) return false
                    if (/\b(washed[\s-]?effect|bell[\s-]?bottom|splatter|uneven[\s-]?hem|twisted[\s-]?seams?)\b/.test(blob)) return false
                    if (formalOcc && /\bpiqu/.test(blob) && /\bpolo/.test(blob)) return false
                    return true
                })
                console.log(`[taste] ${beforeTaste} -> ${wardrobeItems.length} after work/classic veto`)
            }
            wardrobeItems = wardrobeItems.slice(0, 40)
            if (anchorPreRank && !wardrobeItems.some((i: any) => String(i.id) === String(anchorItemId))) {
                console.log(`[anchor] Re-adding locked item ${anchorItemId} after rank trim`)
                wardrobeItems = [anchorPreRank, ...wardrobeItems].slice(0, 40)
            }
            console.log(`[rank] ${before} -> ${wardrobeItems.length} items after style filter for ${styleKey}`)
        }

        console.log(`Outfit gen: ${wardrobeItems.length} items, prompt="${prompt}", style=${stylePreferences}`)

        if (wardrobeItems.length === 0) {
            return new Response(JSON.stringify({
                success: false,
                outfits: [],
                error: 'No clothing items found in your wardrobe. Add items first!',
                debug: {
                    dbClothingItems: dbClothingItems.length,
                    legacyReceived: Array.isArray(legacyWardrobe) ? legacyWardrobe.length : 0,
                    useProvidedWardrobeOnly: !!useProvidedWardrobeOnly,
                    wardrobeOnly: !!wardrobeOnly,
                },
            }), {
                headers: { ...corsHeaders, 'Content-Type': 'application/json' }
            })
        }

        // ── 4. Get API keys ───────────────────────────────────────────────
        let nvidiaKey = Deno.env.get('NVIDIA_API_KEY') || ''
        let nvidiaModel = resolveNvidiaModel(Deno.env.get('NVIDIA_TEXT_MODEL') || Deno.env.get('NVIDIA_MODEL'))
        let geminiKey = Deno.env.get('GEMINI_API_KEY') || ''
        // Preferred Gemini model (env > app_config > product default), with
        // a known-good fallback appended at call time.
        let geminiModelPrimary = (Deno.env.get('GEMINI_MODEL') || '').trim() || 'gemini-3.5-flash-lite'

        if (SUPABASE_SERVICE_ROLE_KEY && (!nvidiaKey || !geminiKey)) {
            const svc = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
            const { data: cfg } = await svc
                .from('app_config')
                .select('key, value')
                .in('key', ['nvidia_token', 'gemma', 'gemini_token', 'gemini_model', 'nvidia_model', 'nvidia_text_model'])
            if (cfg) {
                for (const row of cfg) {
                    if ((row.key === 'nvidia_token' || row.key === 'gemma') && !nvidiaKey) nvidiaKey = row.value
                    if (row.key === 'gemini_token' && !geminiKey) geminiKey = row.value
                    if (row.key === 'gemini_model' && row.value && geminiModelPrimary === 'gemini-3.5-flash-lite') {
                        geminiModelPrimary = row.value.trim()
                    }
                    if ((row.key === 'nvidia_model' || row.key === 'nvidia_text_model') && (!nvidiaModel || nvidiaModel === DEFAULT_NVIDIA_MODEL)) {
                        nvidiaModel = resolveNvidiaModel(row.value)
                    }
                }
            }
        }

        // ── 4b. Subscription tier (server-authoritative) ────────────────────
        // Paid tiers (lite/premium/vip with an active trial/subscription)
        // route to Gemini first; free/anonymous runs NVIDIA first and only
        // falls back to Gemini when NVIDIA fails (rate limits included).
        // Fail-open to 'free' — tier must never hard-error generation.
        let paidTier = false
        if (requestUserId && SUPABASE_SERVICE_ROLE_KEY) {
            try {
                const svc = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
                const { data: subs } = await svc
                    .from('subscriptions')
                    .select('tier, status')
                    .eq('user_id', requestUserId)
                    .order('created_at', { ascending: false })
                    .limit(5)
                if (subs) {
                    paidTier = subs.some((s: any) =>
                        String(s.tier || '').toLowerCase() !== 'free'
                        && ['active', 'trial'].includes(String(s.status || '').toLowerCase()),
                    )
                }
                console.log(`[tier] user=${requestUserId.slice(0, 8)} paid=${paidTier}`)
            } catch (_) {
                // Fail open to free-tier routing.
            }
        }

        // ── 5. Decide layering policy (2-top layered composition vs classic 3-slot) ──
        const layered = needsLayering(stylePreferences, weather, prompt)
        console.log(`[layer] style=${styleKey} weather=${weather?.temp ?? '?'}°C layered=${layered}`)

        // ── 5b. No AI key → local fallback ────────────────────────────────
        if (!nvidiaKey && !geminiKey) {
            const svcClient = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
            let outfits = await localFallback(wardrobeItems, stylePreferences, occasion, limit, layered, svcClient, preferences, wardrobeOnly, deckSalt)
            // Apply anchor enforcement if anchorItemId was provided
            if (anchorItemId && !isShopOnlyDeck(stylePreferences, occasion)) {
                const anchorDbItem = wardrobeItems.find((i: any) => String(i.id) === String(anchorItemId))
                if (anchorDbItem) {
                    console.log(`[anchor] Enforcing anchor item ${anchorItemId} in local fallback outfits`)
                    outfits = enforceAnchorInOutfits(outfits, anchorDbItem)
                }
            }
            return new Response(JSON.stringify({ success: true, outfits, source: 'local', layered, providerAttempts: providerTrace }), {
                headers: { ...corsHeaders, 'Content-Type': 'application/json' }
            })
        }

        // ── 6. Build prompts ───────────────────────────────────────────────
        // When an anchor is locked, tell the model explicitly to build AROUND
        // it (color/style/material match + mandatory id). Post-hoc slot swap
        // alone produced clashing complements.
        const sys = buildSystemPrompt(limit, stylePreferences, layered, preferences, occasion, parsedBudget)
        let usr = buildUserPrompt(wardrobeItems, prompt, stylePreferences, occasion, weather, limit, preferences)
        if (anchorItemId && !isShopOnlyDeck(stylePreferences, occasion)) {
            const anchorForPrompt = wardrobeItems.find((i: any) => String(i.id) === String(anchorItemId))
            if (anchorForPrompt) {
                const anchorDesc = [anchorForPrompt.color, anchorForPrompt.name || anchorForPrompt.type].filter(Boolean).join(' ')
                const anchorHint = `ANCHOR ITEM: Every outfit MUST be built around wardrobe item id="${anchorItemId}" (${anchorDesc || 'the locked piece'}, slot=${anchorForPrompt.macroCategory || 'unknown'}). Include this exact id in every outfit and pick complementary items that match its color, style, and material.`
                usr = `${anchorHint}\n\n${usr}`
            }
        }

        // ── 7. Call AI with validator-retry ─────────────────────────────────
        // After generation, run every outfit through validateOutfitAgainstStyle;
        // if >50% fail, call the model once more with a stricter correction
        // prompt. This is what turns "mostly on-style" into "reliably on-style".
        const itemMap = new Map<string, any>(wardrobeItems.map((i: any) => [String(i.id), i] as [string, any]))
        const geminiModels = [...new Set([geminiModelPrimary, 'gemini-2.0-flash'].filter(Boolean))]
        // Provider order follows the subscription tier: paid (lite/premium/
        // vip) goes Gemini-first with NVIDIA as backup; free/anonymous goes
        // NVIDIA-first with Gemini covering quota failures. Local fallback
        // below guarantees outfits either way.
        let aiProviderUsed: string | null = null
        // Provider-attempt telemetry (status codes only, never keys) so a
        // client can tell a dead key (400/403) from a dead model id (404)
        // from a slow model (timeout) without dashboard access.
        const providerTrace: ProviderAttempt[] = []
        async function tryGeminiChain(gSys: string, gUsr: string): Promise<any[] | null> {
            for (const gm of geminiModels) {
                const r = await callGemini(geminiKey, gm, gSys, gUsr, 0, providerTrace)
                if (r && r.length > 0) {
                    aiProviderUsed = 'gemini:' + gm
                    return r
                }
            }
            return null
        }
        let aiOutfits: any[] | null = null
        if (paidTier) {
            console.log('[provider] paid tier → gemini first')
            if (geminiKey) aiOutfits = await tryGeminiChain(sys, usr)
            if (!aiOutfits && nvidiaKey) {
                aiOutfits = await callNvidia(nvidiaKey, nvidiaModel, sys, usr, 0, providerTrace)
                if (aiOutfits) aiProviderUsed = 'nvidia:' + nvidiaModel
            }
        } else {
            if (nvidiaKey) {
                aiOutfits = await callNvidia(nvidiaKey, nvidiaModel, sys, usr, 0, providerTrace)
                if (aiOutfits) aiProviderUsed = 'nvidia:' + nvidiaModel
            }
            if (!aiOutfits && geminiKey) aiOutfits = await tryGeminiChain(sys, usr)
        }

        if (aiOutfits && aiOutfits.length > 0) {
            const valid = filterValidOutfits(aiOutfits, styleKey, itemMap, layered, preferences, occasion)
            const passRate = valid.length / aiOutfits.length
            console.log(`[validate] ${valid.length}/${aiOutfits.length} outfits passed style check for ${styleKey} (layered=${layered})`)

            if (valid.length >= Math.max(1, Math.min(limit, aiOutfits.length) - 1)) {
                aiOutfits = valid
            } else if (passRate < 0.5 && (nvidiaKey || geminiKey)) {
                console.log(`[validate] pass rate ${passRate.toFixed(2)} too low — retrying with correction`)
                const layeredReqs = layered
                    ? `  1. EVERY outfit MUST be a FOUR-slot layered look with ALL of these macroCategories present: "top" (base layer — t-shirt, shirt, polo, blouse), "outerwear" (main top — sweater, blazer, cardigan, hoodie, jacket), "bottom" (pants/trousers/jeans/shorts/skirt), "shoes". A 3-item outfit is INVALID.\n  2. The base top must sit UNDER the outerwear and color-coordinate with it. Think: white oxford shirt under a navy blazer, or cream tee under a camel cardigan.\n  3. Do NOT return two base tops or two outerwear pieces — exactly one of each.`
                    : `  1. EVERY outfit MUST include an item with macroCategory=shoes. No outfit without shoes is acceptable.\n  2. EVERY outfit MUST include a top (or outerwear) AND a bottom.`
                const correctionUsr = `${usr}\n\nIMPORTANT CORRECTION: Your previous attempt failed because (a) some items violated the ${styleKey} aesthetic, OR (b) outfits were missing required slots. REQUIREMENTS:\n${layeredReqs}\n  ${layered ? '4' : '3'}. Reject any item whose name/description contains: hoodie (unless casual), graphic tee, logo tee, cargo, ripped, sweatpants, neon, sequin, rhinestone, chunky sneaker, basketball, skate sneaker, casual t-shirt (for old_money / business_casual / minimalist / semi_classic).\n  ${layered ? '5' : '4'}. For ${styleKey} shoes, prefer: loafers, penny loafers, bit loafers, dress shoes, boat shoes, derbies, leather shoes. Avoid ALL sneakers if style is old_money.\nTry again and return ${limit} outfits that each have ${layered ? 'outerwear+base-top+bottom+shoes' : 'top+bottom+shoes'}.`
                let retry: any[] | null = null
                if (paidTier) {
                    if (geminiKey) retry = await tryGeminiChain(sys, correctionUsr)
                    if (!retry && nvidiaKey) retry = await callNvidia(nvidiaKey, nvidiaModel, sys, correctionUsr, 0, providerTrace)
                } else {
                    if (nvidiaKey) retry = await callNvidia(nvidiaKey, nvidiaModel, sys, correctionUsr, 0, providerTrace)
                    if (!retry && geminiKey) retry = await tryGeminiChain(sys, correctionUsr)
                }
                if (retry && retry.length > 0) {
                    const retryValid = filterValidOutfits(retry, styleKey, itemMap, layered, preferences, occasion)
                    aiOutfits = retryValid.length > valid.length ? retryValid : (valid.length > 0 ? valid : retry)
                } else if (valid.length > 0) {
                    aiOutfits = valid
                }
            } else if (valid.length > 0) {
                aiOutfits = valid
            }
        }

        // The model may technically return 15 records while repeating the
        // same combination, leaving only a few after signature de-duplication.
        // Fill the remainder with the deterministic combination builder so
        // the Home deck receives as many genuinely distinct looks as possible.
        if (aiOutfits && aiOutfits.length > 0 && aiOutfits.length < limit) {
            const svcClient2 = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
            const generatedFill = await localFallback(
                wardrobeItems,
                stylePreferences,
                occasion,
                limit,
                layered,
                svcClient2,
                preferences,
                wardrobeOnly,
                deckSalt,
            )
            const signatures = new Set(
                aiOutfits.map((outfit: any) => (outfit.items || []).map((item: any) => String(item.id)).sort().join('|')),
            )
            const keptSets: Array<Set<string>> = aiOutfits.map((o: any) => new Set((o.items || []).map((item: any) => String(item.id))))
            const fillTarget = layered ? 4 : 3
            const fillMaxShared = fillTarget - 2
            for (const outfit of generatedFill) {
                const ids = (outfit.items || []).map((item: any) => String(item.id)).sort()
                const signature = ids.join('|')
                if (!signature || signatures.has(signature)) continue
                let maxShared = 0
                for (const ks of keptSets) {
                    let s = 0
                    for (const id of ids) if (ks.has(id)) s++
                    if (s > maxShared) maxShared = s
                }
                if (maxShared > fillMaxShared) {
                    console.log(`[diversity] skipped near-duplicate fill (shares ${maxShared}): ${signature}`)
                    continue
                }
                signatures.add(signature)
                keptSets.push(new Set(ids))
                aiOutfits.push(outfit)
                if (aiOutfits.length >= limit) break
            }
            console.log(`[diversity] Filled daily deck to ${aiOutfits.length}/${limit} unique combinations`)
        }

        if (!aiOutfits || aiOutfits.length === 0) {
            // AI failed or all outfits failed validation — fall through to local generator
            // instead of returning a hard error, so the client always gets outfits.
            console.log('[generate-outfits] AI returned no valid outfits, using localFallback')
            const svcClient2 = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
            let fallbackOutfits = await localFallback(wardrobeItems, stylePreferences, occasion, limit, layered, svcClient2, preferences, wardrobeOnly, deckSalt)
            // Apply anchor enforcement if anchorItemId was provided
            if (anchorItemId && !isShopOnlyDeck(stylePreferences, occasion)) {
                const anchorDbItem = wardrobeItems.find((i: any) => String(i.id) === String(anchorItemId))
                if (anchorDbItem) {
                    console.log(`[anchor] Enforcing anchor item ${anchorItemId} in AI-fallback outfits`)
                    fallbackOutfits = enforceAnchorInOutfits(fallbackOutfits, anchorDbItem)
                }
            }
            return new Response(JSON.stringify({ success: true, outfits: fallbackOutfits, source: 'local', layered, providerAttempts: providerTrace }),
                {
                    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
                },
            )
        }

        // ── 8. Enrich items with imageUrl from DB ──────────────────────────
        // Also replace placeholder items (without valid images) with shop catalog items.
        // wardrobeOnly requests skip the shop query entirely — a wardrobe
        // item without an image stays as-is (client filters imageless items
        // upstream) instead of becoming a shop_* item.
        const svcClient = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        const { data: shopItems, error: shopError } = wardrobeOnly
            ? { data: [], error: null }
            : await svcClient
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
        
        // Shop-only + general enrich uses a deterministic round-robin cursor so
        // repeated missing-image slots don't collapse onto the same random shop
        // item and create duplicate outfits.
        let enrichCursor = 0
        const enriched = aiOutfits.map((outfit: any, outfitIdx: number) => ({
            ...outfit,
            items: (outfit.items || []).map((item: any, itemIdx: number) => {
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
                        const ctx = getStyleContext(stylePreferences)
                        const rejectKeywords = extractRejectKeywords(ctx.reject)
                        const allowedShopItems = shopByMacro[macroNormalized].filter(shopItem => {
                            const nameStr = (shopItem.name || '').toLowerCase()
                            const descStr = (shopItem.description || '').toLowerCase()
                            const blob = `${nameStr} ${descStr} ${shopItem.garment_type || ''} ${shopItem.category || ''}`.toLowerCase()
                            for (const kw of rejectKeywords) {
                                if (blob.includes(kw)) return false
                            }
                            return true
                        })

                        const candidates = allowedShopItems.length > 0 ? allowedShopItems : shopByMacro[macroNormalized]
                        const shopItem = candidates[(enrichCursor + outfitIdx + itemIdx) % candidates.length]
                        enrichCursor += 1
                        console.log(`[enrich] Replacing placeholder ${item.id} (${macro}) with shop item ${shopItem.id}`)
                        finalItem = {
                            ...item,
                            id: String(shopItem.id).startsWith('shop_') ? String(shopItem.id) : `shop_${shopItem.id}`,
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

        // ── 9. Apply anchor-item enforcement ──────────────────────────────
        // If anchorItemId was provided, lock that item into its slot in every outfit.
        // This runs after enrichment so the anchor item's imageUrl is already resolved.
        let finalOutfits = enriched
        if (anchorItemId && !isShopOnlyDeck(stylePreferences, occasion)) {
            // Look up the anchor item from the wardrobe pool (pre-enrichment data)
            const anchorDbItem = wardrobeItems.find((i: any) => String(i.id) === String(anchorItemId))
            if (anchorDbItem) {
                console.log(`[anchor] Enforcing anchor item ${anchorItemId} (${anchorDbItem.type || anchorDbItem.category}) in ${finalOutfits.length} AI outfits`)
                finalOutfits = enforceAnchorInOutfits(finalOutfits, anchorDbItem)
            } else {
                console.warn(`[anchor] anchorItemId ${anchorItemId} not found in wardrobeItems pool — skipping enforcement`)
            }
        }

        return new Response(JSON.stringify({ success: true, outfits: finalOutfits, source: 'ai', layered, provider: aiProviderUsed, providerAttempts }), {
            headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        })

    } catch (error: any) {
        console.error('generate-outfits error:', error)
        return new Response(JSON.stringify({ success: false, error: error.message, outfits: [] }), {
            headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 500
        })
    }
})
