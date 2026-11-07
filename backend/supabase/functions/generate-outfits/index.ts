import { STYLING_PROFILES } from '../../../../src/domain/stylingProfiles.ts'
import { stylingCandidateRepresentatives } from '../../../../src/domain/catalogCandidates.ts'
import { describeGarment, prepareGarment, FASHION_INTELLIGENCE_VERSION } from '../../../../src/domain/garmentDescriptor.ts'
import { outfitStyleIntent } from '../../../../src/domain/stylingProfiles.ts'
import {garmentContextReason,scoreGarmentForContext,compareUnbudgetedValue} from '../../../../src/domain/garmentContext.ts'
import { createOutfitPlan } from '../../../../src/domain/validatedQuartets.ts'
import { stylingContext } from '../../../../src/domain/stylingContext.ts'
import type { CatalogCandidateRpcRequest } from '../../../../src/domain/catalogCandidates.ts'
import * as styleInference from '../../../../src/domain/styleInference.ts'
import { parseRankedOutfitChoices } from '../_shared/outfitChoices.ts'
import { readOutfitGenerationRequest, OutfitRequestError } from '../_shared/outfitGenerationRequest.ts'
import { budgetCandidatesFromText as budgetCandidatesFromTextEdge } from '../_shared/outfitBudgetText.ts'
import { type BudgetFxContext, BudgetFxError, budgetFxContext, fitsBudgetAmount, referenceAmount } from '../_shared/budgetFx.ts'
import { currentBudgetRates, budgetRatesById } from '../_shared/budgetFxCache.ts'
import { minimumSlotSpend } from '../_shared/outfitBudgetFeasibility.ts'
import { strictestOutfitBudget } from '../_shared/outfitBudgetSelection.ts'
import { outfitBudgetFeedback } from '../_shared/outfitBudgetFeedback.ts'
import { normalizeModelOutfits } from '../_shared/modelOutfitResponse.ts'
import { enforceRateLimit } from '../_shared/rateLimit.ts'
import { isFourSlotCandidate, fourSlotOutfitReason, outfitStructureReason, isDressGarment, isNonOutfitItem } from '../_shared/outfitStructure.ts'
import { styleJudgeReason, assessOutfitQuality, spreadQualityFamilies, normalizeGarmentText, isWorkoutOccasion, rejectForStyleOccasion, rejectForClimate, rejectForRequestIntent, rejectForActivity, rejectOutfitIncoherence, isDrivingFootwear } from '../_shared/occasionRules.ts'
import { normalizeProductHistory, spreadSuitableProducts, canonicalProductKey, productIdentityKeys, garmentArchetypeToken, countDistinctProducts, type DiversitySlot } from '../_shared/outfitDiversity.ts'
import {wasRecentlyShown, freshProductCount, createShoeRotationPriority, compareRotationOrder} from '../_shared/outfitRotation.ts'
import { needsLayering as needsLayeringShared, resolveStylistWeather } from '../_shared/outfitComposition.ts'
import { detectSlotMismatch, isOxfordFootwearName } from '../_shared/occasionRules.ts'
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from "https://esm.sh/@supabase/supabase-js@2"

const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}
const GENERATION_VERSION = 'stylist-3-unified-planning-20261107';

// Team Collaboration is always 100% shop — no wardrobe anchor.
// Matches by style (`business_casual`) OR occasion label (`Team Collaboration`).
function parseBudgetFromText(text?: string | null, fx?: BudgetFxContext): { amount: number; currency: string } | null {
    const found = budgetCandidatesFromTextEdge(text)
    if (!found.length || (!fx && new Set(found.map(b => b.currency)).size > 1)) return null
    return found.reduce((best, cur) => convertShopAmountEdge(cur.amount, cur.currency, best.currency, fx) < best.amount ? cur : best)
}

function isShopPricedItem(item: any): boolean {
    if (!item) return false
    if (item.isWardrobe === true && !item.isShopItem) return false
    const id = String(item.id || '')
    return Boolean(item.isShopItem || item.isWardrobe === false || /^(shop_|md-|zara-|classic-m-)/i.test(id))
}

function convertShopAmountEdge(amount: number, from: string | null | undefined, to: string, fx?: BudgetFxContext): number {
    const source = String(from || '').trim().toUpperCase(), target = String(to).trim().toUpperCase()
    if (source === target) return Number.isFinite(amount) && amount >= 0 ? amount : Infinity
    return fx ? referenceAmount(amount, source, target, fx.snapshot, fx.checkedAt) : Infinity
}

function shopPriceOf(item: any, targetCurrency?: string | null, fx?: BudgetFxContext): number {
    if (!isShopPricedItem(item)) return 0
    const n = Number(item?.price)
    return Number.isFinite(n) && n > 0 ? (targetCurrency ? convertShopAmountEdge(n, item.currency, targetCurrency, fx) : n) : (targetCurrency ? Number.POSITIVE_INFINITY : 0)
}

function outfitShopSpend(items: any[], itemMap?: Map<string, any>, targetCurrency?: string | null, fx?: BudgetFxContext): number {
    const currency = targetCurrency || items.map(it => (itemMap?.get(String(it?.id)) || it)?.currency).find(Boolean)
    return (items || []).reduce((sum, it) => {
        const src = itemMap?.get(String(it?.id)) || it
        return sum + shopPriceOf({ ...src, ...it, price: src?.price, currency: src?.currency,
            isShopItem: src?.isShopItem, isWardrobe: src?.isWardrobe, id: src?.id }, currency, fx)
    }, 0)
}

function budgetInCurrencyEdge(
    budget: { amount: number; currency: string; fx?: BudgetFxContext } | null | undefined,
    targetCurrency?: string | null,
): { amount: number; currency: string } | null {
    if (!budget) return null
    const to = String(targetCurrency || '').trim().toUpperCase()
    const from = String(budget.currency || 'USD').trim().toUpperCase()
    if (!to || to === from) return { ...budget, currency: from }
    const converted = convertShopAmountEdge(budget.amount, from, to, budget.fx)
    return Number.isFinite(converted) ? { amount: converted, currency: to } : null
}

function fitOutfitToBudget(parts: any[], pool: any[], budget: { amount: number; currency?: string; fx?: BudgetFxContext } | null | undefined): any[] {
    if (!budget || !Array.isArray(parts) || parts.length === 0) return parts
    const cap = budget.amount
    const slotOf = (p: any) => String(p?.macroCategory || '').toLowerCase()
    const cheaper = (slot: string, exclude: Set<string>, maxPrice: number) => {
        return pool
            .filter((p: any) => isShopPricedItem(p) && slotOf(p) === slot && !exclude.has(String(p.id)))
            .filter((p: any) => {
                const pr = shopPriceOf(p, budget.currency, budget.fx)
                return pr > 0 && pr <= maxPrice
            })
            .sort((a: any, b: any) => shopPriceOf(a, budget.currency, budget.fx) - shopPriceOf(b, budget.currency, budget.fx))[0]
    }
    const next = parts.map((p: any) => ({ ...p }))
    for (let guard = 0; guard < 8 && !fitsBudgetAmount(outfitShopSpend(next, undefined, budget.currency, budget.fx), cap); guard++) {
        let worstIdx = -1
        let worstPrice = -1
        next.forEach((p: any, i: number) => {
            const pr = shopPriceOf(p, budget.currency, budget.fx)
            if (pr > worstPrice) { worstPrice = pr; worstIdx = i }
        })
        if (worstIdx < 0) break
        const used = new Set(next.map((p: any) => String(p.id)))
        const room = cap - (outfitShopSpend(next, undefined, budget.currency, budget.fx) - shopPriceOf(next[worstIdx], budget.currency, budget.fx))
        const repl = cheaper(slotOf(next[worstIdx]), used, room)
        if (repl) {
            next[worstIdx] = { ...repl, recommendation: next[worstIdx].recommendation || 'Suggested from shop' }
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

// Shared shop-row predicate. Mirrors the client's `isShopClothingItem` so the
// server and client can never disagree about whether a row is a catalog pick.
function isShopRow(item: any): boolean {
    if (!item) return false
    return Boolean(item.isShopItem) || String(item.id || '').startsWith('shop_')
}

function macroCategory(category: string, type: string, name?: string): string {
    // Explicit garment names override noisy feed slot labels, as English names do.
    const garmentName = String(name || '').toLowerCase()
    if (/кроссовк|кеды|туфл|ботинк|лофер|сапог|мокасин|сандал/.test(garmentName)) return 'shoes'
    if (/жилет|куртк|пальто|пиджак|жакет|кардиган|свитер|худи|толстовк/.test(garmentName)) return 'outerwear'
    if (/джинс|брюк|шорт|юбк|леггинс/.test(garmentName)) return 'bottom'
    if (/футболк|рубашк|блуз|майк|платье/.test(garmentName)) return 'top'
    // A coating is a surface treatment, not a coat. Explicit trouser
    // names take precedence over noisy catalog labels and substrings.
    if (/\b(pants?|trousers?|jeans|chinos?|slacks?|joggers?)\b/.test(garmentName)
        && !/\b(shirts?|jackets?|coats?|blazers?|cardigans?)\b/.test(garmentName)) return 'bottom'
    const t = normalizeGarmentText(`${category} ${type} ${name || ''}`)
    
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
    
    return result
}

function isDressItem(item: any): boolean {
    return isDressGarment(item)
}

// Layer required except hot weather / explicit no-jacket intent.
function needsLayering(style: string | null | undefined, weather: any, prompt: string | null | undefined): boolean {
    return needsLayeringShared(style, weather, prompt);
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
    return isDrivingFootwear(item) || /\bmules?\b/.test(blob)
}

function isWorkishOccasion(occasion: string | null | undefined): boolean {
    return /work|business|office|meeting|interview|wedding|funeral|formal|conference|corporate/.test((occasion || '').toLowerCase())
}

// Serialization and retrieval thresholds are Edge adapter concerns. The style
// lexicon, scoring and eligibility-before-ranking rules have one domain authority.
function inferAttrsEdge(item: any): { styleTags: string[]; materials: string; patterns: string; formality: number } {
    const attrs = styleInference.inferItemAttributes(item)
    return { styleTags: attrs.styleTags, materials: attrs.materials.join(', '), patterns: attrs.patterns.join(', '), formality: attrs.formality }
}
function scoreItemForStyleEdge(item: any, style: string): number {
    return styleInference.rawStyleScore(item, style)
}
function rankItemsForStyleEdge<T extends styleInference.ItemForInference>(
    items: T[], style: string,
    opts: { minKeep?: number; dropThreshold?: number; perCategoryFloor?: number; eligibility?: (item: T) => boolean } = {},
): T[] {
    return styleInference.rankItemsForStyle(items, styleInference.normalizeStyleId(style), { minKeep: 12, dropThreshold: -2, ...opts })
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
// Preserve the known per-slot budget floor through style ranking and product
// family collapse. This does not claim that unseen catalog rows were searched.
function retainBudgetFloorCandidates(
    source: any[], ranked: any[], cost: (item: any) => number,
    eligible: (item: any) => boolean, anchor?: any, perSlotChoices = 1, qualityOf: (item:any)=>number = ()=>0, context?: ReturnType<typeof stylingContext>,
): any[] {
    const cheapest = new Map<string, { item: any; cost: number }>()
    for (const item of source) {
        if (!eligible(item)) continue
        const slot = isDressGarment(item) ? 'dress' : String(item.macroCategory || '').toLowerCase()
        if (!['top', 'bottom', 'shoes', 'outerwear', 'dress'].includes(slot)) continue
        const value = cost(item)
        if (!Number.isFinite(value) || value < 0) continue
        const prior = cheapest.get(slot)
        if (!prior || value < prior.cost) cheapest.set(slot, { item, cost: value })
    }
    const representatives = Array.from(cheapest.values(), row => row.item)
    const representedIds = new Set(representatives.map(item => String(item.id)))
    const representedProducts = new Set(representatives.flatMap(item=>productIdentityKeys(item)))
    const slotCounts = new Map(Array.from(cheapest.keys(), slot => [slot, 1]))
    // The cheapest floor is mandatory; remaining per-slot choices follow
    // the context-aware ranking, not the catalog's arrival order.
    const available = [...new Map([...ranked,...source].filter(eligible).map(item=>[String(item.id),item])).values()]
    const ordered = context ? ['top','outerwear','bottom','shoes','dress'].flatMap(slot =>
        stylingCandidateRepresentatives(available.filter(item => (isDressGarment(item) ? 'dress' : item.macroCategory) === slot),item=>item,qualityOf,context,perSlotChoices))
        : spreadSuitableProducts(available,item=>item,qualityOf,0.08)
    for (const item of ordered) {
        if (!eligible(item) || representedIds.has(String(item.id)) || productIdentityKeys(item).some(key=>representedProducts.has(key))) continue
        const slot = isDressGarment(item) ? 'dress' : String(item.macroCategory || '').toLowerCase()
        if (!cheapest.has(slot) || (slotCounts.get(slot) || 0) >= perSlotChoices) continue
        representedIds.add(String(item.id)); productIdentityKeys(item).forEach(key=>representedProducts.add(key)); slotCounts.set(slot, (slotCounts.get(slot) || 0) + 1)
        representatives.push(item)
    }
    const combined = [...(anchor ? [anchor] : []), ...representatives, ...ranked]
    const ids = new Set<string>()
    return collapseBySimEdge(combined.filter(item => {
        const id = String(item.id)
        if (ids.has(id)) return false
        ids.add(id)
        return true
    }), item => canonicalProductKey(item)).slice(0, 40)
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
const WORKOUT_STYLE_KNOWLEDGE = `WORKOUT REQUIREMENTS:
- Use verified athletic or performance clothing in every required slot, with activity-appropriate athletic shoes.
- Match the user's fit, color and pattern preferences without substituting dress clothing or tailored footwear.
- Performance synthetics and breathable fabrics are appropriate; prioritize movement and comfort.
- Keep all weather, required-slot and total-budget constraints. Never omit shoes or a required layer.`

const STYLE_CONTEXT: Record<string, { vibe: string; must: string[]; reject: string[]; palette: string[]; fabrics: string[] }> = {
    sport: {
        vibe: 'Functional workout clothing with coordinated, understated colors.',
        must: ['Verified athletic top, training bottom and activity-appropriate athletic shoes. Activity requirements take priority over aesthetic tailoring.'],
        reject: ['NEVER include: dress shirts, blazers, loafers, suede shoes'],
        palette: ['navy', 'grey', 'black', 'white'],
        fabrics: ['performance fabric', 'breathable cotton', 'stretch jersey'],
    },
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
            'NEVER include: cargo shorts, cargo pants, athletic shorts, basketball shorts, track pants, sweatpants, joggers, jogging trousers, shorts/bermudas, drawstring pants, elastic-waist trousers.',
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
    const shopOnly = isShopOnlyDeck(style, occasion) && !preferences?.lockedAnchorId
    const styleKnowledge = style === 'sport' ? WORKOUT_STYLE_KNOWLEDGE : STYLE_KNOWLEDGE
    const slotRules = layered
        ? `- EVERY outfit MUST contain exactly FOUR items, one in each category:
    (1) LAYER (macroCategory="outerwear" — cardigan, blazer, lightweight jacket, coat, vest or sweater)
    (2) BASE SHIRT (macroCategory="top" — shirt, T-shirt or polo shirt, never another sweater or jacket)
    (3) PANTS (macroCategory="bottom" — pants, full-length trousers, chinos or jeans; never shorts or skirts)
    (4) SHOES (macroCategory="shoes")
  Never omit a category, substitute a dress for shirt and pants, or count one garment twice.`
        : `- EVERY outfit MUST contain exactly THREE items, one in each category:
    (1) BASE SHIRT (macroCategory="top" — shirt, T-shirt or polo shirt)
    (2) PANTS (macroCategory="bottom" — pants, full-length trousers, chinos or jeans; never shorts or skirts)
    (3) SHOES (macroCategory="shoes")
  Do NOT include any outerwear/layer/jacket in this request. Never substitute a dress for shirt and pants, or count one garment twice.`

    const ownershipRule = shopOnly
        ? `- THIS IS A SHOP-ONLY DECK (Team Collaboration). EVERY item in EVERY outfit MUST have an id starting with "shop_". NEVER use user wardrobe items. Build all ${limit} looks purely from shop candidates.`
        : `- PRIORITIZE USER-OWNED ITEMS. Items whose id starts with "shop_" are optional catalog suggestions; use them only to complete a missing/sparse slot or add necessary variety. Every outfit should contain owned pieces whenever any compatible owned pieces exist.`
    const budgetRule = budget
        ? `- HARD BUDGET ${budget.amount} ${budget.currency}: for EACH outfit, the sum of budgetCost values (converted to the budget currency) MUST be <= ${budget.amount}. Owned wardrobe items (source:user wardrobe) cost 0. Never pick a shop item whose budgetCost alone exceeds remaining budget. Prefer a cheaper shop substitute over dropping a required slot.`
        : `- Shop items may include a price. If the user prompt names a budget, shop prices in each look MUST sum to that cap or less (owned = 0). Always cite prices for shop picks. Without an explicit budget, prefer lower verified prices among equally suitable alternatives in the same currency; do not assume a premium price improves styling and do not invent a budget ceiling.`
    // PRODUCT UNIQUENESS IS THE ONLY HARD DIVERSITY RULE.
    //
    // The provider's job is to return `limit` valid, high-quality looks
    // whose ACTUAL PRODUCTS do not repeat (unless the slot is genuinely
    // thin). The GLOBAL deck selector then picks the best combination.
    //
    // The old wording ("must differ by at least 2 item ids", "visibly
    // different") contradicted that rule twice: it permitted two looks to
    // share two products, and it forbade the same outfit FORMULA — which is
    // legal and often correct. A prompt that contradicts the architecture
    // produces decks the selector then has to throw away.
    const uniquenessRule = shopOnly
        ? `- NEVER reuse the same actual product in two looks when another valid product is available. Two candidates with the same product name or the same photo are the SAME product; candidates whose names differ only by color are DIFFERENT products.
- Different looks MAY use the same garment types and the same outfit structure. Rotate tops, bottoms AND shoes so each look uses a different actual product in every slot.`
        : `- NEVER reuse the same actual non-anchor item in two looks when another valid product is available. An item is the SAME product when the id, the product name or the photo matches.
- Different looks MAY use the same garment types and the same outfit structure. Rotate tops, bottoms and shoes so each look uses a different actual product in every slot.
- Reusing one anchor item across all looks is allowed ONLY when an ANCHOR ITEM is locked; all other slots MUST still rotate.`
    // Same formula is allowed; same actual product is not. The deck may
    // repeat an outfit structure (blazer + shirt + trousers + loafers x3)
    // when that is the right formula for the occasion — what must never
    // repeat is the actual clothing product, when alternatives exist.
    const directionRule = `- PRODUCT UNIQUENESS ACROSS THE ${limit} LOOKS:
  - Each non-anchor item should appear in only one look. When a slot has fewer candidates than looks, it is acceptable to reuse that slot's product rather than break style, occasion, weather or budget.
  - Different looks MAY use the same garment types or the same overall outfit structure. Three blazer + shirt + trousers + loafer looks are acceptable if every blazer, shirt, trouser and shoe is a different actual product.
  - Do not sacrifice style, occasion, weather or budget correctness just to force different silhouettes or formulas.
  - Prefer some visual variety when equally good alternatives exist, but product uniqueness is the hard diversity requirement.`

    return `You are a world-class fashion stylist curating outfits for a specific aesthetic. You must obey the STYLE DIRECTION below as if your reputation depends on it.

TASK: Create ${limit} complete outfits using ONLY items from the provided wardrobe, referenced by their exact id.

GENERAL STYLING RULES:
${slotRules}
${ownershipRule}
${budgetRule}
${uniquenessRule}
${directionRule}
- If any required slot has no suitable item, omit that outfit. Return fewer outfits or an empty outfits array rather than relaxing the request.
- NEVER invent items. Use ONLY the exact ids listed in "Available wardrobe items".
- Pick items whose styleTags, name, description, or material indicate they FIT the requested style. If an item's tags or description clash with the style, do NOT use it, even when it is the only option for that slot.
- Favor color harmony (tonal, monochromatic, or analogous palettes from the style's preferred palette). When layering, the base top must color-coordinate with the outerwear (tonal or complementary — never clashing).
- PRODUCT ROTATION: spread usage across the available pool. Do not put the same top, bottom, or shoes in all ${limit} outfits when alternatives exist. If a slot has ≥${limit} candidates, each outfit MUST use a different actual product. The same garment TYPE repeating across looks is fine; the same actual PRODUCT repeating is not.
- NEVER pair formal outerwear (blazer, suit jacket, sport coat, overcoat, topcoat, trench, peacoat, tuxedo) with shorts or bermudas. If the only available bottom is shorts and a formal layer is required, omit the outfit.
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

${styleKnowledge}

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
  4. Does any non-anchor item (same id, same product name or same photo) appear in more than one outfit? If so, rebuild the overlapping outfit with a different actual product (a different blazer/shirt/trousers/shoes — the same garment type and the same formula are fine).
  5. Are style, occasion, weather and budget still correct for every outfit? Never trade correctness for variety.
If the answer to any of these is "no", rebuild the outfit.

Respond ONLY in pure JSON (no markdown, no code fences):
{"outfits":[{"description":"brief explanation","items":[{"id":"EXACT_DB_ID"}]}]}`
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
            preferences?.maxShopBudget
                ? `budgetCost:${shopPriceOf(i, preferences.maxShopBudget.currency, preferences.maxShopBudget.fx)} ${preferences.maxShopBudget.currency}`
                : '',
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
            lines.push(`- HARD BUDGET: shop budgetCost values in each outfit must sum to <= ${preferences.maxShopBudget.amount} ${preferences.maxShopBudget.currency || ''}. Owned items cost 0.`)
        }
        if (lines.length > 0) {
            prefText = `\n\nREMINDER OF USER PREFERENCES:\n${lines.join('\n')}`
        }
    }

    const temp = typeof weather?.temp === 'number' ? weather.temp : null
    let climateRules = ''
    if (temp != null && temp < 18) {
        climateRules = style === 'sport'
            ? `CLIMATE HARD RULES: It is ${temp}°C. Choose warm performance layers, training trousers and closed athletic shoes. Avoid shorts, sandals and summer-only fabrics. Include the required warm outer layer.\n`
            : `CLIMATE HARD RULES: It is ${temp}°C (${weather?.condition || 'cool'}). Do NOT pick linen, seersucker, tropical wool, shorts, sandals, or espadrilles. Prefer wool, merino, cashmere, flannel, cotton oxfords, and leather shoes. EVERY outfit MUST include a warm layer (wool blazer, knit, coat, cardigan) — a summer shirt alone is invalid.\n`
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
    const includesDress = items.some((item: any) => isDressItem(itemMap.get(String(item?.id))))
    const minItems = includesDress ? (layered ? 3 : 2) : (layered ? 4 : 3)
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
        const nameIsShoe = isOxfordFootwearName(nameOnly) || /\bshoes?|sneakers?|boots?|loafers?|sandals?|heels?|trainers?|derbys?|derbies|moccasins?|drivers?|mules?\b/.test(nameOnly)
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
    const hasBaseTop = macros.has('top') || hasDress
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

    // Exactly-one-per-slot: Layer -> Top -> Pants -> Shoes. A second bottom
    // (two pants in one outfit), a second base top, or a second outerwear
    // piece is always a generation error — reject so the caller rebuilds.
    // A dress replaces base-top + bottom, so it exempts those two counts.
    {
        const slotCounts = new Map<string, number>()
        for (const it of items) {
            const src = itemMap.get(it.id) || {}
            const macro = String(src.macroCategory || it.macroCategory || '').toLowerCase()
            if (!macro) continue
            slotCounts.set(macro, (slotCounts.get(macro) || 0) + 1)
        }
        const needSingle = hasDress ? ['shoes'] : ['top', 'outerwear', 'bottom', 'shoes']
        // Non-layered looks carry a single upper (top OR outerwear) — never both doubled.
        const singleSlots = layered ? needSingle : (hasDress ? ['shoes'] : ['bottom', 'shoes'])
        for (const slot of singleSlots) {
            const n = slotCounts.get(slot) || 0
            if (n > 1) return { ok: false, reason: `duplicate "${slot}" slot x${n} (macros: ${Array.from(macros).join(',')}) — one ${slot} per outfit` }
        }
        if (!layered && !hasDress) {
            const uppers = (slotCounts.get('top') || 0) + (slotCounts.get('outerwear') || 0)
            if (uppers > 1) return { ok: false, reason: `duplicate upper slot x${uppers} (macros: ${Array.from(macros).join(',')}) — one top per non-layered outfit` }
        }
        if (hasDress) {
            // A dress already covers base-top + bottom: a second dress, an
            // extra pants item, or an extra non-dress top alongside it is the
            // same duplicate-slot defect in another shape.
            const dressCount = items.filter((it: any) => isDressItem(itemMap.get(it.id) || {}) || isDressItem(it)).length
            if (dressCount > 1) return { ok: false, reason: `duplicate dress x${dressCount} — one dress per outfit` }
            if ((slotCounts.get('bottom') || 0) > 0) return { ok: false, reason: `dress + bottom (macros: ${Array.from(macros).join(',')}) — a dress replaces the bottom slot` }
            const extraTops = items.filter((it: any) => {
                const src = itemMap.get(it.id) || {}
                if (String(src.macroCategory || it.macroCategory || '').toLowerCase() !== 'top') return false
                return !(isDressItem(src) || isDressItem(it))
            })
            if (extraTops.length > 0) return { ok: false, reason: `dress + extra top "${extraTops[0]?.name || extraTops[0]?.id}" — a dress replaces the base-top slot` }
            // Dress looks are outer + dress + shoes (layered) or dress + shoes.
            const dressTarget = layered ? 3 : 2
            if (items.length > dressTarget) return { ok: false, reason: `too many items for a dress look (${items.length}, need ${dressTarget}) — duplicate slot` }
        }
        const targetItems = layered ? 4 : 3
        if (!hasDress && items.length > targetItems) return { ok: false, reason: `too many items (${items.length}, need ${targetItems}) — duplicate slot` }
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

    // The selected anchor may override aesthetic taste; the hard validators above still apply.
    const rejectKeywords = extractRejectKeywords(ctx.reject)
    for (const it of items) {
        if (preferences?.lockedAnchorId && String(it.id) === preferences.lockedAnchorId) continue
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
            // Preserve the rule's scope: "loafers with tie laces" must not
            // prohibit every loafer, nor "chunky sneakers" every sneaker.
            out.push(p)
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

/**
 * Validate every candidate outfit INDEPENDENTLY.
 *
 * This function answers exactly one question: "is this individual outfit
 * valid?" It answers NOTHING about which looks should coexist in the final
 * deck. Cross-look product uniqueness is decided ONCE, globally, by
 * `assembleDiverseDeck()`.
 *
 * A greedy cross-look filter here would be actively harmful: if candidate A
 * overlaps B, C and D while B/C/D are mutually unique, accepting A first
 * would reject B, C and D and destroy the only correct deck. The global
 * selector explores combinations, so it can pick B + C + D.
 */
function filterValidOutfits(outfits: any[], style: string, itemMap: Map<string, any>, layered: boolean, preferences?: any, occasion?: string, weather?: any, onReject?: (index: number, reason: string) => void): any[] {
    const kept: any[] = []
    const signatures = new Set<string>()
    const shopOnly = isShopOnlyDeck(style, occasion) && !preferences?.lockedAnchorId
    const hasOwnedCandidates = !shopOnly && Array.from(itemMap.values()).some(
        (item: any) => !item.isShopItem && !String(item.id || '').startsWith('shop_'),
    )
    for (const [outfitIndex, proposal] of outfits.entries()) {
        const reject = (reason: string) => onReject?.(outfitIndex + 1, reason.slice(0, 300))
        const resolved = (proposal.items || []).map((it: any) => itemMap.get(String(it.id))).filter(Boolean)
        const reason = styleJudgeReason(resolved, style, occasion, weather, preferences?.lockedAnchorId,preferences)
        if (reason || resolved.length !== (proposal.items || []).length) {
            reject(reason || 'unknown_item_id')
            void 0
            continue
        }
        // Only selected IDs come from the model; every validation stage sees
        // the same authoritative metadata that will be returned to the client.
        const structural = fourSlotOutfitReason(resolved, layered)
        if (structural) {
            reject(structural)
            void 0
            continue
        }
        const o = { ...proposal, items: resolved }
        const v = validateOutfitAgainstStyle(o, isWorkoutOccasion(occasion) ? 'sport' : style, itemMap, layered, preferences)
        if (!v.ok) {
            reject(v.reason || 'invalid_style_or_slots')
            void 0
            continue
        }
        const c = validateOutfitCompatibility(o, itemMap, isWorkoutOccasion(occasion) ? 'sport' : style)
        if (!c.ok) {
            reject(c.reason || 'incompatible_items')
            void 0
            continue
        }
        const budgetCap = preferences?.maxShopBudget
        if (budgetCap && typeof budgetCap.amount === 'number') {
            const spend = outfitShopSpend(o.items || [], itemMap, budgetCap.currency, budgetCap.fx)
            if (!fitsBudgetAmount(spend, budgetCap.amount)) {
            reject(`shop spend ${spend} exceeds ${budgetCap.amount} ${budgetCap.currency || ''}`)
                void 0
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
            reject('swim bottoms are not allowed for this occasion')
                void 0
                continue
            }
            // Driving moccasins are a tier below loafers — never work wear.
            let hasDriverShoe = false
            for (const it of (o.items || [])) {
                const src = itemMap.get(String(it.id)) || {}
                if (isDriverShoeItem({ ...src, ...it })) { hasDriverShoe = true; break }
            }
            if (hasDriverShoe) {
            reject('driving moccasins are not allowed for this occasion')
                void 0
                continue
            }
        }
        const ids = (o.items || []).map((item: any) => String(item.id)).sort()
        const signature = ids.join('|')
        if (!signature || signatures.has(signature)) {
            reject('duplicate or empty outfit combination')
            void 0
            continue
        }
        if (shopOnly) {
            const allShop = (o.items || []).length > 0 && (o.items || []).every((item: any) => {
                const source = itemMap.get(String(item.id)) || item
                return Boolean(source.isShopItem) || String(source.id || item.id || '').startsWith('shop_')
            })
            if (!allShop) {
            reject('shop-only requests cannot include owned wardrobe items')
                void 0
                continue
            }
        }
        if (hasOwnedCandidates) {
            const usesOwnedItem = (o.items || []).some((item: any) => {
                const source = itemMap.get(String(item.id)) || item
                return !source.isShopItem && !String(source.id || item.id || '').startsWith('shop_')
            })
            if (!usesOwnedItem) {
            reject('use a compatible owned item when wardrobe candidates are available')
                void 0
                continue
            }
        }
        signatures.add(signature)
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
    // Product titles describe the garment; descriptions may mention styling
    // companions (e.g. a navy shirt suggested with white trousers).
    for (const evidence of [row?.color, row?.primary_color, row?.name, row?.description]) {
        const blob = String(evidence || '').toLowerCase().replace(SHOP_BOILERPLATE_RE, ' ')
        const words = SHOP_COLOR_WORDS.join('|')
        const match = blob.match(new RegExp(`\\b(?:(?:dark|light|pale|muted|dusty|pastel|bright|neon|electric|vivid|fluorescent)[ -]+)?(?:${words})\\b`))
        if (match) return match[0]
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
class CatalogLookupError extends Error {
    constructor() { super('catalog_lookup_unavailable') }
}

async function fetchShopItemsForAllSlotsEdge(
    supabaseClient:any, requiredSlots:string[], style:string, perSlot=12,
    excludeSet?:Set<string>, eligibility?:(item:any,discovered?:any[])=>boolean,
    budgetFx?:BudgetFxContext,nativeBudget?:{amount:number;currency:string},requestIntent='',
    _recycleTarget=0,historyPriority?:(items:any[])=>readonly number[],
    _acceptsLayeredTop?:(layer:any,top:any)=>boolean,generationSeed=crypto.randomUUID(),
    context=stylingContext({style,occasion:requestIntent}), trace?:GenerationTrace,
):Promise<any[]> {
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),12000)
    const picks:any[]=[]
    try {
        for(const slot of requiredSlots) {
            const eligible:any[]=[]
            const cap=context.budget || nativeBudget
            const retrievalContext:Record<string,unknown>={dressCode:context.dressCode,
                preferredSubtypes:STYLING_PROFILES[context.style]?.preferred || []}
            if(cap){retrievalContext.budget=cap.amount;retrievalContext.currency=cap.currency}
            if(budgetFx&&cap)retrievalContext.rates=budgetFx.snapshot.uzsPerUnit
            for(let page=0;page<2;page++) {
                const pending=supabaseClient.rpc('catalog_discovery_candidates',{
                    p_slot:slot,p_source:'all',p_search:'',p_seed:generationSeed,p_limit:100,p_offset:page*100,
                    p_currency:nativeBudget?.currency||null,p_max_price:nativeBudget?.amount??null,p_context:retrievalContext,
                } satisfies CatalogCandidateRpcRequest).abortSignal(controller.signal)
                const {data,error}=trace?await trace.wait('catalog',()=>Promise.resolve(pending)):await pending
                if(error||!Array.isArray(data))throw new CatalogLookupError()
                const filterStart=performance.now()
                if(trace)trace.counts.catalogRows=(trace.counts.catalogRows||0)+data.length
                for(const row of data) {
                    const {fashion_intelligence:record,...source}=row
                    const item={...source,id:`shop_${row.id}`,isShopItem:true,isWardrobe:false,
                        macroCategory:row.fashion_slot || slot,type:row.garment_type||slot,
                        imageUrl:row.image_url,image:row.image_url,shopUrl:row.source_url,color:inferShopColor(row)}
                    const cached=record?.version===FASHION_INTELLIGENCE_VERSION&&record.descriptor?.slot===slot?record.descriptor:undefined
                    const prepared=prepareGarment(item,cached)
                    if(trace&&cached)trace.counts.cachedDescriptors=(trace.counts.cachedDescriptors||0)+1
                    if(item.macroCategory!==slot || (eligibility&&!eligibility(prepared,picks)))continue
                    eligible.push(prepared)
                }
                if(trace)trace.syncMs.catalog_filter=(trace.syncMs.catalog_filter||0)+performance.now()-filterStart
                if(data.length<100||countDistinctProducts(eligible)>=perSlot)break
            }
            // Rotation ranks the admitted pool; it never deletes the only
            // compatible bridge garment before complete-look feasibility.
            eligible.sort((a,b)=>compareRotationOrder(historyPriority?.([b])||[],historyPriority?.([a])||[]))
            const ranked=stylingCandidateRepresentatives(eligible,item=>item,item=>scoreGarmentForContext(item,context),context,Math.max(perSlot,12))
            const floors=cap?[...eligible].sort((a,b)=>shopPriceOf(a,cap.currency,budgetFx)-shopPriceOf(b,cap.currency,budgetFx)).slice(0,1):[]
            const keys=new Set<string>()
            for(const item of [...floors,...ranked]) {
                const own=productIdentityKeys(item);if(own.some(key=>keys.has(key)))continue
                own.forEach(key=>keys.add(key));picks.push(item)
                if(picks.filter(row=>row.macroCategory===slot).length>=Math.max(perSlot,12))break
            }
        }
        return picks
    } catch {throw new CatalogLookupError()}finally{clearTimeout(timer)}
}

function requiredShopSupplementSlots(
    items: any[],
    requiredSlots: string[],
    limit: number,
    selectedItemIds: string[] = [],
    eligibility?: (item: any) => boolean,
): { slots: string[]; targetChoicesPerSlot: number } {
    const manualSelection = Array.isArray(selectedItemIds) && selectedItemIds.length > 0
    // Floor of 6/slot: tiny per-slot fetches plus same-product collapsing
    // starved decks to ~7 distinct items (Jaccard 1.0 across taps, measured
    // Oct 2026). Fetching deeper is cheap; rotation needs real choice.
    const targetChoicesPerSlot = manualSelection
        ? 1
        : (limit >= 10 ? 16 : items.some(item => !isShopRow(item)) ? 6 : 12)
    const countBySlot = new Map<string, number>()
    const seenProducts = new Set<string>()

    for (const item of items) {
        if (eligibility && !eligibility(item)) continue
        const slot = String(item.macroCategory || '').toLowerCase()
        if (!slot) continue
        const key = `${slot}:${isShopRow(item) ? canonicalProductKey(item) : item.id}`
        if (seenProducts.has(key)) continue
        seenProducts.add(key)
        countBySlot.set(slot, (countBySlot.get(slot) || 0) + 1)
    }

    return {
        targetChoicesPerSlot,
        slots: requiredSlots.filter((slot) => (countBySlot.get(slot) || 0) < targetChoicesPerSlot),
    }
}



function safeParseOutfits(raw: string): any[] | null {
    try {
        const parsed = JSON.parse(raw)
        if (Array.isArray(parsed?.outfits) && parsed.outfits.length > 0) return normalizeModelOutfits(parsed.outfits)
        if (Array.isArray(parsed) && parsed.length > 0) return normalizeModelOutfits(parsed)
    } catch (_) { /* try regex fallback */ }
    // Regex fallback: extract JSON object containing "outfits"
    const match = raw.match(/\{[\s\S]*"outfits"\s*:\s*\[[\s\S]*\][\s\S]*\}/)
    if (match) {
        try {
            const parsed = JSON.parse(match[0])
            if (Array.isArray(parsed.outfits)) return normalizeModelOutfits(parsed.outfits)
        } catch (_) { /* give up */ }
    }
    // Try extracting just the array
    const arrMatch = raw.match(/\[[\s\S]*\]/)
    if (arrMatch) {
        try {
            const arr = JSON.parse(arrMatch[0])
            if (Array.isArray(arr) && arr.length > 0 && arr[0].items) return normalizeModelOutfits(arr)
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

/**
 * PROVIDER_ROUTING (explicit, single source of truth)
 * ────────────────────────────────────────────────────
 * Outfit generation is NVIDIA-only. There is deliberately NO second AI
 * provider: a Gemini implementation existed but was never reachable, and
 * dead provider-routing branches mislead engineers into believing a
 * fallback protects production. It has been removed.
 *
 *   • Primary provider .............. NVIDIA (one model for every tier)
 *   • Retry policy .................. 1 retry for TRANSIENT failures only
 *                                   (HTTP 429, HTTP 5xx, network error).
 *                                   2s backoff after 429/5xx, 1.5s after a
 *                                   network error. 4xx (other than 429) and
 *                                   unparseable bodies are NOT retried here.
 *   • Timeout policy ................ 45s per HTTP attempt, covering the
 *                                   response body read as well.
 *   • Second provider .............. none
 *   • On provider failure .......... the DETERMINISTIC fallback below is
 *                                   used. Its candidates must pass the SAME
 *                                   independent validation as AI output and
 *                                   the SAME global deck selector. If they
 *                                   do not, the client receives a
 *                                   retryable 503 `ai_generation_unavailable`.
 *   • Retryable client errors ....... 503 `ai_generation_unavailable`,
 *                                   503 `catalog_lookup_unavailable`,
 *                                   503 `budget_rates_unavailable`
 *   • Non-retryable client errors ... 401/403 (auth), 422 (no matching
 *                                   clothing / invalid request)
 */
async function callNvidia(key: string, model: string, sys: string, usr: string, retries = 0, trace?: ProviderAttempt[], layered = false, outputContract?: string): Promise<any[]|null> {
    for (let attempt = 0; attempt <= retries; attempt++) {
        const controller = new AbortController()
        // Two bounded attempts (including quality correction) fit inside the
        // client's 120-second budget with room for catalog queries.
        let timedOut = false
        let timer: ReturnType<typeof setTimeout>
        const deadline = new Promise<never>((_, reject) => {
            timer = setTimeout(() => {
                timedOut = true
                reject(new Error('nvidia_request_timeout'))
                controller.abort()
            }, 45_000)
        })
        const withinDeadline = <T,>(operation: Promise<T>) => Promise.race([operation, deadline])
        try {
            void 0
            const r = await withinDeadline(fetch('https://integrate.api.nvidia.com/v1/chat/completions', {
                method:'POST', headers:{'Content-Type':'application/json','Authorization':`Bearer ${key}`},
                body: JSON.stringify({ model, ...(model.includes('nemotron') ? { chat_template_kwargs: { enable_thinking: false } } : {}), messages:[{role:'system',content:sys + (outputContract || ('\nOUTPUT CONTRACT: Return ' + JSON.stringify({outfits:[{description:'brief explanation',items:(layered ? ['BASE TOP','OUTERWEAR','BOTTOM','SHOES'] : ['TOP','BOTTOM','SHOES']).map(slot=>({id:'EXACT '+slot+' CANDIDATE ID'}))}]}) + '. Every item must contain the id field as a string copied exactly from the candidate list. The example lists required garment slots; replace every placeholder with a real candidate ID for that slot. Never replace a required slot with another top. A dress replaces top and bottom: choose dress plus shoes, and add outerwear when layering is required; never put a separate bottom with a dress. Include additional selected pieces only when requested and compatible; never layer two pullover knits. Do not omit required slots, return names instead of IDs, or invent IDs. Item metadata is attached by the server; only IDs are needed in items.'))},{role:'user',content:usr}], temperature:0.25, max_tokens:outputContract ? 700 : 2000, frequency_penalty:0.6, presence_penalty:0.4, response_format:{type:'json_object'} }),
                signal: controller.signal,
            }))
            if (r.status === 429 || r.status >= 500) {
                void 0
                trace?.push({ provider: 'nvidia', model, status: String(r.status) })
                if (attempt < retries) { await new Promise(r => setTimeout(r, 2000)); continue }
                return null
            }
            if (!r.ok) { void 0; trace?.push({ provider: 'nvidia', model, status: String(r.status) }); return null }
            const d = await withinDeadline(r.json())
            const content = d.choices?.[0]?.message?.content || '{}'
            void 0
            const outfits = (outputContract ? parseRankedOutfitChoices(content) : null) || safeParseOutfits(content)
            if (outfits && outfits.length > 0) {
                void 0
                trace?.push({ provider: 'nvidia', model, status: 'ok' })
                return outfits
            }
            void 0
            trace?.push({ provider: 'nvidia', model, status: 'unparseable' })
            return null
        } catch(e) {
            void 0
            trace?.push({ provider: 'nvidia', model, status: timedOut ? 'timeout' : e instanceof SyntaxError ? 'invalid_response' : 'network_error' })
            if (attempt < retries) { await new Promise(r => setTimeout(r, 1500)); continue }
            return null
        } finally {
            // Reading a response body is part of the provider deadline too.
            clearTimeout(timer!)
        }
    }
    return null
}

// ── Anchor-item enforcement ───────────────────────────────────────────────
// After the AI (or local fallback) selects outfit items, lock the anchor item
// into its matching slot in every outfit. If no slot matches the anchor's
// macroCategory, the anchor is prepended to the items array.
function enforceAnchorInOutfits(outfits: any[], anchorDbItem: any): any[] {
    // Correct a mistagged closet slot from the product name (trousers stored
    // as Tops) so the anchor replaces the right slot, not a shirt.
    const anchorMacro = detectSlotMismatch(anchorDbItem.macroCategory || macroCategory(anchorDbItem.category || '', anchorDbItem.type || ''), anchorDbItem.name || anchorDbItem.type || '', anchorDbItem.type || '')
        || macroCategory(anchorDbItem.category || '', anchorDbItem.type || '')
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

async function generateOutfitsHandler(req: Request, trace: GenerationTrace) {
    if (req.method === 'OPTIONS') {
        return new Response('ok', { headers: corsHeaders })
    }

    if (req.method !== 'POST') return new Response(JSON.stringify({success:false,outfits:[],error:'method_not_allowed'}), {status:405,headers:{...corsHeaders,'Content-Type':'application/json'}})

    const rateLimitResponse = await enforceRateLimit(req, { feature: 'generate-outfits', limit: 10, headers: corsHeaders })
    if (rateLimitResponse) return rateLimitResponse


    const SUPABASE_URL = Deno.env.get('SUPABASE_URL')!
    const SUPABASE_ANON_KEY = Deno.env.get('SUPABASE_ANON_KEY')!
    const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!

    let body: any
    try { body = await readOutfitGenerationRequest(req) } catch (error) {
        return new Response(JSON.stringify({ success: false, outfits: [], error: 'invalid_generation_request' }), {
            status: error instanceof OutfitRequestError ? error.status : 400,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        })
    }

    let {
        prompt = '',
        stylePreferences = 'Casual',
        occasion = 'Everyday',
        weather,
        limit = 3,
        // Deck size the caller will actually DISPLAY. The Personal Stylist
        // requests a few extra candidates (limit) but shows three cards
        // (deckTarget). Capacity must be judged against the DISPLAYED deck
        // size, otherwise a 5-candidate request would mark a 3-shoe slot
        // "thin" and license repeating shoes on the three visible cards.
        deckTarget,
        selectedItemIds = [],
        wardrobeItems: legacyWardrobe = [],
        useProvidedWardrobeOnly = false,
        catalogOnly = false,
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
        // Item ids the user ALREADY SAW in recent decks (rolling window,
        // maintained client-side). Excluded from the candidate pool so
        // regenerate taps bring NEW pieces instead of recombining the same
        // ~40 items forever. Manual selection (selectedItemIds) ignores this.
        excludeItemIds = [],
        excludeProductKeys = [],
        previousDeckProductKeys = [],
        olderRecentProductKeys = [],
        rotationDecks = [],
        previousShoeProductKeys,
        recentShoeProductKeys = [],
        shoeRotationDecks,
    } = body

    if((catalogOnly&&anchorItemId&&!String(anchorItemId).startsWith('shop_'))||(wardrobeOnly&&String(anchorItemId||'').startsWith('shop_')))return new Response(JSON.stringify({success:false,outfits:[],code:'no_matching_clothing',error:'The locked item is outside the selected source.'}),{status:422,headers:{...corsHeaders,'Content-Type':'application/json'}})

    if (catalogOnly) {
        legacyWardrobe = []
        useProvidedWardrobeOnly = true
        wardrobeOnly = false
    }
    const retrievalSeed = body.generationRequestId || (deckSalt >= 0 ? `replay:${deckSalt}` : crypto.randomUUID())
    weather = resolveStylistWeather(prompt, weather)
    const selectionStyle = isWorkoutOccasion(`${occasion} ${prompt}`) ? 'sport' : stylePreferences

    // Cap the window: 200 ids ≈ 15+ decks. Unbounded arrays would bloat
    // the request body and risk excluding so much that a slot starves.
    const excludeSet: Set<string> = normalizeProductHistory(
        (Array.isArray(excludeItemIds) ? excludeItemIds : [])
            .slice(0, 200)
            .map((v: unknown) => String(v ?? '').trim())
            .filter(Boolean),
    )
    for (const key of [...excludeProductKeys,...previousDeckProductKeys,...olderRecentProductKeys]) excludeSet.add(key)
    for (const key of [...recentShoeProductKeys,...(previousShoeProductKeys || [])]) excludeSet.add(key)
    // Reconstruct history when a caller sends deck boundaries but omits the
    // flattened arrays. Never let that payload shape reset consecutive rotation.
    for(const deck of rotationDecks)for(const key of deck.productKeys)excludeSet.add(key)
    for(const key of normalizeProductHistory(excludeSet))excludeSet.add(key)
    const previousSet = normalizeProductHistory(previousDeckProductKeys.length ? previousDeckProductKeys : rotationDecks.at(-1)?.productKeys || [])
    const previousShoeSet = normalizeProductHistory(previousShoeProductKeys ?? previousSet)
    const deckPriority = createShoeRotationPriority(excludeSet,previousSet,rotationDecks,anchorItemId,
        shoeRotationDecks ? {previous:previousShoeSet,decks:shoeRotationDecks} : undefined)
    const rotationTarget = deckTarget === 3 ? 3 : 0

    let parsedBudget: { amount: number; currency: string; fx?: BudgetFxContext } | null
    let requestFx: BudgetFxContext | undefined
    const loadRequestFx = async () => {
        if (!requestFx) {
            try {
                const admin = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {auth: {persistSession: false, autoRefreshToken: false}})
                const id = body.preferences?.budgetRatesId
                const snapshot = id === undefined ? await currentBudgetRates(admin) : await budgetRatesById(admin, id)
                requestFx = budgetFxContext(snapshot)
            } catch { throw new BudgetFxError() }
        }
        return requestFx
    }
    try {
        const selected = strictestOutfitBudget(body.preferences?.maxShopBudget, null, budgetInCurrencyEdge)
        const candidates = budgetCandidatesFromTextEdge(`${prompt} ${occasion}`)
        const needsFx = new Set([...candidates.map(b => b.currency), ...(selected ? [selected.currency] : [])]).size > 1
        const hasPinnedBudgetRatesId = body.preferences?.budgetRatesId !== undefined
        if (hasPinnedBudgetRatesId || needsFx) {
            try { await loadRequestFx() } catch (fxError) {
                // Rates are only required to CONVERT. A single-currency
                // budget (e.g. explicit "$250") must still be enforced in
                // its own currency when no reference snapshot was requested.
                if (!needsFx && !hasPinnedBudgetRatesId) requestFx = undefined
                else throw fxError
            }
        }
        parsedBudget = strictestOutfitBudget(selected, parseBudgetFromText(`${prompt} ${occasion}`, requestFx),
            (budget, currency) => budgetInCurrencyEdge({...budget, fx: requestFx}, currency))
        if (parsedBudget && requestFx) parsedBudget.fx = requestFx
    } catch (error) {
        const unavailable = error instanceof BudgetFxError
        return new Response(JSON.stringify({ success: false, error: unavailable ? 'budget_rates_unavailable' : 'invalid_outfit_budget', outfits: [] }), {
            status: unavailable ? 503 : 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        })
    }
    const preferences = parsedBudget
        ? { ...(body.preferences || {}), maxShopBudget: parsedBudget, lockedAnchorId: anchorItemId }
        : { ...(body.preferences || {}), lockedAnchorId: anchorItemId }

    try {
        // ── 1. Authenticate user via JWT ──────────────────────────────────
        const authHeader = req.headers.get('Authorization') || ''
        let dbClothingItems: any[] = []
        let requestUserId: string | null = null

        const authReply = (status: number) => new Response(JSON.stringify({
            success: false, outfits: [], error: status === 401 ? 'unauthorized' : 'auth_unavailable',
        }), {status, headers: {...corsHeaders, 'Content-Type': 'application/json'}})
        if (!/^Bearer\s+\S+$/i.test(authHeader.trim())) return authReply(401)
        const userSupabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
            global: { headers: { Authorization: authHeader } }
        })
        let user: any = null
        try {
            const result = await userSupabase.auth.getUser()
            if (result.error) {
                const status = Number(result.error.status)
                return authReply(!status || status >= 500 ? 503 : 401)
            }
            user = result.data?.user
        } catch { return authReply(503) }
        if (!user?.id) return authReply(401)
        requestUserId = user.id

        // Worn-item cooldown is a hard constraint, independent of the softer
        // displayed-product rotation and its exhaustion fallback.
        const { data: recentWearLogs, error: wearHistoryError } = await userSupabase
            .from('wear_logs').select('item_ids').eq('user_id', requestUserId)
            .order('created_at', { ascending: false }).order('id', { ascending: false }).limit(5)
        if (wearHistoryError || !Array.isArray(recentWearLogs)) throw new CatalogLookupError()
        const wornItemIds = new Set<string>(recentWearLogs.flatMap((log: any) =>
            Array.isArray(log.item_ids) ? log.item_ids.filter((id: any) => typeof id === 'string') : []))

        // Provided clothing controls the data source, never authentication.
        if (!useProvidedWardrobeOnly) {
            if (user) {
                // ── 2. Fetch clothing items from DB ───────────────────────
                // NOTE: live schema uses `occasions` (JSONB) + `seasons`
                // (JSONB), not singular `occasion`. Selecting a missing column
                // makes PostgREST fail the whole query → empty wardrobe.
                const rows: any[] = []
                const pageSize = 500
                for (let offset = 0; ; offset += pageSize) {
                    let q = userSupabase
                        .from('clothing_items')
                        .select('id, type, category, color, style, brand, season, seasons, occasions, pattern, material, image_url, description, primary_color, sub_category, wear_count, name')
                        .eq('user_id', requestUserId)
                        .order('id', { ascending: true })
                        .range(offset, offset + pageSize - 1)
                    if (selectedItemIds && selectedItemIds.length > 0) q = q.in('id', selectedItemIds)
                    const { data, error } = await q
                    if (error || !Array.isArray(data)) throw new CatalogLookupError()
                    rows.push(...data)
                    if (data.length < pageSize) break
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

        if (anchorItemId && useProvidedWardrobeOnly && !wardrobeItems.some((item: any) => String(item.id) === String(anchorItemId))) {
            const {data: anchor, error: anchorError} = await userSupabase.from('clothing_items')
                .select('id,name,type,sub_category,category,primary_color,brand,description,image_url')
                .eq('id', anchorItemId).eq('user_id', requestUserId).maybeSingle()
            if (anchorError) throw new CatalogLookupError()
            if (!anchor) return new Response(JSON.stringify({success:false,outfits:[],error:'anchor_not_found'}), {status:422,headers:{...corsHeaders,'Content-Type':'application/json'}})
            wardrobeItems.push({...anchor, type: anchor.type || anchor.sub_category || anchor.category,
                macroCategory: macroCategory(anchor.category || '', `${anchor.type || ''} ${anchor.sub_category || ''}`, anchor.name || ''),
                color:anchor.primary_color || 'neutral', imageUrl:anchor.image_url, isWardrobe:true, isShopItem:false})
        }

        // ── 3b. Attach inferred style metadata to every item, regardless of
        // source. If the client already shipped styleTags (shop payload does),
        // we keep theirs; otherwise we compute from name/description/brand.
        trace.stage='catalog_intelligence'
        wardrobeItems = wardrobeItems.map((item: any) => {
            // Repair mistagged closet slots from the product name before any
            // filtering (trousers filed under Tops broke wardrobe generation).
            const fixedMacro = detectSlotMismatch(item.macroCategory, item.name || item.type || '', item.type || '')
            const withSlot = fixedMacro ? { ...item, macroCategory: fixedMacro } : item
            const hasTags = Array.isArray(withSlot.styleTags) && withSlot.styleTags.length > 0
            if (hasTags && withSlot.material !== undefined) return withSlot
            const inferred = inferAttrsEdge(withSlot)
            return {
                ...withSlot,
                styleTags: hasTags ? withSlot.styleTags : inferred.styleTags,
                material: withSlot.material || inferred.materials,
                pattern: withSlot.pattern,
                formality: typeof withSlot.formality === 'number' ? withSlot.formality : inferred.formality,
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
                return !avoid.some((c: string) => color.includes(c) || name.includes(` ${c} `) || desc.includes(` ${c} `))
            })
            void 0
        }

        // Variety changes candidates, never the request constraints.
        wardrobeItems = wardrobeItems.filter((it: any) => rotationTarget ||
            String(it.id) === String(anchorItemId || '') ||
            (selectedItemIds || []).includes(String(it.id)) || !excludeSet.has(String(it.id)))

        trace.stage='catalog_retrieval'
        wardrobeItems=wardrobeItems.map((item:any)=>prepareGarment(item))
        const qualityContext=stylingContext({style:stylePreferences,occasion,request:prompt,weather,anchorId:anchorItemId,preferences,sourceMode:wardrobeOnly?'wardrobe':catalogOnly?'catalog':'mixed'});
        const eligibleCandidate = (item: any) => !garmentContextReason(item,qualityContext) && !wornItemIds.has(String(item.id)) && !wornItemIds.has(`shop_${item.id}`) && isFourSlotCandidate(item) && !isNonOutfitItem(item) && !rejectForStyleOccasion(item, String(item.id) === String(anchorItemId || '') ? 'casual' : selectionStyle, `${occasion} ${prompt}`)
            && !rejectForClimate(item, weather) && !rejectForRequestIntent(item, `${occasion} ${prompt}`, weather) && !rejectForActivity(item, `${occasion} ${prompt}`)
        const requiredIds = new Set([...(selectedItemIds || []).map(String), ...(anchorItemId ? [String(anchorItemId)] : [])])
        const blockedSelectedItemIds = wardrobeItems.filter((item: any) => requiredIds.has(String(item.id)) && !eligibleCandidate(item)).map((item: any) => String(item.id))
        if (blockedSelectedItemIds.length) return new Response(JSON.stringify({ success: false, outfits: [], code: 'no_matching_clothing',
            error: 'A selected or locked item is incompatible with the requested style, occasion or weather.', blockedSelectedItemIds }), {
            status: 422, headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        })
        if (isWorkoutOccasion(`${occasion} ${prompt}`)) {
            const blockedAnchor = wardrobeItems.some((item: any) => String(item.id) === String(anchorItemId || '') && rejectForActivity(item, `${occasion} ${prompt}`))
            if (blockedAnchor) return new Response(JSON.stringify({ success: false, outfits: [], code: 'no_matching_clothing', error: 'The locked item is not suitable for this activity.' }), {
                status: 422, headers: { ...corsHeaders, 'Content-Type': 'application/json' },
            })
            wardrobeItems = wardrobeItems.filter((item: any) => !rejectForActivity(item, `${occasion} ${prompt}`))
        }
        wardrobeItems = wardrobeItems.filter(eligibleCandidate)

        // ── 3c. Supplement sparse wardrobe slots from the shop ─────────────
        // Wardrobe pieces remain in the pool, but a 15-card daily deck needs
        // several choices per slot. If the user owns only a few garments, add
        // style-ranked shop catalog items so the combinations can actually vary.
        // Shop-only decks (Team Collaboration) fetch every slot from the shop.
        // wardrobeOnly requests skip this entirely — Supabase clothing only.
        if (!wardrobeOnly) {
            let requiredSlots = ['top', 'bottom', 'shoes']
            if (needsLayering(stylePreferences, weather, prompt)) requiredSlots.push('outerwear')
            const lockedSlot = wardrobeItems.find((item:any)=>String(item.id)===String(anchorItemId || ''))?.macroCategory
            if (lockedSlot) requiredSlots = requiredSlots.filter(slot=>slot!==lockedSlot)
            const shopOnlyRequest = isShopOnlyDeck(stylePreferences, occasion)
            const { slots: missingSlots, targetChoicesPerSlot } = shopOnlyRequest
                ? { slots: requiredSlots, targetChoicesPerSlot: limit >= 10 ? 16 : 8 }
                : requiredShopSupplementSlots(
                    wardrobeItems,
                    requiredSlots,
                    limit,
                    selectedItemIds,
                    item => eligibleCandidate(item) && (!rotationTarget || !wasRecentlyShown(item, excludeSet)) && (!parsedBudget || !isShopPricedItem(item)),
                )
            try {
                const svcClient = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
                if (parsedBudget && missingSlots.length > 0) {
                    try { parsedBudget.fx = await loadRequestFx() }
                    catch (error) {
                        if (body.preferences?.budgetRatesId || new Set(budgetCandidatesFromTextEdge(`${prompt} ${occasion}`).map(b => b.currency)).size > 1) throw error
                        // Same-currency prices require no exchange rate. Exclude
                        // foreign quotes rather than inventing a conversion.
                        void 0
                    }
                    preferences.maxShopBudget = parsedBudget
                }
                // A banned choice must not count toward a completed page or
                // consume a per-slot pick while later eligible products exist.
                const lockedItem = wardrobeItems.find((item:any)=>String(item.id)===String(anchorItemId || ''))
                // Fetch companions that can actually be worn with the locked
                // garment. Otherwise a page of individually valid blazers can
                // hide the cardigans needed by a piqué polo, leaving no deck.
                const shopEligible = (item: any, discovered: any[] = []) => {
                    if (!eligibleCandidate(item) || filterShopByPrefs([item], preferences).length !== 1
                        || (lockedItem && rejectOutfitIncoherence([lockedItem, item], selectionStyle, `${occasion} ${prompt}`))) return false
                    // Count compatible tops, not merely individually valid
                    // knits that cannot layer with any admitted outerwear.
                    const layers=[...wardrobeItems,...discovered].filter((layer:any)=>layer.macroCategory==='outerwear')
                    return !lockedItem || item.macroCategory!=='top' || !layers.length || layers.some((layer:any)=>
                        !rejectOutfitIncoherence([...(lockedItem.macroCategory==='outerwear' ? [] : [lockedItem]),layer,item],selectionStyle,`${occasion} ${prompt}`))
                }
                const discoverySlots=lockedItem ? [...missingSlots].sort((a,b)=>Number(b==='outerwear')-Number(a==='outerwear')) : missingSlots
                const fetchedShopItems = await fetchShopItemsForAllSlotsEdge(svcClient, discoverySlots, selectionStyle, targetChoicesPerSlot, excludeSet, shopEligible, parsedBudget ? requestFx : undefined, parsedBudget && !requestFx ? parsedBudget : undefined, `${occasion} ${prompt}`, rotationTarget, deckPriority,
                    lockedItem ? (layer,top)=>!rejectOutfitIncoherence([...(lockedItem.macroCategory==='outerwear' ? [] : [lockedItem]),layer,top],selectionStyle,`${occasion} ${prompt}`) : undefined, retrievalSeed,{...qualityContext,budget:parsedBudget||undefined},trace)
                // Shop candidates must respect color/pattern prefs too — the
                // wardrobe-side filter above ran before they were appended.
                const shopItems = filterShopByPrefs(fetchedShopItems, preferences)
                if (String(selectionStyle).toLowerCase().replace(/[ -]+/g,'_') === 'old_money' && /\bdate\b/i.test(`${occasion} ${prompt}`)) {
                    const freshLayers = [...wardrobeItems,...shopItems].filter(item=>item.macroCategory==='outerwear' && !wasRecentlyShown(item,excludeSet));
                    const families = new Set(freshLayers.map(garmentArchetypeToken));
                    if (freshLayers.length >= 3 && families.size === 1) {
                        // Same admitted family can dominate an early catalog page.
                        // Reuse the existing paginated fetch with a quality eligibility
                        // predicate; pagination and product-history rules stay unchanged.
                        const alternatives = await fetchShopItemsForAllSlotsEdge(svcClient,['outerwear'],selectionStyle,3,excludeSet,
                            item=>shopEligible(item) && !families.has(garmentArchetypeToken(item)),parsedBudget?requestFx:undefined,
                            parsedBudget&&!requestFx?parsedBudget:undefined,`${occasion} ${prompt}`,rotationTarget,deckPriority,undefined,retrievalSeed)
                        shopItems.push(...filterShopByPrefs(alternatives,preferences))
                    }
                    const layers = spreadQualityFamilies(shopItems.filter(item=>item.macroCategory==='outerwear'),garmentArchetypeToken)
                    shopItems.splice(0,shopItems.length,...layers,...shopItems.filter(item=>item.macroCategory!=='outerwear'))
                }

                if (shopItems.length > 0) {
                    void 0
                    wardrobeItems = [...wardrobeItems, ...shopItems]
                }
            } catch (error) {
                if (error instanceof CatalogLookupError || error instanceof BudgetFxError) throw error
                void 0
            }
        } else {
            void 0
        }
        const fetchedShoeIds = wardrobeItems.filter((item:any)=>item.macroCategory==='shoes').map((item:any)=>item.id)
        // Recently shown is a planning preference, never a pre-plan deletion.

        // ── 3c2. Shop-only decks (Team Collaboration) are 100% catalog ────
        // The supplement above APPENDS shop rows to the wardrobe pool, which
        // left the user's own garments visible to the LLM as "Available
        // wardrobe items". The prompt forbids using them, but a model that
        // disobeys (or a validation pass that falls back to raw output) put
        // wardrobe pieces on a card labelled CURATED LOOK. Drop owned rows
        // from the pool entirely — before ranking, budget, itemMap and prompt
        // build — so there is nothing to leak.
        if (isShopOnlyDeck(stylePreferences, occasion) && !anchorItemId) {
            const beforeShopOnly = wardrobeItems.length
            wardrobeItems = wardrobeItems.filter((item: any) => isShopRow(item))
            void 0
        }

        if (isWorkoutOccasion(`${occasion} ${prompt}`)) {
            wardrobeItems = wardrobeItems.filter((item: any) => !rejectForActivity(item, `${occasion} ${prompt}`))
        }

        // Keep the user's original cap currency; compare each listed quote
        // using one server-validated reference snapshot for this request.
        if (parsedBudget && wardrobeItems.some((item: any) => isShopPricedItem(item) &&
            Number.isFinite(Number(item.price)) && Number(item.price) > 0 &&
            ['USD','EUR','GBP','UZS'].includes(String(item.currency || '').trim().toUpperCase()) &&
            String(item.currency || '').trim().toUpperCase() !== parsedBudget!.currency)) {
            try { await loadRequestFx() }
            catch (error) {
                if (body.preferences?.budgetRatesId) throw error
                wardrobeItems = wardrobeItems.filter((item: any) => !isShopPricedItem(item) || String(item.currency || '').toUpperCase() === parsedBudget!.currency)
                void 0
            }
            parsedBudget.fx = requestFx
            preferences.maxShopBudget = parsedBudget
        }
        if (parsedBudget && !wardrobeItems.some(isDressGarment)) {
            const layeredBudget = needsLayering(stylePreferences, weather, prompt)
            const requiredSlots = ['top','bottom','shoes', ...(layeredBudget ? ['outerwear'] : [])]
            const cost = (item: any) => {
                if (!isShopPricedItem(item)) return 0
                const value = shopPriceOf(item, parsedBudget!.currency, parsedBudget!.fx)
                return Number.isFinite(value) ? value : null
            }
            const minimum = minimumSlotSpend(wardrobeItems, requiredSlots, cost)
            const anyAffordable = wardrobeItems.some((item: any) => {
                const value = cost(item)
                return value !== null && fitsBudgetAmount(value, parsedBudget!.amount)
            })
            if (anyAffordable && minimum !== null && !fitsBudgetAmount(minimum, parsedBudget.amount)) {
                return new Response(JSON.stringify({success:false,outfits:[],code:'no_outfit_within_budget',
                    error:'No complete outfit in the selected clothing pool fits this budget.',
                    budget:{amount:parsedBudget.amount,currency:parsedBudget.currency},minimumShopSpend:minimum}),
                    {status:422,headers:{...corsHeaders,'Content-Type':'application/json'}})
            }
        }
        let budgetFilteredAll = false
        const unverifiedBudgetItems: any[] = []
        if (parsedBudget) {
            const beforeBudget = wardrobeItems.length
            wardrobeItems = wardrobeItems.filter((item: any) => {
                if (!isShopPricedItem(item)) return true
                const p = shopPriceOf(item, parsedBudget!.currency, parsedBudget!.fx)
                if (!Number.isFinite(p)) { unverifiedBudgetItems.push(item); return false }
                if (p <= 0) return true
                return fitsBudgetAmount(p, parsedBudget!.amount)
            })
            budgetFilteredAll = beforeBudget > 0 && wardrobeItems.length === 0
            void 0
        }

        // ── 3d. Rank + trim the pool for the requested style so the LLM only
        // sees plausible candidates. In manual mode (selectedItemIds) we keep
        // the user's picks as-is. The client already pre-filters shop items,
        // so this mainly helps the wardrobe + legacy branches.
        // The locked anchor is EXEMPT from trimming: dropping it here is what
        // caused "locked item missing / locked to wrong slot" downstream.
        const styleKey = (selectionStyle || 'casual').toLowerCase().replace(/[\s-]+/g, '_')
        const anchorPreRank = anchorItemId
            ? wardrobeItems.find((i: any) => String(i.id) === String(anchorItemId))
            : undefined
        if (!selectedItemIds || selectedItemIds.length === 0) {
            const before = wardrobeItems.length
            const occKey = String(occasion || '').toLowerCase()
            const formalOcc = /\b(work|office|date|formal|wedding)\b/.test(occKey)
            const classicStyle = /old_money|classic|business/.test(styleKey)
            const tasteEligible = (it: any) => {
                if (!(formalOcc || classicStyle)) return true
                    const blob = `${it.name || ''} ${it.type || ''} ${it.description || ''}`.toLowerCase()
                    if (isDriverShoeItem(it)) return false
                    if (/\b(washed[\s-]?effect|bell[\s-]?bottom|splatter|uneven[\s-]?hem|twisted[\s-]?seams?)\b/.test(blob)) return false
                    // Date polos are admitted by the shared occasion judge
                    // and checked against their actual layer. Do not discard
                    // them again here and falsely mark the top slot sparse.
                    if (formalOcc && !/\bdate\b/.test(occKey) && /\bpiqu/.test(blob) && /\bpolo/.test(blob)) return false
                    return true
            }
            if (rotationTarget) wardrobeItems.sort((a:any,b:any)=>compareRotationOrder(deckPriority([b]),deckPriority([a])));
            const budgetSource = wardrobeItems
            const rankEligible = (item: any) => eligibleCandidate(item) && (String(item.id) === String(anchorItemId || '') || tasteEligible(item))
            wardrobeItems = rankItemsForStyleEdge(wardrobeItems, styleKey, { minKeep: 16, dropThreshold: -2, perCategoryFloor: 6, eligibility: rankEligible })
            // Keep style and activity on comparable scales: Classic + Travel
            // should not rank every cashmere blazer above a practical layer.
            const compareRankedCandidates = (a: any, b: any) => {
                const score = (item: any) => 0.6 * scoreGarmentForContext(item, qualityContext)
                    + 0.4 * styleInference.scoreItemForStyle(item, styleInference.normalizeStyleId(styleKey));
                return (rotationTarget ? compareRotationOrder(deckPriority([b]), deckPriority([a])) : 0)
                    || score(b) - score(a)
                    || (!parsedBudget ? compareUnbudgetedValue(a, b) : 0);
            }
            wardrobeItems.sort(compareRankedCandidates)
            wardrobeItems = collapseBySimEdge(wardrobeItems, (i: any) => canonicalProductKey(i))
            if (parsedBudget) {
                wardrobeItems = retainBudgetFloorCandidates(budgetSource, wardrobeItems,
                    item => shopPriceOf(item, parsedBudget!.currency, parsedBudget!.fx), rankEligible, anchorPreRank, 10, item=>scoreGarmentForContext(item,qualityContext),qualityContext)
            } else {
                // A global trim must not discard an entire required slot.
                wardrobeItems = retainBudgetFloorCandidates([...budgetSource].sort(compareRankedCandidates), wardrobeItems, () => 0, rankEligible, anchorPreRank, 10, item=>scoreGarmentForContext(item,qualityContext),qualityContext)
            }
            // Keep the 40-item prompt balanced: up to ten representatives
            // in each slot. Prepending shoes here truncated other slots again.
            wardrobeItems = wardrobeItems.slice(0, 40)
            if (anchorPreRank && !wardrobeItems.some((i: any) => String(i.id) === String(anchorItemId))) {
                void 0
                wardrobeItems = [anchorPreRank, ...wardrobeItems].slice(0, 40)
            }
            void 0
        } else {
            wardrobeItems = wardrobeItems.filter(eligibleCandidate)
        }

        void 0

        // Unknown cost is not proof that a budget is too low. Keep optional
        // unpriced rows out, but fail before provider work if a required or
        // locked piece cannot be priced and no valid composition remains.
        if (parsedBudget && unverifiedBudgetItems.length > 0) {
            const layeredBudget = needsLayering(stylePreferences, weather, prompt)
            const requiredBudgetSlots = ['top', 'bottom', 'shoes', ...(layeredBudget ? ['outerwear'] : [])]
            const hasBudgetSlot = (slot: string) => wardrobeItems.some((item: any) =>
                String(item.macroCategory || '').toLowerCase() === slot ||
                (slot === 'top' && !layeredBudget && String(item.macroCategory || '').toLowerCase() === 'outerwear'))
            const unverifiedRequiredSlot = !wardrobeItems.some(isDressGarment) && requiredBudgetSlots.some(slot =>
                !hasBudgetSlot(slot) && unverifiedBudgetItems.some(item => String(item.macroCategory || '').toLowerCase() === slot))
            const unverifiedAnchor = anchorItemId && unverifiedBudgetItems.some(item => String(item.id) === String(anchorItemId))
            if (wardrobeItems.length === 0 || unverifiedRequiredSlot || unverifiedAnchor) {
                return new Response(JSON.stringify({success: false, outfits: [], code: 'unverified_shop_pricing',
                    error: 'A required shop price or currency is unavailable. Choose items with verified prices.', budget: parsedBudget ? {amount: parsedBudget.amount, currency: parsedBudget.currency} : null}), {
                    status: 422, headers: {...corsHeaders, 'Content-Type': 'application/json'},
                })
            }
        }

        if (wardrobeItems.length > 0) {
            const required = [...(wardrobeItems.some(isDressGarment) ? [] : ['top', 'bottom']), 'shoes', ...(needsLayering(selectionStyle, weather, prompt) ? ['outerwear'] : [])]
            const missingSlots = required.filter(slot => !wardrobeItems.some((item: any) => item.macroCategory === slot))
            if (missingSlots.length) void 0
            if (missingSlots.length) return new Response(JSON.stringify({ success: false, outfits: [], code: 'no_matching_clothing',
                error: 'The selected clothing pool has no complete outfit suitable for the requested style, occasion and weather.', missingSlots }), {
                status: 422, headers: { ...corsHeaders, 'Content-Type': 'application/json' },
            })
        }

        if (wardrobeItems.length === 0) {
            return new Response(JSON.stringify({
                success: false,
                budget: parsedBudget ? {amount: parsedBudget.amount, currency: parsedBudget.currency} : null,
                outfits: [],
                code: budgetFilteredAll ? 'no_items_within_budget' : 'no_matching_clothing',
                error: budgetFilteredAll ? 'No clothing items fit this budget. Increase the budget or use owned clothing.' : 'No clothing items match your selections. Add suitable items or adjust your preferences.',
                debug: {
                    dbClothingItems: dbClothingItems.length,
                    legacyReceived: Array.isArray(legacyWardrobe) ? legacyWardrobe.length : 0,
                    useProvidedWardrobeOnly: !!useProvidedWardrobeOnly,
                    wardrobeOnly: !!wardrobeOnly,
                },
            }), {
                status: 422, headers: { ...corsHeaders, 'Content-Type': 'application/json' }
            })
        }

        // A lower bound above the cap proves no required-slot combination can fit.
        // Dresses are alternative compositions; unknown shop prices/currencies
        // leave the decision to normal validation rather than falsely rejecting.
        const budgetEligibleItems = wardrobeItems.filter((item: any) =>
            !rejectForStyleOccasion(item, String(item.id) === String(anchorItemId || '') ? 'casual' : selectionStyle, `${occasion} ${prompt}`)
                && !rejectForClimate(item, weather)
                && !rejectForActivity(item, `${occasion} ${prompt}`))
        if (parsedBudget && !budgetEligibleItems.some(isDressGarment)) {
            const required = ['top', 'bottom', 'shoes']
            if (needsLayering(stylePreferences, weather, prompt)) required.push('outerwear')
            const minimumShopSpend = minimumSlotSpend(budgetEligibleItems, required, (item: any) => {
                if (!isShopPricedItem(item)) return 0
                const cost = shopPriceOf(item, parsedBudget!.currency, parsedBudget!.fx)
                return Number.isFinite(cost) && typeof item.price === 'number' && item.price > 0 ? cost : null
            })
            if (minimumShopSpend !== null && !fitsBudgetAmount(minimumShopSpend, parsedBudget.amount)) {
                return new Response(JSON.stringify({success: false, outfits: [], code: 'no_outfit_within_budget',
                    error: 'No complete outfit in the selected clothing pool fits this budget.',
                    budget: parsedBudget ? {amount: parsedBudget.amount, currency: parsedBudget.currency} : null, minimumShopSpend}), {
                    status: 422, headers: {...corsHeaders, 'Content-Type': 'application/json'},
                })
            }
        }

        // ── 4. Get API keys ───────────────────────────────────────────────
        let nvidiaKey = Deno.env.get('NVIDIA_API_KEY') || ''
        let nvidiaModel = resolveNvidiaModel(Deno.env.get('NVIDIA_TEXT_MODEL') || Deno.env.get('NVIDIA_MODEL'))
        // Provider routing policy (explicit — see the PROVIDER_ROUTING
        // comment above callNvidia): generation is NVIDIA-only. No second AI
        // provider is wired for outfit generation.

        if (SUPABASE_SERVICE_ROLE_KEY && !nvidiaKey) {
            const svc = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
            const { data: cfg } = await svc
                .from('app_config')
                .select('key, value')
                .in('key', ['nvidia_token', 'gemma', 'nvidia_model', 'nvidia_text_model'])
            if (cfg) {
                for (const row of cfg) {
                    if ((row.key === 'nvidia_token' || row.key === 'gemma') && !nvidiaKey) nvidiaKey = row.value
                    if ((row.key === 'nvidia_model' || row.key === 'nvidia_text_model') && (!nvidiaModel || nvidiaModel === DEFAULT_NVIDIA_MODEL)) {
                        nvidiaModel = resolveNvidiaModel(row.value)
                    }
                }
            }
        }

        // ── 4b. Subscription tier (server-authoritative) ────────────────────
        // The tier is read for entitlement/billing visibility ONLY. It does
        // NOT select a provider: every tier generates through the same
        // NVIDIA path (see PROVIDER_ROUTING). Fail-open to 'free' — the tier
        // must never hard-error generation.
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
                void 0
            } catch (_) {
                // Fail open to free-tier routing.
            }
        }

        nvidiaModel = Deno.env.get('NVIDIA_OUTFIT_MODEL') || nvidiaModel
        // Every generation uses the same complete-look planning protocol.

        // ── 5. Decide layering policy (2-top layered composition vs classic 3-slot) ──
        // A locked anchor that never reached the candidate pool must fail
        // explicitly, not silently produce anchor-less decks.
        if (anchorItemId && (!isShopOnlyDeck(stylePreferences, occasion) || !!anchorItemId)
            && !wardrobeItems.some((i: any) => String(i.id) === String(anchorItemId))) {
            return new Response(JSON.stringify({ success: false, error: 'anchor_not_found', outfits: [] }), { status: 422, headers: { ...corsHeaders, 'Content-Type': 'application/json' } })
        }
        const layered = needsLayering(stylePreferences, weather, `${occasion} ${prompt}`)
        void 0

        // A configured provider ranks the plan; missing provider uses its validated seed.

        // A single plan is the only producer of outfits. The provider never
        // supplies garment IDs, metadata or a second set of combinations.
        trace.stage='planning'
        wardrobeItems = trace.sync('intelligence',()=>wardrobeItems.map((item:any)=>prepareGarment(item)))
        const itemMap = new Map<string,any>(wardrobeItems.map((item:any)=>[String(item.id),item]))
        const requiredChoiceIds=[...new Set([...(selectedItemIds||[]),...(anchorItemId?[anchorItemId]:[])])].map(String)
        const hasOwnedCandidates=wardrobeItems.some((item:any)=>!isShopRow(item))
        const shopOnly=(catalogOnly || isShopOnlyDeck(stylePreferences,occasion)) && !anchorItemId
        const evaluations=new Map<string,ReturnType<typeof assessOutfitQuality>>()
        const rowNumbers=new Map<any,number>(wardrobeItems.map((item:any,index:number)=>[item,index]))
        const evaluationKey=(parts:any[])=>parts.map(item=>rowNumbers.get(item)).sort((a,b)=>Number(a)-Number(b)).join(':')
        const gateCounts={structure:0,budget:0,source:0,quality:0}
        const assess=(parts:any[])=>{
            const key=evaluationKey(parts)
            let result=evaluations.get(key)
            if(!result){result=assessOutfitQuality(parts,styleKey,`${occasion} ${prompt}`,weather,anchorItemId,preferences,qualityContext);evaluations.set(key,result)}
            return result
        }
        const evaluate=(parts:any[])=>{
            const fail=(kind:keyof typeof gateCounts)=>{gateCounts[kind]++;return {valid:false,score:0}}
            if(fourSlotOutfitReason(parts,true) || requiredChoiceIds.some(id=>!parts.some(item=>item.id===id)))return fail('structure')
            if(parts.some(item=>!itemMap.has(item.id)||wornItemIds.has(item.id)))return fail('source')
            if((shopOnly&&parts.some(item=>!isShopRow(item))) || (wardrobeOnly&&parts.some(isShopRow))
                || (!shopOnly&&hasOwnedCandidates&&parts.every(isShopRow)))return fail('source')
            if(parsedBudget&&!fitsBudgetAmount(outfitShopSpend(parts,itemMap,parsedBudget.currency,parsedBudget.fx),parsedBudget.amount))return fail('budget')
            const styleVerdict=validateOutfitAgainstStyle({items:parts},styleKey,itemMap,true,preferences)
            if(!styleVerdict.ok||!validateOutfitCompatibility({items:parts},itemMap,styleKey).ok)return fail('quality')
            const assessment=assess(parts)
            if(!assessment.acceptance.passed || styleJudgeReason(parts,styleKey,`${occasion} ${prompt}`,weather,anchorItemId,preferences,assessment,qualityContext))return fail('quality')
            return {valid:true,score:assessment.score}
        }
        const target=Math.max(1,Math.min(3,Number(deckTarget)||limit))
        const plan=trace.sync('planning',()=>createOutfitPlan(wardrobeItems,qualityContext,evaluate,{
            requiredIds:requiredChoiceIds,target,priority:deckPriority,
            cost:item=>shopPriceOf(item,parsedBudget?.currency,parsedBudget?.fx),budget:parsedBudget?.amount,
            maxEvaluations:256,maxSyncMs:Math.max(50,Math.min(600,800-Object.values(trace.syncMs).reduce((sum,ms)=>sum+ms,0))),
        }))
        trace.counts={...trace.counts,...plan.diagnostics.counts,evaluations:plan.diagnostics.evaluations,
            proposals:plan.diagnostics.proposals,valid:plan.diagnostics.validCandidates,...gateCounts,budgetPruned:plan.diagnostics.budgetRejected}
        if(!plan.seed.length) {
            const missing=Object.values(plan.diagnostics.counts).some(count=>!count)
            const code=missing?'insufficient_candidates':plan.diagnostics.cpuRisk?'planning_cpu_risk'
                :(gateCounts.budget+plan.diagnostics.budgetRejected)>0&&!gateCounts.quality?'no_outfit_within_budget':plan.diagnostics.validCandidates?'incomplete_outfit_selection':'outfit_quality_rejected'
            return new Response(JSON.stringify({success:false,outfits:[],code,stage:'planning',
                diagnostics:{counts:plan.diagnostics.counts,evaluations:plan.diagnostics.evaluations,valid:plan.diagnostics.validCandidates,
                    bounded:!plan.diagnostics.exploredCompletely,cpuRisk:plan.diagnostics.cpuRisk}}),{status:plan.diagnostics.cpuRisk?503:422,headers:{...corsHeaders,'Content-Type':'application/json'}})
        }
        const providerTrace:ProviderAttempt[]=[]
        let ranks=new Map<string,number>(),generationSource:'ai'|'deterministic_fallback'='deterministic_fallback'
        if(nvidiaKey) {
            trace.stage='provider'
            const used=new Set(plan.candidates.flatMap(look=>look.items.map(item=>item.id)))
            const selectionUser=JSON.stringify({request:prompt,occasion,weather:weather||null,
                preferences:{favoriteColors:preferences?.favoriteColors,fitPreference:preferences?.fitPreference,styleGoals:preferences?.styleGoals},
                products:wardrobeItems.filter((item:any)=>used.has(item.id)).map((item:any)=>({id:item.id,name:item.name,
                    description:String(item.description||'').slice(0,160),slot:item.macroCategory,garment:(()=>{const d=describeGarment(item);return {category:d.subtype,fit:d.volume,color:d.color,formality:d.formality,pattern:d.pattern,prominence:d.patternProminence,material:d.materials,texture:d.texture,weight:d.bulk,layering:d.closure,footwear:d.soleProfile}})()})),
                looks:plan.candidates.map(look=>({id:look.id,items:look.items.map(item=>item.id),intent:outfitStyleIntent(look.items.map(describeGarment),qualityContext).id}))})
            const system=`You are a fashion stylist for ${selectionStyle}. Rank the supplied complete outfits for aesthetic quality, silhouette, pattern hierarchy, footwear and occasion. All looks are already validated. Keep creative coherent combinations. Do not change products. Unknown garment evidence remains unknown. Rank every supplied look ID exactly once, best first. Return IDs only, without descriptions or explanations.`
            const selections=await trace.wait('provider',()=>callNvidia(nvidiaKey,nvidiaModel,system,selectionUser,0,providerTrace,false,
                '\nOUTPUT CONTRACT: {"rankedLookIds":["look001","look002"]}. Return every supplied complete look ID exactly once, best first. No individual product IDs or additional fields.'))
            const ids=selections?.map(selection=>selection.items?.length===1?selection.items[0]?.id:null)
            const allowed=new Set(plan.candidates.map(look=>look.id))
            if(ids?.length===allowed.size&&new Set(ids).size===allowed.size&&ids.every(id=>allowed.has(id))) {
                ranks=new Map(ids.map((id,index)=>[id,index]));generationSource='ai'
            } else trace.counts.providerErrors=1
        } else trace.counts.providerErrors=1
        trace.stage='final_selection'
        const selected=trace.sync('selection',()=>plan.select(ranks))
        // The feasible seed survives a missing, malformed or failed rank call.
        const deck=selected.length===target?selected:plan.seed
        if(deck.length!==target)return new Response(JSON.stringify({success:false,outfits:[],code:'incomplete_outfit_selection',stage:'final_selection'}),{status:422,headers:{...corsHeaders,'Content-Type':'application/json'}})
        const forcedSlots=plan.diagnostics.forcedSlots as DiversitySlot[]
        const finalOutfits=trace.sync('response',()=>deck.map((look,index)=>({
            id:`outfit_${index+1}`,items:look.items,description:`${selectionStyle} outfit`,stylingTips:[],
            confidence:look.quality,curationRank:ranks.get(look.id),qualityAssessment:assess(look.items),
        })))
        const previousDeckOverlapBySlot=Object.fromEntries(['outerwear','top','bottom','shoes'].map(slot=>[slot,
            finalOutfits.flatMap(look=>look.items).filter((item:any)=>item.macroCategory===slot&&item.id!==anchorItemId&&wasRecentlyShown(item,slot==='shoes'?previousShoeSet:previousSet)).length]))
        trace.counts.selected=finalOutfits.length
        return new Response(JSON.stringify({success:true,generationVersion:GENERATION_VERSION,generationRequestId:body.generationRequestId,
            rotationScope:body.rotationScope,outfits:finalOutfits,source:generationSource,layered:true,provider:'nvidia:'+nvidiaModel,
            providerAttempts:providerTrace,previousDeckOverlapBySlot,
            diversity:{activeSlots:['top','outerwear','bottom','shoes'],strictSlots:['top','outerwear','bottom','shoes'].filter(slot=>!forcedSlots.includes(slot as DiversitySlot)),forcedRepeatSlots:forcedSlots},
            rotation:{recycledSlots:forcedSlots,freshProducts:freshProductCount(finalOutfits.flatMap(look=>look.items),excludeSet,anchorItemId)},
            diagnostics:{stage:'complete',syncMs:trace.syncMs,counts:trace.counts,cpuRisk:plan.diagnostics.cpuRisk},
            budgetReference:requestFx?{id:requestFx.snapshot.id,asOf:requestFx.snapshot.asOf,source:'CBU'}:undefined,
        }),{headers:{...corsHeaders,'Content-Type':'application/json'}})

    } catch (error: any) {
        if (error instanceof CatalogLookupError) return new Response(JSON.stringify({success:false,outfits:[],error:'catalog_lookup_unavailable'}), {status:503,headers:{...corsHeaders,'Content-Type':'application/json'}})
        if (error instanceof BudgetFxError) return new Response(JSON.stringify({success:false,outfits:[],error:'budget_rates_unavailable'}), {status:503,headers:{...corsHeaders,'Content-Type':'application/json'}})
        return new Response(JSON.stringify({ success: false, code: 'generation_failed', outfits: [] }), {
            headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 500
        })
    }
}

interface GenerationTrace {
  stage:string;
  syncMs:Record<string,number>;
  waitMs:Record<string,number>;
  counts:Record<string,number>;
  sync<T>(stage:string,fn:()=>T):T;
  wait<T>(stage:string,fn:()=>Promise<T>):Promise<T>;
}
serve(async(req:Request)=>{
  const trace:GenerationTrace={stage:'admission',syncMs:{},waitMs:{},counts:{},async wait(stage,fn){const start=performance.now();try{return await fn();}finally{this.waitMs[stage]=(this.waitMs[stage]||0)+performance.now()-start;}},sync(stage,fn){
    const start=performance.now();try{return fn();}finally{this.syncMs[stage]=(this.syncMs[stage]||0)+performance.now()-start;}
  }};
  let response=await generateOutfitsHandler(req,trace);
  if(req.method!=='OPTIONS') {
    if(response.status>=400) {
      let body:any;try{body=await response.json();}catch{body={success:false,outfits:[]};}
      const known=typeof body.code==='string'?body.code:typeof body.error==='string'&&/^[a-z_]+$/.test(body.error)?body.error:'generation_failed';
      body={...body,error:known,code:known,stage:body.stage||trace.stage};
      response=new Response(JSON.stringify(body),{status:response.status,headers:response.headers});
    }
    console.info('stylist_request',{status:response.status,stage:trace.stage,syncMs:trace.syncMs,waitMs:trace.waitMs,counts:trace.counts});
  }
  return response;
});
