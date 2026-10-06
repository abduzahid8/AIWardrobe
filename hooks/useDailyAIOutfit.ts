/**
 * useDailyAIOutfit
 *
 * Generates a batch of AI outfits for **a single named category** (e.g.
 * "business_casual", "old_money") and regenerates that batch **once per
 * calendar day per user per category**.
 *
 * The category is driven by the section title on Home ("Team Collaboration /
 * Business Casual", "Night-Time Dinner / Elegant Classic", etc.) so different
 * sections get their own independent daily batch and their own cache slot.
 *
 *   - Items come from `shop_catalog` via the existing `generate-outfits`
 *     Supabase Edge Function (with a rule-based fallback baked in).
 *   - Cache key = `daily_ai_outfit::<userId>::<style>` so each section refreshes
 *     independently and a user running the same device across accounts doesn't
 *     poison the cache.
 *   - Cache entry holds `dateKey = YYYY-MM-DD` in local time; on the next
 *     calendar day the hook sees the mismatch and triggers a fresh AI call.
 *   - Max AI calls per user per day = number of sections using this hook.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

import useAuthStore from '../store/auth';
import useWardrobeStore from '../store/wardrobeStore';
import useShopCatalogStore from '../store/shopCatalogStore';
import { useStylePreferenceStore } from '../store/stylePreferenceStore';
import {
    generateOutfitsFromDB,
    type GeneratedOutfit,
} from '../src/services/outfitGenerationService';
import { createLogger } from '../src/utils/logger';
import type { ClothingItem } from '../src/types/domain';
import type { ShopCatalogItem } from '../src/types/shop';
import { normalizeStyleId, scoreItemForStyle, type StyleId } from '../features/outfit-generator/utils/styleInference';
import { isFormalLayer, isShortsBottom } from '../src/utils/categoryMapper';

// Static catalogs require()d lazily — ~120KB of data modules must not
// parse/evaluate on app startup (Home imports this hook eagerly). The
// rebuild paths below are the only consumers; first rebuild pays the cost,
// subsequent calls reuse the cached array.
// eslint-disable-next-line @typescript-eslint/no-var-requires
let staticShopPoolCache: ShopCatalogItem[] | null = null;
function getStaticShopPool(): ShopCatalogItem[] {
    if (!staticShopPoolCache) {
        const data = require('../data/shopCatalogItems') as {
            SHOP_CATALOG_ITEMS: ShopCatalogItem[];
        };
        const inspo = require('../data/inspoMensShopItems') as {
            CLASSIC_MENS_ITEMS: ShopCatalogItem[];
        };
        staticShopPoolCache = [...data.SHOP_CATALOG_ITEMS, ...inspo.CLASSIC_MENS_ITEMS];
    }
    return staticShopPoolCache;
}
import { buildCatalogSimilarityKey, spreadSimilarCatalogItems } from '../src/utils/shopCatalogOrder';

const logger = createLogger('useDailyAIOutfit');

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface DailyOutfitWeather {
    temp: number;
    condition: string;
}

export interface UseDailyAIOutfitOptions {
    /**
     * Style id understood by the `generate-outfits` edge function.
     * e.g. `"business_casual" | "old_money" | "semi_classic" | "minimalist" | "casual"`.
     * This is the "category" driving the daily refresh — each distinct value
     * gets its own cache slot.
     */
    style: string;
    /** Human-friendly occasion label, e.g. "Work", "Date night". */
    occasion: string;
    /** Current weather context (optional but improves AI choices). */
    weather?: DailyOutfitWeather | null;
    /** How many variants to generate per day (default 3). */
    variants?: number;
    /** Skip generation until caller is ready. */
    enabled?: boolean;
}

export interface UseDailyAIOutfitResult {
    outfits: GeneratedOutfit[];
    loading: boolean;
    /** Present when the AI call failed outright. */
    error: string | null;
    /** Force a regeneration now (bypasses the daily cache). */
    regenerate: () => Promise<void>;
}

interface CachedEntry {
    dateKey: string;
    style: string;
    occasion: string;
    outfits: GeneratedOutfit[];
    createdAt: string;
}

type DailyOutfitItem = GeneratedOutfit['items'][number];
type DailySlot = 'top' | 'outerwear' | 'bottom' | 'shoes';

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function todayKey(d: Date = new Date()): string {
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${y}-${m}-${day}`;
}

function storageKey(userId: string | null | undefined, style: string): string {
    // v18 invalidates decks built without similarity spreading + per-deck
    // rotation seeds (lookalike listings clustered at top of ranked pools;
    // sibling sections opening on the same garments) — visual repeats with
    // distinct ids.
    // v17 invalidates decks built without pool dedupe + overlap minimization
    // (one garment occupying two rotation indices; near-duplicate 3-of-4
    // shared looks accepted) — the residual ~20% repeats.
    // v16 invalidates decks built with the frozen-slot enumerator
    // (floor((i+attempt)/divisor)+slotIndex*3) that left top/bottom/shoes
    // fixed — 70%+ identical looks with one item repeating every time.
    // v15 invalidates decks created before Team Collaboration / business_casual
    // became shop-only. Old v14 decks contain a wardrobe anchor + shop
    // complements and must not be reused for the shop-only section.
    // v14 invalidates decks created before the rebuild became style-ranked
    // and required all 15 daily combinations to be unique.
    // v13 invalidates any deck created before the Home hook forcibly rebuilt
    // large daily batches as mixed wardrobe+shop outfits.
    // v12 invalidates decks that were unique but shop-only. Daily looks should
    // be mixed when the user has wardrobe items: owned anchor + shop complements.
    // (Team Collaboration is the exception — see isShopOnlyDeck below.)
    // v11 invalidates daily batches generated before the client local fallback
    // could use the bundled shop catalog when Supabase returned no shop rows.
    // It also keeps rejecting same-combination 15-card decks below.
    // v10 invalidates daily batches generated from too-small wardrobe-only
    // pools. Daily Home decks now require enough unique combinations for the
    // requested 15-card carousel before the cache can short-circuit generation.
    return `daily_ai_outfit_v18::${userId ?? 'anon'}::${style}`;
}

/**
 * Team Collaboration must be 100% shopping — no wardrobe anchor.
 * Matches by style (`business_casual`) OR by occasion label
 * (`Team Collaboration`) so an admin rename of one side can't silently
 * re-enable mixed decks.
 */
export function isShopOnlyDeck(style?: string | null, occasion?: string | null): boolean {
    const s = (style || '').toLowerCase().replace(/[\s-]+/g, '_');
    if (s === 'business_casual') return true;
    const o = (occasion || '').toLowerCase();
    if (o.includes('team') && o.includes('collab')) return true;
    if (o.trim() === 'team collaboration') return true;
    return false;
}

/**
 * Formal decks (business_casual / old_money / classic) never contain shorts —
 * not even with a casual layer. Casual decks still allow shorts, but NEVER
 * paired with a formal outerwear piece (blazer, suit jacket, overcoat,
 * trench, peacoat, tuxedo). Single rule behind every guard below.
 */
function isFormalDeckStyle(style?: string | null): boolean {
    const s = (style || '').toLowerCase().replace(/[\s-]+/g, '_');
    return s === 'business_casual' || s === 'old_money' || s === 'classic' || s === 'formal';
}

function outfitHasShortsWithSuit(items: Array<{ name?: string; type?: string; macroCategory?: string }>, style?: string | null): boolean {
    const bottomIsShorts = items.some((it) => {
        if (String(it.macroCategory || '').toLowerCase() !== 'bottom') return false;
        return isShortsBottom(it as unknown as Record<string, unknown>);
    });
    if (!bottomIsShorts) return false;
    if (isFormalDeckStyle(style)) return true;
    return items.some((it) => isFormalLayer(it as unknown as Record<string, unknown>));
}

/**
 * Normalize a heterogeneous macroCategory / type value into the canonical
 * slot the collage understands.
 *
 * Strategy: blob match FIRST (most specific — a "sweater polo" must be
 * outerwear even if the server tagged it macroCategory="top"), then fall
 * back to the raw `macroCategory` field, then declare it `other`.
 *
 * This mirrors `getOutfitItemMacroCategory()` in `outfitPreview.ts` so the
 * filter's classification and the collage's slot-picker can never disagree.
 */
function canonicalMacro(item: GeneratedOutfit['items'][number]): 'top' | 'outerwear' | 'bottom' | 'shoes' | 'other' {
    const blob = `${item.type || ''} ${item.name || ''}`.toLowerCase();

    // Outerwear first — sweaters, blazers, jackets must not be collapsed
    // into the base-top slot otherwise a layered look loses its main layer.
    if (/\b(blazer|overcoat|topcoat|peacoat|trench|parka|puffer|windbreaker|bomber)s?\b/.test(blob)) return 'outerwear';
    if (/\b(coat|jacket|cardigan|sweater|hoodie|vest|pullover|fleece)s?\b/.test(blob)) return 'outerwear';
    if (/\b(pants?|trousers?|jeans|shorts?|skirts?|chinos?|slacks?|joggers?|sweatpants?)\b/.test(blob)) return 'bottom';
    if (/\b(shoes?|sneakers?|boots?|loafers?|sandals?|heels?|trainers?|derbys?|derbies|mules?)\b/.test(blob)) return 'shoes';
    // "oxford" alone is ambiguous (Oxford shirt vs Oxford shoes) — only match
    // as shoes when paired with a shoe qualifier like "shoe" or "flat", or in plural form "oxfords"
    if (/\boxfords?\b/i.test(blob) && !/\bshirts?\b/i.test(blob)) return 'shoes';
    if (/\boxfords?\s*(shoes?|flats?|lace|brogues?|derbys?|derbies)\b/i.test(blob)) return 'shoes';
    if (/\b(t-shirt|tshirt|tee|polo|blouse|shirt|dress(?:es)?)s?\b/.test(blob)) return 'top';
    if (/upper[_\s-]?body/.test(blob)) return 'top';
    if (/lower[_\s-]?body/.test(blob)) return 'bottom';

    // Fallback to the macroCategory tag the server supplied.
    const raw = (item.macroCategory || '').toLowerCase().trim();
    if (raw === 'outerwear' || raw === 'outer_layer' || raw === 'layer') return 'outerwear';
    if (raw === 'top' || raw === 'tops' || raw === 'upper_body' || raw === 'upper-body' || raw === 'shirt' || raw === 'dress' || raw === 'dresses') return 'top';
    if (raw === 'bottom' || raw === 'bottoms' || raw === 'lower_body' || raw === 'lower-body' || raw === 'pants' || raw === 'pant') return 'bottom';
    if (raw === 'shoes' || raw === 'shoe' || raw === 'footwear') return 'shoes';

    return 'other';
}

function needsDailyLayering(style: string, weather?: DailyOutfitWeather | null): boolean {
    const styleKey = style.toLowerCase().replace(/[\s-]+/g, '_');
    const condition = (weather?.condition || '').toLowerCase();
    const temp = typeof weather?.temp === 'number' ? weather.temp : null;
    if (temp != null) return temp < 18;
    if (/\b(cold|chilly|freezing|snow|rain|drizzle|wind|storm|cool)\b/.test(condition)) return true;
    return ['old_money', 'business_casual', 'classic', 'semi_classic'].includes(styleKey);
}

function toWardrobeOutfitItem(item: ClothingItem): DailyOutfitItem {
    return {
        id: item.id,
        type: item.subCategory || item.category || 'clothing',
        macroCategory: canonicalMacro({
            id: item.id,
            type: item.subCategory || item.category || '',
            name: item.name || item.subCategory || item.category || '',
            color: item.primaryColor || 'neutral',
            recommendation: '',
            isShopItem: false,
        } as DailyOutfitItem),
        color: item.primaryColor || 'neutral',
        name: item.name || item.subCategory || item.category || 'Wardrobe item',
        brand: item.brand || '',
        imageUrl: item.imageUrl || item.thumbnailUrl || '',
        image: item.imageUrl || item.thumbnailUrl || '',
        recommendation: 'From your wardrobe',
        isShopItem: false,
    };
}

function toShopOutfitItem(item: ShopCatalogItem): DailyOutfitItem {
    const image = typeof item.imageUrl === 'string' ? item.imageUrl : '';
    const slot = shopSlot(item);
    const id = String(item.id || '').startsWith('shop_') ? String(item.id) : `shop_${item.id}`;
    return {
        id,
        type: item.garmentType || slot,
        macroCategory: slot,
        color: 'neutral',
        name: item.name || 'Shop item',
        brand: item.brand || '',
        imageUrl: image,
        image,
        recommendation: 'Suggested from shop',
        isShopItem: true,
        price: item.price,
        shopUrl: item.sourceUrl,
    };
}

function shopItemStyleScore(item: DailyOutfitItem, style: StyleId): number {
    return scoreItemForStyle(
        {
            name: item.name,
            description: '',
            brand: item.brand,
            color: item.color,
            type: item.type,
            macroCategory: item.macroCategory,
        },
        style,
    );
}

function isHardRejectedShopItem(item: DailyOutfitItem, style: StyleId): boolean {
    const blob = `${item.name || ''} ${item.type || ''} ${item.brand || ''}`.toLowerCase();
    const slot = canonicalMacro(item);

    if (style === 'old_money' || style === 'classic' || style === 'business_casual') {
        if (/\b(graphic|logo|cargo|ripped|distressed|sweatpants?|joggers?|basketball|running|athletic|platform|chunky|square[-\s]?toe)\b/.test(blob)) {
            return true;
        }
        if (slot === 'bottom' && /\b(shorts?|bermudas?)\b/.test(blob)) return true;
        if (style === 'old_money' && slot === 'shoes' && /\bsneakers?\b/.test(blob)) return true;
        if (slot === 'outerwear' && /\b(hoodie|windbreaker|puffer|bomber|track|zip[-\s]?up)\b/.test(blob)) return true;
    }

    return false;
}

function styleRankItems(items: DailyOutfitItem[], style: string): DailyOutfitItem[] {
    const styleKey = normalizeStyleId(style || 'casual');
    return [...items]
        .filter((item) => !isHardRejectedShopItem(item, styleKey))
        .sort((a, b) => shopItemStyleScore(b, styleKey) - shopItemStyleScore(a, styleKey));
}

function shopSlot(item: ShopCatalogItem): DailySlot | 'other' {
    const blob = `${item.name || ''} ${item.description || ''} ${item.garmentType || ''}`.toLowerCase();
    if (item.garmentType === 'shoes') return 'shoes';
    if (item.garmentType === 'lower_body') return 'bottom';
    if (item.garmentType === 'upper_body') {
        if (/\b(blazer|overcoat|topcoat|peacoat|trench|parka|puffer|windbreaker|bomber|coat|jacket|cardigan|sweater|hoodie|vest|pullover|fleece|overshirt)\b/.test(blob)) {
            return 'outerwear';
        }
        return 'top';
    }
    return 'other';
}

function bucketDailyItems(items: DailyOutfitItem[]): Record<DailySlot, DailyOutfitItem[]> {
    const buckets: Record<DailySlot, DailyOutfitItem[]> = {
        top: [],
        outerwear: [],
        bottom: [],
        shoes: [],
    };
    const seen = new Set<string>();
    for (const item of items) {
        const slot = canonicalMacro(item);
        if (slot === 'other') continue;
        const key = `${slot}:${item.id}`;
        if (seen.has(key)) continue;
        seen.add(key);
        buckets[slot].push({ ...item, macroCategory: slot });
    }
    return buckets;
}

function pickFrom(pool: DailyOutfitItem[], index: number, usedIds: Set<string>): DailyOutfitItem | undefined {
    if (pool.length === 0) return undefined;
    for (let offset = 0; offset < pool.length; offset++) {
        const item = pool[(index + offset) % pool.length];
        if (!usedIds.has(String(item.id))) return item;
    }
    return pool[index % pool.length];
}

// Diverse round-robin: each slot rotates with the outfit index (i) plus a
// small per-slot offset (slotIndex*3) so top/bottom/shoes don't freeze on one
// item. Attempt steps are primes coprime to small pool sizes (3/4/5) so a
// signature collision perturbs every slot differently and finds a new unique
// combination instead of repeating the same 70%-identical look.
const DIVERSE_ATTEMPT_STEPS = [7, 11, 13, 17];

function diversePoolIndex(outfitIndex: number, slotIndex: number, attempt: number, poolLength: number, seed = 0, reverse = false): number {
    if (poolLength <= 0) return 0;
    const base = (outfitIndex + seed + slotIndex * 3 + attempt * DIVERSE_ATTEMPT_STEPS[slotIndex % DIVERSE_ATTEMPT_STEPS.length]) % poolLength;
    // Mirrored lane: sibling decks sharing one catalog traverse opposite ends
    // so their 15-card windows overlap as little as the pool size allows.
    return reverse ? (poolLength - 1 - base) : base;
}

// Merge pools without duplicating the same item id twice. The old
// [...primary, ...existing] concat let one id occupy two indices, so the
// modulo rotation could land on the same garment via either copy and repeat
// it across outfits.
function dedupePoolById(pool: DailyOutfitItem[]): DailyOutfitItem[] {
    const seen = new Set<string>();
    const out: DailyOutfitItem[] = [];
    for (const item of pool) {
        const key = String(item.id);
        if (seen.has(key)) continue;
        seen.add(key);
        out.push(item);
    }
    return out;
}

// Score a candidate combination: penalize items already used often and pairs
// sharing several pieces with an already-built outfit. Lower is better.
function scoreCandidateOverlap(
    candidateIds: string[],
    builtIdSets: Set<string>[],
    usageCounts: Map<string, number>,
): number {
    let maxShared = 0;
    for (const built of builtIdSets) {
        let shared = 0;
        for (const id of candidateIds) if (built.has(id)) shared++;
        if (shared > maxShared) maxShared = shared;
    }
    let usage = 0;
    for (const id of candidateIds) usage += usageCounts.get(id) ?? 0;
    return maxShared * 100 + usage;
}

// Product-family key: groups listings that look identical at card size
// ("Brown Shirt" ×3, eight "penny loafer" variants). Score ties cluster
// these at the top of a ranked pool, so a plain rotation walks straight
// through lookalikes — different ids, same look.
function similarityKeyOf(item: DailyOutfitItem): string {
    return buildCatalogSimilarityKey({
        id: String(item.id),
        name: item.name || 'item',
        garmentType: String(item.macroCategory || 'item'),
    });
}

// Reorder a ranked pool so same-family products are spread apart instead of
// occupying adjacent rotation indices.
function spreadBucketBySimilarity(pool: DailyOutfitItem[], slot: DailySlot): DailyOutfitItem[] {
    if (pool.length <= 2) return pool;
    const decorated = pool.map((item) => ({ ...item, name: item.name || 'item', garmentType: slot }));
    const ordered = spreadSimilarCatalogItems(
        decorated as Array<DailyOutfitItem & { name: string; garmentType: string }>,
    );
    return ordered.map((entry) => {
        const { garmentType: _dropped, ...rest } = entry as DailyOutfitItem & { garmentType?: string };
        return rest as DailyOutfitItem;
    });
}

// Per-deck rotation seed + lane: two sections sharing one catalog (Team
// Collaboration + Night-Time Dinner rank the same oxfords/loafers at top)
// must not start at the same pool offset, or card #k in both decks shows
// the same garments. The lane mirrors traversal so sibling decks sample
// opposite ends of small pools.
function deckRotationHash(style: string, occasion: string): number {
    const key = `${style}::${occasion}`.toLowerCase();
    let hash = 0;
    for (let i = 0; i < key.length; i++) hash = (Math.imul(31, hash) + key.charCodeAt(i)) | 0;
    return Math.abs(hash);
}
function deckRotationSeed(style: string, occasion: string): number {
    return deckRotationHash(style, occasion) % 32;
}
function deckReverseLane(style: string, occasion: string): boolean {
    return ((deckRotationHash(style, occasion) >> 5) & 1) === 1;
}

function rebuildShopOnlyDailyDeck(
    inputOutfits: GeneratedOutfit[],
    requestedVariants: number,
    style: string,
    occasion: string,
    weather: DailyOutfitWeather | null | undefined,
    shopCatalogItems: ShopCatalogItem[],
): GeneratedOutfit[] {
    const layered = needsDailyLayering(style, weather);
    const targetSlots: DailySlot[] = layered ? ['outerwear', 'top', 'bottom', 'shoes'] : ['top', 'bottom', 'shoes'];
    const count = Math.min(requestedVariants, 15);

    const existingShopItems = inputOutfits
        .flatMap((outfit) => outfit.items || [])
        .filter((item) => (item as DailyOutfitItem).isShopItem === true);
    const staticShop = getStaticShopPool();
    const shopOutfitItems = styleRankItems(
        [...shopCatalogItems, ...staticShop].map(toShopOutfitItem).filter((item) => canonicalMacro(item) !== 'other'),
        style,
    );

    const shopBuckets = bucketDailyItems(shopOutfitItems);
    const existingBuckets = bucketDailyItems(existingShopItems);

    // Require every target slot to have at least one shop candidate —
    // otherwise we can't build complete shop-only looks.
    const missingSlot = targetSlots.find(
        (slot) => shopBuckets[slot].length === 0 && existingBuckets[slot].length === 0,
    );
    if (missingSlot) return filterRenderableOutfits(inputOutfits).filter((o) =>
        (o.items || []).every((i) => (i as DailyOutfitItem).isShopItem === true),
    );

    const rebuilt: GeneratedOutfit[] = [];
    const signatures = new Set<string>();
    const builtIdSets: Set<string>[] = [];
    const usageCounts = new Map<string, number>();
    const simUsageCounts = new Map<string, number>();
    // Bounded search: pools hold 100+ candidates, so count*3 outer tries ×
    // count*2 inner attempts is plenty for 15 uniques. Was count*6 ×
    // max(30, count*4) = 5400 iterations worst-case per deck × 2 decks on
    // Home — the main-thread stall behind slow Home renders.
    const maxOutfits = count * 3;
    // Each named deck starts at its own rotation offset so two sections
    // sharing one catalog don't open on the same garments.
    const seed = deckRotationSeed(style, occasion);
    const reverse = deckReverseLane(style, occasion);

    // Per-slot pools: deduped (one garment can't occupy two rotation
    // indices) then similarity-spread (same-looking listings can't occupy
    // adjacent rotation indices).
    // Formal decks never see shorts in the bottom pool — `styleRankItems`
    // already strips them from fresh shop candidates, but `existingBuckets`
    // (edge-function output) bypasses that filter, so strip here as well.
    const poolsBySlot = new Map<DailySlot, DailyOutfitItem[]>();
    const formalDeck = isFormalDeckStyle(style);
    for (const slot of targetSlots) {
        const primaryPool = shopBuckets[slot].length > 0 ? shopBuckets[slot] : existingBuckets[slot];
        let combined = [...primaryPool, ...existingBuckets[slot]];
        if (slot === 'bottom' && formalDeck) {
            combined = combined.filter((item) => !isShortsBottom(item as unknown as Record<string, unknown>));
        }
        poolsBySlot.set(slot, spreadBucketBySimilarity(dedupePoolById(combined), slot));
    }

    for (let i = 0; rebuilt.length < count && i < maxOutfits; i++) {
        let items: DailyOutfitItem[] = [];
        let finalSignature = '';
        let bestScore = Number.POSITIVE_INFINITY;
        const maxAttempts = Math.max(15, count * 2);

        for (let attempt = 0; attempt < maxAttempts; attempt++) {
            const usedIds = new Set<string>();
            const itemsBySlot = new Map<DailySlot, DailyOutfitItem>();

            let failed = false;
            targetSlots.forEach((slot, slotIndex) => {
                const pool = poolsBySlot.get(slot) ?? [];
                const poolIndex = diversePoolIndex(i, slotIndex, attempt, pool.length, seed, reverse);
                const pick = pickFrom(pool, poolIndex, usedIds);
                if (!pick) {
                    failed = true;
                    return;
                }
                usedIds.add(String(pick.id));
                itemsBySlot.set(slot, {
                    ...pick,
                    macroCategory: slot,
                    recommendation: 'Suggested from shop',
                    isShopItem: true,
                });
            });
            if (failed) continue;

            const candidateItems = targetSlots
                .map((slot) => itemsBySlot.get(slot))
                .filter(Boolean) as DailyOutfitItem[];
            // Hard styling rule: never shorts with a suit/formal layer, and
            // never shorts at all in formal decks. A pooled shorts item that
            // slipped past the pool filter must not survive as a combination.
            if (outfitHasShortsWithSuit(candidateItems, style)) continue;
            // Shop-only: every piece must be a shop pick, and the combination
            // must never repeat (unique by sorted item ids).
            const allShop = candidateItems.every((item) => item.isShopItem === true);
            const signature = candidateItems.map((item) => String(item.id)).sort().join('|');
            if (candidateItems.length !== targetSlots.length || !allShop || !signature || signatures.has(signature)) continue;

            // Best-of-attempts: keep the unique candidate sharing the fewest
            // pieces with already-built outfits (then least total reuse).
            // Same-looking products count extra: different ids, same look.
            let simReuse = 0;
            for (const item of candidateItems) simReuse += simUsageCounts.get(similarityKeyOf(item)) ?? 0;
            const score = scoreCandidateOverlap(
                candidateItems.map((item) => String(item.id)),
                builtIdSets,
                usageCounts,
            ) + simReuse * 150;
            if (score < bestScore) {
                bestScore = score;
                items = candidateItems;
                finalSignature = signature;
                if (score === 0) break;
            }
        }

        if (!items.length || !finalSignature || signatures.has(finalSignature)) continue;
        // Final guard: never emit a mixed or duplicate outfit from this path —
        // and never a shorts-with-suit combination that won best-of-attempts.
        if (!items.every((item) => item.isShopItem === true)) continue;
        if (outfitHasShortsWithSuit(items, style)) continue;
        signatures.add(finalSignature);
        const idSet = new Set(items.map((item) => String(item.id)));
        builtIdSets.push(idSet);
        for (const id of idSet) usageCounts.set(id, (usageCounts.get(id) ?? 0) + 1);
        for (const item of items) {
            const key = similarityKeyOf(item);
            simUsageCounts.set(key, (simUsageCounts.get(key) ?? 0) + 1);
        }

        const outputIndex = rebuilt.length;
        rebuilt.push({
            id: inputOutfits[outputIndex]?.id || `daily_shop_${style}_${outputIndex}_${Date.now()}`,
            description: `A ${style} look curated from the shop.`,
            style,
            occasion: inputOutfits[outputIndex]?.occasion || occasion,
            confidence: inputOutfits[outputIndex]?.confidence ?? 0.82,
            matchScore: inputOutfits[outputIndex]?.matchScore ?? 0.82,
            items: items.sort((a, b) => targetSlots.indexOf(a.macroCategory as DailySlot) - targetSlots.indexOf(b.macroCategory as DailySlot)),
            stylingTips: inputOutfits[outputIndex]?.stylingTips || ['Shop the full look', 'All pieces available now'],
        });
    }

    logger.debug('Rebuilt shop-only daily deck in hook', {
        style,
        occasion,
        count: rebuilt.length,
        uniqueCombinations: uniqueOutfitCombinationCount(rebuilt),
        shopItemsByOutfit: rebuilt.map((outfit) => outfit.items.filter((item) => item.isShopItem).length),
    });

    return rebuilt;
}

function rebuildMixedDailyDeck(
    inputOutfits: GeneratedOutfit[],
    requestedVariants: number,
    style: string,
    occasion: string,
    weather: DailyOutfitWeather | null | undefined,
    wardrobeItems: ClothingItem[],
    shopCatalogItems: ShopCatalogItem[],
): GeneratedOutfit[] {
    // Team Collaboration is always shop-only — never anchor to wardrobe.
    if (isShopOnlyDeck(style, occasion)) {
        return rebuildShopOnlyDailyDeck(inputOutfits, requestedVariants, style, occasion, weather, shopCatalogItems);
    }
    if (requestedVariants < 10 || wardrobeItems.length === 0) return inputOutfits;

    const layered = needsDailyLayering(style, weather);
    const targetSlots: DailySlot[] = layered ? ['outerwear', 'top', 'bottom', 'shoes'] : ['top', 'bottom', 'shoes'];
    const count = Math.min(requestedVariants, 15);

    const existingItems = inputOutfits.flatMap((outfit) => outfit.items || []);
    const wardrobeOutfitItems = wardrobeItems.map(toWardrobeOutfitItem).filter((item) => canonicalMacro(item) !== 'other');
    const staticShop = getStaticShopPool();
    const shopOutfitItems = styleRankItems(
        [...shopCatalogItems, ...staticShop].map(toShopOutfitItem).filter((item) => canonicalMacro(item) !== 'other'),
        style,
    );

    const wardrobeBuckets = bucketDailyItems(wardrobeOutfitItems);
    const shopBuckets = bucketDailyItems(shopOutfitItems);
    const existingBuckets = bucketDailyItems(existingItems);
    const wardrobeAnchors = targetSlots
        .flatMap((slot) => wardrobeBuckets[slot])
        .concat(wardrobeOutfitItems.filter((item) => !targetSlots.includes(canonicalMacro(item) as DailySlot)));

    if (wardrobeAnchors.length === 0) return inputOutfits;

    const rebuilt: GeneratedOutfit[] = [];
    const signatures = new Set<string>();
    const builtIdSets: Set<string>[] = [];
    const usageCounts = new Map<string, number>();

    // Same bound as the shop-only path above (was count*4 outers ×
    // max(30, count*4) attempts).
    const mixedSeed = deckRotationSeed(style, occasion);
    const mixedReverse = deckReverseLane(style, occasion);
    const mixedSimUsage = new Map<string, number>();
    // Deduped + similarity-spread per-slot pools shared across attempts
    // (anchor excluded at pick time). Same formal-deck shorts strip as the
    // shop-only path — existing/wardrobe buckets bypass style ranking.
    const mixedFormalDeck = isFormalDeckStyle(style);
    const mixedPoolsBySlot = new Map<DailySlot, DailyOutfitItem[]>();
    for (const slot of targetSlots) {
        const primaryPool = shopBuckets[slot].length > 0 ? shopBuckets[slot] : wardrobeBuckets[slot];
        let combined = [...primaryPool, ...existingBuckets[slot], ...wardrobeBuckets[slot]];
        if (slot === 'bottom' && mixedFormalDeck) {
            combined = combined.filter((item) => !isShortsBottom(item as unknown as Record<string, unknown>));
        }
        mixedPoolsBySlot.set(slot, spreadBucketBySimilarity(dedupePoolById(combined), slot));
    }
    for (let i = 0; rebuilt.length < count && i < count * 3; i++) {
        const anchor = wardrobeAnchors[i % wardrobeAnchors.length];
        const anchorSlot = targetSlots.includes(canonicalMacro(anchor) as DailySlot)
            ? canonicalMacro(anchor) as DailySlot
            : targetSlots[i % targetSlots.length];

        let items: DailyOutfitItem[] = [];
        let finalSignature = '';
        let bestScore = Number.POSITIVE_INFINITY;
        const maxAttempts = Math.max(15, count * 2);

        for (let attempt = 0; attempt < maxAttempts; attempt++) {
            const usedIds = new Set<string>([String(anchor.id)]);
            const itemsBySlot = new Map<DailySlot, DailyOutfitItem>();
            itemsBySlot.set(anchorSlot, {
                ...anchor,
                macroCategory: anchorSlot,
                isShopItem: false,
                recommendation: 'From your wardrobe',
            });

            targetSlots.forEach((slot, slotIndex) => {
                if (itemsBySlot.has(slot)) return;
                const pool = mixedPoolsBySlot.get(slot) ?? [];
                const poolIndex = diversePoolIndex(i, slotIndex, attempt, pool.length, mixedSeed, mixedReverse);
                const pick = pickFrom(pool, poolIndex, usedIds);
                if (!pick) return;
                usedIds.add(String(pick.id));
                itemsBySlot.set(slot, {
                    ...pick,
                    macroCategory: slot,
                    recommendation: pick.isShopItem ? 'Suggested from shop' : 'From your wardrobe',
                });
            });

            const candidateItems = targetSlots
                .map((slot) => itemsBySlot.get(slot))
                .filter(Boolean) as DailyOutfitItem[];
            // Same hard rule as shop-only: suits never pair with shorts.
            if (outfitHasShortsWithSuit(candidateItems, style)) continue;
            const isMixed = candidateItems.some((item) => !item.isShopItem)
                && candidateItems.some((item) => item.isShopItem);
            const signature = candidateItems.map((item) => String(item.id)).sort().join('|');
            if (candidateItems.length !== targetSlots.length || !isMixed || !signature || signatures.has(signature)) continue;

            let simReuse = 0;
            for (const item of candidateItems) simReuse += mixedSimUsage.get(similarityKeyOf(item)) ?? 0;
            const score = scoreCandidateOverlap(
                candidateItems.map((item) => String(item.id)),
                builtIdSets,
                usageCounts,
            ) + simReuse * 150;
            if (score < bestScore) {
                bestScore = score;
                items = candidateItems;
                finalSignature = signature;
                if (score === 0) break;
            }
        }

        if (!items.length || !finalSignature || signatures.has(finalSignature)) continue;
        if (outfitHasShortsWithSuit(items, style)) continue;
        signatures.add(finalSignature);
        const idSet = new Set(items.map((item) => String(item.id)));
        builtIdSets.push(idSet);
        for (const id of idSet) usageCounts.set(id, (usageCounts.get(id) ?? 0) + 1);
        for (const item of items) {
            const key = similarityKeyOf(item);
            mixedSimUsage.set(key, (mixedSimUsage.get(key) ?? 0) + 1);
        }

        const outputIndex = rebuilt.length;
        rebuilt.push({
            id: inputOutfits[outputIndex]?.id || `daily_mixed_${style}_${outputIndex}_${Date.now()}`,
            description: `A ${style} look mixing your wardrobe with shop suggestions.`,
            style,
            occasion: inputOutfits[outputIndex]?.occasion || occasion,
            confidence: inputOutfits[outputIndex]?.confidence ?? 0.82,
            matchScore: inputOutfits[outputIndex]?.matchScore ?? 0.82,
            items: items.sort((a, b) => targetSlots.indexOf(a.macroCategory as DailySlot) - targetSlots.indexOf(b.macroCategory as DailySlot)),
            stylingTips: inputOutfits[outputIndex]?.stylingTips || ['Use your owned piece as the anchor', 'Shop items complete the silhouette'],
        });
    }

    logger.debug('Rebuilt mixed daily deck in hook', {
        style,
        count: rebuilt.length,
        uniqueCombinations: uniqueOutfitCombinationCount(rebuilt),
        shopItemsByOutfit: rebuilt.map((outfit) => outfit.items.filter((item) => item.isShopItem).length),
        wardrobeItemsByOutfit: rebuilt.map((outfit) => outfit.items.filter((item) => !item.isShopItem).length),
    });

    return rebuilt;
}

/**
 * Normalize AI outfit items so Home can always display *something*.
 *
 * Previously we dropped any item without a matching http/file/data image
 * URL, then required an exact top+bottom+shoes composition. In production
 * that rejected 100% of valid AI outfits because the edge function
 * sometimes returns items whose IDs were hallucinated by the LLM (so
 * wardrobe lookup fails and imageUrl ends up empty) — the empty-card bug
 * the user was seeing.
 *
 * New policy:
 *   1. Keep every item the server returned. If an item has no usable
 *      image, stamp its `macroCategory` on so the collage can render a
 *      labelled placeholder tile ("Top", "Bottom", "Shoes") instead of a
 *      blank card.
 *   2. Deduplicate within each canonical slot so the collage never gets
 *      two bottoms or two pairs of shoes.
 *   3. Keep any outfit that ended up with at least 1 item. The collage
 *      already fills missing slots with shirt-icon placeholders.
 */
function filterRenderableOutfits(outfits: GeneratedOutfit[]): GeneratedOutfit[] {
    return outfits
        .map((o) => {
            const seenMacro = new Set<string>();
            const items: GeneratedOutfit['items'] = [];
            for (const rawItem of o.items ?? []) {
                const macro = canonicalMacro(rawItem);

                // Force a canonical macroCategory so the collage's slot
                // logic (`top` / `outerwear` / `bottom` / `shoes`) always
                // has something to bucket the item into.
                const item: GeneratedOutfit['items'][number] = {
                    ...rawItem,
                    macroCategory: macro,
                };

                // Items without a clearly renderable image still render — the
                // collage's CollageImage component will attempt to load the
                // URL and fall back to a shirt-icon placeholder via onError
                // if it fails.  We no longer blank out imageUrl/image here
                // because that prevented the Image component from even trying
                // to load potentially valid URLs (e.g. non-http schemes,
                // relative paths that RN might resolve, etc.).
                // if (!itemHasRenderableImage(item)) { item.imageUrl = ''; item.image = ''; }

                if (macro !== 'other') {
                    if (seenMacro.has(macro)) continue;
                    seenMacro.add(macro);
                }
                items.push(item);
            }
            return { ...o, items };
        })
        .filter((o) => o.items.length >= 1);
}

function outfitSignature(outfit: GeneratedOutfit): string {
    return (outfit.items || [])
        .map((item) => String(item.id || item.name || item.type || '').trim())
        .filter(Boolean)
        .sort()
        .join('|');
}

function uniqueOutfitCombinationCount(outfits: GeneratedOutfit[]): number {
    return new Set(outfits.map(outfitSignature).filter(Boolean)).size;
}

function errorMessage(err: unknown): string {
    return err instanceof Error ? err.message : 'Failed to generate daily outfit';
}

function hasUsableDailyDeck(
    outfits: GeneratedOutfit[],
    requestedVariants: number,
    style?: string,
    occasion?: string,
): boolean {
    if (outfits.length === 0) return false;

    // Small callers can keep the older "any renderable outfit" behavior.
    if (requestedVariants < 10) return true;

    // Home asks for 15. Do not reuse a stale deck unless it can actually fill
    // the carousel with distinct combinations.
    const expectedCount = Math.min(requestedVariants, 15);
    const expectedUnique = expectedCount;
    if (outfits.length < expectedCount) return false;
    if (uniqueOutfitCombinationCount(outfits) < expectedUnique) return false;
    // Team Collaboration is shop-only: every piece must be a shop pick and
    // every combination must be unique — never mixed, never repeated.
    if (isShopOnlyDeck(style, occasion)) {
        return outfits.every((outfit) => {
            const items = outfit.items || [];
            return items.length > 0 && items.every((item) => (item as DailyOutfitItem).isShopItem === true);
        });
    }
    const everyOutfitIsMixed = outfits.every((outfit) => {
        const items = outfit.items || [];
        return items.some((item) => !item.isShopItem && item.id !== 'placeholder_shoes')
            && items.some((item) => item.isShopItem);
    });
    return everyOutfitIsMixed;
}

// ─────────────────────────────────────────────────────────────────────────────
// Hook
// ─────────────────────────────────────────────────────────────────────────────

// Stable selector — selecting a primitive (string | undefined) avoids the
// React 18 "getSnapshot should be cached" warning.
const selectUserId = (s: { user?: { id?: string } | null }) => s.user?.id;

export function useDailyAIOutfit({
    style,
    occasion,
    weather,
    variants = 3,
    enabled = true,
}: UseDailyAIOutfitOptions): UseDailyAIOutfitResult {
    const userId = useAuthStore(selectUserId);
    const preferences = useStylePreferenceStore((state) => state.preferences);
    const wardrobeItems = useWardrobeStore((state) => state.items);
    const loadedShopItems = useShopCatalogStore((state) => state.items);

    const [outfits, setOutfits] = useState<GeneratedOutfit[]>([]);
    // Start as false — the pre-check effect below will set it to true only
    // when a fresh cache entry is NOT available and the edge function must run.
    const [loading, setLoading] = useState(false);
    const [error, setError]     = useState<string | null>(null);

    const inflightRef   = useRef<Promise<void> | null>(null);
    // Tracks whether the pre-cache-check has completed so `run` is only
    // invoked once the check has had a chance to short-circuit.
    const cacheCheckedRef = useRef(false);

    // Keep weather in a ref so `run` can read the latest value without
    // needing weather in its dependency array. If weather were a dep, `run`
    // would be recreated every time weather changes, which would trigger the
    // generation useEffect → setLoading(true) → re-render → new run → loop.
    const weatherRef = useRef(weather);
    weatherRef.current = weather;

    const preferencesRef = useRef(preferences);
    preferencesRef.current = preferences;

    const wardrobeItemsRef = useRef(wardrobeItems);
    wardrobeItemsRef.current = wardrobeItems;

    const loadedShopItemsRef = useRef(loadedShopItems);
    loadedShopItemsRef.current = loadedShopItems;

    const run = useCallback(
        async (forceRegenerate: boolean) => {
            if (!enabled) {
                setLoading(false);
                return;
            }
            // Never generate for an anonymous/stale session. During delete ->
            // create there is a window with no userId where the old code built
            // 'anon'-keyed decks (edge call + 1350-iteration rebuild) that were
            // immediately thrown away once the real userId arrived — pure
            // freeze. The Home gate (dailyDecksEnabled) already prevents this,
            // this is defense-in-depth.
            if (!userId) {
                setLoading(false);
                return;
            }

            const key     = storageKey(userId, style);
            const dateKey = todayKey();

            if (!forceRegenerate) {
                try {
                    const raw = await AsyncStorage.getItem(key);
                    if (raw) {
                        const cached = JSON.parse(raw) as CachedEntry;
                        const cachedRenderable = rebuildMixedDailyDeck(filterRenderableOutfits(
                            Array.isArray(cached.outfits) ? cached.outfits : [],
                        ), variants, style, occasion, weatherRef.current, wardrobeItemsRef.current, loadedShopItemsRef.current);
                        if (
                            cached.dateKey === dateKey
                            && hasUsableDailyDeck(cachedRenderable, variants, style, occasion)
                        ) {
                            logger.debug('Using cached daily outfits', {
                                style,
                                count: cachedRenderable.length,
                                uniqueCombinations: uniqueOutfitCombinationCount(cachedRenderable),
                                dateKey,
                            });
                            setOutfits(cachedRenderable);
                            setLoading(false);
                            return;
                        } else if (cached.dateKey === dateKey && cachedRenderable.length > 0) {
                            logger.debug('Ignoring sparse daily outfit cache', {
                                style,
                                count: cachedRenderable.length,
                                uniqueCombinations: uniqueOutfitCombinationCount(cachedRenderable),
                                requestedVariants: variants,
                                dateKey,
                            });
                        }
                    }
                } catch (cacheErr) {
                    logger.warn('Failed to read daily outfit cache', cacheErr);
                }
            }

            setLoading(true);
            setError(null);
            try {
                // Read weather from the ref so this callback doesn't need
                // weather in its dependency array (avoids infinite loop).
                const currentWeather = weatherRef.current;
                // Default to a "cool" (15°C) weather when none is available.
                // Without this, the edge function's `needsLayering()` returns
                // false and builds 3-item non-layered outfits — which then
                // fail our 4-slot renderability filter because there is no
                // base top underneath the blazer/jacket. See
                // `supabase/functions/generate-outfits/index.ts::needsLayering`.
                const effectiveWeather = currentWeather
                    ? { temp: currentWeather.temp, condition: currentWeather.condition }
                    : { temp: 15, condition: 'cool' };
                const result = await generateOutfitsFromDB({
                    stylePreferences: style,
                    occasion,
                    weather: effectiveWeather,
                    limit: variants,
                    preferences: preferencesRef.current,
                });

                const rawOutfits = (result.outfits ?? []).filter(
                    (o) => Array.isArray(o.items) && o.items.length > 0,
                );
                const fresh = rebuildMixedDailyDeck(
                    filterRenderableOutfits(rawOutfits),
                    variants,
                    style,
                    occasion,
                    effectiveWeather,
                    wardrobeItemsRef.current,
                    loadedShopItemsRef.current,
                );

                logger.debug('Outfit generation result', {
                    style,
                    source: result.source,
                    raw: rawOutfits.length,
                    fresh: fresh.length,
                    uniqueCombinations: new Set(
                        fresh.map((outfit) => outfit.items.map((item) => String(item.id)).sort().join('|')),
                    ).size,
                    shopItemsByOutfit: fresh.map(
                        (outfit) => outfit.items.filter((item) => item.isShopItem).length,
                    ),
                    sampleSignatures: fresh.slice(0, 3).map(
                        (outfit) => outfit.items.map((item) => String(item.id).slice(0, 8)).sort().join('+'),
                    ),
                    sampleItem: rawOutfits[0]?.items?.[0]
                        ? {
                            id: rawOutfits[0].items[0].id,
                            hasImageUrl: Boolean(rawOutfits[0].items[0].imageUrl),
                            hasImage: Boolean(rawOutfits[0].items[0].image),
                            macroCategory: rawOutfits[0].items[0].macroCategory,
                            type: rawOutfits[0].items[0].type,
                            name: rawOutfits[0].items[0].name,
                        }
                        : null,
                });

                if (fresh.length === 0) {
                    logger.warn('No renderable outfits returned; showing empty state', {
                        style,
                        raw: rawOutfits.length,
                    });
                    setOutfits([]);
                    setLoading(false);
                    return;
                }

                setOutfits(fresh);

                const entry: CachedEntry = {
                    dateKey,
                    style,
                    occasion,
                    outfits: fresh,
                    createdAt: new Date().toISOString(),
                };
                try {
                    if (hasUsableDailyDeck(fresh, variants, style, occasion)) {
                        await AsyncStorage.setItem(key, JSON.stringify(entry));
                    } else {
                        await AsyncStorage.removeItem(key);
                        logger.debug('Skipped caching sparse daily outfit deck', {
                            style,
                            count: fresh.length,
                            uniqueCombinations: uniqueOutfitCombinationCount(fresh),
                            requestedVariants: variants,
                        });
                    }
                } catch (writeErr) {
                    logger.warn('Failed to persist daily outfit cache', writeErr);
                }
            } catch (err: unknown) {
                logger.error('Daily outfit generation failed', { style, err });
                setError(errorMessage(err));
            } finally {
                setLoading(false);
            }
        },
        // weather is read from weatherRef inside the callback so it does NOT
        // need to be listed here. Adding weather to deps would recreate `run`
        // on every weather update → trigger the generation effect → setLoading
        // → re-render → new run → infinite loop.
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [enabled, userId, style, occasion, variants],
    );

    /**
     * Pre-check effect — runs BEFORE the main generation effect.
     *
     * Reads AsyncStorage synchronously (as early as possible) and, if a
     * valid cache entry for today already exists, populates state and marks
     * the check as done WITHOUT ever setting `loading = true` or calling
     * the edge function.  Only when the cache is stale or absent does it
     * set `cacheCheckedRef.current = true` so the generation effect below
     * can proceed.
     *
     * This satisfies the spec requirement: "this check SHALL happen before
     * the useEffect fires the run() function, not inside it after the
     * loading state is set."
     */
    useEffect(() => {
        if (!enabled) return;
        // Same anonymous-session guard as run(): don't even start the
        // staggered timer when there's no userId yet.
        if (!userId) return;

        let cancelled = false;
        let timeoutId: ReturnType<typeof setTimeout> | null = null;

        // Defer outfit generation past app initialization (wardrobe rehydrate,
        // subscription checks, catalog fetch) to avoid competing for
        // JS thread / network right after splash. Staggered per style so the
        // two Home sections (business_casual + old_money) don't fire their
        // edge-function + fallback queries in the same tick (thundering herd
        // that stalled cold start). Plain setTimeout — InteractionManager is
        // deprecated (see startup warning) and runAfterInteractions still
        // fires during the herd.
        let hash = 0;
        for (let i = 0; i < style.length; i++) hash = (Math.imul(31, hash) + style.charCodeAt(i)) | 0;
        const startupDelayMs = 1500 + (Math.abs(hash) % 1500);

        timeoutId = setTimeout(() => {
                if (cancelled) return;

                (async () => {
                    const key     = storageKey(userId, style);
                    const dateKey = todayKey();

                    try {
                        const raw = await AsyncStorage.getItem(key);
                        if (!cancelled && raw) {
                            const cached = JSON.parse(raw) as CachedEntry;
                            const cachedRenderable = rebuildMixedDailyDeck(filterRenderableOutfits(
                                Array.isArray(cached.outfits) ? cached.outfits : [],
                            ), variants, style, occasion, weatherRef.current, wardrobeItemsRef.current, loadedShopItemsRef.current);
                            if (cached.dateKey === dateKey && hasUsableDailyDeck(cachedRenderable, variants, style, occasion)) {
                                logger.debug('Pre-check: serving cached daily outfits (edge function skipped)', {
                                    style,
                                    count: cachedRenderable.length,
                                    uniqueCombinations: uniqueOutfitCombinationCount(cachedRenderable),
                                    dateKey,
                                });
                                setOutfits(cachedRenderable);
                                // loading stays false — no edge function call needed.
                                cacheCheckedRef.current = true;
                                return;
                            } else if (cached.dateKey === dateKey && cachedRenderable.length > 0) {
                                logger.debug('Pre-check: ignoring sparse daily outfit cache', {
                                    style,
                                    count: cachedRenderable.length,
                                    uniqueCombinations: uniqueOutfitCombinationCount(cachedRenderable),
                                    requestedVariants: variants,
                                    dateKey,
                                });
                            }
                        }
                    } catch (cacheErr) {
                        logger.warn('Pre-check: failed to read daily outfit cache', cacheErr);
                    }

                    // Cache is absent or stale — allow the generation effect to run.
                    if (!cancelled) {
                        cacheCheckedRef.current = true;
                        // Trigger the generation effect by setting loading now so
                        // the UI shows a spinner while the edge function is called.
                        setLoading(true);
                    }
                })();
        }, startupDelayMs);

        return () => {
            cancelled = true;
            if (timeoutId) clearTimeout(timeoutId);
        };
        // Re-run whenever the identity of the cache slot changes.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [enabled, userId, style]);

    useEffect(() => {
        if (!enabled) return;
        if (!userId) return;
        // Wait until the pre-check has completed before deciding whether to
        // call the edge function.  If the pre-check already served cached
        // data it will NOT have set loading=true, so this effect is a no-op.
        if (!cacheCheckedRef.current) return;
        if (!loading) return;
        if (inflightRef.current) return;
        const p = run(false).finally(() => {
            if (inflightRef.current === p) inflightRef.current = null;
        });
        inflightRef.current = p;
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [enabled, userId, style, occasion, loading, run]);

    const regenerate = useCallback(async () => {
        cacheCheckedRef.current = true;
        // Register this call in `inflightRef` (same guard the auto-run effect
        // above uses) BEFORE awaiting anything. `run(true)` synchronously sets
        // `loading = true`, which by itself would satisfy that effect's
        // `if (!loading) return;` check on the next render and fire a second,
        // redundant `run(false)` alongside this forced one — wasting a full
        // generation call (a real cost when it hits the AI edge function
        // rather than the local fallback). Claiming `inflightRef` here makes
        // that effect's `if (inflightRef.current) return;` guard skip it.
        const p = run(true).finally(() => {
            if (inflightRef.current === p) inflightRef.current = null;
        });
        inflightRef.current = p;
        await p;
    }, [run]);

    return { outfits, loading, error, regenerate };
}

export default useDailyAIOutfit;
