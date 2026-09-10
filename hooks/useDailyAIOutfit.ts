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
import { InteractionManager } from 'react-native';

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
import type { ShopCatalogItem } from '../features/try-on/types';
import { SHOP_CATALOG_ITEMS } from '../data/shopCatalogItems';
import { CLASSIC_MENS_ITEMS } from '../data/inspoMensShopItems';
import { normalizeStyleId, scoreItemForStyle, type StyleId } from '../features/outfit-generator/utils/styleInference';

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
    // v14 invalidates decks created before the rebuild became style-ranked
    // and required all 15 daily combinations to be unique.
    // v13 invalidates any deck created before the Home hook forcibly rebuilt
    // large daily batches as mixed wardrobe+shop outfits.
    // v12 invalidates decks that were unique but shop-only. Daily looks should
    // be mixed when the user has wardrobe items: owned anchor + shop complements.
    // v11 invalidates daily batches generated before the client local fallback
    // could use the bundled shop catalog when Supabase returned no shop rows.
    // It also keeps rejecting same-combination 15-card decks below.
    // v10 invalidates daily batches generated from too-small wardrobe-only
    // pools. Daily Home decks now require enough unique combinations for the
    // requested 15-card carousel before the cache can short-circuit generation.
    return `daily_ai_outfit_v14::${userId ?? 'anon'}::${style}`;
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

function rebuildMixedDailyDeck(
    inputOutfits: GeneratedOutfit[],
    requestedVariants: number,
    style: string,
    occasion: string,
    weather: DailyOutfitWeather | null | undefined,
    wardrobeItems: ClothingItem[],
    shopCatalogItems: ShopCatalogItem[],
): GeneratedOutfit[] {
    if (requestedVariants < 10 || wardrobeItems.length === 0) return inputOutfits;

    const layered = needsDailyLayering(style, weather);
    const targetSlots: DailySlot[] = layered ? ['outerwear', 'top', 'bottom', 'shoes'] : ['top', 'bottom', 'shoes'];
    const count = Math.min(requestedVariants, 15);

    const existingItems = inputOutfits.flatMap((outfit) => outfit.items || []);
    const wardrobeOutfitItems = wardrobeItems.map(toWardrobeOutfitItem).filter((item) => canonicalMacro(item) !== 'other');
    const staticShop = [...SHOP_CATALOG_ITEMS, ...CLASSIC_MENS_ITEMS];
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

    for (let i = 0; rebuilt.length < count && i < count * 4; i++) {
        const anchor = wardrobeAnchors[i % wardrobeAnchors.length];
        const anchorSlot = targetSlots.includes(canonicalMacro(anchor) as DailySlot)
            ? canonicalMacro(anchor) as DailySlot
            : targetSlots[i % targetSlots.length];

        let items: DailyOutfitItem[] = [];
        let finalSignature = '';
        const maxAttempts = Math.max(30, count * 4);

        for (let attempt = 0; attempt < maxAttempts; attempt++) {
            const usedIds = new Set<string>([String(anchor.id)]);
            const itemsBySlot = new Map<DailySlot, DailyOutfitItem>();
            itemsBySlot.set(anchorSlot, {
                ...anchor,
                macroCategory: anchorSlot,
                isShopItem: false,
                recommendation: 'From your wardrobe',
            });

            let divisor = 1;
            targetSlots.forEach((slot, slotIndex) => {
                if (itemsBySlot.has(slot)) return;
                const primaryPool = shopBuckets[slot].length > 0 ? shopBuckets[slot] : wardrobeBuckets[slot];
                const fallbackPool = [
                    ...primaryPool,
                    ...existingBuckets[slot],
                    ...wardrobeBuckets[slot],
                ];
                const pool = fallbackPool.length > 0 ? fallbackPool : primaryPool;
                const poolIndex = Math.floor((i + attempt) / divisor) + slotIndex * 3;
                const pick = pickFrom(pool, poolIndex, usedIds);
                divisor *= Math.max(pool.length, 1);
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
            const isMixed = candidateItems.some((item) => !item.isShopItem)
                && candidateItems.some((item) => item.isShopItem);
            const signature = candidateItems.map((item) => String(item.id)).sort().join('|');

            if (candidateItems.length === targetSlots.length && isMixed && signature && !signatures.has(signature)) {
                items = candidateItems;
                finalSignature = signature;
                break;
            }

            if (candidateItems.length > items.length) {
                items = candidateItems;
                finalSignature = signature;
            }
        }

        if (!items.length || !finalSignature || signatures.has(finalSignature)) continue;
        signatures.add(finalSignature);

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

function hasUsableDailyDeck(outfits: GeneratedOutfit[], requestedVariants: number): boolean {
    if (outfits.length === 0) return false;

    // Small callers can keep the older "any renderable outfit" behavior.
    if (requestedVariants < 10) return true;

    // Home asks for 15. Do not reuse a stale deck unless it can actually fill
    // the carousel with distinct combinations.
    const expectedCount = Math.min(requestedVariants, 15);
    const expectedUnique = expectedCount;
    if (outfits.length < expectedCount) return false;
    const everyOutfitIsMixed = outfits.every((outfit) => {
        const items = outfit.items || [];
        return items.some((item) => !item.isShopItem && item.id !== 'placeholder_shoes')
            && items.some((item) => item.isShopItem);
    });
    return everyOutfitIsMixed && uniqueOutfitCombinationCount(outfits) >= expectedUnique;
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
                            && hasUsableDailyDeck(cachedRenderable, variants)
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
                    if (hasUsableDailyDeck(fresh, variants)) {
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

        let cancelled = false;

        // Defer outfit generation by 1500ms on startup to avoid competing
        // with app initialization (wardrobe rehydrate, subscription checks,
        // notification scheduling). This spreads CPU/network load and
        // eliminates the UI freeze users see right after splash screen.
        const startupDelayMs = 1500;

        const task = InteractionManager.runAfterInteractions(() => {
            const timeoutId = setTimeout(() => {
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
                            if (cached.dateKey === dateKey && hasUsableDailyDeck(cachedRenderable, variants)) {
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

            // Return cleanup for the timeout
            return () => clearTimeout(timeoutId);
        });

        return () => {
            cancelled = true;
            task.cancel();
        };
        // Re-run whenever the identity of the cache slot changes.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [enabled, userId, style]);

    useEffect(() => {
        if (!enabled) return;
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
