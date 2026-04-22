/**
 * outfitGenerationService.ts
 *
 * Single entry point for AI outfit generation.
 * Calls the generate-outfits Supabase Edge Function which:
 *   1. Authenticates the user via JWT
 *   2. Fetches their clothing_items directly from the DB
 *   3. Calls NVIDIA / Gemini to build outfits from the real wardrobe
 *   4. Returns enriched outfits with image URLs
 *
 * Falls back to a local Supabase query + rule-based matching if the
 * edge function is unavailable.
 */

import { supabase } from '../../lib/supabase';
import { fillMissingSlots, fetchShopPoolForStyle, type OutfitSlotId, type ShopFillItem } from './shoppingService';
import type { ClothingItem, ClothingCategory, Season, Occasion } from '../types/domain';
import { mapDbCategory as canonicalMapDbCategory, getMacroCategory as canonicalGetMacroCategory, canonicalizeMacroCategory } from '../utils/categoryMapper';
import { rankItemsForStyle, scoreItemForStyle, normalizeStyleId, type StyleId } from '../../features/outfit-generator/utils/styleInference';

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

export interface GenerateOutfitsParams {
    /** Free-text user prompt, e.g. "beach trip" or "important meeting tomorrow" */
    prompt?: string;
    /** Style preset, e.g. 'old_money' | 'streetwear' | 'minimalist' */
    stylePreferences?: string;
    /** Occasion label, e.g. 'Casual' | 'Formal' | 'Sport' */
    occasion?: string;
    /** Current weather context */
    weather?: { temp: number; condition: string };
    /** How many distinct outfits to generate (default 3) */
    limit?: number;
    /** IDs of specific items the user has pre-selected (manual mode) */
    selectedItemIds?: string[];
}

export interface GeneratedOutfitItem {
    id: string;
    type: string;
    macroCategory?: string;
    color: string;
    name?: string;
    brand?: string;
    /** URL for remote images */
    imageUrl?: string;
    /** Alias kept for backward compat with OutfitItem in the screen */
    image?: string | number;
    recommendation: string;
    isShopItem: boolean;
    price?: number;
    shopUrl?: string;
}

export interface GeneratedOutfit {
    id: string;
    description: string;
    style: string;
    occasion: string;
    confidence: number;
    matchScore: number;
    items: GeneratedOutfitItem[];
    stylingTips: string[];
}

export interface GenerateOutfitsResult {
    success: boolean;
    outfits: GeneratedOutfit[];
    error?: string;
    source?: 'ai' | 'local';
    /** Whether the backend applied the 4-slot layered schema. */
    layered?: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main entry point
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Generate outfits via the Supabase Edge Function.
 * The edge function handles DB fetch + AI call server-side.
 * Falls back to local DB fetch + rule-based matching on failure.
 */
export async function generateOutfitsFromDB(
    params: GenerateOutfitsParams
): Promise<GenerateOutfitsResult> {
    const TIMEOUT_MS = 45_000;
    try {
        const invokePromise = supabase.functions.invoke('generate-outfits', {
            body: {
                prompt: params.prompt || '',
                stylePreferences: params.stylePreferences || 'Casual',
                occasion: params.occasion || 'Everyday',
                weather: params.weather,
                limit: params.limit ?? 3,
                selectedItemIds: params.selectedItemIds ?? [],
            },
        });

        const timeoutPromise = new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error('outfit_generation_timeout')), TIMEOUT_MS)
        );

        const { data, error } = await Promise.race([invokePromise, timeoutPromise]) as Awaited<typeof invokePromise>;

        if (error) {
            console.warn('[outfitGenerationService] Edge function error, using local fallback:', error);
            return generateOutfitsLocally(params);
        }

        if (!data?.success || !Array.isArray(data?.outfits) || data.outfits.length === 0) {
            console.warn('[outfitGenerationService] Empty AI response, using local fallback');
            return generateOutfitsLocally(params);
        }

        // Check that outfits actually contain items — edge function may return outfits with 0 items
        const hasItems = data.outfits.some((o: any) => Array.isArray(o.items) && o.items.length > 0);
        if (!hasItems) {
            console.warn('[outfitGenerationService] AI returned outfits with no items, using local fallback');
            return generateOutfitsLocally(params);
        }

        const outfits: GeneratedOutfit[] = data.outfits.map((o: any) => mapRawOutfit(o, params));
        return { success: true, outfits, source: data.source ?? 'ai', layered: Boolean(data.layered) };

    } catch (err: any) {
        if (err?.message === 'outfit_generation_timeout') {
            console.warn('[outfitGenerationService] Edge function timed out, using local fallback');
            return generateOutfitsLocally(params);
        }
        console.error('[outfitGenerationService] Unexpected error:', err);
        return generateOutfitsLocally(params);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Local fallback — fetches from DB and applies rule-based matching
// ─────────────────────────────────────────────────────────────────────────────

async function generateOutfitsLocally(
    params: GenerateOutfitsParams
): Promise<GenerateOutfitsResult> {
    try {
        const items = await fetchClothingItemsFromDB(params.selectedItemIds);

        const layered = isLayeredWeather(params.weather, params.prompt);
        const requiredSlots: OutfitSlotId[] = ['top', 'bottom', 'shoes'];
        if (layered) requiredSlots.push('outerwear');

        // ── Fetch style-matched shop items for ALL slots ──────────────────
        // Previously we only fetched shop items for MISSING macro-category
        // slots. That meant a user with casual sneakers would get those
        // sneakers in a "business_casual" outfit instead of shop loafers.
        // Now we fetch style-ranked shop items for every slot so the local
        // builder can pick the best match regardless of wardrobe contents.
        let shopPool: ShopFillItem[] = [];
        try {
            shopPool = await fetchShopPoolForStyle(requiredSlots, params.stylePreferences, 5);
        } catch (_) {
            // Shop catalog unreachable — continue with wardrobe only.
        }

        // Also fill any slots the shop pool couldn't cover (e.g. no shoes
        // in shop_catalog at all) via the legacy missing-slot path.
        const shopMacros = new Set(shopPool.map(sf => sf.macroCategory.toLowerCase()));
        const wardrobeMacros = new Set(items.map(i => getMacroCategory(i.category, i.subCategory)).map(m => m.toLowerCase()));
        const stillMissing = requiredSlots.filter(s => !shopMacros.has(s) && !wardrobeMacros.has(s)) as OutfitSlotId[];
        let legacyFills: ShopFillItem[] = [];
        if (stillMissing.length > 0) {
            try {
                legacyFills = await fillMissingSlots(stillMissing, params.stylePreferences);
            } catch (_) { /* ignore */ }
        }

        const allShopFills = [...shopPool, ...legacyFills];

        // Merge shop items into the wardrobe pool so buildLocalOutfits can
        // pick them alongside real wardrobe items.
        const shopAsClothing = allShopFills.map(sf => ({
            id: sf.id,
            userId: '',
            imageUrl: sf.image || '',
            category: sf.macroCategory as ClothingCategory,
            subCategory: sf.type || sf.macroCategory,
            primaryColor: sf.color || 'neutral',
            colorHex: '#000000',
            pattern: 'solid' as const,
            material: '',
            brand: sf.brand || undefined,
            name: sf.name || sf.type || 'Shop item',
            seasons: [] as Season[],
            occasions: [] as Occasion[],
            wearCount: 0,
            lastWornAt: null as string | null,
            isFavorite: false,
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
        }));

        // Style-rank the combined pool so the best-matching items are
        // picked first by buildLocalOutfits. Shop items that match the
        // style (e.g. loafers for business_casual) should appear before
        // wardrobe items that clash (e.g. graphic tees).
        const styleKey: StyleId = normalizeStyleId(params.stylePreferences || 'casual');
        const mergedItems = rankItemsForStyle([...shopAsClothing, ...items], styleKey);

        // If wardrobe is completely empty but we have shop items, still proceed.
        if (items.length === 0 && shopAsClothing.length === 0) {
            return { success: false, outfits: [], error: 'No clothing items found in your wardrobe.', source: 'local' };
        }

        const outfits = buildLocalOutfits(mergedItems, params);

        // Tag shop items so the UI can display a price / shop badge.
        for (const outfit of outfits) {
            for (const item of outfit.items) {
                if (item.id.startsWith('shop_')) {
                    item.isShopItem = true;
                    const match = allShopFills.find((sf: ShopFillItem) => sf.id === item.id);
                    if (match) {
                        item.price = match.price;
                        item.shopUrl = match.shopUrl;
                    }
                }
            }
        }

        return { success: true, outfits, source: 'local' };
    } catch (err: any) {
        return { success: false, outfits: [], error: err.message, source: 'local' };
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Supabase DB fetch — used by both local fallback and the UI item picker
// ─────────────────────────────────────────────────────────────────────────────

export interface WardrobeDisplayItem {
    id: string;
    image: string;
    imageUrl: string;
    type: string;
    color: string;
    name: string;
    brand?: string;
    macroCategory: string;
    category: string;
}

/**
 * Fetch the current user's clothing items from Supabase.
 * Returns them mapped to a display-friendly shape for the item picker.
 */
export async function fetchWardrobeDisplayItems(
    selectedIds?: string[]
): Promise<WardrobeDisplayItem[]> {
    const { data: sessionData } = await supabase.auth.getSession();
    const userId = sessionData?.session?.user?.id;
    if (!userId) return [];

    let query = supabase
        .from('clothing_items')
        .select('id, type, category, color, primary_color, sub_category, image_url, brand, wear_count')
        .eq('user_id', userId)
        .order('created_at', { ascending: false })
        .limit(100);

    if (selectedIds && selectedIds.length > 0) {
        query = query.in('id', selectedIds);
    }

    const { data, error } = await query;
    if (error || !data) {
        console.error('[outfitGenerationService] fetchWardrobeDisplayItems error:', error);
        return [];
    }

    return data
        .filter((row: any) => row.image_url)
        .map((row: any) => {
            const typeLabel = row.type || row.sub_category || row.category || 'Clothing';
            const cat = row.category || '';
            const mc = getMacroCategory(cat, typeLabel);
            return {
                id: row.id,
                image: row.image_url,
                imageUrl: row.image_url,
                type: typeLabel,
                color: Array.isArray(row.color) ? (row.color[0] || '') : (row.primary_color || row.color || ''),
                name: typeLabel,
                brand: row.brand || undefined,
                macroCategory: mc,
                category: cat,
            };
        });
}

// ─────────────────────────────────────────────────────────────────────────────
// Internals
// ─────────────────────────────────────────────────────────────────────────────

async function fetchClothingItemsFromDB(selectedIds?: string[]): Promise<ClothingItem[]> {
    const { data: sessionData } = await supabase.auth.getSession();
    const userId = sessionData?.session?.user?.id;
    if (!userId) return [];

    let query = supabase
        .from('clothing_items')
        .select('*')
        .eq('user_id', userId)
        .order('created_at', { ascending: false })
        .limit(100);

    if (selectedIds && selectedIds.length > 0) {
        query = query.in('id', selectedIds);
    }

    const { data, error } = await query;
    if (error || !data) return [];

    return data.map((row: any): ClothingItem => ({
        id: row.id,
        userId: row.user_id,
        imageUrl: row.image_url || '',
        thumbnailUrl: row.thumbnail_url ?? undefined,
        category: mapDbCategory(row.category),
        subCategory: row.type || row.sub_category || '',
        primaryColor: Array.isArray(row.color) ? (row.color[0] || '') : (row.primary_color || row.color || ''),
        colorHex: row.color_hex || '#000000',
        pattern: row.pattern || 'solid',
        material: row.material || '',
        brand: row.brand ?? undefined,
        name: row.type || row.sub_category || row.category,
        seasons: Array.isArray(row.season) ? row.season : (Array.isArray(row.seasons) ? row.seasons : []),
        occasions: Array.isArray(row.occasion) ? row.occasion : (Array.isArray(row.occasions) ? row.occasions : []),
        wearCount: row.wear_count || 0,
        lastWornAt: row.last_worn_date || row.last_worn_at || null,
        isFavorite: row.is_favorite || false,
        createdAt: row.created_at,
        updatedAt: row.updated_at,
    }));
}

function mapDbCategory(category: string): ClothingCategory {
    return canonicalMapDbCategory(category);
}

function getMacroCategory(category: string, type: string): string {
    return canonicalGetMacroCategory(category, type);
}

// Formal outerwear keywords (blazer, suit jacket, sport coat, overcoat,
// topcoat, trench, peacoat, tuxedo) — never pair with shorts per styling rule.
function isFormalLayerItem(item: { name?: string; type?: string; brand?: string; macroCategory?: string } | null | undefined): boolean {
    if (!item) return false;
    const blob = `${item.type || ''} ${item.name || ''} ${item.brand || ''}`.toLowerCase();
    const macro = (item.macroCategory || '').toLowerCase();
    const isOuter = macro === 'outerwear' || /jacket|coat|blazer|vest|outerwear/.test(blob);
    if (!isOuter) return false;
    return /\b(blazer|suit\s*jacket|sport\s*coat|sports\s*coat|overcoat|top\s*coat|topcoat|trench|peacoat|pea\s*coat|tuxedo)\b/.test(blob);
}

function isShortsItem(item: { name?: string; type?: string; brand?: string; macroCategory?: string; subCategory?: string } | null | undefined): boolean {
    if (!item) return false;
    const blob = `${item.type || ''} ${item.name || ''} ${item.subCategory || ''}`.toLowerCase();
    const macro = (item.macroCategory || '').toLowerCase();
    const isBottom = macro === 'bottom' || /pant|trouser|jeans|bottom|shorts?|skirt/.test(blob);
    if (!isBottom) return false;
    return /\b(shorts?|bermudas?)\b/.test(blob);
}

function isLayeredWeather(weather?: { temp?: number | null; condition?: string | null }, prompt?: string | null): boolean {
    const promptBlob = (prompt || '').toLowerCase();
    if (/\b(summer|hot|heatwave|tee[-\s]?only|no jacket|no outerwear|beach)\b/.test(promptBlob)) return false;
    const condition = (weather?.condition || '').toString().toLowerCase();
    const temp = typeof weather?.temp === 'number' ? weather.temp : null;
    const coldTemp = temp != null && temp < 18;
    const coldCondition = /\b(cold|chilly|freezing|snow|rain|drizzle|wind|storm)\b/.test(condition);
    return coldTemp || coldCondition;
}

function getItemStyleScore(item: ClothingItem, style: StyleId): number {
    return scoreItemForStyle(
        {
            name: item.name,
            description: '',
            brand: item.brand,
            color: item.primaryColor,
            type: item.subCategory,
            category: item.category,
            macroCategory: getMacroCategory(item.category, item.subCategory),
        },
        style,
    );
}

function strictRejectsForStyleSlot(item: ClothingItem, style: StyleId, slot: 'top' | 'outerwear' | 'bottom' | 'shoes'): boolean {
    const blob = `${item.name || ''} ${item.subCategory || ''} ${item.category || ''} ${item.brand || ''}`.toLowerCase();

    if (slot === 'bottom' && /\b(shorts?|bermudas?|cargo|sweatpants?|joggers?)\b/.test(blob)) return true;

    if (style === 'business_casual' || style === 'old_money') {
        if (slot === 'top' && /\b(t-?shirt|tee|graphic|tank|sleeveless|mesh)\b/.test(blob)) return true;
        if (slot === 'outerwear' && /\b(hoodie|puffer|windbreaker|track)\b/.test(blob)) return true;
        if (slot === 'shoes' && /\b(chunky|basketball|skate|running|trainer|retro sneaker|retro sneakers)\b/.test(blob)) return true;
        if (slot === 'shoes' && /\b(sneaker|sneakers)\b/.test(blob) && !/\b(leather sneaker|leather sneakers|minimal sneaker|minimal sneakers)\b/.test(blob)) return true;
    }

    if (style === 'old_money') {
        if (slot === 'top' && /\b(logo|oversized)\b/.test(blob)) return true;
    }

    return false;
}

function filterSlotByStyle(
    items: ClothingItem[],
    style: StyleId,
    slot: 'top' | 'outerwear' | 'bottom' | 'shoes',
    minScore = 0.15,
): ClothingItem[] {
    if (items.length === 0) return items;

    // Tier 1: items that pass strict reject AND meet the style score threshold.
    const tier1 = items.filter((item) => {
        if (strictRejectsForStyleSlot(item, style, slot)) return false;
        return getItemStyleScore(item, style) >= minScore;
    });
    if (tier1.length > 0) return tier1;

    // Tier 2: items that aren't hard-rejected (may have low style score
    // but are still acceptable — e.g. a plain shirt with no brand info).
    const tier2 = items.filter((item) => !strictRejectsForStyleSlot(item, style, slot));
    if (tier2.length > 0) return tier2;

    // Tier 3: last resort — return all items so the outfit builder can
    // always compose something. Better a t-shirt than an empty card.
    return items;
}

// Placeholder shoes item injected when the wardrobe has no shoes at all.
// Ensures every outfit satisfies the "SHOES must be in every outfit" contract.
const PLACEHOLDER_SHOES: GeneratedOutfitItem = {
    id: 'placeholder_shoes',
    type: 'shoes',
    macroCategory: 'shoes',
    color: 'neutral',
    name: 'Shoes',
    imageUrl: 'basic_clothing_shoes',
    image: 'basic_clothing_shoes',
    recommendation: 'Add shoes to your wardrobe for better outfits',
    isShopItem: false,
};

function normalizeGeneratedOutfitItems(
    items: GeneratedOutfitItem[],
    params: GenerateOutfitsParams,
): GeneratedOutfitItem[] {
    const normalizedItems = items.map((item, index) => {
        // Canonicalize macroCategory so downstream pick() / classifier calls
        // can rely on 'top' | 'bottom' | 'outerwear' | 'shoes'. AI responses
        // sometimes carry 'upper_body' / 'lower_body' / 'tops' / 'footwear',
        // which used to silently drop items from the outfit grid.
        const rawMacro = item.macroCategory || '';
        const canonical = canonicalizeMacroCategory(rawMacro);
        const resolvedMacro = canonical !== 'other'
            ? canonical
            : (getMacroCategory('', item.type || '') || 'other').toLowerCase();
        return {
            ...item,
            id: item.id || `item_${index}`,
            macroCategory: resolvedMacro,
        };
    });

    const usedIds = new Set<string>();
    // Layering is weather-only. Style/item-presence must not force a jacket.
    const layered = isLayeredWeather(params.weather, params.prompt);

    const pick = (predicate: (item: GeneratedOutfitItem) => boolean): GeneratedOutfitItem | undefined => {
        const found = normalizedItems.find((item) => {
            const key = String(item.id || '');
            return !usedIds.has(key) && predicate(item);
        });
        if (!found) return undefined;
        usedIds.add(String(found.id || ''));
        return found;
    };

    // Pick bottom first so we can reason about shorts + formal-layer conflicts.
    const bottom =
        pick((item) => item.macroCategory === 'bottom')
        || pick((item) => item.macroCategory !== 'shoes' && item.macroCategory !== 'top' && item.macroCategory !== 'outerwear');
    const bottomIsShorts = isShortsItem(bottom);

    // Pick outerwear only when layered; skip formal outerwear if bottom is shorts.
    const outerwear = layered
        ? (pick((item) => item.macroCategory === 'outerwear' && !(bottomIsShorts && isFormalLayerItem(item)))
            || (!bottomIsShorts ? pick((item) => item.macroCategory === 'outerwear') : undefined))
        : undefined;

    const baseTop =
        pick((item) => item.macroCategory === 'top')
        || pick((item) => item.macroCategory === 'accessory' || item.macroCategory === 'other')
        || pick((item) => item.macroCategory !== 'bottom' && item.macroCategory !== 'shoes' && item.macroCategory !== 'outerwear');

    // Only pick a second top when layered (outerwear present). A 3-item look
    // has exactly one top, no extras.
    const secondTop = layered && outerwear
        ? (pick((item) => item.macroCategory === 'top')
            || pick((item) => item.macroCategory === 'accessory' || item.macroCategory === 'other'))
        : undefined;

    const shoes =
        pick((item) => item.macroCategory === 'shoes')
        || { ...PLACEHOLDER_SHOES };

    return [outerwear, baseTop, secondTop, bottom, shoes].filter(Boolean) as GeneratedOutfitItem[];
}

function buildLocalOutfits(items: ClothingItem[], params: GenerateOutfitsParams): GeneratedOutfit[] {
    const style = params.stylePreferences || 'Casual';
    const occ = params.occasion || 'Everyday';
    const limit = params.limit ?? 3;
    const layered = isLayeredWeather(params.weather, params.prompt);
    const styleKey: StyleId = normalizeStyleId(style);

    const rawBaseTops = items.filter(i => getMacroCategory(i.category, i.subCategory) === 'top');
    const rawOuterwear = items.filter(i => getMacroCategory(i.category, i.subCategory) === 'outerwear');
    const rawBottoms = items.filter(i => getMacroCategory(i.category, i.subCategory) === 'bottom');
    const rawShoes = items.filter(i => getMacroCategory(i.category, i.subCategory) === 'shoes');

    const baseTops = filterSlotByStyle(rawBaseTops, styleKey, 'top', styleKey === 'casual' ? 0.1 : 0.18);
    const outerwear = filterSlotByStyle(rawOuterwear, styleKey, 'outerwear', styleKey === 'casual' ? 0.1 : 0.18);
    const bottoms = filterSlotByStyle(rawBottoms, styleKey, 'bottom', styleKey === 'casual' ? 0.1 : 0.18);
    const shoes = filterSlotByStyle(rawShoes, styleKey, 'shoes', styleKey === 'casual' ? 0.1 : 0.18);
    const legacyTops = [...outerwear, ...baseTops];
    const nonShortsBottoms = bottoms.filter(b => !isShortsItem({
        name: b.name, type: b.subCategory, macroCategory: getMacroCategory(b.category, b.subCategory), subCategory: b.subCategory,
    }));
    const casualOuterwear = outerwear.filter(o => !isFormalLayerItem({
        name: o.name, type: o.subCategory, macroCategory: getMacroCategory(o.category, o.subCategory),
    }));

    const outfits: GeneratedOutfit[] = [];
    const targetItems = layered ? 4 : 3;

    for (let i = 0; i < limit; i++) {
        const outfitItems: GeneratedOutfitItem[] = [];

        // Pre-pick bottom to check for shorts / formal-layer conflict.
        const candidateBottom = bottoms[i % Math.max(bottoms.length, 1)] || bottoms[0] || items[0];
        const candidateBottomIsShorts = !!candidateBottom && isShortsItem({
            name: candidateBottom.name, type: candidateBottom.subCategory,
            macroCategory: getMacroCategory(candidateBottom.category, candidateBottom.subCategory),
            subCategory: candidateBottom.subCategory,
        });

        if (layered) {
            // Prefer casual outerwear if the bottom is shorts; else default outerwear rotation.
            let outer = outerwear[i % Math.max(outerwear.length, 1)];
            const outerIsFormal = !!outer && isFormalLayerItem({
                name: outer.name, type: outer.subCategory, macroCategory: getMacroCategory(outer.category, outer.subCategory),
            });
            if (candidateBottomIsShorts && outerIsFormal) {
                outer = casualOuterwear[i % Math.max(casualOuterwear.length, 1)] || outer;
            }
            const mainTop = outer || legacyTops[i % Math.max(legacyTops.length, 1)];
            const base = baseTops[i % Math.max(baseTops.length, 1)] || legacyTops[(i + 1) % Math.max(legacyTops.length, 1)];
            if (mainTop) outfitItems.push(toDisplayItem(mainTop, 'Main top / outerwear layer'));
            if (base && base.id !== mainTop?.id) outfitItems.push(toDisplayItem(base, 'Base top worn underneath'));
        } else {
            // Non-layered: single top only.
            const top = legacyTops[i % Math.max(legacyTops.length, 1)] || items[0];
            if (top) outfitItems.push(toDisplayItem(top, 'Key piece for this look'));
        }

        // If we still have a formal-layer + shorts conflict, swap bottom for a non-shorts option.
        let finalBottom = candidateBottom;
        if (layered && candidateBottomIsShorts) {
            const firstPick = outfitItems[0];
            if (firstPick && isFormalLayerItem(firstPick) && nonShortsBottoms.length > 0) {
                finalBottom = nonShortsBottoms[i % nonShortsBottoms.length];
            }
        }

        const shoe = shoes[i % Math.max(shoes.length, 1)] || shoes[0];

        if (finalBottom) outfitItems.push(toDisplayItem(finalBottom, 'Pairs well with the top'));
        if (shoe) outfitItems.push(toDisplayItem(shoe, 'Completes the look'));
        else outfitItems.push({ ...PLACEHOLDER_SHOES });

        // Pad only when layered and we still have fewer than 4 items.
        if (layered && outfitItems.length < targetItems && items.length > 0) {
            let fillIndex = 0;
            while (outfitItems.length < targetItems && fillIndex < items.length) {
                const fillItem = items[fillIndex];
                const alreadyAdded = outfitItems.some(oi => oi.id === fillItem.id);
                if (!alreadyAdded) {
                    outfitItems.push(toDisplayItem(fillItem, 'Complementary piece'));
                }
                fillIndex++;
            }
        }
        // Trim accidental extras down to the contract size.
        if (outfitItems.length > targetItems) outfitItems.length = targetItems;

        if (outfitItems.length === 0) continue;

        const normalizedOutfitItems = normalizeGeneratedOutfitItems(outfitItems, params);

        outfits.push({
            id: `local_${i}_${Date.now()}`,
            description: `A ${style} look built from your wardrobe.`,
            style,
            occasion: occ,
            confidence: 0.75,
            matchScore: 0.75,
            items: normalizedOutfitItems,
            stylingTips: layered
                ? ['Layer the base top under the outerwear for depth', 'Keep the palette tonal for a refined finish']
                : ['Add accessories to personalize', 'Roll sleeves or cuffs for a lived-in feel'],
        });
    }

    // If no outfits were created but we have items, create at least one outfit
    if (outfits.length === 0 && items.length > 0) {
        const outfitItems: GeneratedOutfitItem[] = [];
        if (layered) {
            const candidateBottom = bottoms[0] || items[1] || items[0];
            const candidateBottomIsShorts = !!candidateBottom && isShortsItem({
                name: candidateBottom.name, type: candidateBottom.subCategory,
                macroCategory: getMacroCategory(candidateBottom.category, candidateBottom.subCategory),
                subCategory: candidateBottom.subCategory,
            });
            let outer = outerwear[0] || legacyTops[0];
            const outerIsFormal = !!outer && isFormalLayerItem({
                name: outer?.name, type: outer?.subCategory, macroCategory: outer ? getMacroCategory(outer.category, outer.subCategory) : '',
            });
            if (candidateBottomIsShorts && outerIsFormal) {
                outer = casualOuterwear[0] || legacyTops.find(t => !isFormalLayerItem({
                    name: t.name, type: t.subCategory, macroCategory: getMacroCategory(t.category, t.subCategory),
                })) || outer;
            }
            const base = baseTops[0] || legacyTops[0] || outer;
            const finalBottom = (candidateBottomIsShorts && outer && isFormalLayerItem({
                name: outer.name, type: outer.subCategory, macroCategory: getMacroCategory(outer.category, outer.subCategory),
            }) && nonShortsBottoms[0]) ? nonShortsBottoms[0] : candidateBottom;
            const shoe = shoes[0];
            if (outer) outfitItems.push(toDisplayItem(outer, 'Main top / outerwear layer'));
            if (base && base.id !== outer?.id) outfitItems.push(toDisplayItem(base, 'Base top worn underneath'));
            if (finalBottom && finalBottom.id !== base?.id) outfitItems.push(toDisplayItem(finalBottom, 'Pairs well with the top'));
            if (shoe && shoe.id !== finalBottom?.id) outfitItems.push(toDisplayItem(shoe, 'Completes the look'));
            else outfitItems.push({ ...PLACEHOLDER_SHOES });
        } else {
            const top = legacyTops[0] || items[0];
            const bottom = bottoms[0] || items[1] || items[0];
            const shoe = shoes[0];
            if (top) outfitItems.push(toDisplayItem(top, 'Key piece for this look'));
            if (bottom && bottom.id !== top?.id) outfitItems.push(toDisplayItem(bottom, 'Pairs well with the top'));
            if (shoe && shoe.id !== bottom?.id) outfitItems.push(toDisplayItem(shoe, 'Completes the look'));
            else outfitItems.push({ ...PLACEHOLDER_SHOES });
            if (outfitItems.length > 3) outfitItems.length = 3;
        }

        if (outfitItems.length > 0) {
            const normalizedOutfitItems = normalizeGeneratedOutfitItems(outfitItems, params);
            outfits.push({
                id: `local_0_${Date.now()}`,
                description: `A ${style} look built from your wardrobe.`,
                style,
                occasion: occ,
                confidence: 0.75,
                matchScore: 0.75,
                items: normalizedOutfitItems,
                stylingTips: layered
                    ? ['Layer the base top under the outerwear for depth', 'Keep the palette tonal for a refined finish']
                    : ['Add accessories to personalize', 'Roll sleeves or cuffs for a lived-in feel'],
            });
        }
    }

    return outfits;
}

function toDisplayItem(item: ClothingItem, recommendation: string): GeneratedOutfitItem {
    return {
        id: item.id,
        type: item.subCategory || item.category,
        macroCategory: getMacroCategory(item.category, item.subCategory),
        color: item.primaryColor || 'neutral',
        name: item.name || item.subCategory,
        brand: item.brand,
        imageUrl: item.imageUrl,
        image: item.imageUrl,
        recommendation,
        isShopItem: false,
    };
}

function mapRawOutfit(o: any, params: GenerateOutfitsParams): GeneratedOutfit {
    const items: GeneratedOutfitItem[] = (o.items || []).map((item: any): GeneratedOutfitItem => {
        const resolvedImage = item.imageUrl || item.image_url || item.image || item.thumbnailUrl || item.thumbnail_url || '';
        const resolvedMacroCategory = item.macroCategory || item.macro_category || item.category || item.garmentType || item.garment_type || '';

        return {
            id: item.id || `item_${Date.now()}_${Math.random()}`,
            type: item.type || item.garmentType || item.garment_type || 'clothing',
            macroCategory: resolvedMacroCategory,
            color: item.color || item.primaryColor || item.primary_color || 'neutral',
            name: item.name || item.title || item.type || item.garmentType || 'Item',
            brand: item.brand || '',
            imageUrl: typeof resolvedImage === 'string' ? resolvedImage : '',
            image: resolvedImage,
            recommendation: item.recommendation || 'Selected for this outfit',
            isShopItem: item.isShopItem ?? item.is_shop_item ?? false,
            price: item.price,
            shopUrl: item.shopUrl || item.shop_url,
        };
    });

    // De-duplicate exact duplicate ids first.
    const seen = new Set<string>();
    const dedupedItems = items.filter(item => {
        const key = String(item.id || '');
        if (!key) return true;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
    });

    const normalizedItems = normalizeGeneratedOutfitItems(dedupedItems, params);

    return {
        id: o.id || `outfit_${Date.now()}_${Math.random()}`,
        description: o.description || `A ${o.style || params.stylePreferences || 'stylish'} look`,
        style: o.style || params.stylePreferences || 'Casual',
        occasion: o.occasion || params.occasion || 'Everyday',
        confidence: o.confidence ?? 0.82,
        matchScore: o.confidence ?? o.matchScore ?? 0.82,
        items: normalizedItems,
        stylingTips: Array.isArray(o.stylingTips) ? o.stylingTips : ['Style to your preference'],
    };
}
