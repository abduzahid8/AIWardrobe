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
import useWardrobeStore from '../../store/wardrobeStore';

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

export interface GenerateOutfitsParams {
    /** Free-text user prompt, e.g. "beach trip" or "important meeting tomorrow" */
    prompt?: string;
    /** Style preset, e.g. 'old_money' | 'semi_classic' | 'minimalist' | 'casual' */
    stylePreferences?: string;
    /** Occasion label, e.g. 'Casual' | 'Formal' | 'Sport' */
    occasion?: string;
    /** Current weather context */
    weather?: { temp: number; condition: string };
    /** How many distinct outfits to generate (default 3) */
    limit?: number;
    /** IDs of specific items the user has pre-selected (manual mode) */
    selectedItemIds?: string[];
    /** When set, the edge function locks this item into its slot in every outfit. */
    anchorItemId?: string;
    /** User's style quiz preferences to customize recommendations */
    preferences?: {
        favoriteColors?: string[];
        avoidColors?: string[];
        fitPreference?: 'loose' | 'fitted' | 'balanced';
        likedPatterns?: string[];
        dislikedPatterns?: string[];
        primaryOccasions?: string[];
        prefersSustainable?: boolean;
        styleGoals?: string[];
    };
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

/**
 * Team Collaboration is always 100% shop — no wardrobe anchor.
 * Matches by style (`business_casual`) OR occasion label
 * (`Team Collaboration`) so an admin rename of one side can't
 * silently re-enable mixed decks.
 */
export function isShopOnlyDeck(style?: string | null, occasion?: string | null): boolean {
    const s = (style || '').toLowerCase().replace(/[\s-]+/g, '_');
    if (s === 'business_casual') return true;
    const o = (occasion || '').toLowerCase();
    if (o.includes('team') && o.includes('collab')) return true;
    if (o.trim() === 'team collaboration') return true;
    return false;
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
        // Shop-only decks (Team Collaboration) never accept a wardrobe anchor —
        // every item in every outfit must come from shopping.
        const shopOnlyRequest = isShopOnlyDeck(params.stylePreferences, params.occasion);
        const invokePromise = supabase.functions.invoke('generate-outfits', {
            body: {
                prompt: params.prompt || '',
                stylePreferences: params.stylePreferences || 'Casual',
                occasion: params.occasion || 'Everyday',
                weather: params.weather,
                limit: params.limit ?? 3,
                selectedItemIds: params.selectedItemIds ?? [],
                anchorItemId: shopOnlyRequest ? undefined : params.anchorItemId,
                preferences: params.preferences,
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

export async function generateOutfitsLocally(
    params: GenerateOutfitsParams
): Promise<GenerateOutfitsResult> {
    try {
        const items = await fetchClothingItemsFromDB(params.selectedItemIds);

        const layered = isLayeredWeather(params.weather, params.prompt);
        const requiredSlots: OutfitSlotId[] = ['top', 'bottom', 'shoes'];
        if (layered) requiredSlots.push('outerwear');

        let filteredItems = [...items];

        if (params.preferences) {
            const avoidColors = params.preferences.avoidColors || [];
            if (avoidColors.length > 0) {
                const avoid = avoidColors.map((c: string) => c.toLowerCase().trim()).filter(Boolean);
                filteredItems = filteredItems.filter(item => {
                    const color = (item.primaryColor || '').toLowerCase();
                    const name = (item.name || '').toLowerCase();
                    const desc = (item.subCategory || '').toLowerCase();
                    return !avoid.some(c => color.includes(c) || name.includes(` ${c} `) || desc.includes(` ${c} `));
                });
            }
        }

        // Wardrobe-first, with enough catalog supplement for large daily decks.
        // One owned item per slot can form an outfit, but it cannot produce
        // 15 visibly different outfits on Home.
        // Exception: Team Collaboration / business_casual is always shop-only —
        // fetch the full shop pool for every required slot, ignore wardrobe counts.
        const shopOnly = isShopOnlyDeck(params.stylePreferences, params.occasion);
        const wardrobeCountBySlot = new Map<string, number>();
        filteredItems.forEach((item) => {
            const slot = getItemMacroCategory(item).toLowerCase();
            wardrobeCountBySlot.set(slot, (wardrobeCountBySlot.get(slot) ?? 0) + 1);
        });
        const isManualSelection = Array.isArray(params.selectedItemIds) && params.selectedItemIds.length > 0;
        const minimumChoicesPerSlot = isManualSelection
            ? 1
            : ((params.limit ?? 3) >= 10 ? 5 : ((params.limit ?? 3) > 1 ? 2 : 1));
        const missingSlots = shopOnly
            ? [...requiredSlots] as OutfitSlotId[]
            : requiredSlots.filter(
                slot => (wardrobeCountBySlot.get(slot) ?? 0) < minimumChoicesPerSlot,
            ) as OutfitSlotId[];

        let shopPool: ShopFillItem[] = [];
        try {
            shopPool = await fetchShopPoolForStyle(missingSlots, params.stylePreferences, minimumChoicesPerSlot);
        } catch (_) {
            // Shop catalog unreachable — continue with the usable wardrobe.
        }

        const coveredShopSlots = new Set(shopPool.map(sf => sf.macroCategory.toLowerCase()));
        const stillMissing = missingSlots.filter(slot => !coveredShopSlots.has(slot)) as OutfitSlotId[];
        let legacyFills: ShopFillItem[] = [];
        if (stillMissing.length > 0) {
            try {
                legacyFills = await fillMissingSlots(stillMissing, params.stylePreferences);
            } catch (_) { /* ignore */ }
        }

        const allShopFills = [...shopPool, ...legacyFills];
        const shopAsClothing = allShopFills.map(sf => ({
            id: sf.id,
            userId: '',
            imageUrl: sf.image || '',
            category: sf.macroCategory as ClothingCategory,
            subCategory: sf.type || sf.macroCategory,
            macroCategory: sf.macroCategory,
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
            isShopItem: true,
            price: sf.price,
            shopUrl: sf.shopUrl,
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
        }));

        let filteredShopAsClothing = [...shopAsClothing];

        if (params.preferences) {
            const avoidColors = params.preferences.avoidColors || [];
            if (avoidColors.length > 0) {
                const avoid = avoidColors.map((c: string) => c.toLowerCase().trim()).filter(Boolean);
                filteredShopAsClothing = filteredShopAsClothing.filter(item => {
                    const color = (item.primaryColor || '').toLowerCase();
                    const name = (item.name || '').toLowerCase();
                    return !avoid.some(c => color.includes(c) || name.includes(` ${c} `));
                });
            }
        }

        // Rank within the wardrobe-first pool. Shop candidates exist only for
        // absent or sparse categories, instead of every category by default.
        // Exception: shop-only decks (Team Collaboration) build purely from
        // shop candidates so no wardrobe piece can leak in.
        const styleKey: StyleId = normalizeStyleId(params.stylePreferences || 'casual');
        const rankedItems = shopOnly
            ? rankItemsForStyle([...filteredShopAsClothing], styleKey)
            : rankItemsForStyle([...filteredShopAsClothing, ...filteredItems], styleKey);
        const rankedIds = new Set(rankedItems.map((item) => item.id));
        const mergedItems = shopOnly
            ? [...rankedItems] as LocalClothingItem[]
            : [
                ...rankedItems,
                ...filteredItems.filter((item) => !rankedIds.has(item.id)),
            ] as LocalClothingItem[];

        // If wardrobe is completely empty but we have shop items, still proceed.
        if (items.length === 0 && shopAsClothing.length === 0) {
            return { success: false, outfits: [], error: 'No clothing items found in your wardrobe.', source: 'local' };
        }
        // Shop-only decks need shop candidates — without them we can't build
        // a valid all-shop outfit.
        if (shopOnly && filteredShopAsClothing.length === 0) {
            return { success: false, outfits: [], error: 'No shop items found for this style.', source: 'local' };
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
    // Primary path: read from the Zustand wardrobeStore (already fetched on app start).
    // This avoids a redundant Supabase round-trip when the store is already populated.
    const storeItems = useWardrobeStore.getState().items;
    if (storeItems.length > 0) {
        if (selectedIds && selectedIds.length > 0) {
            const idSet = new Set(selectedIds);
            return storeItems.filter(item => idSet.has(item.id));
        }
        return storeItems;
    }

    // Fallback: store is empty (e.g. first launch before rehydration completes),
    // so fetch directly from Supabase.
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

function getItemMacroCategory(item: ClothingItem & { macroCategory?: string }): string {
    const explicit = canonicalizeMacroCategory(item.macroCategory);
    if (explicit !== 'other') return explicit;
    return getMacroCategory(item.category, item.subCategory);
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
    // Explicit hot/summer keywords override to non-layered, even without weather.
    if (/\b(summer|hot|heatwave|tee[-\s]?only|no jacket|no outerwear|beach)\b/.test(promptBlob)) return false;
    const condition = (weather?.condition || '').toString().toLowerCase();
    const temp = typeof weather?.temp === 'number' ? weather.temp : null;
    // When we have NEITHER a temperature nor a weather description, default
    // to layered=true. This matches the client's `needsOuterwear` default
    // and keeps the pipeline from producing 3-item outfits that render as
    // "layer + pants + shoes with no base top" on the card.
    if (temp == null && !condition) return true;
    const coldTemp = temp != null && temp < 18;
    const coldCondition = /\b(cold|chilly|freezing|snow|rain|drizzle|wind|storm|cool)\b/.test(condition);
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
            macroCategory: getItemMacroCategory(item),
        },
        style,
    );
}

function strictRejectsForStyleSlot(item: ClothingItem, style: StyleId, slot: 'top' | 'outerwear' | 'bottom' | 'shoes'): boolean {
    const blob = `${item.name || ''} ${item.subCategory || ''} ${item.category || ''} ${item.brand || ''} ${item.pattern || ''}`.toLowerCase();

    if (slot === 'bottom' && /\b(shorts?|bermudas?|cargo|sweatpants?|joggers?)\b/.test(blob)) return true;

    if (style === 'business_casual' || style === 'old_money' || style === 'classic') {
        // Reject loud/sporty/graphic/floral pattern types in formal/classic styles
        if (/\b(print|printed|floral|paisley|camo|tie-dye|graphic|logo|patterns?)\b/.test(blob)) return true;

        // Reject sporty outerwear layers
        if (slot === 'outerwear' && /\b(hoodie|hooded|puffer|windbreaker|track|bomber|zip-?up|casual-?jacket)\b/.test(blob)) return true;

        // Reject sporty bottoms
        if (slot === 'bottom' && /\b(shorts?|bermudas?|cargo|sweatpants?|joggers?)\b/.test(blob)) return true;

        // Reject casual base layers
        if (slot === 'top' && /\b(t-?shirt|tee|graphic|tank|sleeveless|mesh)\b/.test(blob)) return true;

        // Reject non-dress shoes
        if (slot === 'shoes' && /\b(chunky|basketball|skate|running|trainer|retro sneaker|retro sneakers|athletic)\b/.test(blob)) return true;
        if (style === 'old_money') {
            if (slot === 'shoes' && /\b(sneaker|sneakers)\b/.test(blob)) return true;
        } else {
            if (slot === 'shoes' && /\b(sneaker|sneakers)\b/.test(blob) && !/\b(leather sneaker|leather sneakers|minimal sneaker|minimal sneakers)\b/.test(blob)) return true;
        }
    }

    if (style === 'old_money' || style === 'classic') {
        if (slot === 'top' && /\b(logo|oversized|t-shirt|tee|graphic)\b/.test(blob)) return true;
        if (slot === 'outerwear' && /\b(zip-up|windbreaker|casual jacket)\b/.test(blob)) return true;
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
    // Tier 2: items that aren't hard-rejected (may have low style score but
    // are still acceptable). Keep them behind Tier 1 rather than dropping
    // them: collapsing a slot to one high-scoring piece caused all 15 daily
    // outfits to contain the exact same garments.
    const tier2 = items.filter((item) => !strictRejectsForStyleSlot(item, style, slot));
    if (tier2.length > 0) {
        const tier1Ids = new Set(tier1.map((item) => item.id));
        return [...tier1, ...tier2.filter((item) => !tier1Ids.has(item.id))];
    }

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
    imageUrl: '',
    image: '',
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

    const shopOnlyDeck = isShopOnlyDeck(params.stylePreferences, params.occasion);
    const shoes =
        pick((item) => item.macroCategory === 'shoes')
        || (shopOnlyDeck ? undefined : { ...PLACEHOLDER_SHOES });

    return [outerwear, baseTop, secondTop, bottom, shoes].filter(Boolean) as GeneratedOutfitItem[];
}

// Shuffle array for outfit variety
function shuffleArray<T>(array: T[]): T[] {
    const arr = [...array];
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
}

type LocalClothingItem = ClothingItem & {
    isShopItem?: boolean;
    price?: number;
    shopUrl?: string;
    macroCategory?: string;
};

function isShopClothingItem(item: LocalClothingItem | GeneratedOutfitItem | null | undefined): boolean {
    if (!item) return false;
    return Boolean(item.isShopItem || String(item.id || '').startsWith('shop_'));
}

function isOwnedOutfitItem(item: GeneratedOutfitItem | null | undefined): boolean {
    return Boolean(item && !isShopClothingItem(item) && item.id !== PLACEHOLDER_SHOES.id);
}

function replaceSlotItem(
    outfitItems: GeneratedOutfitItem[],
    replacement: GeneratedOutfitItem,
    targetItems: number,
    protectId?: string,
): void {
    const replacementSlot = String(replacement.macroCategory || '').toLowerCase();
    const sameSlotIndex = outfitItems.findIndex((item) => (
        String(item.macroCategory || '').toLowerCase() === replacementSlot
        && item.id !== replacement.id
        && item.id !== protectId
    ));

    if (sameSlotIndex >= 0) {
        outfitItems[sameSlotIndex] = replacement;
        return;
    }

    if (outfitItems.length < targetItems) {
        outfitItems.push(replacement);
        return;
    }

    const replaceableShopIndex = outfitItems.findIndex((item) => isShopClothingItem(item) && item.id !== protectId);
    if (replaceableShopIndex >= 0) {
        outfitItems[replaceableShopIndex] = replacement;
        return;
    }

    const replaceableIndex = outfitItems.findIndex((item) => item.id !== protectId);
    if (replaceableIndex >= 0) {
        outfitItems[replaceableIndex] = replacement;
    }
}

function enforceMixedDailyOutfit(
    outfitItems: GeneratedOutfitItem[],
    ownedItems: LocalClothingItem[],
    shopItems: LocalClothingItem[],
    index: number,
    targetItems: number,
): GeneratedOutfitItem[] {
    const nextItems = [...outfitItems];
    let protectedOwnedId: string | undefined;

    if (ownedItems.length > 0 && !nextItems.some(isOwnedOutfitItem)) {
        const owned = ownedItems[index % ownedItems.length];
        const ownedDisplay = toDisplayItem(owned, 'From your wardrobe');
        protectedOwnedId = ownedDisplay.id;
        replaceSlotItem(nextItems, ownedDisplay, targetItems);
    } else {
        protectedOwnedId = nextItems.find(isOwnedOutfitItem)?.id;
    }

    if (shopItems.length > 0 && !nextItems.some(isShopClothingItem)) {
        const shop = shopItems[index % shopItems.length];
        replaceSlotItem(nextItems, toDisplayItem(shop, 'Suggested from shop'), targetItems, protectedOwnedId);
    }

    if (nextItems.length > targetItems) nextItems.length = targetItems;
    return nextItems;
}

function enforceShopOnlyDailyOutfit(
    outfitItems: GeneratedOutfitItem[],
    shopItems: LocalClothingItem[],
    index: number,
    targetItems: number,
): GeneratedOutfitItem[] {
    const nextItems = [...outfitItems];
    // Strip any wardrobe piece that leaked in — replace slot-for-slot with
    // a rotating shop pick so the outfit stays complete and unique.
    for (let i = nextItems.length - 1; i >= 0; i--) {
        if (!isShopClothingItem(nextItems[i]) && shopItems.length > 0) {
            const shop = shopItems[(index + i) % shopItems.length];
            const shopDisplay = toDisplayItem(shop, 'Suggested from shop');
            // Only replace when slots match to avoid breaking top/bottom/shoes coverage.
            if (String(shopDisplay.macroCategory || '').toLowerCase() === String(nextItems[i].macroCategory || '').toLowerCase()) {
                nextItems[i] = shopDisplay;
            } else {
                nextItems.splice(i, 1);
            }
        }
    }
    if (shopItems.length > 0 && !nextItems.some(isShopClothingItem)) {
        const shop = shopItems[index % shopItems.length];
        replaceSlotItem(nextItems, toDisplayItem(shop, 'Suggested from shop'), targetItems);
    }

    if (nextItems.length > targetItems) nextItems.length = targetItems;
    return nextItems.filter((item) => isShopClothingItem(item));
}

function buildLocalOutfits(items: LocalClothingItem[], params: GenerateOutfitsParams): GeneratedOutfit[] {
    const style = params.stylePreferences || 'Casual';
    const occ = params.occasion || 'Everyday';
    const limit = params.limit ?? 3;
    const layered = isLayeredWeather(params.weather, params.prompt);
    const styleKey: StyleId = normalizeStyleId(style);
    const shopOnly = isShopOnlyDeck(style, occ);
    // Shop-only decks ignore wardrobe pieces entirely.
    const effectiveItems = shopOnly ? items.filter(isShopClothingItem) : items;
    const ownedItems = shopOnly ? [] : effectiveItems.filter((item) => !isShopClothingItem(item));
    const shopItems = effectiveItems.filter(isShopClothingItem);

    const sortByFavorite = (pool: ClothingItem[]) => {
        if (!params.preferences || !Array.isArray(params.preferences.favoriteColors) || params.preferences.favoriteColors.length === 0) {
            return pool;
        }
        const favorites = params.preferences.favoriteColors.map((c: string) => c.toLowerCase().trim()).filter(Boolean);
        return [...pool].sort((a, b) => {
            const colorA = (a.primaryColor || '').toLowerCase();
            const colorB = (b.primaryColor || '').toLowerCase();
            const matchA = favorites.some(c => colorA.includes(c)) ? 1 : 0;
            const matchB = favorites.some(c => colorB.includes(c)) ? 1 : 0;
            return matchB - matchA;
        });
    };

    // Shuffle items to ensure variety even with limited wardrobes
    // Shop-only decks shuffle only shop candidates.
    let rawBaseTops = shuffleArray(effectiveItems.filter(i => getItemMacroCategory(i) === 'top'));
    let rawOuterwear = shuffleArray(effectiveItems.filter(i => getItemMacroCategory(i) === 'outerwear'));
    let rawBottoms = shuffleArray(effectiveItems.filter(i => getItemMacroCategory(i) === 'bottom'));
    let rawShoes = shuffleArray(effectiveItems.filter(i => getItemMacroCategory(i) === 'shoes'));

    if (params.preferences) {
        rawBaseTops = sortByFavorite(rawBaseTops);
        rawOuterwear = sortByFavorite(rawOuterwear);
        rawBottoms = sortByFavorite(rawBottoms);
        rawShoes = sortByFavorite(rawShoes);
    }

    const baseTops = filterSlotByStyle(rawBaseTops, styleKey, 'top', styleKey === 'casual' ? 0.1 : 0.18);
    const outerwear = filterSlotByStyle(rawOuterwear, styleKey, 'outerwear', styleKey === 'casual' ? 0.1 : 0.18);
    const bottoms = filterSlotByStyle(rawBottoms, styleKey, 'bottom', styleKey === 'casual' ? 0.1 : 0.18);
    const shoes = filterSlotByStyle(rawShoes, styleKey, 'shoes', styleKey === 'casual' ? 0.1 : 0.18);
    const legacyTops = [...outerwear, ...baseTops];
    const nonShortsBottoms = bottoms.filter(b => !isShortsItem({
        name: b.name, type: b.subCategory, macroCategory: getItemMacroCategory(b), subCategory: b.subCategory,
    }));
    const casualOuterwear = outerwear.filter(o => !isFormalLayerItem({
        name: o.name, type: o.subCategory, macroCategory: getItemMacroCategory(o),
    }));

    // For old_money, classic, and business_casual styles, completely exclude shorts
    // when formal outerwear is available. This prevents the illogical coat + shorts combination.
    const isFormalStyle = ['old_money', 'classic', 'business_casual'].includes(styleKey);
    if (isFormalStyle && outerwear.some(o => isFormalLayerItem({
        name: o.name, type: o.subCategory, macroCategory: getItemMacroCategory(o),
    }))) {
        // Replace bottoms with only non-shorts options
        bottoms.length = 0;
        bottoms.push(...nonShortsBottoms);
    }

    const outfits: GeneratedOutfit[] = [];
    const targetItems = layered ? 4 : 3;
    const slotSize = (pool: ClothingItem[]) => Math.max(pool.length, 1);
    const seenSignatures = new Set<string>();
    const outfitSignatureFor = (list: GeneratedOutfitItem[]) =>
        list.map((it) => String(it.id)).sort().join('|');

    for (let i = 0; i < limit; i++) {
        const outfitItems: GeneratedOutfitItem[] = [];

        // Mixed-radix indexes enumerate a new combination before repeating.
        // Random offsets frequently produced the same four pieces many times.
        const topIndex = i;
        const baseIndex = Math.floor(i / slotSize(outerwear));
        const bottomIndex = Math.floor(i / (slotSize(outerwear) * slotSize(baseTops)));
        const shoeIndex = Math.floor(i / (slotSize(outerwear) * slotSize(baseTops) * slotSize(bottoms)));

        // Pre-pick bottom to check for shorts / formal-layer conflict.
        const candidateBottom = bottoms[(layered ? bottomIndex : Math.floor(i / slotSize(legacyTops))) % slotSize(bottoms)] || bottoms[0] || effectiveItems[0];
        const candidateBottomIsShorts = !!candidateBottom && isShortsItem({
            name: candidateBottom.name, type: candidateBottom.subCategory,
            macroCategory: getItemMacroCategory(candidateBottom),
            subCategory: candidateBottom.subCategory,
        });

        if (layered) {
            // Prefer casual outerwear if the bottom is shorts; else default outerwear rotation.
            let outer = outerwear[topIndex % slotSize(outerwear)];
            const outerIsFormal = !!outer && isFormalLayerItem({
                name: outer.name, type: outer.subCategory, macroCategory: getItemMacroCategory(outer),
            });
            if (candidateBottomIsShorts && outerIsFormal) {
                outer = casualOuterwear[topIndex % slotSize(casualOuterwear)] || outer;
            }
            const mainTop = outer || legacyTops[topIndex % slotSize(legacyTops)];
            const base = baseTops[baseIndex % slotSize(baseTops)] || legacyTops[(baseIndex + 1) % slotSize(legacyTops)];
            if (mainTop) outfitItems.push(toDisplayItem(mainTop, shopOnly ? 'Suggested from shop' : 'Main top / outerwear layer'));
            if (base && base.id !== mainTop?.id) outfitItems.push(toDisplayItem(base, shopOnly ? 'Suggested from shop' : 'Base top worn underneath'));
        } else {
            // Non-layered: single top only.
            const top = legacyTops[topIndex % slotSize(legacyTops)] || effectiveItems[0];
            if (top) outfitItems.push(toDisplayItem(top, shopOnly ? 'Suggested from shop' : 'Key piece for this look'));
        }

        // If we still have a formal-layer + shorts conflict, swap bottom for a non-shorts option.
        let finalBottom = candidateBottom;
        if (layered && candidateBottomIsShorts) {
            const firstPick = outfitItems[0];
            if (firstPick && isFormalLayerItem(firstPick) && nonShortsBottoms.length > 0) {
                finalBottom = nonShortsBottoms[bottomIndex % nonShortsBottoms.length];
            }
        }

        const nonLayeredShoeIndex = Math.floor(i / (slotSize(legacyTops) * slotSize(bottoms)));
        const shoe = shoes[(layered ? shoeIndex : nonLayeredShoeIndex) % slotSize(shoes)] || shoes[0];

        if (finalBottom) outfitItems.push(toDisplayItem(finalBottom, shopOnly ? 'Suggested from shop' : 'Pairs well with the top'));
        if (shoe) outfitItems.push(toDisplayItem(shoe, shopOnly ? 'Suggested from shop' : 'Completes the look'));
        else if (!shopOnly) outfitItems.push({ ...PLACEHOLDER_SHOES });

        // Pad only when layered and we still have fewer than 4 items.
        if (layered && outfitItems.length < targetItems && effectiveItems.length > 0) {
            let fillIndex = 0;
            while (outfitItems.length < targetItems && fillIndex < effectiveItems.length) {
                const fillItem = effectiveItems[(fillIndex + i) % effectiveItems.length];
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

        const enforcedItems = shopOnly
            ? enforceShopOnlyDailyOutfit(outfitItems, shopItems, i, targetItems)
            : enforceMixedDailyOutfit(outfitItems, ownedItems, shopItems, i, targetItems);
        // Shop-only: drop incomplete looks instead of padding with wardrobe/placeholder.
        if (shopOnly && enforcedItems.length < targetItems) continue;
        if (shopOnly && !enforcedItems.every(isShopClothingItem)) continue;
        const normalizedOutfitItems = normalizeGeneratedOutfitItems(enforcedItems, params);
        // Never emit the same combination twice — uniqueness is required,
        // especially for the 15-card shop-only carousel.
        const signature = outfitSignatureFor(normalizedOutfitItems);
        if (!signature || seenSignatures.has(signature)) continue;
        seenSignatures.add(signature);

        outfits.push({
            id: `local_${i}_${Date.now()}_${Math.floor(Math.random() * 1000)}`,
            description: shopOnly ? `A ${style} look curated from the shop.` : `A ${style} look built around your wardrobe.`,
            style,
            occasion: occ,
            confidence: 0.75 + (Math.random() * 0.1),
            matchScore: 0.75 + (Math.random() * 0.1),
            items: normalizedOutfitItems,
            stylingTips: layered
                ? ['Layer the base top under the outerwear for depth', 'Keep the palette tonal for a refined finish']
                : ['Add accessories to personalize', 'Roll sleeves or cuffs for a lived-in feel'],
        });
    }

    // If no outfits were created but we have items, create at least one outfit
    if (outfits.length === 0 && effectiveItems.length > 0) {
        const outfitItems: GeneratedOutfitItem[] = [];
        if (layered) {
            const candidateBottom = bottoms[0] || effectiveItems[1] || effectiveItems[0];
            const candidateBottomIsShorts = !!candidateBottom && isShortsItem({
                name: candidateBottom.name, type: candidateBottom.subCategory,
                macroCategory: getItemMacroCategory(candidateBottom),
                subCategory: candidateBottom.subCategory,
            });
            let outer = outerwear[0] || legacyTops[0];
            const outerIsFormal = !!outer && isFormalLayerItem({
                name: outer?.name, type: outer?.subCategory, macroCategory: outer ? getItemMacroCategory(outer) : '',
            });
            if (candidateBottomIsShorts && outerIsFormal) {
                outer = casualOuterwear[0] || legacyTops.find(t => !isFormalLayerItem({
                    name: t.name, type: t.subCategory, macroCategory: getItemMacroCategory(t),
                })) || outer;
            }
            const base = baseTops[0] || legacyTops[0] || outer;
            const finalBottom = (candidateBottomIsShorts && outer && isFormalLayerItem({
                name: outer.name, type: outer.subCategory, macroCategory: getItemMacroCategory(outer),
            }) && nonShortsBottoms[0]) ? nonShortsBottoms[0] : candidateBottom;
            const shoe = shoes[0];
            if (outer) outfitItems.push(toDisplayItem(outer, shopOnly ? 'Suggested from shop' : 'Main top / outerwear layer'));
            if (base && base.id !== outer?.id) outfitItems.push(toDisplayItem(base, shopOnly ? 'Suggested from shop' : 'Base top worn underneath'));
            if (finalBottom && finalBottom.id !== base?.id) outfitItems.push(toDisplayItem(finalBottom, shopOnly ? 'Suggested from shop' : 'Pairs well with the top'));
            if (shoe && shoe.id !== finalBottom?.id) outfitItems.push(toDisplayItem(shoe, shopOnly ? 'Suggested from shop' : 'Completes the look'));
            else if (!shopOnly) outfitItems.push({ ...PLACEHOLDER_SHOES });
        } else {
            const top = legacyTops[0] || effectiveItems[0];
            const bottom = bottoms[0] || effectiveItems[1] || effectiveItems[0];
            const shoe = shoes[0];
            if (top) outfitItems.push(toDisplayItem(top, shopOnly ? 'Suggested from shop' : 'Key piece for this look'));
            if (bottom && bottom.id !== top?.id) outfitItems.push(toDisplayItem(bottom, shopOnly ? 'Suggested from shop' : 'Pairs well with the top'));
            if (shoe && shoe.id !== bottom?.id) outfitItems.push(toDisplayItem(shoe, shopOnly ? 'Suggested from shop' : 'Completes the look'));
            else if (!shopOnly) outfitItems.push({ ...PLACEHOLDER_SHOES });
            if (outfitItems.length > 3) outfitItems.length = 3;
        }

        if (outfitItems.length > 0) {
            const enforcedItems = shopOnly
                ? enforceShopOnlyDailyOutfit(outfitItems, shopItems, 0, targetItems)
                : enforceMixedDailyOutfit(outfitItems, ownedItems, shopItems, 0, targetItems);
            if (shopOnly && (enforcedItems.length < targetItems || !enforcedItems.every(isShopClothingItem))) {
                return outfits;
            }
            const normalizedOutfitItems = normalizeGeneratedOutfitItems(enforcedItems, params);
            outfits.push({
                id: `local_0_${Date.now()}`,
                description: shopOnly ? `A ${style} look curated from the shop.` : `A ${style} look built around your wardrobe.`,
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

function toDisplayItem(
    item: ClothingItem & { isShopItem?: boolean; price?: number; shopUrl?: string },
    recommendation: string,
): GeneratedOutfitItem {
    const isShopItem = Boolean(item.isShopItem || String(item.id || '').startsWith('shop_'));

    return {
        id: item.id,
        type: item.subCategory || item.category,
        macroCategory: getItemMacroCategory(item),
        color: item.primaryColor || 'neutral',
        name: item.name || item.subCategory,
        brand: item.brand,
        imageUrl: item.imageUrl,
        image: item.imageUrl,
        recommendation,
        isShopItem,
        price: item.price,
        shopUrl: item.shopUrl,
    };
}

function mapRawOutfit(o: any, params: GenerateOutfitsParams): GeneratedOutfit {
    const items: GeneratedOutfitItem[] = (o.items || []).map((item: any): GeneratedOutfitItem => {
        const resolvedImage = item.imageUrl || item.image_url || item.image || item.thumbnailUrl || item.thumbnail_url || '';
        const resolvedMacroCategory = item.macroCategory || item.macro_category || item.category || item.garmentType || item.garment_type || '';

        if (__DEV__) {
            console.log('[mapRawOutfit] item:', {
                id: String(item.id || '').slice(0, 8),
                hasImageUrl: Boolean(item.imageUrl),
                hasImage_url: Boolean(item.image_url),
                hasImage: Boolean(item.image),
                resolvedImgLen: typeof resolvedImage === 'string' ? resolvedImage.length : typeof resolvedImage,
                macro: resolvedMacroCategory,
                type: item.type,
                name: item.name,
            });
        }

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
