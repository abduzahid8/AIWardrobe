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
import type { ClothingItem, ClothingCategory } from '../types/domain';
import { mapDbCategory as canonicalMapDbCategory, getMacroCategory as canonicalGetMacroCategory } from '../utils/categoryMapper';

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
        if (items.length === 0) {
            return { success: false, outfits: [], error: 'No clothing items found in your wardrobe.', source: 'local' };
        }
        const outfits = buildLocalOutfits(items, params);
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

function buildLocalOutfits(items: ClothingItem[], params: GenerateOutfitsParams): GeneratedOutfit[] {
    const style = params.stylePreferences || 'Casual';
    const occ = params.occasion || 'Everyday';
    const limit = params.limit ?? 3;

    const tops = items.filter(i => ['top', 'outerwear'].includes(getMacroCategory(i.category, i.subCategory)));
    const bottoms = items.filter(i => getMacroCategory(i.category, i.subCategory) === 'bottom');
    const shoes = items.filter(i => getMacroCategory(i.category, i.subCategory) === 'shoes');

    const outfits: GeneratedOutfit[] = [];

    for (let i = 0; i < limit; i++) {
        const outfitItems: GeneratedOutfitItem[] = [];

        // Always include top (reuse if necessary)
        const top = tops[i % Math.max(tops.length, 1)] || tops[0];
        // Always include bottom (reuse if necessary)
        const bottom = bottoms[i % Math.max(bottoms.length, 1)] || bottoms[0] || items[0];
        // Always include shoes (reuse if necessary)
        const shoe = shoes[i % Math.max(shoes.length, 1)] || shoes[0] || items[1] || items[0];

        if (top) outfitItems.push(toDisplayItem(top, 'Key piece for this look'));
        if (bottom) outfitItems.push(toDisplayItem(bottom, 'Pairs well with the top'));
        if (shoe) outfitItems.push(toDisplayItem(shoe, 'Completes the look'));

        // Ensure at least 3 items for a complete outfit
        if (outfitItems.length < 3 && items.length > 0) {
            // Fill remaining slots with available items
            let fillIndex = 0;
            while (outfitItems.length < 3 && fillIndex < items.length) {
                const fillItem = items[fillIndex];
                // Avoid adding duplicate items
                const alreadyAdded = outfitItems.some(oi => oi.id === fillItem.id);
                if (!alreadyAdded) {
                    outfitItems.push(toDisplayItem(fillItem, 'Complementary piece'));
                }
                fillIndex++;
            }
        }

        if (outfitItems.length === 0) continue;

        outfits.push({
            id: `local_${i}_${Date.now()}`,
            description: `A ${style} look built from your wardrobe.`,
            style,
            occasion: occ,
            confidence: 0.75,
            matchScore: 0.75,
            items: outfitItems,
            stylingTips: ['Add accessories to personalize', 'Experiment with layering for visual depth'],
        });
    }

    // If no outfits were created but we have items, create at least one outfit
    if (outfits.length === 0 && items.length > 0) {
        const outfitItems: GeneratedOutfitItem[] = [];
        const top = tops[0] || items[0];
        const bottom = bottoms[0] || items[1] || items[0];
        const shoe = shoes[0] || items[2] || items[0];
        
        if (top) outfitItems.push(toDisplayItem(top, 'Key piece for this look'));
        if (bottom && bottom.id !== top?.id) outfitItems.push(toDisplayItem(bottom, 'Pairs well with the top'));
        if (shoe && shoe.id !== bottom?.id) outfitItems.push(toDisplayItem(shoe, 'Completes the look'));
        
        if (outfitItems.length > 0) {
            outfits.push({
                id: `local_0_${Date.now()}`,
                description: `A ${style} look built from your wardrobe.`,
                style,
                occasion: occ,
                confidence: 0.75,
                matchScore: 0.75,
                items: outfitItems,
                stylingTips: ['Add accessories to personalize', 'Experiment with layering for visual depth'],
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
    const items: GeneratedOutfitItem[] = (o.items || []).map((item: any): GeneratedOutfitItem => ({
        id: item.id || `item_${Date.now()}_${Math.random()}`,
        type: item.type || 'clothing',
        macroCategory: item.macroCategory || '',
        color: item.color || 'neutral',
        name: item.name || item.type || 'Item',
        brand: item.brand || '',
        imageUrl: item.imageUrl || item.image_url || '',
        image: item.imageUrl || item.image_url || '',
        recommendation: item.recommendation || 'Selected for this outfit',
        isShopItem: item.isShopItem ?? false,
        price: item.price,
        shopUrl: item.shopUrl,
    }));

    // De-duplicate exact duplicate ids first.
    const seen = new Set<string>();
    const dedupedItems = items.filter(item => {
        const key = String(item.id || '');
        if (!key) return true;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
    });

    return {
        id: o.id || `outfit_${Date.now()}_${Math.random()}`,
        description: o.description || `A ${o.style || params.stylePreferences || 'stylish'} look`,
        style: o.style || params.stylePreferences || 'Casual',
        occasion: o.occasion || params.occasion || 'Everyday',
        confidence: o.confidence ?? 0.82,
        matchScore: o.confidence ?? o.matchScore ?? 0.82,
        items: dedupedItems,
        stylingTips: Array.isArray(o.stylingTips) ? o.stylingTips : ['Style to your preference'],
    };
}
