import { budgetFxContext, fitsBudgetAmount, type BudgetFxContext } from '../../backend/supabase/functions/_shared/budgetFx';
import { strictestOutfitBudget } from '../../backend/supabase/functions/_shared/outfitBudgetSelection';
import { validateOutfitGenerationRequest } from '../../backend/supabase/functions/_shared/outfitGenerationRequest';
import { getBudgetRates } from './budgetRates';
import { outfitGenerationFailure, type OutfitGenerationFailure } from './outfitGenerationFailure';
import readinessLogger from '../utils/logger';
import { fourSlotOutfitReason } from '../../backend/supabase/functions/_shared/outfitStructure';
import { needsLayering as needsLayeringShared } from '../../backend/supabase/functions/_shared/outfitComposition';
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
 * Never substitutes a local outfit when the edge function fails.
 */

import { passesStyleJudge, detectSlotMismatch } from '../../features/outfit-generator/utils/occasionRules';
import { supabase } from '../../lib/supabase';
import { fillMissingSlots, fetchShopPoolForStyle, type OutfitSlotId, type ShopFillItem } from './shoppingService';
import type { ClothingItem, ClothingCategory, Season, Occasion } from '../types/domain';
import { mapDbCategory as canonicalMapDbCategory, getMacroCategory as canonicalGetMacroCategory, canonicalizeMacroCategory } from '../utils/categoryMapper';
import { rankItemsForStyle, scoreItemForStyle, normalizeStyleId, type StyleId } from '../../features/outfit-generator/utils/styleInference';
import { AI_STYLES } from '../../features/outfit-generator/types';
import { buildCatalogSimilarityKey, spreadSimilarCatalogItems } from '../utils/shopCatalogOrder';
import { assembleDiverseDeck, productIdentityKeys } from '../../features/outfit-generator/utils/outfitDiversity';
import { parseBudgetFromText, budgetCandidatesFromText, budgetInCurrency, outfitShopSpend, itemShopPrice, isShopPricedItem, type ParsedBudget } from '../utils/outfitBudget';
import { getRecentShownIds, getRecentShownProductKeys, recordShownOutfits } from './recentOutfitIds';
import useWardrobeStore from '../../store/wardrobeStore';

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

export interface GenerateOutfitsParams {
    budgetFx?: BudgetFxContext;
    /** Free-text user prompt, e.g. "beach trip" or "important meeting tomorrow" */
    prompt?: string;
    conversationContext?: string;
    /** Set false when the caller records only after committing a visible result. */
    recordShown?: boolean;
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
        maxShopBudget?: ParsedBudget;
        budgetRatesId?: string;
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
    currency?: string;
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
    errorTranslationKey?: OutfitGenerationFailure['translationKey'];
    source?: 'ai' | 'local' | 'deterministic_fallback';
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
 * Returns an explicit failure when generation or validation fails.
 */
export async function generateOutfitsFromDB(
    params: GenerateOutfitsParams
): Promise<GenerateOutfitsResult> {
    // Covers one provider attempt plus one quality correction and DB work.
    const TIMEOUT_MS = 120_000;
    let timeout: ReturnType<typeof setTimeout> | undefined;
    const controller = new AbortController();
    let subscription: { unsubscribe: () => void } | undefined;
    let rejectCancelled!: (reason: Error) => void;
    let cancelled = false;
    const cancellation = new Promise<never>((_, reject) => { rejectCancelled = reject; });
    const cancel = (reason: string) => {
        if (cancelled) return;
        cancelled = true;
        rejectCancelled(new Error(reason));
        controller.abort();
    };
    const wait = <T,>(operation: PromiseLike<T>) => Promise.race([operation, cancellation]);
    timeout = setTimeout(() => cancel('outfit_generation_timeout'), TIMEOUT_MS);
    try {
        // Shop-only decks (Team Collaboration) never accept a wardrobe anchor —
        // every item in every outfit must come from shopping.
        const shopOnlyRequest = isShopOnlyDeck(params.stylePreferences, params.occasion);
        const { data: sessionData, error: sessionError } = await wait(supabase.auth.getSession());
        const session = sessionData.session;
        if (sessionError || !session?.user.id || !session.access_token) {
            return { success: false, outfits: [], error: 'authentication_required' };
        }
        const recencyScope = session.user.id;
        subscription = supabase.auth.onAuthStateChange((_event, nextSession) => {
            // Keep this callback synchronous: Supabase auth events hold an auth lock.
            if (nextSession?.user.id !== recencyScope) cancel('generation_account_changed');
        }).data.subscription;
        // Recheck after subscribing so a change during the first session read
        // cannot fall between the captured session and event registration.
        const current = await wait(supabase.auth.getSession());
        if (current.error || current.data.session?.user.id !== recencyScope) {
            cancel('generation_account_changed');
            throw new Error('generation_account_changed');
        }
        const excludeItemIds = await wait(getRecentShownIds(recencyScope));
        const excludeProductKeys = await wait(getRecentShownProductKeys(recencyScope));
        if (cancelled) throw new Error('generation_account_changed');
        const body = {
            prompt: params.prompt || '',
            conversationContext: params.conversationContext,
            stylePreferences: params.stylePreferences || 'Casual',
            occasion: params.occasion || 'Everyday',
            weather: params.weather,
            limit: params.limit ?? 3,
            selectedItemIds: params.selectedItemIds ?? [],
            anchorItemId: shopOnlyRequest ? undefined : params.anchorItemId,
            preferences: params.preferences,
            // Recently-shown window → edge skips these when building the
            // candidate pool, so regenerates surface new pieces.
            excludeItemIds,
            excludeProductKeys,
        };
        try { validateOutfitGenerationRequest(body); }
        catch {
            return { success: false, outfits: [], error: 'invalid_generation_request', errorTranslationKey: 'outfitMaker.invalidRequest' };
        }
        const invokeOnce = () => supabase.functions.invoke('generate-outfits', {
            signal: controller.signal,
            headers: { Authorization: `Bearer ${session.access_token}` },
            body,
        });

        let { data, error } = await wait(invokeOnce());
        if (cancelled) throw new Error('generation_account_changed');
        // One retry for transient provider/transport failures (observed
        // 503s and lost connections); 4xx and generic errors are not retried.
        if (error || data?.error === 'ai_generation_unavailable') {
            const status = (error as any)?.context?.status ?? (data?.error === 'ai_generation_unavailable' ? 503 : 0);
            const isFetchFailure = (error as any)?.name === 'FunctionsFetchError' || /fetch|network|connection/i.test(String((error as any)?.message || ''));
            if (status >= 500 || isFetchFailure) {
                console.warn('[outfitGenerationService] Retrying generate-outfits once after transient failure');
                ({ data, error } = await wait(invokeOnce()));
                if (cancelled) throw new Error('generation_account_changed');
            }
        }

        if (error) {
            console.warn('[outfitGenerationService] Edge function error, generation rejected:', error);
            return { success: false, outfits: [], error: 'stylist_generation_failed', errorTranslationKey: (await outfitGenerationFailure(data, error)).translationKey };
        }

        if (data?.source === 'local' || !data?.success || !Array.isArray(data?.outfits) || data.outfits.length === 0) {
            console.warn('[outfitGenerationService] Empty AI response, generation rejected');
            return { success: false, outfits: [], error: 'stylist_generation_failed', errorTranslationKey: (await outfitGenerationFailure(data, error)).translationKey };
        }

        const outfits: GeneratedOutfit[] = data.outfits
            .filter((o: any) => o && !fourSlotOutfitReason(o.items, data.layered !== false))
            .map((o: any) => mapRawOutfit(o, params))
            .filter((o: GeneratedOutfit) => passesStyleJudge(o.items, params.stylePreferences, `${params.occasion || ''} ${params.prompt || ''}`, params.weather));
        if (!outfits.length) return { success: false, outfits: [], error: 'no_valid_outfits_for_request' };
        if (params.recordShown !== false) recordShownOutfits(outfits, recencyScope);
        return { success: true, outfits, source: data.source ?? 'ai', layered: Boolean(data.layered) };

    } catch (err: any) {
        if (err?.message === 'generation_account_changed') {
            return { success: false, outfits: [], error: 'generation_account_changed' };
        }
        if (err?.message === 'outfit_generation_timeout') {
            console.warn('[outfitGenerationService] Edge function timed out, generation rejected');
            return { success: false, outfits: [], error: 'outfit_generation_timeout' };
        }
        console.error('[outfitGenerationService] Unexpected error:', err);
        return { success: false, outfits: [], error: 'stylist_generation_failed' };
    } finally {
        if (timeout) clearTimeout(timeout);
        subscription?.unsubscribe();
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
            : ((params.limit ?? 3) >= 10 ? 10 : ((params.limit ?? 3) > 1 ? 2 : 1));
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
            currency: sf.currency,
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

        const candidates = budgetCandidatesFromText(`${params.prompt || ''} ${params.occasion || ''}`);
        const cap = params.preferences?.maxShopBudget || parseBudgetFromText(`${params.prompt || ''} ${params.occasion || ''}`);
        const needsRates = new Set([...candidates.map(value => value.currency), ...(cap ? [cap.currency] : [])]).size > 1 ||
            (cap && mergedItems.some(item => isShopPricedItem(item) && Number(item.price) > 0 && item.currency && item.currency !== cap.currency));
        const fx = params.budgetFx || (needsRates ? budgetFxContext(await getBudgetRates()) : undefined);
        const outfits = buildLocalOutfits(mergedItems, {...params, budgetFx: fx});

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

        // Record what the user will see so the next edge call rotates away
        // from it (the local path uses shuffle, so exclusion is belt-and-braces).
        recordShownOutfits(outfits);

        return outfits.length ? { success: true, outfits, source: 'local' } : { success: false, outfits: [], error: 'no_valid_outfits_for_request', source: 'local' };
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
            const mc = detectSlotMismatch(getMacroCategory(cat, typeLabel), typeLabel, typeLabel) ?? getMacroCategory(cat, typeLabel);
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
    // 'short-sleeved' describes SLEEVES on tops — strip it first, otherwise
    // every short-sleeve tee mislabeled as bottoms reads as shorts.
    const blob = `${item.type || ''} ${item.name || ''} ${item.subCategory || ''}`.toLowerCase().replace(/\bshort[\s-]?sleeve(?:s|d)?\b/g, '');
    const macro = (item.macroCategory || '').toLowerCase();
    const isBottom = macro === 'bottom' || /pant|trouser|jeans|bottom|shorts?|skirt/.test(blob);
    if (!isBottom) return false;
    return /\b(shorts?|bermudas?)\b/.test(blob);
}

function isLayeredWeather(weather?: { temp?: number | null; condition?: string | null }, prompt?: string | null): boolean {
    return needsLayeringShared(undefined, weather, prompt);
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
        if (slot === 'shoes' && /\b(chunky|basketball|skate|running|trainer|retro sneaker|retro sneakers|athletic|moccasins?|drivers?|driving|gommino|loafer[s]?\s+with\s+(?:tie|laces?))\b/.test(blob)) return true;
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
    shopOnlyDeckOverride?: boolean,
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

    // Non-layered rotation picks the "top" from legacyTops (outerwear + tops),
    // so the pick can land on an outerwear piece. That must NOT be dropped —
    // a jacket-as-top is a valid 3-item summer look (mirrors the edge
    // validator, which accepts hasBaseTop || hasOuter when not layered).
    // Dropping it used to emit broken 2-item cards (bottom + shoes, no top).
    // Layered mode keeps excluding outerwear here: the layer slot above
    // already claimed it, and baseTop must be a true base layer.
    const baseTop =
        pick((item) => item.macroCategory === 'top')
        || (!layered ? pick((item) => item.macroCategory === 'outerwear') : undefined)
        || pick((item) => item.macroCategory === 'accessory' || item.macroCategory === 'other')
        || pick((item) => item.macroCategory !== 'bottom' && item.macroCategory !== 'shoes' && (layered ? item.macroCategory !== 'outerwear' : true));

    // Contract is Layer -> Top -> Pants -> Shoes: exactly one base top per
    // outfit. A second top would emit a 5-item look (or duplicate the top
    // slot), so it is never picked — extra tops from AI responses are dropped.
    const shopOnlyDeck = shopOnlyDeckOverride ?? isShopOnlyDeck(params.stylePreferences, params.occasion);
    const shoes =
        pick((item) => item.macroCategory === 'shoes')
        || (shopOnlyDeck ? undefined : { ...PLACEHOLDER_SHOES });

    // Exactly-one-per-slot: [layer, top, pants, shoes]. Extra items sharing
    // a slot (e.g. two pants) are dropped by construction — pick() takes the
    // first unused item per slot and never revisits it.
    return [outerwear, baseTop, bottom, shoes].filter(Boolean) as GeneratedOutfitItem[];
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

function deckHashFor(style: string, occasion: string): number {
    const key = `${style}::${occasion}`.toLowerCase();
    let hash = 0;
    for (let i = 0; i < key.length; i++) hash = (Math.imul(31, hash) + key.charCodeAt(i)) | 0;
    return Math.abs(hash);
}

function deckSeedFor(style: string, occasion: string): number {
    return deckHashFor(style, occasion) % 32;
}

function deckReverseFor(style: string, occasion: string): boolean {
    return ((deckHashFor(style, occasion) >> 5) & 1) === 1;
}

// Spread same-looking listings apart so the rotation window covers varied
// products instead of adjacent score-tied lookalikes.
function spreadSlotPool(pool: LocalClothingItem[], slot: string): LocalClothingItem[] {
    if (pool.length <= 2) return pool;
    const decorated = pool.map((item) => ({ ...item, name: item.name || 'item', garmentType: slot }));
    return spreadSimilarCatalogItems(
        decorated as Array<LocalClothingItem & { name: string; garmentType: string }>,
    ).map((entry) => {
        const { garmentType: _dropped, ...rest } = entry as LocalClothingItem & { garmentType?: string };
        return rest as LocalClothingItem;
    });
}

export type LocalClothingItem = ClothingItem & {
    isShopItem?: boolean;
    price?: number;
    currency?: string;
    shopUrl?: string;
    macroCategory?: string;
};

function isShopClothingItem(item: LocalClothingItem | GeneratedOutfitItem | null | undefined): boolean {
    if (!item) return false;
    return Boolean(item.isShopItem || String(item.id || '').startsWith('shop_'));
}

function fitGeneratedOutfitToBudget(
    items: GeneratedOutfitItem[],
    pool: LocalClothingItem[],
    budget: ParsedBudget,
): GeneratedOutfitItem[] {
    const slotOf = (p: { macroCategory?: string }) => String(p.macroCategory || '').toLowerCase();
    const next = [...items];
    for (let guard = 0; guard < 8 && !fitsBudgetAmount(outfitShopSpend(next, undefined, budget.currency, budget.fx), budget.amount); guard++) {
        let worstIdx = -1;
        let worstPrice = -1;
        next.forEach((p, i) => {
            const pr = itemShopPrice(p, budget.currency, budget.fx);
            if (pr > worstPrice) {
                worstPrice = pr;
                worstIdx = i;
            }
        });
        if (worstIdx < 0) break;
        const used = new Set(next.map((p) => String(p.id)));
        const room = budget.amount - (outfitShopSpend(next, undefined, budget.currency, budget.fx) - itemShopPrice(next[worstIdx], budget.currency, budget.fx));
        const slot = slotOf(next[worstIdx]);
        const repl = pool
            .filter((p) => isShopPricedItem(p) && slotOf(p) === slot && !used.has(String(p.id)))
            .filter((p) => {
                const pr = itemShopPrice(p, budget.currency, budget.fx);
                return pr > 0 && pr <= room;
            })
            .sort((a, b) => itemShopPrice(a, budget.currency, budget.fx) - itemShopPrice(b, budget.currency, budget.fx))[0];
        if (repl) {
            next[worstIdx] = toDisplayItem(repl, next[worstIdx].recommendation || 'Suggested from shop');
        } else if (slot === 'outerwear' && next.length > 3) {
            next.splice(worstIdx, 1);
        } else {
            break;
        }
    }
    return next;
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

export function buildLocalOutfits(items: LocalClothingItem[], params: GenerateOutfitsParams): GeneratedOutfit[] {
    const limit = params.limit ?? 3;
    // A non-positive limit means "no outfits" — bail before the last-resort
    // single-outfit block below, which otherwise emits 1 look for limit 0.
    if (limit <= 0) return [];
    const style = params.stylePreferences || 'Casual';
    const occ = params.occasion || 'Everyday';
    const layered = isLayeredWeather(params.weather, params.prompt);
    const styleKey: StyleId = normalizeStyleId(style);
    // Human-readable style name for user-facing copy ("old_money" -> "Old Money").
    const styleLabel = AI_STYLES.find((s) => s.id === styleKey)?.label
        ?? styleKey.split('_').map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
    const styleArticle = /^[aeiou]/i.test(styleLabel) ? 'An' : 'A';
    const shopOnly = isShopOnlyDeck(style, occ);
    const shopCandidates = items.filter(isShopClothingItem);
    const budgetText = `${params.prompt || ''} ${occ}`;
    const fx = params.budgetFx || params.preferences?.maxShopBudget?.fx;
    const textBudget = parseBudgetFromText(budgetText, fx);
    if (budgetCandidatesFromText(budgetText).length && !textBudget) return [];
    let budget: ParsedBudget | null;
    try {
        const selected = strictestOutfitBudget(params.preferences?.maxShopBudget, textBudget,
            (value, currency) => budgetInCurrency({...value, fx}, currency));
        budget = selected ? {...selected, fx} : null;
    } catch { return []; }
    const affordableShop = budget
        ? shopCandidates.filter((item) => {
            const p = itemShopPrice(item, budget.currency, budget.fx);
            return p <= 0 || fitsBudgetAmount(p, budget.amount);
        })
        : shopCandidates;
    const shopPoolForDeck = affordableShop;
    // STRICT shop-only (Team Collaboration / business_casual): new clothing
    // must come ONLY from new clothing (shop). Never fall back to wardrobe
    // when the shop pool is empty — emit 0 looks / error instead of mixing.
    const deckShopOnly = shopOnly;
    const effectiveItems = shopOnly
        ? shopPoolForDeck
        : budget
            ? items.filter((item) => !isShopClothingItem(item) || shopPoolForDeck.includes(item))
            : items;
    if (effectiveItems.length === 0) return [];
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

    const baseTops = spreadSlotPool(filterSlotByStyle(rawBaseTops, styleKey, 'top', styleKey === 'casual' ? 0.1 : 0.18), 'top');
    const outerwear = spreadSlotPool(filterSlotByStyle(rawOuterwear, styleKey, 'outerwear', styleKey === 'casual' ? 0.1 : 0.18), 'outerwear');
    const bottoms = spreadSlotPool(filterSlotByStyle(rawBottoms, styleKey, 'bottom', styleKey === 'casual' ? 0.1 : 0.18), 'bottom');
    const shoes = spreadSlotPool(filterSlotByStyle(rawShoes, styleKey, 'shoes', styleKey === 'casual' ? 0.1 : 0.18), 'shoes');
    const legacyTops = [...outerwear, ...baseTops];
    const nonShortsBottoms = bottoms.filter(b => !isShortsItem({
        name: b.name, type: b.subCategory, macroCategory: getItemMacroCategory(b), subCategory: b.subCategory,
    }));
    const casualOuterwear = outerwear.filter(o => !isFormalLayerItem({
        name: o.name, type: o.subCategory, macroCategory: getItemMacroCategory(o),
    }));

    // Formal decks (old_money, classic, business_casual) never contain
    // shorts — unconditionally, not just when a formal layer happens to be
    // in the pool. A suit + shorts card must be impossible by construction.
    const isFormalStyle = ['old_money', 'classic', 'business_casual'].includes(styleKey);
    if (isFormalStyle && nonShortsBottoms.length > 0) {
        // Replace bottoms with only non-shorts options
        bottoms.length = 0;
        bottoms.push(...nonShortsBottoms);
    }

    const outfits: GeneratedOutfit[] = [];
    const targetItems = layered ? 4 : 3;
    const slotSize = (pool: ClothingItem[]) => Math.max(pool.length, 1);
    const seenSignatures = new Set<string>();
    const builtIdSets: Set<string>[] = [];
    /**
     * Every valid candidate look. The deck is chosen FROM this list by the
     * diversity gate below — accepting looks one at a time would let the
     * three best-looking but visually identical outfits win every slot.
     */
    const collectedCandidates: GeneratedOutfitItem[][] = [];
    let candidateOrdinal = 0;
    const outfitSignatureFor = (list: GeneratedOutfitItem[]) =>
        list.map((it) => String(it.id)).sort().join('|');
    const maxSharedWithBuilt = (ids: string[]): number => {
        let max = 0;
        for (const built of builtIdSets) {
            let shared = 0;
            for (const id of ids) if (built.has(id)) shared++;
            if (shared > max) max = shared;
        }
        return max;
    };

    const deckSeed = deckSeedFor(style, occ);
    const deckReverse = deckReverseFor(style, occ);

    // Build one candidate deck entry. Returns normalized items, or null when
    // the pool cannot complete a valid four-slot outfit.
    const buildCandidateItems = (i: number): GeneratedOutfitItem[] | null => {
        const outfitItems: GeneratedOutfitItem[] = [];

        // Mix the candidate ordinal independently for each slot. Linear or
        // mixed-radix rotations visited only a short diagonal of the
        // combination space for small inventories, starving the deck picker
        // of a valid three-look set despite sufficient products.
        const ordinal = candidateOrdinal++;
        const outerSize = slotSize(outerwear);
        const baseSize = slotSize(baseTops);
        const bottomSize = slotSize(bottoms);
        const shoeSize = slotSize(shoes);
        const mixIndex = (input: number): number => {
            let value = input >>> 0;
            value ^= value >>> 16;
            value = Math.imul(value, 0x7feb352d) >>> 0;
            value ^= value >>> 15;
            value = Math.imul(value, 0x846ca68b) >>> 0;
            return (value ^ (value >>> 16)) >>> 0;
        };
        const rotationIndex = (slotIndex: number, poolLength: number): number => {
            const salt = Math.imul(deckSeed + 1, 0x9e3779b1) ^ Math.imul(slotIndex + 1, 0x85ebca6b);
            const rotated = mixIndex((ordinal + 1) ^ salt) % poolLength;
            return deckReverse ? poolLength - 1 - rotated : rotated;
        };
        const topIndex = rotationIndex(0, outerSize);
        const baseIndex = rotationIndex(1, baseSize);
        const bottomIndex = rotationIndex(2, bottomSize);
        const shoeIndex = rotationIndex(3, shoeSize);
        const legacyTopIndex = rotationIndex(0, slotSize(legacyTops));
        const legacyBottomIndex = rotationIndex(2, bottomSize);

        // Pre-pick bottom to check for shorts / formal-layer conflict.
        const candidateBottom = bottoms[(layered ? bottomIndex : legacyBottomIndex) % slotSize(bottoms)] || bottoms[0] || effectiveItems[0];
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
            const mainTop = outer || legacyTops[legacyTopIndex % slotSize(legacyTops)];
            const base = baseTops[baseIndex % slotSize(baseTops)] || legacyTops[(baseIndex + 1) % Math.max(slotSize(legacyTops), 1)];
            if (mainTop) outfitItems.push(toDisplayItem(mainTop, deckShopOnly ? 'Suggested from shop' : 'Main top / outerwear layer'));
            if (base && base.id !== mainTop?.id) outfitItems.push(toDisplayItem(base, deckShopOnly ? 'Suggested from shop' : 'Base top worn underneath'));
        } else {
            // Non-layered: single top only.
            const top = legacyTops[legacyTopIndex % slotSize(legacyTops)] || effectiveItems[0];
            if (top) outfitItems.push(toDisplayItem(top, deckShopOnly ? 'Suggested from shop' : 'Key piece for this look'));
        }

        // If we still have a formal-layer + shorts conflict, swap bottom for a non-shorts option.
        // Formal decks swap unconditionally — shorts never appear there.
        let finalBottom = candidateBottom;
        if (candidateBottomIsShorts && nonShortsBottoms.length > 0) {
            const firstPick = outfitItems[0];
            const firstIsFormal = !!firstPick && isFormalLayerItem(firstPick);
            if (isFormalStyle || (layered && firstIsFormal)) {
                finalBottom = nonShortsBottoms[bottomIndex % Math.max(nonShortsBottoms.length, 1)];
            }
        }

        const nonLayeredShoeIndex = rotationIndex(3, shoeSize);
        const shoe = shoes[(layered ? shoeIndex : nonLayeredShoeIndex) % slotSize(shoes)] || shoes[0];

        if (finalBottom) outfitItems.push(toDisplayItem(finalBottom, deckShopOnly ? 'Suggested from shop' : 'Pairs well with the top'));
        if (shoe) outfitItems.push(toDisplayItem(shoe, deckShopOnly ? 'Suggested from shop' : 'Completes the look'));
        else if (!deckShopOnly) outfitItems.push({ ...PLACEHOLDER_SHOES });

        // Pad only when layered and we still have fewer than 4 items.
        // Slot-aware: only fill slots that are actually missing — never add
        // a second bottom/top to reach the count (the "2 pants" bug).
        if (layered && outfitItems.length < targetItems && effectiveItems.length > 0) {
            const presentMacros = new Set(outfitItems.map(oi => String(oi.macroCategory || '').toLowerCase()));
            let fillIndex = 0;
            while (outfitItems.length < targetItems && fillIndex < effectiveItems.length) {
                const fillItem = effectiveItems[(fillIndex + i) % effectiveItems.length];
                const alreadyAdded = outfitItems.some(oi => oi.id === fillItem.id);
                const fillMacro = getItemMacroCategory(fillItem).toLowerCase();
                if (!alreadyAdded && !presentMacros.has(fillMacro)) {
                    outfitItems.push(toDisplayItem(fillItem, 'Complementary piece'));
                    presentMacros.add(fillMacro);
                }
                fillIndex++;
            }
        }
        // Trim accidental extras down to the contract size.
        if (outfitItems.length > targetItems) outfitItems.length = targetItems;

        if (outfitItems.length === 0) return null;

        const enforcedItems = deckShopOnly
            ? enforceShopOnlyDailyOutfit(outfitItems, shopItems, i, targetItems)
            : enforceMixedDailyOutfit(outfitItems, ownedItems, shopItems, i, targetItems);
        // Shop-only: drop incomplete looks instead of padding with wardrobe/placeholder.
        if (deckShopOnly && enforcedItems.length < targetItems) return null;
        if (deckShopOnly && !enforcedItems.every(isShopClothingItem)) return null;
        let normalizedOutfitItems = normalizeGeneratedOutfitItems(enforcedItems, params, deckShopOnly);
        if (budget) {
            normalizedOutfitItems = fitGeneratedOutfitToBudget(normalizedOutfitItems, shopItems, budget);
            if (!fitsBudgetAmount(outfitShopSpend(normalizedOutfitItems, undefined, budget.currency, budget.fx), budget.amount)) return null;
        }
        // Hard rule: never emit shorts with a suit, never shorts in formal decks.
        {
            const hasShorts = normalizedOutfitItems.some((it) => it.macroCategory === 'bottom' && isShortsItem({
                name: it.name, type: it.type, macroCategory: it.macroCategory,
            }));
            const hasFormal = normalizedOutfitItems.some((it) => isFormalLayerItem({
                name: it.name, type: it.type, macroCategory: it.macroCategory,
            }));
            if (hasShorts && (isFormalStyle || hasFormal)) return null;
        }
        return normalizedOutfitItems;
    };

    // Rotating styling tips (transcript doctrine): socks match trousers,
    // bottom button undone, cuff shows, three colors max. Mirrors the edge
    // pickStylingTips pool so local + server cards teach the same rules.
    const pickStylingTips = (isLayered: boolean): string[] => {
        const pool = isLayered
            ? [
                'Layer the base top under the outerwear for depth',
                'Keep the palette tonal for a refined finish',
                'Match your socks to your trousers, not your shoes',
                'Leave the bottom jacket button undone',
                'Show 1-2cm of shirt cuff beyond the jacket sleeve',
                'Texture elevates: linen, flannel or suede over flat cotton',
            ]
            : [
                'Add accessories to personalize',
                'Roll sleeves or cuffs for a lived-in feel',
                'Match your socks to your trousers, not your shoes',
                "Keep it to three colors — white doesn't count",
                'Dark top + light bottom (or the reverse) reads most masculine',
                'Trousers sit at mid-rise; low-rise breaks the proportions',
            ];
        const a = Math.floor(Math.random() * pool.length);
        let b = Math.floor(Math.random() * pool.length);
        if (b === a) b = (b + 1) % pool.length;
        return [pool[a], pool[b]];
    };

    const pushOutfit = (normalizedOutfitItems: GeneratedOutfitItem[], tag: number): void => {
        if (fourSlotOutfitReason(normalizedOutfitItems, layered)) return;
        outfits.push({
            id: `local_${tag}_${Date.now()}_${Math.floor(Math.random() * 1000)}`,
            description: deckShopOnly ? `${styleArticle} ${styleLabel} look curated from the shop.` : `${styleArticle} ${styleLabel} look built around your wardrobe.`,
            style,
            occasion: occ,
            confidence: 0.75 + (Math.random() * 0.1),
            matchScore: 0.75 + (Math.random() * 0.1),
            items: normalizedOutfitItems,
            stylingTips: pickStylingTips(layered),
        });
    };

    const tryKeepCandidate = (normalizedOutfitItems: GeneratedOutfitItem[], relaxed: boolean): boolean => {
        if (fourSlotOutfitReason(normalizedOutfitItems, layered)) return false;
        // Never emit the same combination twice — uniqueness is required,
        // especially for the 15-card shop-only carousel.
        const signature = outfitSignatureFor(normalizedOutfitItems);
        if (!signature || seenSignatures.has(signature)) return false;
        if (!relaxed) {
            // Skip near-duplicates while rotation tries remain: any two
            // looks may share at most 1 item, whatever the slot count.
            const ids = normalizedOutfitItems.map((it) => String(it.id));
            if (builtIdSets.length > 0 && maxSharedWithBuilt(ids) >= 2) return false;
        }
        seenSignatures.add(signature);
        builtIdSets.push(new Set(normalizedOutfitItems.map((it) => String(it.id))));
        collectedCandidates.push(normalizedOutfitItems);
        return true;
    };

    // Strict pass: decks must differ by >= 2 items. Rejections perturb the
    // attempt so retries explore new slot combinations.
    //
    // More candidates are collected than the deck needs: the diversity gate
    // below picks the best COMBINATION, so it needs a choice to pick from —
    // three candidates and three cards would leave it no freedom at all.
    // Collect enough alternatives that the deck selector can form a full
    // deck even when several slot products are shared across the first
    // few rotations. A short candidate pool occasionally produced only two
    // looks despite sufficient distinct inventory in every strict slot.
    const collectionTarget = Math.min(36, Math.max(limit * 8, limit + 6));
    let attempt = 0;
    for (let i = 0, tag = 0; collectedCandidates.length < collectionTarget && i < limit * 48; i++, tag++) {
        const candidate = buildCandidateItems(i);
        if (!candidate) { attempt++; continue; }
        if (tryKeepCandidate(candidate, false)) {
            attempt = 0;
        } else {
            attempt++;
        }
    }

    // Relaxed pass: thin pools (a single shoes/bottom/outerwear choice) can
    // NEVER satisfy the strict rule — every second look shares 3 of 4 by
    // construction. Fill the remaining slots with exact-dupe-only rejection
    // so the user gets 3 looks varying the pieces that DO exist instead of 1.
    if (collectedCandidates.length < collectionTarget) {
        // Drive rotation by attempt alone (never reset): incrementing both
        // i and attempt per failure made the bottom index advance by
        // 1 + 13 ≡ 0 (mod 2), so thin pools cycled the same bottom and
        // only ~3 signatures were ever explored before giving up at 2.
        let relaxedAttempt = attempt;
        const relaxedAttemptEnd = attempt + limit * 48;
        for (let tag = 1000; collectedCandidates.length < collectionTarget && relaxedAttempt < relaxedAttemptEnd; tag++) {
            const candidate = buildCandidateItems(limit * 12);
            relaxedAttempt++;
            if (!candidate) continue;
            tryKeepCandidate(candidate, true);
        }
    }

    // VISUAL / STYLISTIC diversity — the layer product dedupe cannot express.
    // "navy blazer + white oxford + grey tailored trousers + brown loafers"
    // and "charcoal blazer + blue poplin + black trousers + penny loafers"
    // are one outfit to a human, no matter how many SKUs they use. The deck
    // is now chosen as the COMBINATION that maximises quality + diversity
    // instead of accepting looks one at a time.
    if (collectedCandidates.length > 0) {
        const selected = assembleDiverseDeck<GeneratedOutfitItem[]>({
            candidates: collectedCandidates,
            itemsOf: (items) => items,
            qualityOf: () => 0.9,
            keysOf: (item: any) => productIdentityKeys(item),
            usedKeys: new Set<string>(),
            target: limit,
            poolItems: effectiveItems as any[],
            // If randomized rotations leave the first candidate window
            // without a full unique deck, keep exploring combinations rather
            // than returning two looks from an inventory that can support
            // three. The fill remains bounded and every candidate still
            // passes the same four-slot, style and budget checks above.
            maxFillAttempts: limit <= 3 ? 24 : 0,
            buildFill: limit <= 3
                ? (_usedKeys, fillAttempt) => {
                    const candidate = buildCandidateItems(limit * 48 + fillAttempt);
                    return candidate && !fourSlotOutfitReason(candidate, layered) ? candidate : null;
                }
                : undefined,
        });
        let tag = 0;
        for (const items of selected.deck) pushOutfit(items, tag++);
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
            if (outer) outfitItems.push(toDisplayItem(outer, deckShopOnly ? 'Suggested from shop' : 'Main top / outerwear layer'));
            if (base && base.id !== outer?.id) outfitItems.push(toDisplayItem(base, deckShopOnly ? 'Suggested from shop' : 'Base top worn underneath'));
            // Never let the pants slot reuse the layer/base ids, and never let
            // a non-bottom macro occupy it — both produced "2 pants" cards.
            if (finalBottom && finalBottom.id !== base?.id && finalBottom.id !== outer?.id
                && getItemMacroCategory(finalBottom).toLowerCase() === 'bottom') outfitItems.push(toDisplayItem(finalBottom, deckShopOnly ? 'Suggested from shop' : 'Pairs well with the top'));
            if (shoe && shoe.id !== finalBottom?.id) outfitItems.push(toDisplayItem(shoe, deckShopOnly ? 'Suggested from shop' : 'Completes the look'));
            else if (!deckShopOnly) outfitItems.push({ ...PLACEHOLDER_SHOES });
        } else {
            const top = legacyTops[0] || effectiveItems[0];
            const bottom = bottoms[0] || effectiveItems[1] || effectiveItems[0];
            const shoe = shoes[0];
            if (top) outfitItems.push(toDisplayItem(top, deckShopOnly ? 'Suggested from shop' : 'Key piece for this look'));
            if (bottom && bottom.id !== top?.id) outfitItems.push(toDisplayItem(bottom, deckShopOnly ? 'Suggested from shop' : 'Pairs well with the top'));
            if (shoe && shoe.id !== bottom?.id) outfitItems.push(toDisplayItem(shoe, deckShopOnly ? 'Suggested from shop' : 'Completes the look'));
            else if (!deckShopOnly) outfitItems.push({ ...PLACEHOLDER_SHOES });
            if (outfitItems.length > 3) outfitItems.length = 3;
        }

        if (outfitItems.length > 0) {
            const enforcedItems = deckShopOnly
                ? enforceShopOnlyDailyOutfit(outfitItems, shopItems, 0, targetItems)
                : enforceMixedDailyOutfit(outfitItems, ownedItems, shopItems, 0, targetItems);
            if (deckShopOnly && (enforcedItems.length < targetItems || !enforcedItems.every(isShopClothingItem))) {
                return outfits;
            }
            let normalizedOutfitItems = normalizeGeneratedOutfitItems(enforcedItems, params);
            if (budget) {
                normalizedOutfitItems = fitGeneratedOutfitToBudget(normalizedOutfitItems, shopItems, budget);
                if (!fitsBudgetAmount(outfitShopSpend(normalizedOutfitItems, undefined, budget.currency, budget.fx), budget.amount)) return outfits;
            }
            outfits.push({
                id: `local_0_${Date.now()}`,
                description: deckShopOnly ? `${styleArticle} ${styleLabel} look curated from the shop.` : `${styleArticle} ${styleLabel} look built around your wardrobe.`,
                style,
                occasion: occ,
                confidence: 0.75,
                matchScore: 0.75,
                items: normalizedOutfitItems,
                stylingTips: pickStylingTips(layered),
            });
        }
    }

    return outfits.filter(outfit => !fourSlotOutfitReason(outfit.items, layered));
}

function toDisplayItem(
    item: ClothingItem & { isShopItem?: boolean; price?: number; currency?: string; shopUrl?: string },
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
        currency: item.currency,
        shopUrl: item.shopUrl,
    };
}

function mapRawOutfit(o: any, params: GenerateOutfitsParams): GeneratedOutfit {
    const items: GeneratedOutfitItem[] = (o.items || []).map((item: any): GeneratedOutfitItem => {
        const resolvedImage = item.imageUrl || item.image_url || item.image || item.thumbnailUrl || item.thumbnail_url || '';
        const resolvedMacroCategory = item.macroCategory || item.macro_category || item.category || item.garmentType || item.garment_type || '';

        if (__DEV__) {
            if (__DEV__) readinessLogger.debug(String('[mapRawOutfit] item:'), [{
                id: String(item.id || '').slice(0, 8),
                hasImageUrl: Boolean(item.imageUrl),
                hasImage_url: Boolean(item.image_url),
                hasImage: Boolean(item.image),
                resolvedImgLen: typeof resolvedImage === 'string' ? resolvedImage.length : typeof resolvedImage,
                macro: resolvedMacroCategory,
                type: item.type,
                name: item.name,
            }]);
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
            currency: item.currency,
            shopUrl: item.shopUrl || item.shop_url || item.sourceUrl || item.source_url,
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

    const normalizedItems = dedupedItems;

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
