/**
 * Shopping Integration Service
 * Provides product search, affiliate links, and "Complete Your Look" features
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import { createLogger } from '../utils/logger';
import {
    scoreItemForStyle,
    normalizeStyleId,
    type StyleId,
} from '../../features/outfit-generator/utils/styleInference';
import type { ShopCatalogItem } from '../types/shop';
import { getShopDestination } from '../utils/shopDestination';
import { createFixtureCatalogRepository, createSupabaseCatalogRepository } from './catalogRepository';
import { catalogTitleColor } from '../utils/catalogIdentity';
import { inferShopRowMacro } from '../utils/shopCatalogOrder';
import { rankDiverseShoppingCandidates } from './shoppingIntelligence';
import { loadCatalogDiscovery } from './catalogDiscovery';
import { findBetterShoppingProducts, evaluateShoppingPurchase } from './shoppingAdvisor';
import { hasProductOnlyCatalogProvenance } from '../utils/productOnlyCatalog';

// Static catalogs require()d lazily — ~120KB of data modules must not
// parse/evaluate on app startup (this service is reachable from eager
// components). Only localCatalogRowsForSlot consumes them.
// eslint-disable-next-line @typescript-eslint/no-var-requires
let staticSlotPoolCache: ShopCatalogItem[] | null = null;
function getStaticSlotPool(): ShopCatalogItem[] {
    if (!staticSlotPoolCache) {
        const data = require('../../data/shopCatalogItems') as {
            SHOP_CATALOG_ITEMS: ShopCatalogItem[];
        };
        const inspo = require('../../data/inspoMensShopItems') as {
            CLASSIC_MENS_ITEMS: ShopCatalogItem[];
        };
        staticSlotPoolCache = [...data.SHOP_CATALOG_ITEMS, ...inspo.CLASSIC_MENS_ITEMS];
    }
    return staticSlotPoolCache;
}

const logger = createLogger('Shopping');

const API_URL = process.env.EXPO_PUBLIC_API_URL || 'https://aiwardrobe-api.onrender.com';

// Affiliate networks (would need real API keys in production)
const AFFILIATE_CONFIG = {
    shopstyle: {
        enabled: false,
        apiKey: process.env.EXPO_PUBLIC_SHOPSTYLE_API_KEY,
        baseUrl: 'https://api.shopstyle.com/api/v2',
    },
    amazon: {
        enabled: true,
        affiliateTag: 'aiwardrobe-20', // Replace with real tag
    },
    // Add more networks as needed
};

// Product categories for search
export const PRODUCT_CATEGORIES = {
    tops: ['shirt', 'blouse', 'top', 't-shirt', 'sweater', 'cardigan', 'jacket', 'blazer'],
    bottoms: ['pants', 'jeans', 'trousers', 'skirt', 'shorts'],
    dresses: ['dress', 'gown', 'jumpsuit', 'romper'],
    shoes: ['shoes', 'sneakers', 'boots', 'heels', 'sandals', 'loafers'],
    accessories: ['bag', 'watch', 'belt', 'scarf', 'hat', 'jewelry', 'sunglasses'],
};

// ============================================
// INTERFACES
// ============================================

export interface ProductSearchParams {
    query: string;
    category?: string;
    color?: string;
    currency?: string;
    priceMin?: number;
    priceMax?: number;
    brand?: string;
    limit?: number;
}

export interface Product {
    id: string;
    name: string;
    brand: string;
    price: number;
    originalPrice?: number;
    currency: string;
    imageUrl: string;
    productUrl: string;
    affiliateUrl: string;
    category: string;
    color?: string;
    inStock?: boolean;
    rating?: number;
    reviewCount?: number;
    source: string;
}

export interface CompleteYourLookSuggestion {
    missingCategory: string;
    suggestedProducts: Product[];
    reason: string;
}

// ============================================
// Real catalog reads with a lazy bundled fallback; no fabricated stock/ratings.
// ============================================
export const catalogRepository = createSupabaseCatalogRepository(createFixtureCatalogRepository(getStaticSlotPool));

// ============================================
// SHOPPING SERVICE
// ============================================

class ShoppingService {
    private recentSearches: string[] = [];
    private wishlist: Product[] = [];

    constructor() {
        this.loadSavedData();
    }

    private async loadSavedData() {
        try {
            const searches = await AsyncStorage.getItem('recentSearches');
            const wishlistData = await AsyncStorage.getItem('shoppingWishlist');

            if (searches) this.recentSearches = JSON.parse(searches);
            if (wishlistData) this.wishlist = JSON.parse(wishlistData);
        } catch (error) {
            console.error('Failed to load shopping data:', error);
        }
    }

    private async saveRecentSearches() {
        try {
            await AsyncStorage.setItem('recentSearches', JSON.stringify(this.recentSearches.slice(0, 10)));
        } catch (error) {
            console.error('Failed to save searches:', error);
        }
    }

    /**
     * Search for products based on query and filters
     */
    async searchProducts(params: ProductSearchParams): Promise<Product[]> {
        const { query } = params;

        // Add to recent searches
        if (query && !this.recentSearches.includes(query)) {
            this.recentSearches.unshift(query);
            this.saveRecentSearches();
        }

        const result = await catalogRepository.search(params);
        return result.items.map(item => ({
            id:item.id, name:item.name, brand:item.brand, price:item.price,
            currency:item.currency || '', imageUrl:typeof item.imageUrl === 'string' ? item.imageUrl : '',
            productUrl:getShopDestination({...item, shopUrl:item.sourceUrl}), affiliateUrl:getShopDestination({...item, shopUrl:item.sourceUrl}),
            category:({upper_body:'tops',lower_body:'bottoms',accessory:'accessories'} as Record<string,string>)[item.garmentType] || item.garmentType,
            color:catalogTitleColor(item), source:item.source || result.origin,
        }));
    }

    /** Goal-specific shopping intelligence; generation contracts remain unchanged. */
    async findBetter(reference: ShopCatalogItem, owned: readonly ShopCatalogItem[], options: Parameters<typeof findBetterShoppingProducts>[2]) {
        return findBetterShoppingProducts(reference, owned, {...options, repository:catalogRepository});
    }

    evaluatePurchase(candidate: ShopCatalogItem, owned: readonly ShopCatalogItem[], context: Parameters<typeof evaluateShoppingPurchase>[2] = {}) {
        return evaluateShoppingPurchase(candidate, owned, context);
    }

    /**
     * Find similar products based on an item in wardrobe
     */
    async findSimilarProducts(item: {
        type: string;
        color?: string;
        style?: string;
    }): Promise<Product[]> {
        // Determine category from item type
        const itemType = (item.type || '').toLowerCase();
        let category: string | undefined;

        for (const [cat, keywords] of Object.entries(PRODUCT_CATEGORIES)) {
            if (keywords.some(k => itemType.includes(k))) {
                category = cat;
                break;
            }
        }

        return this.searchProducts({
            query: item.type,
            category,
            color: item.color,
            limit: 10,
        });
    }

    /**
     * Get "Complete Your Look" suggestions based on current outfit
     */
    async getCompleteYourLookSuggestions(currentItems: any[]): Promise<CompleteYourLookSuggestion[]> {
        const suggestions: CompleteYourLookSuggestion[] = [];

        // Analyze what categories are already in the outfit
        const presentCategories = new Set<string>();

        currentItems.forEach(item => {
            const itemType = (item.type || item.itemType || '').toLowerCase();

            for (const [cat, keywords] of Object.entries(PRODUCT_CATEGORIES)) {
                if (keywords.some(k => itemType.includes(k))) {
                    presentCategories.add(cat);
                    break;
                }
            }
        });

        // Suggest missing categories
        const missingCategories: { category: string; reason: string }[] = [];

        if (!presentCategories.has('shoes')) {
            missingCategories.push({
                category: 'shoes',
                reason: 'Complete your outfit with the perfect footwear',
            });
        }

        if (!presentCategories.has('accessories')) {
            missingCategories.push({
                category: 'accessories',
                reason: 'Add a finishing touch with an accessory',
            });
        }

        // If only has top, suggest bottom
        if (presentCategories.has('tops') && !presentCategories.has('bottoms') && !presentCategories.has('dresses')) {
            missingCategories.push({
                category: 'bottoms',
                reason: 'Pair your top with these bottoms',
            });
        }

        // If only has bottom, suggest top
        if (presentCategories.has('bottoms') && !presentCategories.has('tops') && !presentCategories.has('dresses')) {
            missingCategories.push({
                category: 'tops',
                reason: 'These tops would complement your bottoms',
            });
        }

        // Get product suggestions for each missing category
        for (const { category, reason } of missingCategories) {
            const products = await this.searchProducts({
                query: '',
                category,
                limit: 5
            });

            if (products.length > 0) {
                suggestions.push({
                    missingCategory: category,
                    suggestedProducts: products,
                    reason,
                });
            }
        }

        return suggestions;
    }

    /**
     * Get affiliate link for product
     */
    getAffiliateLink(product: Product): string {
        // Feed URLs are untrusted: only http(s) links reach Linking.openURL;
        // anything else becomes a Shopping search for the product.
        return getShopDestination({
            brand: product.brand,
            name: product.name,
            shopUrl: product.affiliateUrl || product.productUrl,
        });
    }

    /**
     * Track product click for analytics
     */
    async trackProductClick(product: Product, source: string) {
        logger.info(`Product clicked: ${product.name} from ${source}`);

        // In production, send to analytics
        try {
            await fetch(`${API_URL}/api/analytics/product-click`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    productId: product.id,
                    productName: product.name,
                    brand: product.brand,
                    price: product.price,
                    source,
                    timestamp: new Date().toISOString(),
                }),
            }).catch(() => { }); // Non-blocking
        } catch (error) {
            // Silent fail
        }
    }

    /**
     * Add product to wishlist
     */
    async addToWishlist(product: Product): Promise<void> {
        if (!this.wishlist.find(p => p.id === product.id)) {
            this.wishlist.push(product);
            await AsyncStorage.setItem('shoppingWishlist', JSON.stringify(this.wishlist));
        }
    }

    /**
     * Remove from wishlist
     */
    async removeFromWishlist(productId: string): Promise<void> {
        this.wishlist = this.wishlist.filter(p => p.id !== productId);
        await AsyncStorage.setItem('shoppingWishlist', JSON.stringify(this.wishlist));
    }

    /**
     * Get wishlist
     */
    getWishlist(): Product[] {
        return this.wishlist;
    }

    /**
     * Get recent searches
     */
    getRecentSearches(): string[] {
        return this.recentSearches;
    }

    /**
     * Clear recent searches
     */
    async clearRecentSearches(): Promise<void> {
        this.recentSearches = [];
        await AsyncStorage.removeItem('recentSearches');
    }
}

// Export singleton instance
export const shoppingService = new ShoppingService();
export default shoppingService;

// ─────────────────────────────────────────────────────────────────────────────
// fillMissingSlots — queries shop_catalog for 1 style-matching item per
// missing outfit slot. Used by the outfit generator to auto-complete
// looks when the wardrobe lacks, say, an outerwear piece for an
// Old-Money layered look.
// ─────────────────────────────────────────────────────────────────────────────

export type OutfitSlotId = 'outerwear' | 'top' | 'bottom' | 'shoes';

export interface ShopFillItem {
    id: string;
    name: string;
    image: string;
    type: string;
    macroCategory: OutfitSlotId;
    color: string;
    brand?: string;
    price?: number;
    currency?: string;
    shopUrl?: string;
    isShopItem: true;
    recommendation: string;
    missingSlot: OutfitSlotId;
}

interface ShopCatalogRow {
    id: string;
    brand?: string | null;
    name?: string | null;
    price?: number | null;
    currency?: string | null;
    image_url?: string | null;
    garment_type?: string | null;
    category?: string | null;
    description?: string | null;
    source?: string | null;
    source_url?: string | null;
}

const SHOP_COLOR_WORDS = ['black','white','offwhite','ecru','cream','beige','stone','sand','taupe','camel','khaki','olive','green','sage','mint','blue','navy','indigo','brown','chocolate','grey','gray','silver','charcoal','red','burgundy','maroon','pink','purple','lilac','yellow','mustard','orange','rust','tan','forest'];

function inferShopColor(row: Pick<ShopCatalogRow, 'name' | 'description'>): string {
    const blob = `${row?.name || ''} ${row?.description || ''}`.toLowerCase();
    for (const c of SHOP_COLOR_WORDS) {
        if (new RegExp(`\\b${c}\\b`).test(blob)) return c;
    }
    return 'neutral';
}

function rowToShopFillItem(row: ShopCatalogRow, slot: OutfitSlotId): ShopFillItem | null {
    if (!row?.id || !row?.image_url || !hasProductOnlyCatalogProvenance(row)) return null;
    if (inferShopRowMacro(row.garment_type || '', row.name || '') !== slot) return null;
    return {
        id: `shop_${row.id}`,
        name: row.name || row.brand || 'Shop pick',
        image: row.image_url,
        type: row.garment_type || row.category || slot,
        macroCategory: slot,
        color: inferShopColor(row),
        brand: row.brand || undefined,
        price: typeof row.price === 'number' ? row.price : undefined,
        currency: row.currency || undefined,
        shopUrl: row.source_url || undefined,
        isShopItem: true,
        recommendation: `Suggested from shop to complete your ${slot === 'outerwear' ? 'main-top layer' : slot}`,
        missingSlot: slot,
    };
}

function isOuterwearCatalogItem(item: Pick<ShopCatalogItem, 'name' | 'description' | 'garmentType'>): boolean {
    const blob = `${item.name || ''} ${item.description || ''} ${item.garmentType || ''}`.toLowerCase();
    return /\b(jacket|coat|blazer|cardigan|sweater|hoodie|puffer|bomber|vest|outerwear|trench|peacoat|overshirt)\b/.test(blob);
}

function localCatalogRowsForSlot(slot: OutfitSlotId): ShopCatalogRow[] {
    const seen = new Set<string>();
    const rows: ShopCatalogRow[] = [];

    const allItems = getStaticSlotPool();
    for (const item of allItems) {
        if (!item?.id || !item?.imageUrl) continue;
        if (seen.has(item.id)) continue;

        const isOuter = isOuterwearCatalogItem(item);
        const matchesSlot =
            (slot === 'outerwear' && item.garmentType === 'upper_body' && isOuter)
            || (slot === 'top' && item.garmentType === 'upper_body' && !isOuter)
            || (slot === 'bottom' && item.garmentType === 'lower_body')
            || (slot === 'shoes' && item.garmentType === 'shoes');

        if (!matchesSlot) continue;
        seen.add(item.id);
        rows.push({
            id: item.id,
            brand: item.brand,
            name: item.name,
            price: item.price,
            currency: item.currency,
            image_url: typeof item.imageUrl === 'string' ? item.imageUrl : '',
            garment_type: item.garmentType,
            category: slot === 'top' ? 'tops' : slot === 'bottom' ? 'bottoms' : slot,
            description: item.description,
            source: 'local_catalog',
            source_url: item.sourceUrl,
        });
    }

    return rows;
}

/**
 * Fetch multiple style-ranked shop items for ALL outfit slots, not just
 * missing ones. Used by the local outfit fallback so that style-specific
 * sections on Home ("Business Casual", "Old Money") get shop items that
 * actually match the aesthetic, even when the user's wardrobe has items
 * in those macro-categories that clash with the style.
 *
 * Returns up to `perSlot` items per slot, style-ranked best-first.
 */
export async function fetchShopPoolForStyle(
    slots: OutfitSlotId[],
    style: string | null | undefined,
    perSlot = 5,
): Promise<ShopFillItem[]> {
    if (!slots || slots.length === 0) return [];
    const normalizedStyle: StyleId = normalizeStyleId(style || 'casual');

    // Parallelize per-slot queries — the old sequential for-await loop issued
    // up to 4 Supabase round-trips back-to-back (~4x latency). Promise.all
    // collapses them into a single network flight.
    const perSlotResults = await Promise.all(
        slots.map(async (slot): Promise<ShopFillItem[]> => {
            try {
                const { data, error } = await loadCatalogDiscovery({slot,limit:30})
                    .then(page=>({data:page.rows,error:null})).catch(error => ({ data:null, error }));
                const sourceRows = (!error && data && data.length > 0)
                    ? (data as ShopCatalogRow[])
                    : localCatalogRowsForSlot(slot);
                if (sourceRows.length === 0) return [];

                const scored = sourceRows
                    .map((row) => ({
                        row,
                        score: scoreItemForStyle(
                            {
                                name: row.name || undefined,
                                description: row.description || undefined,
                                brand: row.brand || undefined,
                                color: inferShopColor(row),
                                type: row.garment_type || undefined,
                                category: row.category || undefined,
                                macroCategory: slot,
                            },
                            normalizedStyle,
                        ),
                    }))
                    .sort((a, b) => b.score - a.score);

                const fillsById = new Map<string, ShopFillItem>();
                const candidates = scored.flatMap(candidate => {
                    const fill = rowToShopFillItem(candidate.row, slot);
                    if (!fill) return [];
                    fillsById.set(candidate.row.id, fill);
                    return [{ score:candidate.score, item:{
                        id:candidate.row.id, name:fill.name, brand:fill.brand || '', price:fill.price || 0,
                        currency:fill.currency, imageUrl:fill.image,
                        garmentType:candidate.row.garment_type as ShopCatalogItem['garmentType'],
                        source:candidate.row.source || undefined, sourceUrl:candidate.row.source_url || undefined,
                    } }];
                });
                // Eligibility and final outfit validation remain in the existing
                // pipeline. Only spread admitted physical candidates here.
                const floors=new Map<string,typeof candidates[number]>();
                for(const candidate of candidates) {
                    const currency=candidate.item.currency;
                    if(currency && candidate.item.price>0 && Number.isFinite(candidate.item.price) &&
                       (!floors.has(currency) || candidate.item.price<floors.get(currency)!.item.price)) floors.set(currency,candidate);
                }
                const floorIds=new Set(floors.size<=perSlot?[...floors.values()].map(candidate=>candidate.item.id):[]);
                const fills = rankDiverseShoppingCandidates(candidates.map(candidate=>({...candidate,budgetRepresentative:floorIds.has(candidate.item.id)})), {
                    limit:Math.max(0,Math.min(30,perSlot)), eligible:() => true,
                }).map(item => fillsById.get(item.id)!);
                return fills;
            } catch (err) {
                logger.warn(`fetchShopPoolForStyle failed for slot=${slot}: ${String(err)}`);
                return [];
            }
        }),
    );
    return perSlotResults.flat();
}

/**
 * Return 1 shop item per missing slot, style-ranked. Safe to call with an
 * empty list — returns an empty array. Errors are swallowed so outfit
 * generation never fails because the shop table is unreachable.
 */
export async function fillMissingSlots(
    missingSlots: OutfitSlotId[],
    style: string | null | undefined,
): Promise<ShopFillItem[]> {
    if (!missingSlots || missingSlots.length === 0) return [];
    const normalizedStyle: StyleId = normalizeStyleId(style || 'casual');

    // Parallelize — same reasoning as fetchShopPoolForStyle above.
    const perSlotPicks = await Promise.all(
        missingSlots.map(async (slot): Promise<ShopFillItem | null> => {
            try {
                const { data, error } = await loadCatalogDiscovery({slot,limit:30})
                    .then(page=>({data:page.rows,error:null})).catch(error => ({ data:null, error }));
                const sourceRows = (!error && data && data.length > 0)
                    ? (data as ShopCatalogRow[])
                    : localCatalogRowsForSlot(slot);
                if (sourceRows.length === 0) return null;

                const scored = sourceRows
                    .map((row) => ({
                        row,
                        score: scoreItemForStyle(
                            {
                                name: row.name || undefined,
                                description: row.description || undefined,
                                brand: row.brand || undefined,
                                color: inferShopColor(row),
                                type: row.garment_type || undefined,
                                category: row.category || undefined,
                                macroCategory: slot,
                            },
                            normalizedStyle,
                        ),
                    }))
                    .sort((a, b) => b.score - a.score);

                for (const candidate of scored) {
                    const fill = rowToShopFillItem(candidate.row, slot);
                    if (fill) return fill;
                }
                return null;
            } catch (err) {
                logger.warn(`fillMissingSlots failed for slot=${slot}: ${String(err)}`);
                return null;
            }
        }),
    );
    return perSlotPicks.filter((p): p is ShopFillItem => p !== null);
}
