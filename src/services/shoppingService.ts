/**
 * Shopping Integration Service
 * Provides product search, affiliate links, and "Complete Your Look" features
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import { createLogger } from '../utils/logger';
import { supabase } from '../../lib/supabase';
import {
    scoreItemForStyle,
    normalizeStyleId,
    type StyleId,
} from '../../features/outfit-generator/utils/styleInference';

const logger = createLogger('Shopping');

const API_URL = process.env.EXPO_PUBLIC_API_URL || 'https://aiwardrobe-ivh4.onrender.com';

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
    inStock: boolean;
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
// MOCK DATA (Replace with real API in production)
// ============================================

const MOCK_PRODUCTS: Product[] = [
    {
        id: 'prod-1',
        name: 'Classic White Button-Down Shirt',
        brand: 'Everlane',
        price: 78,
        currency: 'USD',
        imageUrl: 'https://example.com/shirt.jpg',
        productUrl: 'https://everlane.com/shirt',
        affiliateUrl: 'https://everlane.com/shirt?ref=aiwardrobe',
        category: 'tops',
        color: 'white',
        inStock: true,
        rating: 4.5,
        reviewCount: 234,
        source: 'everlane',
    },
    {
        id: 'prod-2',
        name: 'High-Rise Straight Jeans',
        brand: 'Levi\'s',
        price: 98,
        originalPrice: 128,
        currency: 'USD',
        imageUrl: 'https://example.com/jeans.jpg',
        productUrl: 'https://levis.com/jeans',
        affiliateUrl: 'https://levis.com/jeans?ref=aiwardrobe',
        category: 'bottoms',
        color: 'blue',
        inStock: true,
        rating: 4.7,
        reviewCount: 567,
        source: 'levis',
    },
    {
        id: 'prod-3',
        name: 'Leather Crossbody Bag',
        brand: 'Madewell',
        price: 148,
        currency: 'USD',
        imageUrl: 'https://example.com/bag.jpg',
        productUrl: 'https://madewell.com/bag',
        affiliateUrl: 'https://madewell.com/bag?ref=aiwardrobe',
        category: 'accessories',
        color: 'brown',
        inStock: true,
        rating: 4.3,
        reviewCount: 89,
        source: 'madewell',
    },
    {
        id: 'prod-4',
        name: 'Minimalist Leather Sneakers',
        brand: 'Common Projects',
        price: 425,
        currency: 'USD',
        imageUrl: 'https://example.com/sneakers.jpg',
        productUrl: 'https://commonprojects.com/sneakers',
        affiliateUrl: 'https://commonprojects.com/sneakers?ref=aiwardrobe',
        category: 'shoes',
        color: 'white',
        inStock: true,
        rating: 4.8,
        reviewCount: 312,
        source: 'commonprojects',
    },
    {
        id: 'prod-5',
        name: 'Wool Blazer',
        brand: 'Theory',
        price: 495,
        originalPrice: 595,
        currency: 'USD',
        imageUrl: 'https://example.com/blazer.jpg',
        productUrl: 'https://theory.com/blazer',
        affiliateUrl: 'https://theory.com/blazer?ref=aiwardrobe',
        category: 'tops',
        color: 'navy',
        inStock: true,
        rating: 4.6,
        reviewCount: 156,
        source: 'theory',
    },
];

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
        const { query, category, color, priceMin, priceMax, brand, limit = 20 } = params;

        // Add to recent searches
        if (query && !this.recentSearches.includes(query)) {
            this.recentSearches.unshift(query);
            this.saveRecentSearches();
        }

        // In production, call actual shopping APIs
        // For now, filter mock data
        let results = [...MOCK_PRODUCTS];

        if (query) {
            const q = query.toLowerCase();
            results = results.filter(p =>
                p.name.toLowerCase().includes(q) ||
                p.brand.toLowerCase().includes(q) ||
                p.category.toLowerCase().includes(q)
            );
        }

        if (category) {
            results = results.filter(p => p.category === category);
        }

        if (color) {
            results = results.filter(p => p.color?.toLowerCase() === color.toLowerCase());
        }

        if (priceMin !== undefined) {
            results = results.filter(p => p.price >= priceMin);
        }

        if (priceMax !== undefined) {
            results = results.filter(p => p.price <= priceMax);
        }

        if (brand) {
            results = results.filter(p => p.brand.toLowerCase() === brand.toLowerCase());
        }

        return results.slice(0, limit);
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
        // In production, generate proper affiliate links with tracking
        return product.affiliateUrl || product.productUrl;
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
    primary_color?: string | null;
    source?: string | null;
}

/**
 * Map a missing outfit slot to the Supabase query that narrows shop_catalog
 * to candidates for that slot. We target both `category` and `garment_type`
 * because the two columns evolved independently across ingestion scripts.
 */
function buildShopQueryForSlot(slot: OutfitSlotId, limit: number) {
    const base = supabase
        .from('shop_catalog')
        .select('id, brand, name, price, currency, image_url, garment_type, category, description, primary_color, source')
        .eq('is_active', true)
        .limit(limit);

    switch (slot) {
        case 'outerwear':
            return base.or('category.eq.outerwear,garment_type.eq.outerwear');
        case 'top':
            return base.or('category.eq.tops,garment_type.eq.upper_body');
        case 'bottom':
            return base.or('category.eq.bottoms,garment_type.eq.lower_body');
        case 'shoes':
            return base.or('category.eq.shoes,garment_type.eq.shoes');
        default:
            return base;
    }
}

function rowToShopFillItem(row: ShopCatalogRow, slot: OutfitSlotId): ShopFillItem | null {
    if (!row?.id || !row?.image_url) return null;
    return {
        id: `shop_${row.id}`,
        name: row.name || row.brand || 'Shop pick',
        image: row.image_url,
        type: row.garment_type || row.category || slot,
        macroCategory: slot,
        color: row.primary_color || 'neutral',
        brand: row.brand || undefined,
        price: typeof row.price === 'number' ? row.price : undefined,
        shopUrl: undefined,
        isShopItem: true,
        recommendation: `Suggested from shop to complete your ${slot === 'outerwear' ? 'main-top layer' : slot}`,
        missingSlot: slot,
    };
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
    const pool: ShopFillItem[] = [];

    for (const slot of slots) {
        try {
            const { data, error } = await buildShopQueryForSlot(slot, 30);
            if (error || !data || data.length === 0) continue;

            const scored = (data as ShopCatalogRow[])
                .map((row) => ({
                    row,
                    score: scoreItemForStyle(
                        {
                            name: row.name || undefined,
                            description: row.description || undefined,
                            brand: row.brand || undefined,
                            color: row.primary_color || undefined,
                            type: row.garment_type || undefined,
                            category: row.category || undefined,
                            macroCategory: slot,
                        },
                        normalizedStyle,
                    ),
                }))
                .sort((a, b) => b.score - a.score);

            let added = 0;
            for (const candidate of scored) {
                if (added >= perSlot) break;
                const fill = rowToShopFillItem(candidate.row, slot);
                if (fill) {
                    pool.push(fill);
                    added++;
                }
            }
        } catch (err) {
            logger.warn(`fetchShopPoolForStyle failed for slot=${slot}: ${String(err)}`);
        }
    }
    return pool;
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
    const picks: ShopFillItem[] = [];

    for (const slot of missingSlots) {
        try {
            const { data, error } = await buildShopQueryForSlot(slot, 30);
            if (error || !data || data.length === 0) continue;

            const scored = (data as ShopCatalogRow[])
                .map((row) => ({
                    row,
                    score: scoreItemForStyle(
                        {
                            name: row.name || undefined,
                            description: row.description || undefined,
                            brand: row.brand || undefined,
                            color: row.primary_color || undefined,
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
                if (fill) {
                    picks.push(fill);
                    break;
                }
            }
        } catch (err) {
            logger.warn(`fillMissingSlots failed for slot=${slot}: ${String(err)}`);
        }
    }
    return picks;
}
