/**
 * Shopping Integration Service
 * Provides product search, affiliate links, and "Complete Your Look" features
 */

import AsyncStorage from '@react-native-async-storage/async-storage';

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
        imageUrl: 'https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=400',
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
        imageUrl: 'https://images.unsplash.com/photo-1541099649105-f69ad21f3246?w=400',
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
        name: 'Slim Fit Chino Trousers',
        brand: 'Uniqlo',
        price: 49,
        currency: 'USD',
        imageUrl: 'https://images.unsplash.com/photo-1624378439575-d8705ad7ae80?w=400',
        productUrl: 'https://uniqlo.com/chinos',
        affiliateUrl: 'https://uniqlo.com/chinos?ref=aiwardrobe',
        category: 'bottoms',
        color: 'beige',
        inStock: true,
        rating: 4.3,
        reviewCount: 412,
        source: 'uniqlo',
    },
    {
        id: 'prod-4',
        name: 'Minimalist Leather Sneakers',
        brand: 'Common Projects',
        price: 425,
        currency: 'USD',
        imageUrl: 'https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400',
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
        imageUrl: 'https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=400',
        productUrl: 'https://theory.com/blazer',
        affiliateUrl: 'https://theory.com/blazer?ref=aiwardrobe',
        category: 'tops',
        color: 'navy',
        inStock: true,
        rating: 4.6,
        reviewCount: 156,
        source: 'theory',
    },
    {
        id: 'prod-6',
        name: 'Running Sneakers Pro',
        brand: 'Nike',
        price: 110,
        currency: 'USD',
        imageUrl: 'https://images.unsplash.com/photo-1608231387042-66d1773070a5?w=400',
        productUrl: 'https://nike.com/runners',
        affiliateUrl: 'https://nike.com/runners?ref=aiwardrobe',
        category: 'shoes',
        color: 'black',
        inStock: true,
        rating: 4.7,
        reviewCount: 890,
        source: 'nike',
    },
    {
        id: 'prod-7',
        name: 'Oversized Hoodie',
        brand: 'Champion',
        price: 65,
        currency: 'USD',
        imageUrl: 'https://images.unsplash.com/photo-1556821840-3a63f15732ce?w=400',
        productUrl: 'https://champion.com/hoodie',
        affiliateUrl: 'https://champion.com/hoodie?ref=aiwardrobe',
        category: 'tops',
        color: 'grey',
        inStock: true,
        rating: 4.4,
        reviewCount: 623,
        source: 'champion',
    },
    {
        id: 'prod-8',
        name: 'Ankle Boots',
        brand: 'Dr. Martens',
        price: 180,
        currency: 'USD',
        imageUrl: 'https://images.unsplash.com/photo-1638247025967-b4e38f787b76?w=400',
        productUrl: 'https://drmartens.com/boots',
        affiliateUrl: 'https://drmartens.com/boots?ref=aiwardrobe',
        category: 'shoes',
        color: 'black',
        inStock: true,
        rating: 4.9,
        reviewCount: 1204,
        source: 'drmartens',
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
        console.log(`[Shopping] Product clicked: ${product.name} from ${source}`);

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
