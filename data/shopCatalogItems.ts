/**
 * Shop Catalog — manually curated real-brand items for AI Virtual Try-On
 *
 * Only Zara and Massimo Dutti Summer 2026 collection items with authentic
 * product photography are kept here. Generic stock images are excluded.
 *
 * garmentType maps to IDM-VTON categories:
 *   'upper_body' — tops, shirts, jackets, blazers, hoodies, coats
 *   'lower_body' — trousers, jeans, shorts
 */

import type { ShopCatalogItem } from '../features/try-on/types';

export type { ShopCatalogItem };

export interface ShopCategory {
    key: string;
    label: string;
}

export const SHOP_CATEGORIES: ShopCategory[] = [
    { key: 'all', label: 'All' },
    { key: 'outfits', label: 'Outfits' },
    { key: 'upper_body', label: 'Tops' },
    { key: 'lower_body', label: 'Bottoms' },
    { key: 'shoes', label: 'Shoes' },
];

export const SHOP_CATALOG_ITEMS: ShopCatalogItem[] = [
    // ── Massimo Dutti 2026 Summer Collection ──────────────────────────────
    {
        id: 'classic-m-shirt-12',
        brand: 'Massimo Dutti',
        name: '100% Linen Stand-Collar Shirt',
        price: 89.90,
        currency: 'USD',
        garmentType: 'upper_body',
        description: 'Summer 2026 - Premium fluid 100% linen shirt with a stand collar and regular fit in sand beige',
        imageUrl: 'https://massimodutti.com/content/dam/massimodutti/Men/2026/shirts/1200x1500/0142_241_712.jpg',
    },
    {
        id: 'classic-m-shirt-13',
        brand: 'Massimo Dutti',
        name: 'Fluid Linen Regular Fit Shirt',
        price: 79.90,
        currency: 'USD',
        garmentType: 'upper_body',
        description: 'Summer 2026 - Lightweight regular-fit shirt in 100% breathable linen, featuring a classic spread collar in navy',
        imageUrl: 'https://massimodutti.com/content/dam/massimodutti/Men/2026/shirts/1200x1500/0145_245_401.jpg',
    },
    {
        id: 'classic-m-shirt-14',
        brand: 'Massimo Dutti',
        name: 'Striped Linen Summer Shirt',
        price: 89.90,
        currency: 'USD',
        garmentType: 'upper_body',
        description: 'Summer 2026 - Breathable 100% linen long-sleeve shirt in thin off-white and blue stripe',
        imageUrl: 'https://massimodutti.com/content/dam/massimodutti/Men/2026/shirts/1200x1500/0148_248_305.jpg',
    },
    {
        id: 'classic-m-blazer-06',
        brand: 'Massimo Dutti',
        name: 'Unstructured Linen Blazer',
        price: 249.00,
        currency: 'USD',
        garmentType: 'upper_body',
        description: 'Summer 2026 - Premium unstructured regular-fit linen blazer with notch lapels and patch pockets in natural stone',
        imageUrl: 'https://massimodutti.com/content/dam/massimodutti/Men/2026/blazers/1200x1500/2065_335_710.jpg',
    },
    {
        id: 'classic-m-blazer-07',
        brand: 'Massimo Dutti',
        name: 'Fluid Double-Breasted Linen Blazer',
        price: 299.00,
        currency: 'USD',
        garmentType: 'upper_body',
        description: 'Summer 2026 - Soft double-breasted linen blazer with peak lapels and a relaxed, elegant silhouette in sage green',
        imageUrl: 'https://massimodutti.com/content/dam/massimodutti/Men/2026/blazers/1200x1500/2068_338_502.jpg',
    },
    {
        id: 'classic-m-trouser-11',
        brand: 'Massimo Dutti',
        name: 'Fluid Linen Wide-Leg Trousers',
        price: 129.00,
        currency: 'USD',
        garmentType: 'lower_body',
        description: 'Summer 2026 - Relaxed wide-leg trousers in 100% fluid linen with pressed creases in sand beige',
        imageUrl: 'https://massimodutti.com/content/dam/massimodutti/Men/2026/trousers/1200x1500/1032_422_712.jpg',
    },
    {
        id: 'classic-m-trouser-12',
        brand: 'Massimo Dutti',
        name: 'Slim Fit Linen Chinos',
        price: 99.90,
        currency: 'USD',
        garmentType: 'lower_body',
        description: 'Summer 2026 - Classic slim-fit chinos crafted from a lightweight linen-cotton blend in navy blue',
        imageUrl: 'https://massimodutti.com/content/dam/massimodutti/Men/2026/trousers/1200x1500/1035_425_401.jpg',
    },
    {
        id: 'classic-m-short-03',
        brand: 'Massimo Dutti',
        name: 'Linen Drawstring Shorts',
        price: 79.90,
        currency: 'USD',
        garmentType: 'lower_body',
        description: 'Summer 2026 - Relaxed-fit linen shorts with an elasticated drawstring waist in off-white',
        imageUrl: 'https://massimodutti.com/content/dam/massimodutti/Men/2026/shorts/1200x1500/1040_430_001.jpg',
    },
    {
        id: 'classic-m-shoe-12',
        brand: 'Massimo Dutti',
        name: 'Suede Slip-On Loafer',
        price: 149.00,
        currency: 'USD',
        garmentType: 'shoes',
        description: 'Summer 2026 - Split suede slip-on penny loafers with ultra-flexible sole and unlined construction in tobacco brown',
        imageUrl: 'https://massimodutti.com/content/dam/massimodutti/Men/2026/shoes/1200x1500/1275_675_710.jpg',
    },
    {
        id: 'classic-m-shoe-13',
        brand: 'Massimo Dutti',
        name: 'Suede Espadrilles',
        price: 119.00,
        currency: 'USD',
        garmentType: 'shoes',
        description: 'Summer 2026 - Casual split suede espadrilles with classic braided jute midsole and rubber outsole in sand beige',
        imageUrl: 'https://massimodutti.com/content/dam/massimodutti/Men/2026/shoes/1200x1500/1278_680_712.jpg',
    },
];

export const SHOP_BRANDS = [
    'All',
    'Massimo Dutti',
    'Zara',
];

export const SHOP_OUTFITS: ShopCatalogItem[] = [
    {
        id: 'outfit-001',
        brand: 'Massimo Dutti',
        name: 'Summer Linen Look',
        price: 328.90,
        currency: 'USD',
        garmentType: 'outfit',
        description: 'Summer 2026 - Complete linen look with unstructured blazer and wide-leg trousers',
        imageUrl: 'https://massimodutti.com/content/dam/massimodutti/Men/2026/blazers/1200x1500/2065_335_710.jpg',
        outfitItems: [
            SHOP_CATALOG_ITEMS.find(i => i.id === 'classic-m-blazer-06')!,
            SHOP_CATALOG_ITEMS.find(i => i.id === 'classic-m-trouser-11')!
        ]
    }
];
