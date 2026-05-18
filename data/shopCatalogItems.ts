/**
 * Shop Catalog — curated items for AI Virtual Try-On
 *
 * garmentType maps to IDM-VTON categories:
 *   'upper_body' — tops, shirts, jackets, blazers, hoodies, coats
 *   'lower_body' — trousers, jeans, skirts, shorts
 *   'dresses'    — full-length dresses, jumpsuits
 *
 * imageUrl must be a publicly accessible HTTPS URL so the Replicate API
 * can fetch the garment image for composite generation.
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
    { key: 'dresses', label: 'Dresses' },
];

export const SHOP_CATALOG_ITEMS: ShopCatalogItem[] = [
    // ── Massimo Dutti ──────────────────────────────────────────────────────
    {
        id: 'md-001',
        brand: 'Massimo Dutti',
        name: 'Oxford Cotton Shirt',
        price: 79.90,
        currency: 'USD',
        garmentType: 'upper_body',
        description: 'classic oxford cotton button-down shirt with spread collar',
        imageUrl: 'https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=600&h=800&fit=crop&q=85',
    },
    {
        id: 'md-002',
        brand: 'Massimo Dutti',
        name: 'Slim Wool Blazer',
        price: 299.00,
        currency: 'USD',
        garmentType: 'upper_body',
        description: 'slim-fit structured wool blazer with notch lapels',
        imageUrl: 'https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=600&h=800&fit=crop&q=85',
    },
    {
        id: 'md-003',
        brand: 'Massimo Dutti',
        name: 'Tailored Slim Trousers',
        price: 129.00,
        currency: 'USD',
        garmentType: 'lower_body',
        description: 'slim-fit tailored trousers in stretch wool blend',
        imageUrl: 'https://images.unsplash.com/photo-1473966968600-fa801b869a1a?w=600&h=800&fit=crop&q=85',
    },
    {
        id: 'md-004',
        brand: 'Massimo Dutti',
        name: 'Linen Midi Dress',
        price: 189.00,
        currency: 'USD',
        garmentType: 'dresses',
        description: 'relaxed-fit midi linen dress with V-neckline',
        imageUrl: 'https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=600&h=800&fit=crop&q=85',
    },
    // ── COS ────────────────────────────────────────────────────────────────
    {
        id: 'cos-001',
        brand: 'COS',
        name: 'Oversized Wool Coat',
        price: 350.00,
        currency: 'USD',
        garmentType: 'upper_body',
        description: 'oversized structured wool overcoat with front seam detail',
        imageUrl: 'https://images.unsplash.com/photo-1539533018447-63fcce2678e3?w=600&h=800&fit=crop&q=85',
    },
    {
        id: 'cos-002',
        brand: 'COS',
        name: 'Wide-Leg Linen Trousers',
        price: 115.00,
        currency: 'USD',
        garmentType: 'lower_body',
        description: 'wide-leg relaxed linen trousers with elasticated waistband',
        imageUrl: 'https://images.unsplash.com/photo-1594938298603-c8148c4b5ea4?w=600&h=800&fit=crop&q=85',
    },
    {
        id: 'cos-003',
        brand: 'COS',
        name: 'Merino Crew Neck Knit',
        price: 95.00,
        currency: 'USD',
        garmentType: 'upper_body',
        description: 'structured relaxed crew-neck knit sweater in merino blend',
        imageUrl: 'https://images.unsplash.com/photo-1567401893414-76b7b1e5a7a5?w=600&h=800&fit=crop&q=85',
    },
    {
        id: 'cos-004',
        brand: 'COS',
        name: 'Shift Midi Dress',
        price: 145.00,
        currency: 'USD',
        garmentType: 'dresses',
        description: 'shift silhouette midi dress with clean seam lines',
        imageUrl: 'https://images.unsplash.com/photo-1585487000160-6ebcfceb0d03?w=600&h=800&fit=crop&q=85',
    },
    // ── Zara ───────────────────────────────────────────────────────────────
    {
        id: 'zara-001',
        brand: 'Zara',
        name: 'Relaxed Denim Jacket',
        price: 89.90,
        currency: 'USD',
        garmentType: 'upper_body',
        description: 'relaxed-fit denim jacket with button front and washed finish',
        imageUrl: 'https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=600&h=800&fit=crop&q=85',
    },
    {
        id: 'zara-002',
        brand: 'Zara',
        name: 'High-Rise Straight Jeans',
        price: 59.90,
        currency: 'USD',
        garmentType: 'lower_body',
        description: 'high-rise straight-leg jeans in mid-blue wash',
        imageUrl: 'https://images.unsplash.com/photo-1555689502-c4b22d76c56f?w=600&h=800&fit=crop&q=85',
    },
    {
        id: 'zara-003',
        brand: 'Zara',
        name: 'Satin Slip Dress',
        price: 69.90,
        currency: 'USD',
        garmentType: 'dresses',
        description: 'slim satin slip dress with thin shoulder straps and midi length',
        imageUrl: 'https://images.unsplash.com/photo-1551163943-3f6a855d1153?w=600&h=800&fit=crop&q=85',
    },
    // ── Mango ──────────────────────────────────────────────────────────────
    {
        id: 'mango-001',
        brand: 'Mango',
        name: 'Linen Blazer',
        price: 119.99,
        currency: 'USD',
        garmentType: 'upper_body',
        description: 'regular-fit linen blazer with structured shoulders',
        imageUrl: 'https://images.unsplash.com/photo-1593030761757-71fae45fa0e7?w=600&h=800&fit=crop&q=85',
    },
    {
        id: 'mango-002',
        brand: 'Mango',
        name: 'Flowy Midi Skirt',
        price: 59.99,
        currency: 'USD',
        garmentType: 'lower_body',
        description: 'flowy midi skirt with elastic waistband and airy fabric',
        imageUrl: 'https://images.unsplash.com/photo-1583496661160-fb5886a0aaaa?w=600&h=800&fit=crop&q=85',
    },
    // ── Arket ──────────────────────────────────────────────────────────────
    {
        id: 'arket-001',
        brand: 'Arket',
        name: 'Classic Poplin Shirt',
        price: 95.00,
        currency: 'USD',
        garmentType: 'upper_body',
        description: 'classic regular-fit poplin shirt in organic cotton',
        imageUrl: 'https://images.unsplash.com/photo-1620012253295-c15cc3e65df4?w=600&h=800&fit=crop&q=85',
    },
    {
        id: 'arket-002',
        brand: 'Arket',
        name: 'Slim Cotton Chinos',
        price: 110.00,
        currency: 'USD',
        garmentType: 'lower_body',
        description: 'slim-fit cotton chinos with clean front and back welt pockets',
        imageUrl: 'https://images.unsplash.com/photo-1624378439575-d8705ad7ae80?w=600&h=800&fit=crop&q=85',
    },
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
    'COS',
    'Zara',
    'Mango',
    'Arket',
];

export const SHOP_OUTFITS: ShopCatalogItem[] = [
    {
        id: 'outfit-001',
        brand: 'Curated Look',
        name: 'Summer Linen Suit',
        price: 239.99,
        currency: 'USD',
        garmentType: 'outfit',
        description: 'Complete summer look with Mango linen blazer and wide trousers',
        imageUrl: 'https://images.unsplash.com/photo-1483985988355-763728e1935b?w=600&h=800&fit=crop&q=85',
        outfitItems: [
            SHOP_CATALOG_ITEMS.find(i => i.id === 'mango-001')!,
            SHOP_CATALOG_ITEMS.find(i => i.id === 'cos-002')!
        ]
    },
    {
        id: 'outfit-002',
        brand: 'Curated Look',
        name: 'Casual Semi-Classic',
        price: 149.80,
        currency: 'USD',
        garmentType: 'outfit',
        description: 'Relaxed denim jacket with classic high-rise jeans',
        imageUrl: 'https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?w=600&h=800&fit=crop&q=85',
        outfitItems: [
            SHOP_CATALOG_ITEMS.find(i => i.id === 'zara-001')!,
            SHOP_CATALOG_ITEMS.find(i => i.id === 'zara-002')!
        ]
    }
];
