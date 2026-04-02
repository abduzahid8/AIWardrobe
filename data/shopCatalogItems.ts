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
    { key: 'all',        label: 'All'     },
    { key: 'upper_body', label: 'Tops'    },
    { key: 'lower_body', label: 'Bottoms' },
    { key: 'dresses',    label: 'Dresses' },
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
        imageUrl: 'https://images.unsplash.com/photo-1544441892-794166f1e3be?w=600&h=800&fit=crop&q=85',
    },
    {
        id: 'zara-002',
        brand: 'Zara',
        name: 'High-Rise Straight Jeans',
        price: 59.90,
        currency: 'USD',
        garmentType: 'lower_body',
        description: 'high-rise straight-leg jeans in mid-blue wash',
        imageUrl: 'https://images.unsplash.com/photo-1542272604-787c3835535d?w=600&h=800&fit=crop&q=85',
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
];

export const SHOP_BRANDS = [
    'All',
    'Massimo Dutti',
    'COS',
    'Zara',
    'Mango',
    'Arket',
];
