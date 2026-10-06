/**
 * Shop Catalog — local static catalog removed.
 *
 * All shop items now come live from Supabase `shop_catalog`
 * (imported from /Users/v.zahid/startups/website-downloader).
 * These exports are kept as empty arrays so existing imports don't break.
 */

import type { ShopCatalogItem } from '../src/types/shop';

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
    { key: 'accessory', label: 'Accessories' },
];

export const SHOP_CATALOG_ITEMS: ShopCatalogItem[] = [];

export const SHOP_BRANDS = ['All'];

export const SHOP_OUTFITS: ShopCatalogItem[] = [];
