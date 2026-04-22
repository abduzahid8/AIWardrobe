/**
 * Try-On feature — shared types
 */

export interface WardrobeItem {
    id: string;
    type?: string;
    category?: string;
    color?: string;
    imageUrl?: string;
}

export interface ShopCatalogItem {
    id: string;
    brand: string;
    name: string;
    price: number;
    currency?: string;
    imageUrl: string | any;
    garmentType: 'upper_body' | 'lower_body' | 'dresses' | 'shoes' | 'outfit';
    description?: string;
    outfitItems?: ShopCatalogItem[];
}

export type TryOnMode = 'model';
export type TryOnStep = 1 | 2 | 3;
export type PhotoTab = 'upload' | 'wardrobe' | 'shop';
