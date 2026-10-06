/**
 * Shop catalog item — shared type.
 *
 * Previously lived in `features/try-on/types.ts` (deleted with the Virtual
 * Try-On feature). It is the catalog shape consumed by Home, Inspo, shop
 * stores, and the outfit pipeline, so it lives here now.
 */
import type { GarmentPhysicalProfile } from './garment';

export interface ShopCatalogItem {
    id: string;
    brand: string;
    name: string;
    price: number;
    currency?: string;
    imageUrl: string | any;
    garmentType: 'upper_body' | 'lower_body' | 'dresses' | 'shoes' | 'outfit' | 'accessory';
    description?: string;
    sourceUrl?: string;
    outfitItems?: ShopCatalogItem[];
    /**
     * Per-size physical measurements. Optional — when present, drives the
     * fit engine.
     */
    physicalProfiles?: GarmentPhysicalProfile[];
    /** Default size label to preselect. */
    defaultSize?: string;
    /** Available size labels for this garment (S/M/L, 30/32/34, EU 41/42/43, etc.). */
    availableSizes?: string[];
}
