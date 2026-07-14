/**
 * Try-On feature — shared types
 */
import type { GarmentPhysicalProfile } from '../../src/types/garment';

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
    garmentType: 'upper_body' | 'lower_body' | 'dresses' | 'shoes' | 'outfit' | 'accessory';
    description?: string;
    sourceUrl?: string;
    outfitItems?: ShopCatalogItem[];
    /**
     * Body-fit addition (Month 1 of docs/AIWARDROBE_6_MONTH_BODY_FIT_VTON_PLAN.md).
     * Per-size physical measurements. Optional — when present, drives the
     * fit engine. Month 5 will wire this to a real size-chart ingestion
     * pipeline; for now it's populated manually for seed garments.
     */
    physicalProfiles?: GarmentPhysicalProfile[];
    /** Default size label to preselect in the try-on UI. */
    defaultSize?: string;
    /** Available size labels for this garment (S/M/L, 30/32/34, EU 41/42/43, etc.). */
    availableSizes?: string[];
}

export type TryOnMode = 'model';
export type TryOnStep = 1 | 2 | 3;
export type PhotoTab = 'upload' | 'wardrobe' | 'shop';
