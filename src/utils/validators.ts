/**
 * Zod schemas for validating API responses and DB rows.
 * Prevents corrupted data from silently entering the store.
 */

import { z } from 'zod';

export const ClothingItemRowSchema = z.object({
    id: z.string().uuid(),
    user_id: z.string().uuid(),
    image_url: z.string().min(1),
    thumbnail_url: z.string().nullable().optional(),
    category: z.enum(['top', 'bottom', 'dress', 'shoes', 'outerwear', 'accessory', 'other']),
    sub_category: z.string().nullable().optional(),
    type: z.string().nullable().optional(),
    primary_color: z.string().nullable().optional(),
    color: z.any().optional(),
    color_hex: z.string().nullable().optional(),
    pattern: z.string().nullable().optional(),
    material: z.string().nullable().optional(),
    brand: z.string().nullable().optional(),
    name: z.string().nullable().optional(),
    seasons: z.array(z.string()).nullable().optional(),
    occasions: z.array(z.string()).nullable().optional(),
    wear_count: z.number().int().min(0).default(0),
    last_worn_date: z.string().nullable().optional(),
    last_worn_at: z.string().nullable().optional(),
    is_favorite: z.boolean().default(false),
    detection_confidence: z.number().min(0).max(1).nullable().optional(),
    created_at: z.string(),
    updated_at: z.string(),
});

export type ValidatedClothingRow = z.infer<typeof ClothingItemRowSchema>;

export const OutfitResponseSchema = z.object({
    id: z.string().optional(),
    description: z.string().optional(),
    style: z.string().optional(),
    occasion: z.string().optional(),
    confidence: z.number().min(0).max(1).optional(),
    items: z.array(z.object({
        id: z.string(),
        type: z.string().optional(),
        macroCategory: z.string().optional(),
        color: z.string().optional(),
        name: z.string().optional(),
        brand: z.string().optional(),
        imageUrl: z.string().optional(),
        image_url: z.string().optional(),
        recommendation: z.string().optional(),
        isShopItem: z.boolean().optional(),
        price: z.number().optional(),
        shopUrl: z.string().optional(),
    })).default([]),
    stylingTips: z.array(z.string()).optional(),
});

export const GenerateOutfitsResponseSchema = z.object({
    success: z.boolean(),
    outfits: z.array(OutfitResponseSchema),
    source: z.enum(['ai', 'local']).optional(),
    error: z.string().optional(),
});

/**
 * Validate and filter an array of DB rows, logging warnings for invalid ones.
 */
export function validateClothingRows(rows: unknown[]): ValidatedClothingRow[] {
    const valid: ValidatedClothingRow[] = [];
    for (const row of rows) {
        const result = ClothingItemRowSchema.safeParse(row);
        if (result.success) {
            valid.push(result.data);
        } else {
            console.warn('[validators] Invalid clothing row skipped:', result.error.issues[0]?.message);
        }
    }
    return valid;
}
