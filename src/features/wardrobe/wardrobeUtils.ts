/**
 * Wardrobe Video Analysis — pure utility functions
 * Extracted from WardrobeVideoScreen for testability and reuse.
 */

import { DetectedItem } from './types';

export const CLOTHING_IMAGES: Record<string, string> = {
    'jacket':   'https://images.unsplash.com/photo-1551028719-00167b16eac5?w=400&h=500&fit=crop',
    'denim':    'https://images.unsplash.com/photo-1551028719-00167b16eac5?w=400&h=500&fit=crop',
    'shirt':    'https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=400&h=500&fit=crop',
    't-shirt':  'https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400&h=500&fit=crop',
    'jeans':    'https://images.unsplash.com/photo-1542272454315-4c01d7abdf4a?w=400&h=500&fit=crop',
    'pants':    'https://images.unsplash.com/photo-1624378439575-d8705ad7ae80?w=400&h=500&fit=crop',
    'dress':    'https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=400&h=500&fit=crop',
    'sweater':  'https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=400&h=500&fit=crop',
    'hoodie':   'https://images.unsplash.com/photo-1556821840-3a63f95609a7?w=400&h=500&fit=crop',
    'coat':     'https://images.unsplash.com/photo-1539533018447-63fcce2678e3?w=400&h=500&fit=crop',
    'shoes':    'https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400&h=500&fit=crop',
    'sneakers': 'https://images.unsplash.com/photo-1460353581641-37baddab0fa2?w=400&h=500&fit=crop',
};

const CATEGORY_DISPLAY_MAP: Record<string, string> = {
    'upper_clothes': 'Top',
    'left_shoe': 'Shoes',
    'right_shoe': 'Shoes',
    'pants': 'Pants',
    'dress pants': 'Dress Pants',
    'dress_pants': 'Dress Pants',
    'chinos': 'Chinos',
    'jeans': 'Jeans',
    'skinny jeans': 'Skinny Jeans',
    'joggers': 'Joggers',
    't-shirt': 'T-Shirt',
    'tshirt': 'T-Shirt',
    'sport coat': 'Sport Coat',
    'sport_coat': 'Sport Coat',
    'blazer': 'Blazer',
    'denim jacket': 'Denim Jacket',
    'leather jacket': 'Leather Jacket',
    'cardigan': 'Cardigan',
    'sweater': 'Sweater',
    'hoodie': 'Hoodie',
    'sneakers': 'Sneakers',
    'running shoes': 'Running Shoes',
    'dress shoes': 'Dress Shoes',
    'boots': 'Boots',
    'loafers': 'Loafers',
};

/** Converts a raw category string to a human-readable display name. */
export const formatCategoryName = (category: string): string => {
    if (!category) return 'Clothing';
    const lower = category.toLowerCase();
    if (CATEGORY_DISPLAY_MAP[lower]) return CATEGORY_DISPLAY_MAP[lower];
    return category
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
};

/** Returns the body position for a given clothing category. */
export const getItemPosition = (category: string): string => {
    const cat = (category || '').toLowerCase();
    if (['shirt', 'blouse', 'sweater', 'jacket', 'coat', 'top', 't-shirt', 'hoodie',
        'upper-clothes', 'cardigan', 'polo', 'tank'].some(u => cat.includes(u))) return 'upper';
    if (['pants', 'jeans', 'shorts', 'skirt', 'trousers', 'leggings'].some(l => cat.includes(l))) return 'lower';
    if (['dress', 'jumpsuit', 'romper', 'overalls', 'suit'].some(f => cat.includes(f))) return 'full';
    if (['shoe', 'boot', 'sneaker', 'sandal', 'heel', 'loafer', 'slipper'].some(f => cat.includes(f))) return 'feet';
    if (['bag', 'hat', 'scarf', 'belt', 'watch', 'glasses', 'sunglasses'].some(a => cat.includes(a))) return 'accessory';
    return 'upper';
};

/** Returns a fallback stock image URL for a given item type. */
export const getClothingFallbackImage = (itemType: string): string => {
    const type = itemType.toLowerCase();
    for (const [key, url] of Object.entries(CLOTHING_IMAGES)) {
        if (type.includes(key)) return url;
    }
    return 'https://images.unsplash.com/photo-1489987707025-afc232f7ea0f?w=400&h=500&fit=crop';
};

/** Calculates Intersection over Union for two bounding boxes [x, y, w, h]. */
const iou = (box1: number[] | undefined, box2: number[] | undefined): number => {
    if (!box1 || !box2 || box1.length < 4 || box2.length < 4) return 0;
    const [x1, y1, w1, h1] = box1;
    const [x2, y2, w2, h2] = box2;
    const xi1 = Math.max(x1, x2);
    const yi1 = Math.max(y1, y2);
    const xi2 = Math.min(x1 + w1, x2 + w2);
    const yi2 = Math.min(y1 + h1, y2 + h2);
    if (xi2 <= xi1 || yi2 <= yi1) return 0;
    const inter = (xi2 - xi1) * (yi2 - yi1);
    const union = w1 * h1 + w2 * h2 - inter;
    return union > 0 ? inter / union : 0;
};

/**
 * Removes duplicate items detected in the same frame with overlapping bounding boxes.
 * Items from different frames are preserved (they represent different outfits).
 */
export const deduplicateItems = (items: DetectedItem[]): DetectedItem[] => {
    if (items.length <= 1) return items;
    const sorted = [...items].sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
    const unique: DetectedItem[] = [];
    for (const item of sorted) {
        let isDuplicate = false;
        const itemType = (item.itemType || item.specificType || '').toLowerCase();
        const itemFrame = item.frameIndex || 0;
        for (const existing of unique) {
            const existingType = (existing.itemType || existing.specificType || '').toLowerCase();
            const existingFrame = existing.frameIndex || 0;
            if (itemFrame !== existingFrame) continue;
            const sameCategory =
                (itemType.includes('shirt') && existingType.includes('shirt')) ||
                (itemType.includes('pants') && existingType.includes('pants')) ||
                (itemType.includes('jeans') && existingType.includes('jeans')) ||
                (itemType.includes('shoe') && existingType.includes('shoe')) ||
                (itemType.includes('jacket') && existingType.includes('jacket')) ||
                (itemType.includes('sweater') && existingType.includes('sweater')) ||
                (itemType === existingType);
            if (sameCategory && item.bbox && existing.bbox && iou(item.bbox, existing.bbox) > 0.5) {
                isDuplicate = true;
                break;
            }
        }
        if (!isDuplicate) unique.push(item);
    }
    return unique;
};

/**
 * Merges left/right shoe detections into a single Shoes item,
 * preserving the specificType for display.
 */
export const mergeShoeCategories = (items: DetectedItem[]): DetectedItem[] => {
    const shoeItems: DetectedItem[] = [];
    const otherItems: DetectedItem[] = [];
    items.forEach(item => {
        const cat = (item.itemType || '').toLowerCase();
        if (cat.includes('shoe') || cat.includes('left_shoe') || cat.includes('right_shoe') ||
            cat.includes('sneaker') || cat.includes('boot') || cat.includes('sandal') || cat.includes('loafer')) {
            shoeItems.push(item);
        } else {
            otherItems.push(item);
        }
    });
    if (shoeItems.length > 0) {
        const firstShoe = shoeItems[0];
        let shoeDisplayName = 'Shoes';
        if (firstShoe.specificType) {
            shoeDisplayName = formatCategoryName(firstShoe.specificType);
        } else if (firstShoe.itemType && !firstShoe.itemType.toLowerCase().includes('shoe')) {
            shoeDisplayName = firstShoe.itemType;
        }
        otherItems.push({
            itemType: shoeDisplayName,
            specificType: firstShoe.specificType,
            color: firstShoe.color || 'Unknown',
            style: 'Casual',
            description: `${firstShoe.color || ''} ${shoeDisplayName}`.trim(),
            position: 'feet',
            confidence: firstShoe.confidence,
            bbox: firstShoe.bbox,
            colorHex: firstShoe.colorHex || '#000000',
        });
    }
    return otherItems;
};
