import type { ClothingItem } from '../../src/types/domain';

let n = 0;
export function item(over: Partial<ClothingItem> = {}): ClothingItem {
  n++;
  return {
    id: over.id || `item_${n}`,
    userId: 'u1',
    imageUrl: 'img',
    category: 'top',
    subCategory: 't-shirt',
    primaryColor: 'black',
    colorHex: '#000',
    pattern: 'solid',
    material: 'cotton',
    seasons: ['spring', 'summer', 'fall', 'winter'],
    occasions: ['casual'],
    wearCount: 0,
    lastWornAt: null,
    isFavorite: false,
    createdAt: new Date(Date.now() - 90 * 86_400_000).toISOString(),
    updatedAt: new Date().toISOString(),
    ...over,
  };
}

export const NOW = Date.parse('2026-11-03T00:00:00Z');
export const daysAgo = (d: number) => new Date(NOW - d * 86_400_000).toISOString();
