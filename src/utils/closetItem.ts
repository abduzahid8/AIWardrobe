import type { ClothingItem } from '../types/domain';
import type { ClosetClothingItem } from '../../features/outfit-generator/types';
import { normalizeCategory } from './categoryMapper';

/** Preserve original and compact media references when adapting persisted garments for the closet. */
export function toClosetItem(item: ClothingItem): ClosetClothingItem {
  const category = normalizeCategory(item.subCategory || item.category || '');
  return {
    _id: item.id, id: item.id, type: category, itemType: category,
    color: item.primaryColor || 'various',
    imageUrl: item.imageUrl, image: item.imageUrl, thumbnailUrl: item.thumbnailUrl,
    category: normalizeCategory(item.category || ''), wearCount: item.wearCount,
    createdAt: item.createdAt, isFavorite: item.isFavorite ?? false,
  };
}

/** Compact cards use a supplied thumbnail; original references remain available for detail views. */
export function compactClothingImage(item: Pick<ClosetClothingItem, 'thumbnailUrl' | 'imageUrl' | 'image'>) {
  return item.thumbnailUrl || item.imageUrl || item.image;
}
