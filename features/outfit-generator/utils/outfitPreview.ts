import { getMacroCategory } from '../../../src/utils/categoryMapper';
import type { OutfitItem } from '../types';

type PreviewMacroCategory = 'top' | 'outerwear' | 'bottom' | 'shoes' | 'accessory' | 'other';

export interface OutfitPreviewSlot {
  item: OutfitItem;
  macroCategory: PreviewMacroCategory;
  label: string;
}

const PREVIEW_ORDER: PreviewMacroCategory[] = ['top', 'outerwear', 'bottom', 'shoes', 'accessory', 'other'];

const PREVIEW_LABELS: Record<PreviewMacroCategory, string> = {
  top: 'Top',
  outerwear: 'Outerwear',
  bottom: 'Bottom',
  shoes: 'Shoes',
  accessory: 'Accessory',
  other: 'Item',
};

const LEGACY_CATEGORY_MAP: Record<string, PreviewMacroCategory> = {
  top: 'top',
  tops: 'top',
  outerwear: 'outerwear',
  bottom: 'bottom',
  bottoms: 'bottom',
  pant: 'bottom',
  pants: 'bottom',
  shoe: 'shoes',
  shoes: 'shoes',
  accessory: 'accessory',
  accessories: 'accessory',
  other: 'other',
};

export function getOutfitItemMacroCategory(item: OutfitItem): PreviewMacroCategory {
  const normalizedRawCategory = LEGACY_CATEGORY_MAP[String(item.macroCategory || '').toLowerCase().trim()];
  if (normalizedRawCategory) {
    return normalizedRawCategory;
  }

  const inferredCategory = getMacroCategory(
    item.macroCategory || item.type || item.name || '',
    item.name || item.type
  ) as PreviewMacroCategory;

  if (PREVIEW_ORDER.includes(inferredCategory)) {
    return inferredCategory;
  }

  return 'other';
}

export function getOutfitPreviewSlots(items: OutfitItem[]): OutfitPreviewSlot[] {
  return items
    .map((item, index) => ({
      item,
      index,
      macroCategory: getOutfitItemMacroCategory(item),
    }))
    .map(({ item, macroCategory }) => ({
      item,
      macroCategory,
      label: PREVIEW_LABELS[macroCategory] || item.type || item.name || PREVIEW_LABELS.other,
    }));
}

export function getOutfitPreviewTitle(item: OutfitItem): string {
  return item.name || item.type || PREVIEW_LABELS[getOutfitItemMacroCategory(item)] || PREVIEW_LABELS.other;
}
