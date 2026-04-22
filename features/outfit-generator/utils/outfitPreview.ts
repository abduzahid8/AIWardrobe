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
  upper_body: 'top',
  'upper-body': 'top',
  outerwear: 'outerwear',
  bottom: 'bottom',
  bottoms: 'bottom',
  lower_body: 'bottom',
  'lower-body': 'bottom',
  pant: 'bottom',
  pants: 'bottom',
  shoe: 'shoes',
  shoes: 'shoes',
  dress: 'top',
  dresses: 'top',
  accessory: 'accessory',
  accessories: 'accessory',
  other: 'other',
};

// Raw garmentType strings we must never surface to users as labels.
const RAW_GARMENT_TYPE_RE = /^(upper|lower)[_\s-]?body$|^dresses?$/i;

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
  // Prefer a human-friendly name. If the only available fallback is a raw
  // garmentType string (e.g. "upper_body" / "lower_body" / "dresses"), surface
  // the macroCategory label ("Top", "Bottom", etc.) instead so the collage
  // never shows machine tags to the user.
  const name = typeof item.name === 'string' ? item.name.trim() : '';
  if (name) return name;
  const type = typeof item.type === 'string' ? item.type.trim() : '';
  if (type && !RAW_GARMENT_TYPE_RE.test(type)) return type;
  return PREVIEW_LABELS[getOutfitItemMacroCategory(item)] || PREVIEW_LABELS.other;
}
