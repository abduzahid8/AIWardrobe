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

// Raw garmentType / macroCategory strings we must never surface to users as
// labels. These are backend tags (e.g. "upper_body", "tops", "lower_body")
// that sometimes end up in an item's `name` or `type` fields when the LLM
// couldn't find a friendlier value — without this check they'd leak into
// the collage label text.
const RAW_GARMENT_TYPE_RE = /^(upper|lower)[_\s-]?body$|^dresses?$|^tops?$|^bottoms?$|^shoes?$|^outerwear$|^footwear$|^pants?$|^shirts?$|^item$|^clothing$/i;

function isRawGarmentTag(value: string | null | undefined): boolean {
  if (typeof value !== 'string') return false;
  return RAW_GARMENT_TYPE_RE.test(value.trim());
}

export function getOutfitItemMacroCategory(item: OutfitItem): PreviewMacroCategory {
  // If the item already has a canonical macroCategory, trust it directly.
  // This is important for shop catalog items that have been explicitly tagged
  // by the caller, so we don't let name-regex override explicit slot assignments.
  const explicit = String(item.macroCategory || '').toLowerCase().trim();
  if (explicit === 'top' || explicit === 'outerwear' || explicit === 'bottom' || explicit === 'shoes' || explicit === 'accessory') {
    return explicit as PreviewMacroCategory;
  }

  // Prefer a blob-based match first — item names like "Striped Sweater Polo"
  // or types like "jacket" are more reliable than the `macroCategory` field
  // which the edge function sometimes collapses to `top` even for outerwear.
  const blob = `${item.type || ''} ${item.name || ''}`.toLowerCase();
  if (/\b(blazer|overcoat|topcoat|peacoat|trench|parka|puffer|windbreaker|bomber)\b/.test(blob)) return 'outerwear';
  if (/\b(coat|jacket|cardigan|sweater|hoodie|vest|pullover|fleece)\b/.test(blob)) return 'outerwear';
  if (/\b(pant|trouser|jeans|short|skirt|chino|slack|jogger|sweatpant)\b/.test(blob)) return 'bottom';
  // Check tops BEFORE shoes — "Oxford Shirt" must not match the shoe regex
  if (/\b(t-shirt|tshirt|tee|polo|blouse|shirt|dress)\b/.test(blob)) return 'top';
  if (/\b(shoe|sneaker|boot|loafer|sandal|heel|trainer|derby|mule)\b/.test(blob)) return 'shoes';
  // "oxford" alone is ambiguous (Oxford shirt vs Oxford shoes) — only match
  // as shoes when paired with a shoe qualifier like "shoe" or "flat"
  if (/\boxford\s*(shoe|flat|lace|brogue|derby)\b/i.test(blob)) return 'shoes';
  if (/upper[_\s-]?body/.test(blob)) return 'top';
  if (/lower[_\s-]?body/.test(blob)) return 'bottom';

  // Fall back to the LEGACY_CATEGORY_MAP on macroCategory.
  const normalizedRawCategory = LEGACY_CATEGORY_MAP[String(item.macroCategory || '').toLowerCase().trim()];
  if (normalizedRawCategory) {
    return normalizedRawCategory;
  }

  // Final fallback: let the shared categoryMapper try.
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
  // Prefer a human-friendly name / type, but never surface raw garmentType
  // tags ("upper_body", "tops", "shoes", etc.). If both name and type are
  // raw machine tags, we fall back to the canonical slot label ("Top",
  // "Bottom", "Shoes") so the collage never shows tokenised backend values
  // to the user.
  const name = typeof item.name === 'string' ? item.name.trim() : '';
  if (name && !isRawGarmentTag(name)) return name;
  const type = typeof item.type === 'string' ? item.type.trim() : '';
  if (type && !isRawGarmentTag(type)) return type;
  return PREVIEW_LABELS[getOutfitItemMacroCategory(item)] || PREVIEW_LABELS.other;
}
