/**
 * Canonical category mapper — single source of truth for all category normalization.
 *
 * Used by: wardrobeStore, outfitGenerationService, useItemSelection,
 *          useWardrobeLoader, OutfitCollageDisplay, OutfitResultsView.
 */

import type { ClothingCategory } from '../types/domain';

export type MacroCategory = 'top' | 'bottom' | 'outerwear' | 'shoes' | 'accessory' | 'other';

const CATEGORY_MAP: Record<string, ClothingCategory> = {
  top: 'top', tops: 'top',
  bottom: 'bottom', bottoms: 'bottom',
  dress: 'dress', dresses: 'dress',
  shoe: 'shoes', shoes: 'shoes',
  outerwear: 'outerwear',
  accessory: 'accessory', accessories: 'accessory',
  other: 'other',
};

/**
 * Normalize any category string (PascalCase, lowercase, plural)
 * to the canonical ClothingCategory enum value that matches the DB CHECK constraint.
 */
export function normalizeCategory(raw: string): ClothingCategory {
  return CATEGORY_MAP[raw.toLowerCase().trim()] ?? 'other';
}

/**
 * Map a DB row category back to the canonical domain type.
 * Handles both legacy PascalCase ("Tops") and current lowercase ("top").
 */
export function mapDbCategory(category: string): ClothingCategory {
  return CATEGORY_MAP[category.toLowerCase().trim()] ?? 'other';
}

const OUTERWEAR_RE = /jacket|coat|blazer|hoodie|cardigan|sweater|pullover|vest|puffer|zip|outerwear/;
const TOP_RE = /shirt|t-shirt|tee|blouse|polo|top|dress/;
const BOTTOM_RE = /pant|trouser|jeans|jean|bottom|shorts?|skirt/;
const SHOES_RE = /shoe|sneaker|boot|loafer|sandal/;
const ACCESSORY_RE = /hat|scarf|belt|watch|bag|glasses|sunglasses|necklace|bracelet|earring|ring|tie|gloves/;

/**
 * Infer a macro category from a combination of category + sub-type strings.
 * Useful when the DB `category` column is too broad and the `type` / `sub_category`
 * column provides more specificity (e.g. category="top" but type="blazer" → outerwear).
 */
export function getMacroCategory(categoryOrType: string, subType?: string): MacroCategory {
  const t = `${categoryOrType} ${subType ?? ''}`.toLowerCase();
  if (OUTERWEAR_RE.test(t)) return 'outerwear';
  if (TOP_RE.test(t)) return 'top';
  if (BOTTOM_RE.test(t)) return 'bottom';
  if (SHOES_RE.test(t)) return 'shoes';
  if (ACCESSORY_RE.test(t)) return 'accessory';
  return 'other';
}
