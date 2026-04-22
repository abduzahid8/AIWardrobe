/**
 * Canonical category mapper — single source of truth for all category normalization.
 *
 * Used by: wardrobeStore, outfitGenerationService, useItemSelection,
 *          useWardrobeLoader, OutfitCollageDisplay, OutfitResultsView.
 */

import type { ClothingCategory } from '../types/domain';

export type MacroCategory = 'top' | 'bottom' | 'outerwear' | 'shoes' | 'accessory' | 'other';

/**
 * Canonicalize any macro-category-ish string to one of the six canonical values.
 * Handles aliases produced upstream (AI, shop_catalog.garment_type, legacy DB):
 *   - upper_body / upper-body / tops / dress / dresses → top
 *   - lower_body / lower-body / bottoms / pants        → bottom
 *   - shoe / footwear                                  → shoes
 *   - accessories                                      → accessory
 * Unknown values pass through as 'other'.
 */
const MACRO_ALIAS_MAP: Record<string, MacroCategory> = {
  top: 'top', tops: 'top', upper_body: 'top', 'upper-body': 'top', upperbody: 'top',
  dress: 'top', dresses: 'top',
  bottom: 'bottom', bottoms: 'bottom', lower_body: 'bottom', 'lower-body': 'bottom', lowerbody: 'bottom',
  pant: 'bottom', pants: 'bottom',
  outerwear: 'outerwear', outer: 'outerwear',
  shoe: 'shoes', shoes: 'shoes', footwear: 'shoes',
  accessory: 'accessory', accessories: 'accessory',
  other: 'other',
};

export function canonicalizeMacroCategory(raw: string | null | undefined): MacroCategory {
  if (!raw) return 'other';
  const key = String(raw).toLowerCase().trim();
  return MACRO_ALIAS_MAP[key] ?? 'other';
}

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
  // Shop-catalog garmentType strings come first so "upper_body" doesn't fall
  // into the TOP_RE branch via the generic "top" keyword (it would anyway) —
  // but "lower_body" has no "bottom" keyword so we need an explicit mapping.
  if (/\bupper[_\s-]?body\b/.test(t)) return 'top';
  if (/\blower[_\s-]?body\b/.test(t)) return 'bottom';
  if (OUTERWEAR_RE.test(t)) return 'outerwear';
  if (TOP_RE.test(t)) return 'top';
  if (BOTTOM_RE.test(t)) return 'bottom';
  if (SHOES_RE.test(t)) return 'shoes';
  if (ACCESSORY_RE.test(t)) return 'accessory';
  return 'other';
}

// ── Outfit composition helpers ────────────────────────────────────────────
// Formal layers = blazer, suit jacket, overcoat, topcoat, trench, peacoat,
// sport coat. Casual layers (denim jacket, bomber, cardigan, hoodie, puffer,
// windbreaker, fleece) are intentionally NOT matched so they can still pair
// with shorts. Keep this predicate SEMANTIC — it reads across name / type /
// subCategory / macroCategory.

const FORMAL_LAYER_RE =
  /\b(blazer|suit\s*jacket|sport\s*coat|sports\s*coat|overcoat|top\s*coat|topcoat|trench|peacoat|pea\s*coat|tuxedo)\b/;

const SHORTS_RE = /\b(shorts?|bermudas?|chino\s*shorts?)\b/;

function itemTextBlob(item: Record<string, unknown> | null | undefined): string {
  if (!item) return '';
  const parts = [
    (item as any).name,
    (item as any).type,
    (item as any).subCategory,
    (item as any).sub_category,
    (item as any).category,
    (item as any).macroCategory,
    (item as any).description,
  ];
  return parts.filter(Boolean).map((p) => String(p).toLowerCase()).join(' ');
}

/**
 * True if the item is a formal outerwear piece that must not pair with shorts.
 * Casual layers (denim jacket, cardigan, hoodie, bomber, puffer, windbreaker,
 * fleece) return false so they can still pair with shorts in cool weather.
 */
export function isFormalLayer(item: Record<string, unknown> | null | undefined): boolean {
  const blob = itemTextBlob(item);
  if (!blob) return false;
  // Must be an outerwear-ish item in the first place.
  const macro = String((item as any)?.macroCategory || '').toLowerCase();
  const isOuter = macro === 'outerwear' || OUTERWEAR_RE.test(blob);
  if (!isOuter) return false;
  return FORMAL_LAYER_RE.test(blob);
}

/**
 * True if the item is a shorts-style bottom (excludes skirts and pants).
 */
export function isShortsBottom(item: Record<string, unknown> | null | undefined): boolean {
  const blob = itemTextBlob(item);
  if (!blob) return false;
  const macro = String((item as any)?.macroCategory || '').toLowerCase();
  const isBottom = macro === 'bottom' || /\blower[_\s-]?body\b/.test(blob) || BOTTOM_RE.test(blob);
  if (!isBottom) return false;
  return SHORTS_RE.test(blob);
}

/**
 * True if the weather/conditions indicate an outerwear layer should be added.
 * When weather is missing we err on the side of NO layer (3-item warm default)
 * so Home never surprises a user with a jacket on a hot day; AIOutfitmaker
 * provides its own default if it wants the safer "cold" behaviour.
 */
export function needsLayerForWeather(weather?: { temp?: number | null; condition?: string | null } | null): boolean {
  if (!weather) return false;
  const temp = typeof weather.temp === 'number' ? weather.temp : null;
  const condition = String(weather.condition || '').toLowerCase();
  if (temp != null && temp < 18) return true;
  if (/\b(cold|chilly|freezing|snow|rain|drizzle|wind|storm)\b/.test(condition)) return true;
  return false;
}
