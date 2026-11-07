import { fourSlotOutfitReason } from '../domain/outfitStructure';
import { productIdentityKeys } from '../domain/outfitDiversity';
import { getMacroCategory, canonicalizeMacroCategory } from '../utils/categoryMapper';
/** Read-only normalization at the save boundary; never substitutes garments. */
export function savedQuartetReason(items: unknown): string | null {
  if (!Array.isArray(items)) return 'requires_four_categories';
  const normalized = items.map((item) => {
    const explicit = canonicalizeMacroCategory(item.macroCategory || '');
    const macro =
      explicit !== 'other'
        ? explicit
        : getMacroCategory(item.category || item.type || '', item.type || item.subCategory || '');
    return {
      ...item,
      macroCategory: macro,
      name: item.name || item.subCategory,
      type: item.type || item.subCategory,
    };
  });
  const reason = fourSlotOutfitReason(normalized);
  if (reason) return reason;
  const keys = new Set<string>();
  for (const item of normalized) {
    const own = productIdentityKeys(item);
    if (own.some((key) => keys.has(key))) return 'duplicate_physical_item';
    own.forEach((key) => keys.add(key));
  }
  return null;
}
