import { outfitChoices } from '../../../backend/supabase/functions/_shared/outfitChoices';
import { isDressGarment, isNonOutfitItem, fourSlotOutfitReason } from '../../../backend/supabase/functions/_shared/outfitStructure';
import { assessOutfitQuality, rejectForStyleOccasion, rejectForClimate, detectSlotMismatch } from './occasionRules';
import { recentStylingPenalty } from '../../../src/domain/outfitVisualSignature';
import { canonicalizeMacroCategory, getMacroCategory } from '../../../src/utils/categoryMapper';

export interface AlternativeItem {
  id: string; name?: string; type?: string; category?: string; macroCategory?: string;
  image?: string; imageUrl?: string; isWardrobe?: boolean; isShopItem?: boolean;
  color?: string; brand?: string; price?: number; currency?: string; shopUrl?: string;
}

/** Rank imperfect matches while retaining source, anchor, budget and real garment slots. */
export function closestAvailableOutfits(
  input: AlternativeItem[],
  options: { anchorId?: string; style: string; occasion: string; weather?: { temp?: number | null; condition?: string | null }; limit?: number; source?: 'wardrobe' | 'shop'; accepts?: (items: AlternativeItem[]) => boolean; recentLookSignatures?: string[] },
) {
  const seen = new Set<string>();
  const pool = input.filter(item => {
    const allowedSource = options.source === 'shop'
      ? (item.isShopItem === true && item.isWardrobe !== true) || (item.id === options.anchorId && item.isWardrobe === true)
      : item.isWardrobe !== false && item.isShopItem !== true;
    // Catalog rows explicitly marked unclassified must not be resurrected
    // from a product title when their structured category conflicts.
    if (options.source === 'shop' && item.isShopItem === true && canonicalizeMacroCategory(item.macroCategory || item.category) === 'other') return false;
    if (!item.id || !allowedSource || seen.has(item.id) || isNonOutfitItem(item)) return false;
    seen.add(item.id);
    return true;
  }).map(item => {
    const macro = canonicalizeMacroCategory(item.macroCategory || item.category);
    return { ...item, name: item.name || item.type || item.category || 'Clothing item', macroCategory: detectSlotMismatch(macro, item.name || item.type || '', item.type || '')
      || (macro !== 'other' ? macro : getMacroCategory(item.category || '', item.type || item.name)),
      image: item.imageUrl || item.image || '',
      isWardrobe: options.source === 'shop' ? item.isWardrobe === true : true,
      isShopItem: options.source === 'shop' && item.isShopItem === true && item.isWardrobe !== true };
  });
  const required = options.anchorId ? [options.anchorId] : [];
  const layered = true;
  const itemPenalty = (item: typeof pool[number]) =>
    (rejectForStyleOccasion(item, options.style, options.occasion) ? 2 : 0)
    + (rejectForClimate(item, options.weather) ? 1 : 0);
  // Put the best candidates in each slot first before the bounded search.
  pool.sort((a, b) => itemPenalty(a) - itemPenalty(b));
  const choices = outfitChoices(pool, layered, parts => !fourSlotOutfitReason(parts)
    && (!options.accepts || options.accepts(parts)), isDressGarment, required, 128, 12000, 1);
  const penalty = (parts: typeof pool) => parts.reduce((sum, item) => sum
    + (rejectForStyleOccasion(item, options.style, options.occasion) ? 2 : 0)
    + (rejectForClimate(item, options.weather) ? 1 : 0), 0);
  // Recently shown semantic looks rank lower so the fallback surfaces a
  // fresher alternative first. Hard validity (penalty) still dominates.
  const recent = options.recentLookSignatures ?? [];
  return choices.map(choice => ({ ...choice,
    penalty: penalty(choice.items),
    quality: Math.max(0, assessOutfitQuality(choice.items, options.style, options.occasion, options.weather, options.anchorId).score - recentStylingPenalty(choice.items, recent)),
  })).sort((a, b) => a.penalty - b.penalty || b.quality - a.quality).slice(0, options.limit ?? 3);
}

/** Backwards-compatible wardrobe-only entry point. */
export function closestWardrobeOutfits(input: AlternativeItem[], options: Omit<Parameters<typeof closestAvailableOutfits>[1], 'source'>) {
  return closestAvailableOutfits(input, { ...options, source: 'wardrobe' });
}
