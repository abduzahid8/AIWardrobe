import { outfitChoices } from '../../../backend/supabase/functions/_shared/outfitChoices';
import { isDressGarment, isNonOutfitItem, fourSlotOutfitReason } from '../../../backend/supabase/functions/_shared/outfitStructure';
import { assessOutfitQuality, rejectForStyleOccasion, rejectForClimate, detectSlotMismatch } from './occasionRules';
import { buildOutfitFingerprint, visualSimilarity } from '../../../src/domain/outfitDiversity';
import { recentLookSimilarity, recentStylingPenalty } from '../../../src/domain/outfitVisualSignature';
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
  const ranked = choices.map(choice => {
    const items = choice.items as AlternativeItem[];
    return { ...choice,
      penalty: penalty(choice.items),
      quality: Math.max(0, assessOutfitQuality(items, options.style, options.occasion, options.weather, options.anchorId).score - recentStylingPenalty(items, recent)),
      fingerprint: buildOutfitFingerprint(items),
      recency: recentLookSimilarity(items, recent),
    };
  }).sort((a, b) => a.penalty - b.penalty || b.quality - a.quality);
  const limit = options.limit ?? 3;
  // Greedy MMR over the validity-ranked list: the best look is always
  // first; each next pick maximizes quality minus resemblance to already
  // picked looks and recently shown looks. Looks more than one quality
  // band below the best only fill the deck when nothing closer exists,
  // so a weak-but-novel look can never outrank a strong one.
  const MMR_BAND = 0.12;
  const MMR_LAMBDA = 0.5;
  const picked: typeof ranked = [];
  const remaining = [...ranked];
  const bestQuality = remaining.length ? remaining[0].quality : 0;
  while (picked.length < limit && remaining.length > 0) {
    let eligible = remaining.filter((c) => c.quality >= bestQuality - MMR_BAND);
    if (eligible.length === 0) eligible = remaining;
    let bestIdx = 0;
    let bestVal = -Infinity;
    for (const candidate of eligible) {
      let maxSim = candidate.recency;
      for (const kept of picked) {
        maxSim = Math.max(maxSim, visualSimilarity(candidate.fingerprint, kept.fingerprint));
      }
      const value = candidate.quality - MMR_LAMBDA * maxSim;
      if (value > bestVal) {
        bestVal = value;
        bestIdx = remaining.indexOf(candidate);
      }
    }
    picked.push(...remaining.splice(bestIdx, 1));
  }
  return picked;
}

/** Backwards-compatible wardrobe-only entry point. */
export function closestWardrobeOutfits(input: AlternativeItem[], options: Omit<Parameters<typeof closestAvailableOutfits>[1], 'source'>) {
  return closestAvailableOutfits(input, { ...options, source: 'wardrobe' });
}
