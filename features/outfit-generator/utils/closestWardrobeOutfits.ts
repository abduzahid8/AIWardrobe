import { validatedQuartets } from '../../../src/domain/validatedQuartets';
import { stylingContext } from '../../../src/domain/stylingContext';
import { isNonOutfitItem, fourSlotOutfitReason } from '../../../backend/supabase/functions/_shared/outfitStructure';
import { assessOutfitQuality, rejectForStyleOccasion, rejectForClimate, detectSlotMismatch, passesStyleJudge } from './occasionRules';
import { buildOutfitFingerprint } from '../../../src/domain/outfitDiversity';
import { recentLookSimilarity, recentStylingPenalty } from '../../../src/domain/outfitVisualSignature';
import type { StylingPreferences } from '../../../src/domain/stylingContext';
import { canonicalizeMacroCategory, getMacroCategory } from '../../../src/utils/categoryMapper';
import {displayedRotation, rotationPool, type DisplayedRotationHistory} from '../../../src/domain/outfitRotation';
import {assembleDiverseDeck, productIdentityKeys, diversitySlotOf} from '../../../src/domain/outfitDiversity';

export interface AlternativeItem {
  id: string; name?: string; type?: string; category?: string; macroCategory?: string;
  image?: string; imageUrl?: string; isWardrobe?: boolean; isShopItem?: boolean;
  color?: string; brand?: string; price?: number; currency?: string; shopUrl?: string;
  description?: string; material?: string | string[]; pattern?: string;
}

/** Rank imperfect matches while retaining source, anchor, budget and real garment slots. */
export function closestAvailableOutfits(
  input: AlternativeItem[],
  options: { anchorId?: string; style: string; occasion: string; weather?: { temp?: number | null; condition?: string | null }; limit?: number; source?: 'wardrobe' | 'shop'; accepts?: (items: AlternativeItem[]) => boolean; recentLookSignatures?: string[]; preferences?: StylingPreferences; history?: DisplayedRotationHistory; reserve?: boolean },
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
  const itemPenalty = (item: typeof pool[number]) =>
    (rejectForStyleOccasion(item, options.style, options.occasion) ? 2 : 0)
    + (rejectForClimate(item, options.weather) ? 1 : 0);
  // Shortfalls must not be filled with garments the request cannot wear.
  const eligiblePool = pool.filter(item => !rejectForStyleOccasion(item, item.id === options.anchorId ? 'casual' : options.style, options.occasion)
    && !rejectForClimate(item, options.weather));
  // Put the best candidates in each slot first before the bounded search.
  eligiblePool.sort((a, b) => itemPenalty(a) - itemPenalty(b));
  const rotation = displayedRotation(options.history, options.anchorId);
  const rotatedPool = rotationPool(eligiblePool, rotation.recent, 3, new Set(required)).items;
  const choices = validatedQuartets(rotatedPool, stylingContext({style: options.style, occasion: options.occasion,
    weather: options.weather, anchorId: options.anchorId, preferences: options.preferences}),
    parts => !fourSlotOutfitReason(parts)
      && passesStyleJudge(parts, options.style, options.occasion, options.weather, options.anchorId, options.preferences)
      && (!options.accepts || options.accepts(parts)),
    {requiredIds: required, maximum: 64, searchLimit: 6000});
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
  // Recovery supplies a reserve to the screen's deck search. A displayed
  // fallback uses that same physical-product deck gate, never a loose MMR list.
  if (options.reserve) return ranked.slice(0, limit);
  const anchor = rotatedPool.find(item => item.id === options.anchorId);
  return assembleDiverseDeck({candidates: ranked, itemsOf: look => look.items,
    qualityOf: look => look.quality, recencyOf: look => look.recency,
    priorityOfDeck: deck => rotation.priority(deck.flatMap(look => look.items)),
    target: Math.min(3, limit), usedKeys: new Set(), poolItems: rotatedPool,
    anchorKeys: anchor ? new Set(productIdentityKeys(anchor)) : undefined,
    anchorSlot: anchor ? diversitySlotOf(anchor) : undefined,
  }).deck;

}

/** Backwards-compatible wardrobe-only entry point. */
export function closestWardrobeOutfits(input: AlternativeItem[], options: Omit<Parameters<typeof closestAvailableOutfits>[1], 'source'>) {
  return closestAvailableOutfits(input, { ...options, source: 'wardrobe' });
}
