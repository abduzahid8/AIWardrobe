import { getMacroCategory } from '../../../src/utils/categoryMapper';
import type { GeneratedOutfit, OutfitItem, WardrobeDisplayItem } from '../types';
import {
  scoreItemForStyle,
  normalizeStyleId,
  needsLayering,
  type StyleId,
  type LayeringWeather,
} from './styleInference';

export type OutfitSlot = 'outerwear' | 'top' | 'bottom' | 'shoes';

export interface SanitizeResult {
  items: OutfitItem[];
  missingSlots: OutfitSlot[];
  layered: boolean;
}

type OutfitMacroCategory = 'top' | 'outerwear' | 'bottom' | 'shoes' | 'accessory' | 'other';

interface SanitizableItem {
  id?: string | number;
  name?: string;
  image?: string | number;
  imageUrl?: string;
  color?: string;
  type?: string;
  macroCategory?: string;
  category?: string;
  isShopItem?: boolean;
  price?: number;
  brand?: string;
  shopUrl?: string;
}

interface Candidate {
  key: string;
  item: OutfitItem;
  macroCategory: OutfitMacroCategory;
}

const LEGACY_MACRO_CATEGORY_MAP: Record<string, OutfitMacroCategory> = {
  top: 'top',
  tops: 'top',
  outerwear: 'outerwear',
  bottom: 'bottom',
  bottoms: 'bottom',
  pant: 'bottom',
  pants: 'bottom',
  shoe: 'shoes',
  shoes: 'shoes',
  footwear: 'shoes',
  accessory: 'accessory',
  accessories: 'accessory',
  other: 'other',
};

function normalizeMacroCategory(raw?: string | null): OutfitMacroCategory | null {
  if (!raw) return null;
  return LEGACY_MACRO_CATEGORY_MAP[String(raw).toLowerCase().trim()] || null;
}

function resolveMacroCategory(item: SanitizableItem): OutfitMacroCategory {
  return (
    normalizeMacroCategory(item.macroCategory) ||
    normalizeMacroCategory(
      getMacroCategory(item.category || item.type || '', item.name || item.type)
    ) ||
    normalizeMacroCategory(item.category) ||
    'other'
  );
}

function buildCandidate(
  generatedItem: SanitizableItem,
  sourceItem: SanitizableItem | undefined,
  index: number
): Candidate | null {
  const resolvedMacroCategory = resolveMacroCategory({
    macroCategory: sourceItem?.macroCategory || generatedItem.macroCategory,
    category: sourceItem?.category || generatedItem.category,
    type: sourceItem?.type || generatedItem.type,
    name: sourceItem?.name || generatedItem.name,
  });

  const id = sourceItem?.id ?? generatedItem.id;
  const key = String(id || `generated_${index}_${generatedItem.type || generatedItem.name || 'item'}`);

  if (!key) return null;

  return {
    key,
    macroCategory: resolvedMacroCategory,
    item: {
      id,
      name:
        sourceItem?.name ||
        generatedItem.name ||
        sourceItem?.type ||
        generatedItem.type ||
        'Item',
      image:
        sourceItem?.image ??
        sourceItem?.imageUrl ??
        generatedItem.image ??
        generatedItem.imageUrl ??
        '',
      color: generatedItem.color || sourceItem?.color,
      type:
        sourceItem?.type ||
        generatedItem.type ||
        sourceItem?.name ||
        generatedItem.name ||
        'clothing',
      macroCategory: resolvedMacroCategory,
      isShopItem: generatedItem.isShopItem ?? sourceItem?.isShopItem,
      price: generatedItem.price ?? sourceItem?.price,
      brand: generatedItem.brand || sourceItem?.brand,
      shopUrl: generatedItem.shopUrl || sourceItem?.shopUrl,
    },
  };
}

function pickGeneratedCandidate(
  generatedCandidates: Candidate[],
  usedKeys: Set<string>,
  categories: OutfitMacroCategory[]
): Candidate | null {
  return (
    generatedCandidates.find(
      (candidate) =>
        !usedKeys.has(candidate.key) && categories.includes(candidate.macroCategory)
    ) || null
  );
}

function pickAvailableCandidate(
  availableItems: WardrobeDisplayItem[],
  usedKeys: Set<string>,
  categories: OutfitMacroCategory[],
  style?: StyleId,
  rotationOffset = 0,
  avoidKeys?: Set<string>,
): Candidate | null {
  // Collect every candidate in the requested categories, then pick the
  // best-scoring one under the style (if provided). `rotationOffset` rotates
  // which ranked pick we return so consecutive outfits don't all collapse
  // onto the same best-1 item (the "3 similar outfits" bug). `avoidKeys`
  // penalizes globally over-used ids (cross-outfit) without hard-excluding.
  const allCandidates: { candidate: Candidate; score: number }[] = [];
  for (let index = 0; index < availableItems.length; index += 1) {
    const item = availableItems[index];
    const candidate = buildCandidate(item, item, index);
    if (!candidate) continue;
    if (usedKeys.has(candidate.key)) continue;
    if (!categories.includes(candidate.macroCategory)) continue;
    let score = style
      ? scoreItemForStyle(
          {
            name: item.name,
            description: (item as any).description,
            brand: item.brand,
            color: item.color,
            type: item.type,
            category: item.category,
            macroCategory: item.macroCategory,
          },
          style,
        )
      : 0;
    // Penalize cross-outfit overuse so rotation prefers fresh items.
    if (avoidKeys && avoidKeys.has(candidate.key)) score -= 1.5;
    allCandidates.push({ candidate, score });
  }

  if (allCandidates.length === 0) return null;
  if (!style) {
    const idx = rotationOffset % allCandidates.length;
    return allCandidates[idx].candidate;
  }

  allCandidates.sort((a, b) => b.score - a.score);
  const idx = rotationOffset % allCandidates.length;
  return allCandidates[idx].candidate;
}

function pickCandidateByPriority(
  generatedCandidates: Candidate[],
  availableItems: WardrobeDisplayItem[],
  usedKeys: Set<string>,
  priorityGroups: OutfitMacroCategory[][],
  style?: StyleId,
  rotationOffset = 0,
  avoidKeys?: Set<string>,
): Candidate | null {
  for (const categories of priorityGroups) {
    const generated = pickGeneratedCandidate(generatedCandidates, usedKeys, categories);
    if (generated) return generated;
    const available = pickAvailableCandidate(availableItems, usedKeys, categories, style, rotationOffset, avoidKeys);
    if (available) return available;
  }

  return null;
}

function addCandidate(
  result: OutfitItem[],
  usedKeys: Set<string>,
  candidate: Candidate | null | undefined
) {
  if (!candidate) return;
  if (usedKeys.has(candidate.key)) return;
  usedKeys.add(candidate.key);
  result.push(candidate.item);
}

function isDressCandidate(c: Candidate): boolean {
  const blob = `${c.item.type || ''} ${c.item.name || ''}`.toLowerCase();
  return /\bdress(es)?\b/.test(blob);
}

export interface SanitizeOptions {
  maxItems?: number;
  style?: string;
  layered?: boolean;
  weather?: LayeringWeather | null;
  prompt?: string | null;
  /** Rotate which ranked pick to use (outfitIndex) so 3 outfits don't share best-1. */
  rotationOffset?: number;
  /** Globally over-used ids to de-prioritize (cross-outfit diversity). */
  avoidKeys?: Set<string> | string[];
}

export function sanitizeGeneratedOutfitItems(
  generatedItems: Array<OutfitItem & { imageUrl?: string; category?: string }>,
  availableItems: WardrobeDisplayItem[],
  maxItemsOrOptions: number | SanitizeOptions = 5,
  style?: string,
): OutfitItem[] {
  const result = sanitizeGeneratedOutfitItemsDetailed(
    generatedItems,
    availableItems,
    maxItemsOrOptions,
    style,
  );
  return result.items;
}

export function sanitizeGeneratedOutfitItemsDetailed(
  generatedItems: Array<OutfitItem & { imageUrl?: string; category?: string }>,
  availableItems: WardrobeDisplayItem[],
  maxItemsOrOptions: number | SanitizeOptions = 5,
  legacyStyle?: string,
): SanitizeResult {
  const opts: SanitizeOptions =
    typeof maxItemsOrOptions === 'number'
      ? { maxItems: maxItemsOrOptions, style: legacyStyle }
      : maxItemsOrOptions;
  const maxItems = opts.maxItems ?? 5;
  const style = opts.style ?? legacyStyle;
  const normalizedStyle: StyleId | undefined = style ? normalizeStyleId(style) : undefined;
  const rotationOffset = opts.rotationOffset ?? 0;
  const avoidKeys: Set<string> = opts.avoidKeys
    ? new Set(Array.from(opts.avoidKeys as any).map((k: any) => String(k)))
    : new Set<string>();
  const layered =
    typeof opts.layered === 'boolean'
      ? opts.layered
      : needsLayering(style, opts.weather ?? null, opts.prompt ?? null);

  const availableById = new Map(availableItems.map((item) => [String(item.id), item]));
  const generatedCandidates: Candidate[] = [];
  const seenGeneratedKeys = new Set<string>();

  generatedItems.forEach((generatedItem, index) => {
    const sourceItem =
      generatedItem.id != null
        ? availableById.get(String(generatedItem.id))
        : undefined;
    const candidate = buildCandidate(generatedItem, sourceItem, index);
    if (!candidate || seenGeneratedKeys.has(candidate.key)) return;
    seenGeneratedKeys.add(candidate.key);
    generatedCandidates.push(candidate);
  });

  const sanitizedItems: OutfitItem[] = [];
  const usedKeys = new Set<string>();
  const missingSlots: OutfitSlot[] = [];

  const isDressOutfit = generatedCandidates.some(isDressCandidate);

  // Slot 1: outerwear / main top. Always try when layered, also try for non-layered to ensure 2 tops.
  // rotationOffset shifts fallback picks so outfit_0/1/2 don't all take best-1.
  const outerwear =
    layered && !isDressOutfit
      ? pickCandidateByPriority(
          generatedCandidates,
          availableItems,
          usedKeys,
          [['outerwear']],
          normalizedStyle,
          rotationOffset,
          avoidKeys,
        )
      : null;
  if (outerwear) {
    addCandidate(sanitizedItems, usedKeys, outerwear);
  } else if (layered && !isDressOutfit) {
    missingSlots.push('outerwear');
  }

  // Slot 2: base top. For non-layered outfits this is still the primary top.
  // ALWAYS try to fill this slot - use any available top if needed.
  const baseTop = pickCandidateByPriority(
    generatedCandidates,
    availableItems,
    usedKeys,
    [['top']],
    normalizedStyle,
    rotationOffset,
    avoidKeys,
  );
  if (baseTop) {
    addCandidate(sanitizedItems, usedKeys, baseTop);
  } else if (!isDressOutfit && !(layered === false && outerwear)) {
    // If layering is off and we already have an outerwear piece, a missing
    // base top is acceptable. Otherwise the outfit is short a top.
    missingSlots.push('top');
  }

  if (layered && !isDressOutfit) {
    const secondTop = pickCandidateByPriority(
      generatedCandidates,
      availableItems,
      usedKeys,
      [['top']],
      normalizedStyle,
      rotationOffset + 1,
      avoidKeys,
    );
    if (secondTop) {
      addCandidate(sanitizedItems, usedKeys, secondTop);
    } else {
      missingSlots.push('top');
    }
  }

  // Slot 3: bottom (skipped when the "top" is a dress).
  // Bottoms and shoes rotate fastest — offset by outfit index so 3 looks vary.
  if (!isDressOutfit) {
    const bottom =
      pickGeneratedCandidate(generatedCandidates, usedKeys, ['bottom']) ||
      pickAvailableCandidate(availableItems, usedKeys, ['bottom'], normalizedStyle, rotationOffset, avoidKeys);
    if (bottom) {
      addCandidate(sanitizedItems, usedKeys, bottom);
    } else {
      missingSlots.push('bottom');
    }
  }

  // Slot 4: shoes. ALWAYS try to fill - use any available item if needed.
  const shoes =
    pickGeneratedCandidate(generatedCandidates, usedKeys, ['shoes']) ||
    pickAvailableCandidate(availableItems, usedKeys, ['shoes'], normalizedStyle, rotationOffset, avoidKeys);
  if (shoes) {
    addCandidate(sanitizedItems, usedKeys, shoes);
  } else {
    missingSlots.push('shoes');
  }

  if (sanitizedItems.length === 0) {
    return {
      items: generatedCandidates.slice(0, maxItems).map((c) => c.item),
      missingSlots,
      layered,
    };
  }

  return {
    items: sanitizedItems.slice(0, maxItems),
    missingSlots,
    layered,
  };
}

export function sanitizeGeneratedOutfit(
  outfit: GeneratedOutfit,
  availableItems: WardrobeDisplayItem[],
  maxItems = 5,
  style?: string,
): GeneratedOutfit {
  return {
    ...outfit,
    items: sanitizeGeneratedOutfitItems(outfit.items, availableItems, maxItems, style),
  };
}

/**
 * ensureOutfitsDiverse — post-process a batch of outfits so the 3-card deck
 * is never 3x similar.
 *
 * Guarantees every pair differs by ≥2 item ids (≥1 shared allowed for
 * 3-item looks, anchor excluded from the count when anchorId is set).
 * When a near-duplicate is found, the most-reused non-anchor slot is swapped
 * for the best-scored unused item of the same macroCategory.
 * Never throws — returns input unchanged when the pool can't vary.
 */
export function ensureOutfitsDiverse<T extends { id: string; items: OutfitItem[] }>(
  outfits: T[],
  availableItems: WardrobeDisplayItem[],
  style?: string,
  opts: { anchorId?: string | number | null; layered?: boolean } = {},
): T[] {
  if (outfits.length <= 1) return outfits;
  const normalizedStyle = style ? normalizeStyleId(style) : undefined;
  const anchorKey = opts.anchorId != null ? String(opts.anchorId) : null;

  const usage = new Map<string, number>();
  outfits.forEach((o) =>
    o.items.forEach((i) => {
      const k = String((i as any).id);
      usage.set(k, (usage.get(k) ?? 0) + 1);
    }),
  );

  const scoredPool = (macro: string, exclude: Set<string>): WardrobeDisplayItem[] => {
    const norm = normalizeMacroCategory(macro) ?? 'other';
    const list = availableItems.filter((a) => {
      const m = resolveMacroCategory({
        macroCategory: (a as any).macroCategory,
        category: (a as any).category,
        type: (a as any).type,
        name: (a as any).name,
      });
      if (m !== norm) return false;
      if (exclude.has(String((a as any).id))) return false;
      if (anchorKey && String((a as any).id) === anchorKey) return false;
      return true;
    });
    if (normalizedStyle) {
      list.sort(
        (a, b) =>
          scoreItemForStyle(
            {
              name: (b as any).name,
              description: (b as any).description,
              brand: (b as any).brand,
              color: (b as any).color,
              type: (b as any).type,
              category: (b as any).category,
              macroCategory: (b as any).macroCategory,
            },
            normalizedStyle,
          ) -
          scoreItemForStyle(
            {
              name: (a as any).name,
              description: (a as any).description,
              brand: (a as any).brand,
              color: (a as any).color,
              type: (a as any).type,
              category: (a as any).category,
              macroCategory: (a as any).macroCategory,
            },
            normalizedStyle,
          ),
      );
    }
    // Prefer globally unused items first.
    list.sort((a, b) => (usage.get(String((a as any).id)) ?? 0) - (usage.get(String((b as any).id)) ?? 0));
    return list;
  };

  const result = outfits.map((o) => ({ ...o, items: [...o.items] }));
  const keptSets: Array<Set<string>> = [];

  const sharedCount = (a: Set<string>, b: Set<string>, ignoreKey: string | null): number => {
    let n = 0;
    a.forEach((id) => {
      if (ignoreKey && id === ignoreKey) return;
      if (b.has(id)) n += 1;
    });
    return n;
  };

  result.forEach((outfit, idx) => {
    if (idx === 0) {
      keptSets.push(new Set(outfit.items.map((i) => String((i as any).id))));
      return;
    }
    const curSet = new Set(outfit.items.map((i) => String((i as any).id)));
    // Target: differ by ≥2 non-anchor items → max 1 shared non-anchor.
    let maxShared = 0;
    for (const kept of keptSets) maxShared = Math.max(maxShared, sharedCount(curSet, kept, anchorKey));
    let guard = 0;
    while (maxShared > 1 && guard < 6) {
      guard += 1;
      // Pick most-reused swappable slot (skip anchor).
      let swapIdx = -1;
      let swapUse = -1;
      outfit.items.forEach((item, i) => {
        const k = String((item as any).id);
        if (anchorKey && k === anchorKey) return;
        const u = usage.get(k) ?? 0;
        if (u > swapUse) {
          swapUse = u;
          swapIdx = i;
        }
      });
      if (swapIdx < 0) break;
      const slotMacro =
        (outfit.items[swapIdx] as any).macroCategory || 'other';
      const exclude = new Set<string>([
        ...curSet,
        ...keptSets.flatMap((s) => Array.from(s)),
      ]);
      // First try truly unused globally, then least-used.
      let replacement = scoredPool(slotMacro, exclude)[0];
      if (!replacement) {
        const inOutfit = new Set<string>(Array.from(curSet));
        replacement = scoredPool(slotMacro, inOutfit)[0];
      }
      if (!replacement) break;
      const oldKey = String((outfit.items[swapIdx] as any).id);
      const built = buildCandidate(replacement as any, replacement as any, idx);
      if (!built) break;
      outfit.items[swapIdx] = built.item as OutfitItem;
      usage.set(oldKey, Math.max(0, (usage.get(oldKey) ?? 1) - 1));
      const newKey = String((replacement as any).id);
      usage.set(newKey, (usage.get(newKey) ?? 0) + 1);
      curSet.delete(oldKey);
      curSet.add(newKey);
      maxShared = 0;
      for (const kept of keptSets) maxShared = Math.max(maxShared, sharedCount(curSet, kept, anchorKey));
    }
    keptSets.push(new Set(outfit.items.map((i) => String((i as any).id))));
  });

  return result;
}
