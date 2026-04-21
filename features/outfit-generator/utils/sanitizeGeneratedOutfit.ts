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
): Candidate | null {
  // Collect every candidate in the requested categories, then pick the
  // best-scoring one under the style (if provided). If no style, return
  // the first eligible candidate. Either way, if at least one candidate
  // exists we MUST return one — never null — so required slots like shoes
  // are always filled.
  const allCandidates: { candidate: Candidate; score: number }[] = [];
  for (let index = 0; index < availableItems.length; index += 1) {
    const item = availableItems[index];
    const candidate = buildCandidate(item, item, index);
    if (!candidate) continue;
    if (usedKeys.has(candidate.key)) continue;
    if (!categories.includes(candidate.macroCategory)) continue;
    const score = style
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
    allCandidates.push({ candidate, score });
  }

  if (allCandidates.length === 0) return null;
  if (!style) return allCandidates[0].candidate;

  allCandidates.sort((a, b) => b.score - a.score);
  return allCandidates[0].candidate;
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
  const outerwear =
    pickGeneratedCandidate(generatedCandidates, usedKeys, ['outerwear']) ||
    (layered && !isDressOutfit
      ? pickAvailableCandidate(availableItems, usedKeys, ['outerwear'], normalizedStyle)
      : null);
  if (outerwear) {
    addCandidate(sanitizedItems, usedKeys, outerwear);
  } else if (layered && !isDressOutfit) {
    missingSlots.push('outerwear');
  }

  // Slot 2: base top. For non-layered outfits this is still the primary top.
  // ALWAYS try to fill this slot - use any available top if needed.
  let baseTop =
    pickGeneratedCandidate(generatedCandidates, usedKeys, ['top']) ||
    pickAvailableCandidate(availableItems, usedKeys, ['top'], normalizedStyle);
  // If no base top found but we have outerwear, try to find a different top item
  if (!baseTop && outerwear && availableItems.length > 0) {
    // Find any item that could be a base top (different from outerwear)
    const fallbackTop = availableItems.find(i => 
      i.macroCategory === 'top' || 
      (!usedKeys.has(String(i.id)) && i.macroCategory !== 'outerwear')
    );
    if (fallbackTop) {
      baseTop = buildCandidate(fallbackTop, fallbackTop, 0);
    }
  }
  // If still no base top but we have available items, use any available item
  if (!baseTop && availableItems.length > 0) {
    const anyAvailable = availableItems.find(i => !usedKeys.has(String(i.id)));
    if (anyAvailable) {
      baseTop = buildCandidate(anyAvailable, anyAvailable, 0);
    }
  }
  if (baseTop) {
    addCandidate(sanitizedItems, usedKeys, baseTop);
  } else if (!isDressOutfit && !(layered === false && outerwear)) {
    // If layering is off and we already have an outerwear piece, a missing
    // base top is acceptable. Otherwise the outfit is short a top.
    missingSlots.push('top');
  }

  // Slot 3: bottom (skipped when the "top" is a dress).
  if (!isDressOutfit) {
    let bottom =
      pickGeneratedCandidate(generatedCandidates, usedKeys, ['bottom']) ||
      pickAvailableCandidate(availableItems, usedKeys, ['bottom'], normalizedStyle);
    // If no bottom found, try any available item
    if (!bottom && availableItems.length > 0) {
      const fallbackBottom = availableItems.find(i => !usedKeys.has(String(i.id)));
      if (fallbackBottom) {
        bottom = buildCandidate(fallbackBottom, fallbackBottom, 0);
      }
    }
    if (bottom) {
      addCandidate(sanitizedItems, usedKeys, bottom);
    } else {
      missingSlots.push('bottom');
    }
  }

  // Slot 4: shoes. ALWAYS try to fill - use any available item if needed.
  let shoes =
    pickGeneratedCandidate(generatedCandidates, usedKeys, ['shoes']) ||
    pickAvailableCandidate(availableItems, usedKeys, ['shoes'], normalizedStyle);
  // If no shoes found, try any available item that hasn't been used
  if (!shoes && availableItems.length > 0) {
    const fallbackShoe = availableItems.find(i => !usedKeys.has(String(i.id)));
    if (fallbackShoe) {
      shoes = buildCandidate(fallbackShoe, fallbackShoe, 0);
    }
  }
  // If still no shoes, we MUST have something - duplicate an item if necessary
  if (!shoes && availableItems.length > 0) {
    const duplicateItem = availableItems[0];
    shoes = buildCandidate(duplicateItem, duplicateItem, 0);
  }
  if (shoes) {
    addCandidate(sanitizedItems, usedKeys, shoes);
  } else {
    missingSlots.push('shoes');
  }

  // Optional accessory / other — only surfaced if room remains.
  addCandidate(
    sanitizedItems,
    usedKeys,
    pickGeneratedCandidate(generatedCandidates, usedKeys, ['accessory'])
  );
  addCandidate(
    sanitizedItems,
    usedKeys,
    pickGeneratedCandidate(generatedCandidates, usedKeys, ['other'])
  );

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
