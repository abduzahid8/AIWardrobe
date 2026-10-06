/**
 * normalizeSlots — outfit slot normalization extracted from
 * screens/AIOutfitmaker.tsx so the REAL pick code (not a mirror) is
 * unit-testable. Pure except for __DEV__ logging.
 *
 * createSlotNormalizer(ctx) binds screen state once per render and
 * returns a normalizeTo4Slots with the original positional signature,
 * so all call sites behave byte-identically to before.
 */
import { canonicalizeMacroCategory } from '../../../src/utils/categoryMapper';
import { occasionFormality, rejectForStyleOccasion, rejectForClimate, rejectOutfitClimate, rejectOutfitIncoherence, matchesOccasionSilhouette, detectSlotMismatch, isUnderwearItem, isAccessoryPropItem, SHORT_SLEEVE_RE, STRIP_SHORT_SLEEVE_RE, humanizeProductName } from './occasionRules';
import { orderSlotCandidates, selectSlotPick } from './deckDedupe';
import { parseBudgetFromText, resolveShopPrice, outfitShopSpend, type ParsedBudget } from '../../../src/utils/outfitBudget';
import { parseStylistIntent } from './stylistIntent';
import { normalizeCatalogNameKey, isTrustedProductImage, isBrandCatalogImage, collapseDuplicateProductImages, productImageKey, garmentVarietyFamily } from '../../../src/utils/shopCatalogOrder';

// ── Shared helpers: formal-layer + shorts rule ─────────────────────────
export function isFormalLayer(item: { type?: string; name?: string; brand?: string; macroCategory?: string } | null | undefined): boolean {
  if (!item) return false;
  const blob = `${item.type || ''} ${item.name || ''} ${item.brand || ''}`.toLowerCase();
  const macro = (item.macroCategory || '').toLowerCase();
  const isOuter = macro === 'outerwear' || /jacket|coat|blazer|vest|outerwear/.test(blob);
  if (!isOuter) return false;
  return /\b(blazer|suit\s*jacket|sport\s*coat|sports\s*coat|overcoat|top\s*coat|topcoat|trench|peacoat|pea\s*coat|tuxedo)\b/.test(blob);
}

export function isShortsBottom(item: { type?: string; name?: string; macroCategory?: string; subCategory?: string } | null | undefined): boolean {
  if (!item) return false;
  // 'short-sleeved' describes SLEEVES on tops — strip it first, otherwise
  // every short-sleeve tee mislabeled as bottoms reads as shorts.
  const blob = `${item.type || ''} ${item.name || ''} ${item.subCategory || ''}`.toLowerCase().replace(STRIP_SHORT_SLEEVE_RE, '');
  const macro = (item.macroCategory || '').toLowerCase();
  const isBottom = macro === 'bottom' || /pant|trouser|jeans|bottom|shorts?|skirt|trunks?|swim/.test(blob);
  if (!isBottom) return false;
  // NOTE: must include swim/trunks — "Swimming trunks" contains neither
  // "shorts" nor "bermuda" and previously slipped through every guard,
  // producing suit-jacket + swim-trunks outfits.
  return /\b(shorts?|bermudas?|trunks?|swim\w*|swimwear|board\s*shorts?|bathing\s*suit)\b/.test(blob);
}

// ── Layer-vs-base distinction ──────────────────────────────────────
// MAIN TOP (slot 1) must be a true outer layer, SECOND TOP (slot 2) a
// lightweight base. Two knits / two dark long-sleeves read as "same
// category" (user bug report). These helpers force different fabrics.
export function fabricFamily(item: { type?: string; name?: string; brand?: string; description?: string } | null | undefined): string {
  if (!item) return 'unknown';
  const blob = `${item.type || ''} ${item.name || ''} ${item.brand || ''} ${(item as any).description || ''}`.toLowerCase();
  if (/\b(knit|cardigan|sweater|pullover|jumper|fleece|hoodie)\b/.test(blob)) return 'knit';
  if (/\b(denim|jean|chambray)\b/.test(blob)) return 'denim';
  if (/\b(blazer|suit|coat|jacket|trench|overcoat|peacoat|bomber|puffer|vest|outerwear)\b/.test(blob)) return 'tailored-outer';
  if (/\b(linen|poplin|oxford|flannel|twill|broadcloth)\b/.test(blob)) return 'woven';
  if (/\b(t-shirt|tshirt|\btee\b|jersey|tank|sleeveless)\b/.test(blob)) return 'tee';
  if (/\b(polo|pique)\b/.test(blob)) return 'polo';
  if (/\b(blouse|silk|satin)\b/.test(blob)) return 'blouse';
  return 'other-top';
}

export function colorFamily(color?: string, name?: string): string {
  const blob = `${color || ''} ${name || ''}`.toLowerCase();
  if (/\b(black|noir|charcoal|graphite)\b/.test(blob)) return 'dark';
  if (/\b(navy|midnight|dark blue|ink)\b/.test(blob)) return 'dark';
  if (/\b(white|off-white|cream|ivory|ecru|milk)\b/.test(blob)) return 'light';
  if (/\b(beige|sand|camel|tan|khaki|stone|oat)\b/.test(blob)) return 'light';
  if (/\b(grey|gray|greige|slate)\b/.test(blob)) return 'mid';
  if (/\b(blue|light blue|sky|chambray)\b/.test(blob)) return 'mid';
  if (/\b(brown|chocolate|coffee|espresso|cognac)\b/.test(blob)) return 'earth';
  if (/\b(olive|green|forest|khaki)\b/.test(blob)) return 'earth';
  if (/\b(burgundy|wine|maroon|rust)\b/.test(blob)) return 'warm';
  return 'other';
}

export const REAL_IMAGE_RE = /^(https?:|file:|data:|asset:|content:)/i;

/** True when layer + base would read as the same category. */
export function isSameLayerBaseCategory(
  layer: { type?: string; name?: string; brand?: string; color?: string; description?: string } | null | undefined,
  base: { type?: string; name?: string; brand?: string; color?: string; description?: string } | null | undefined,
): boolean {
  if (!layer || !base) return false;
  const lf = fabricFamily(layer);
  const bf = fabricFamily(base);
  // Same fabric family worn as layer + base (knit-on-knit etc.)
  if (lf !== 'other-top' && lf === bf) return true;
  // Identical dark color on both tops with no texture contrast
  const lc = colorFamily((layer as any).color, layer.name);
  const bc = colorFamily((base as any).color, base.name);
  const lName = (layer.name || '').toLowerCase();
  const bName = (base.name || '').toLowerCase();
  const sameColorWord = (layer as any).color && (base as any).color &&
    String((layer as any).color).toLowerCase().trim() === String((base as any).color).toLowerCase().trim();
  if ((lc === bc && (lc === 'dark' || lc === 'other')) && sameColorWord) {
    // Allow only if fabrics clearly differ (tailored shell over tee is fine)
    if (lf === 'tailored-outer' && (bf === 'tee' || bf === 'woven' || bf === 'polo')) return false;
    // Knit cardigan over dark shirt in the same navy = the reported bug
    if (lf === 'knit' || bf === 'knit') return true;
    if (lName.includes('shirt') && bName.includes('shirt')) return true;
  }
  return false;
}

export function looksLikeLayerGarment(item: { type?: string; name?: string; macroCategory?: string; garmentType?: string; category?: string } | null | undefined): boolean {
  if (!item) return false;
  const raw = `${item.type || ''} ${item.name || ''}`.toLowerCase();
  if (SHORT_SLEEVE_RE.test(raw)) return false;
  const blob = raw.replace(STRIP_SHORT_SLEEVE_RE, ' ');
  const stored = `${item.type || ''} ${item.category || ''} ${(item as any).garmentType || ''} ${item.macroCategory || ''}`.toLowerCase();
  if (/\b(bottom|bottoms|lower_body)\b/.test(stored)) return false;
  const macro = (item.macroCategory || '').toLowerCase();
  if (macro === 'outerwear') return true;
  return /\b(jacket|coat|blazer|hoodie|cardigan|sweater|pullover|puffer|bomber|vest|outerwear|trench|peacoat|overshirt|shacket|total[\s-]?look|zip[\s-]?up)\b/.test(blob);
}

/** Base clothing tops only — ties, sunglasses, wallets must never occupy TOP. */
export function looksLikeRealTop(item: { type?: string; name?: string; macroCategory?: string } | null | undefined): boolean {
  if (!item) return false;
  if (isUnderwearItem(item) || isAccessoryPropItem(item)) return false;
  const blob = `${item.type || ''} ${item.name || ''}`.toLowerCase();
  if (/\b(pyjama|pajama)\b/.test(blob)) return false;
  const macro = (item.macroCategory || '').toLowerCase();
  if (macro === 'bottom' || macro === 'shoes' || macro === 'outerwear' || macro === 'accessory') return false;
  if (looksLikeLayerGarment(item) && !/\b(shirt|polo|tee|t-shirt|henley|blouse)\b/.test(blob)) return false;
  return macro === 'top'
    || /\b(t-?shirts?|tees?|polos?|blouses?|tanks?|henleys?|shirts?|oxfords?|button[\s-]?downs?)\b/.test(blob);
}

/** True bottoms only — shirts/overshirts/jackets must never occupy pants. */
export function looksLikeRealBottom(item: { type?: string; name?: string; macroCategory?: string; subCategory?: string } | null | undefined): boolean {
  if (!item) return false;
  const blob = `${item.type || ''} ${item.name || ''} ${item.subCategory || ''}`.toLowerCase().replace(STRIP_SHORT_SLEEVE_RE, ' ');
  if (SHORT_SLEEVE_RE.test(`${item.name || ''} ${item.type || ''}`)) return false;
  if (/\b(pyjama|pajama|nightwear)\b/.test(blob)) return false;
  if (/\b(t-?shirt|tee\b|polo|blouse|overshirt|shirt|jacket|coat|blazer)\b/.test(blob) && !/\b(pants?|trousers?|jeans?|chinos?|slacks?|shorts|skirt)\b/.test(blob)) {
    return false;
  }
  return /\b(pants?|trousers?|jeans?|chinos?|slacks?|shorts|bermudas?|skirts?|joggers?)\b/.test(blob);
}

export const PLACEHOLDER_SHOES = {
  id: 'placeholder_shoes',
  type: 'shoes',
  macroCategory: 'shoes',
  name: 'Shoes',
  image: 'placeholder_shoes',
  imageUrl: 'placeholder_shoes',
  color: 'neutral',
  brand: '',
};

export const PLACEHOLDER_PANTS = {
  id: 'placeholder_pants',
  type: 'pants',
  macroCategory: 'bottom',
  name: 'Pants',
  image: 'placeholder_pants',
  imageUrl: 'placeholder_pants',
  color: 'neutral',
  brand: '',
};


  // Backend-compatible macroCategory matching the edge function's slot model:
  // 'top' (base layer), 'outerwear' (jacket/blazer/sweater), 'bottom', 'shoes'
  export const getBackendMacroCategory = (type: string, category?: string, name?: string) => {
    // Outerwear keywords take precedence over a generic alias: a blazer
    // stored as category "top"/"tops" is still a layer (main-top slot),
    // not a base top. Without this the anchor locks into second-top.
    const t = `${type || ''} ${category || ''} ${name || ''}`.toLowerCase();
    if (SHORT_SLEEVE_RE.test(t) && !/\b(pant|trouser|jeans?|shorts?|skirt)\b/.test(t.replace(STRIP_SHORT_SLEEVE_RE, ' '))) return 'top';
    if (t.match(/\b(jacket|coat|blazer|hoodie|cardigan|sweater|pullover|vest|puffer|outerwear|trench|peacoat|bomber)\b/)) return 'outerwear';

    // Prefer canonical aliases (upper_body / lower_body / tops / footwear / …)
    // so items coming back from shop_catalog are slotted correctly.
    const aliasHit = canonicalizeMacroCategory(type) !== 'other'
      ? canonicalizeMacroCategory(type)
      : canonicalizeMacroCategory(category || '');
    if (aliasHit !== 'other') return aliasHit;

    if (t.match(/\b(pant|trouser|jeans?|bottom|shorts?|skirt|lower[_\s-]?body)\b/) && !SHORT_SLEEVE_RE.test(t) && !/\b(shirt|tee|polo|blouse|overshirt|jacket|coat|blazer)\b/.test(t)) return 'bottom';
    if (t.match(/\b(shoe|sneaker|boot|loafer|sandal|footwear|trainer|derby|derbies|moccasin|driver|mule)\b/)) return 'shoes';
    if (t.match(/\b(shirt|t-shirt|tee|blouse|polo|top|dress|upper[_\s-]?body)\b/)) return 'top';
    return 'top'; // fallback keeps the backend happy
  };

  export const isCompleteNormalizedOutfit = (items: any[]) => {
    const macros = items.map((item) => canonicalizeMacroCategory(item.macroCategory || ''));
    const topCount = macros.filter((macro) => macro === 'top').length;
    const hasBottom = macros.includes('bottom');
    const hasShoes = macros.includes('shoes');
    const hasOuterwear = macros.includes('outerwear');
    // A complete 4-slot outfit needs: top-like (top or outerwear), bottom, shoes.
    // With the new 4-item model: [outerwear|top, top, bottom, shoes]
    const hasTopLike = topCount >= 1 || hasOuterwear;
    return hasTopLike && hasBottom && hasShoes;
  };

/** Repair Edge-returned items only: names + slot macros. Never picks from a local pool. */
export function repairEdgeItems(items: any[]): any[] {
  return (items || []).map((it: any) => {
    if (!it || isUnderwearItem(it) || isAccessoryPropItem(it)) return null;
    const name = humanizeProductName(it.name) || it.name;
    const fixed = detectSlotMismatch(it.macroCategory, name, it.type);
    const own = canonicalizeMacroCategory(it.macroCategory || '');
    return {
      ...it,
      name,
      macroCategory: fixed || (own !== 'other' ? own : it.macroCategory),
    };
  }).filter(Boolean);
}

const SLOT_FLOOR_MACROS = ['top', 'bottom', 'shoes', 'outerwear'] as const;

/** LAYER + shirt + pants + shoes — the 4-tile card, not “jacket as TOP”. */
export function isFourSlotLook(items: any[] | null | undefined): boolean {
  if (!items || items.length < 4) return false;
  const macros = new Set(
    items.map((item) => canonicalizeMacroCategory(item?.macroCategory || '')),
  );
  return SLOT_FLOOR_MACROS.every((m) => macros.has(m));
}

/**
 * After taste-filter a random Zara window can leave ~20 rows and no shirts.
 * Merge extras (bundled MD catalog) until each clothing slot has a floor.
 */
export function ensureSlotFloors(
  kept: any[],
  extras: any[],
  styleId: string | null | undefined,
  occasionId: string | null | undefined,
  floors: Record<(typeof SLOT_FLOOR_MACROS)[number], number> = {
    top: 12,
    bottom: 10,
    shoes: 8,
    outerwear: 8,
  },
  weather?: { temp?: number | null; condition?: string | null } | null,
): any[] {
  const seen = new Set((kept || []).map((i) => String(i?.id || '')));
  const out = [...(kept || [])];
  const countOf = (slot: string) =>
    out.filter((i) => canonicalizeMacroCategory(i?.macroCategory || '') === slot).length;
  const tryAdd = (item: any): boolean => {
    if (!item || seen.has(String(item.id))) return false;
    const macro = canonicalizeMacroCategory(item.macroCategory || '');
    if (!SLOT_FLOOR_MACROS.includes(macro as any)) return false;
    if (rejectForStyleOccasion(item, styleId, occasionId) !== null) return false;
    if (rejectForClimate(item, weather) !== null) return false;
    seen.add(String(item.id));
    out.push({ ...item, macroCategory: macro });
    return true;
  };
  for (const slot of SLOT_FLOOR_MACROS) {
    for (const extra of extras || []) {
      if (countOf(slot) >= floors[slot]) break;
      if (canonicalizeMacroCategory(extra?.macroCategory || '') !== slot) continue;
      tryAdd(extra);
    }
  }
  return out;
}

export interface SlotNormalizerContext {
    weather?: { temp?: number | null; condition?: string | null } | undefined;
    baseItem?: any;
    baseItemId?: string | null;
    currentSource?: string;
    selectedOccasion?: string | null;
    generationNonceRef?: { current?: number | null } | null;
    poolFallback?: any[];
    /** Free-text prompt (occasion + user budget), parsed for a spend cap. */
    budgetText?: string | null;
    maxShopBudget?: ParsedBudget | null;
    /** Cross-tap recency: skip these ids/name/image keys while the pool still has unused pieces. */
    avoidIds?: Set<string> | ReadonlySet<string> | null;
}

export function createSlotNormalizer(ctx: SlotNormalizerContext) {
    const normalizeTo4Slots = (items: any[], styleId?: string, rotationOffset = 0, tapSeed?: number, pool?: any[], deckUsedIds?: Set<string>): any[] => {
    // Pool override: generation passes the per-tap pool (memoized first page
    // + random slice, collapsed + recency-ordered). Defaults to the memoized
    // pool so pure-anchor / offline paths are unaffected.
    const pickPool: any[] = pool ?? ctx.poolFallback ?? [];
    // ── Anchor resolution (single source of truth for locking) ──────────
    // The anchor MUST win its macro slot regardless of AI order. We compute
    // its true macro with outerwear-keyword priority (blazer stored as "top"
    // is still a layer), then pick anchor-first per slot + post-enforce.
    // Fallback: route params can be lost on re-render — if ctx.baseItem is gone
    // but ctx.baseItemId survives, rebuild it from pool/store so the lock holds.
    // Shop ("New clothing") mode NEVER locks an anchor: the anchor is a
    // wardrobe piece and forcing it in would build every shop outfit around
    // the user's own clothing.
    let effectiveBaseItem = ctx.currentSource === 'wardrobe' ? ctx.baseItem : null;
    if (!effectiveBaseItem && ctx.baseItemId && ctx.currentSource === 'wardrobe') {
      const found: any = [...(items || []), ...(pool ?? ctx.poolFallback ?? [])].find(
        (i: any) => String(i?.id) === String(ctx.baseItemId),
      );
      if (found) {
        effectiveBaseItem = {
          id: ctx.baseItemId,
          imageUrl: found.imageUrl || (typeof found.image === 'string' ? found.image : undefined),
          name: found.name,
          type: found.type,
          macroCategory: found.macroCategory,
          color: found.color,
          category: found.category,
          brand: found.brand,
          description: found.description,
        } as any;
      }
    }
    let anchorMacro: string | null = null;
    let anchorany: any | null = null;
    if (effectiveBaseItem && ctx.baseItemId) {
      const recomputed = getBackendMacroCategory(
        effectiveBaseItem.type || '',
        (effectiveBaseItem as any).category || effectiveBaseItem.macroCategory || '',
        effectiveBaseItem.name || '',
      );
      const stored = canonicalizeMacroCategory(effectiveBaseItem.macroCategory || '');
      // Outerwear wins on conflict: a blazer mis-tagged as "top" upstream
      // must lock into the main-top/layer slot, not second-top.
      anchorMacro = recomputed === 'outerwear' ? 'outerwear' : (stored !== 'other' ? stored : recomputed);
      anchorany = {
        id: ctx.baseItemId,
        name: effectiveBaseItem.name || effectiveBaseItem.type || 'Item',
        image: effectiveBaseItem.imageUrl || '',
        imageUrl: effectiveBaseItem.imageUrl || '',
        type: effectiveBaseItem.type || anchorMacro,
        macroCategory: anchorMacro,
        color: effectiveBaseItem.color || 'neutral',
        isWardrobe: true,
      };
      // Pre-inject so AI-order-independent picks still see it first.
      // NOTE: the anchor id intentionally stays OUT of deckUsedIds — it
      // repeats in every card by design (dedupe already skips it).
      const alreadyPresent = items.some(i => String(i.id) === String(ctx.baseItemId));
      if (!alreadyPresent) {
        const slotIdx = items.findIndex(
          i => (canonicalizeMacroCategory(i.macroCategory || '') === anchorMacro
            || canonicalizeMacroCategory(getBackendMacroCategory(i.type || '', i.category || '', i.name || '')) === anchorMacro)
        );
        if (slotIdx >= 0) {
          items = [...items];
          (items as any[])[slotIdx] = anchorany;
        } else {
          items = [anchorany, ...items];
        }
      }
    }
    // Slot-integrity repair: feed rows are mistagged at scale (a T-shirt
    // stored as bottoms renders as PANTS). When the NAME definitively
    // belongs to another slot, trust the name. Anchor items carry an
    // explicit lock and are never remapped. Applied to BOTH incoming and
    // pool items so every downstream pick sees truthful macros.
    const repairItemMacro = (it: any): any => {
      if (!it || (ctx.baseItemId && String(it?.id) === String(ctx.baseItemId))) return it;
      const fixed = detectSlotMismatch(it?.macroCategory, it?.name || '', it?.type || '');
      if (!fixed) return it;
      if (__DEV__) console.log('[normalizeTo4Slots] slot repair:', { id: it.id, was: it.macroCategory, now: fixed, name: (it.name || '').slice(0, 50) });
      return { ...it, macroCategory: fixed };
    };
    const pickPoolMapped: any[] = collapseDuplicateProductImages(
      (pickPool || []).map(repairItemMacro).filter((it: any) => !isUnderwearItem(it) && !isAccessoryPropItem(it)),
    );
    const pickPoolCdn = pickPoolMapped.filter((it: any) => isBrandCatalogImage(it.imageUrl || it.image));
    const pickPoolTrusted = pickPoolMapped.filter((it: any) => isTrustedProductImage(it.imageUrl || it.image));
    const pickPoolFixed: any[] = ctx.currentSource === 'shop'
      ? (pickPoolCdn.length >= 8 ? pickPoolCdn : (pickPoolTrusted.length > 0 ? pickPoolTrusted : pickPoolMapped))
      : pickPoolMapped;
    const budget: ParsedBudget | null =
      ctx.maxShopBudget
      || parseBudgetFromText(ctx.budgetText || '')
      || null;
    const priceOf = (it: any) => resolveShopPrice(it, pickPoolFixed);
    const withListedPrice = (it: any) => {
      if (!it) return it;
      const listed = priceOf(it);
      if (listed <= 0) return it;
      return { ...it, price: listed, isShopItem: it.isShopItem ?? (ctx.currentSource === 'shop') };
    };
    let incomingItems = items;
    // Helper: prefer an item's own macroCategory (canonicalized) over the
    // keyword-based fallback. Keeps items whose type/category are raw
    // garmentType strings (`upper_body`, `lower_body`) from being dropped.
    const resolveMacroCategory = (it: any): string => {
      const ownCanonical = canonicalizeMacroCategory(it?.macroCategory || '');
      if (ownCanonical !== 'other') return ownCanonical;
      return getBackendMacroCategory(it?.type || '', it?.category || '', it?.name || '');
    };

    const resolveImage = (it: any) => (typeof it.image === 'string' ? it.image : undefined) || it.imageUrl || '';

    const incomingMapped = (incomingItems || []).map((raw) => {
      const it = repairItemMacro(raw);
      if (isUnderwearItem(it) || isAccessoryPropItem(it)) return null;
      const image = it.image || it.imageUrl || '';
      return withListedPrice({
        ...it,
        image,
        macroCategory: resolveMacroCategory(it),
        _hasImage: (it as any)._hasImage ?? isTrustedProductImage(image || (it as any).imageUrl || ''),
      });
    }).filter(Boolean);
    const incomingDeduped = collapseDuplicateProductImages(incomingMapped as any[]);
    const incomingTrusted = ctx.currentSource === 'shop'
      ? incomingDeduped.filter((it: any) => isTrustedProductImage(it.image || it.imageUrl))
      : incomingDeduped;
    const incoming = ctx.currentSource === 'shop'
      ? incomingTrusted
      : incomingDeduped;
    const commentIntent = parseStylistIntent(ctx.budgetText || '');
    const occasionId = commentIntent.occasion || ctx.selectedOccasion;
    const formalOccasion = occasionFormality(occasionId) === 'formal';
    // A locked outerwear anchor forces layering even in warm weather —
    // otherwise its slot would never be picked and it would "fall" into top.
    const anchorForcesLayer = anchorMacro === 'outerwear';
    const incomingWantsLayer = incoming.some((it: any) =>
      canonicalizeMacroCategory(it?.macroCategory || '') === 'outerwear'
      && looksLikeLayerGarment(it)
      && matchesOccasionSilhouette(it, occasionId),
    );
    const travelSoftLayer = String(occasionId || '').toLowerCase() === 'travel'
      && [...incoming, ...pickPoolFixed].some((it: any) => {
        const m = canonicalizeMacroCategory(it?.macroCategory || '');
        return m === 'outerwear'
          && looksLikeLayerGarment(it)
          && matchesOccasionSilhouette(it, occasionId)
          && rejectForStyleOccasion(it, styleId, occasionId) === null;
      });
    const climateNeedsLayer = (ctx.weather?.temp != null && ctx.weather.temp < 18)
      || /\b(cold|chilly|freezing|snow|rain|drizzle|wind|storm)\b/.test((ctx.weather?.condition || '').toLowerCase());
    const needsLayer = anchorForcesLayer
      || incomingWantsLayer
      || formalOccasion
      || travelSoftLayer
      || climateNeedsLayer;
    const usedIds = new Set<string>();

    if (__DEV__) console.log('[normalizeTo4Slots] incoming items:', incoming.map((it: any) => ({ id: it.id, name: it.name, macro: it.macroCategory, image: (it.image || '').substring(0, 40) })));
    if (__DEV__) console.log('[normalizeTo4Slots] pool size:', pickPool.length, 'sample:', pickPool.slice(0, 5).map((it: any) => ({ id: it.id, macro: it.macroCategory })));
    if (__DEV__) console.log('[normalizeTo4Slots] needsLayer:', needsLayer, 'weather:', ctx.weather, 'anchor:', anchorMacro, ctx.baseItemId);

    const pickedForBudget: any[] = [];
    const remainingBudget = () =>
      budget ? budget.amount - outfitShopSpend(pickedForBudget, pickPoolFixed) : Number.POSITIVE_INFINITY;
    const fitsBudget = (candidate: any) => {
      if (!budget) return true;
      const p = priceOf(candidate);
      if (p <= 0) return false;
      return p <= budget.amount && p <= remainingBudget() + 0.009;
    };

    const nameKeyOf = (candidate: any) => normalizeCatalogNameKey(candidate?.name || '');
    const deckKeysOf = (candidate: any): string[] => {
      const keys = [String(candidate?.id || '')];
      const nk = nameKeyOf(candidate);
      if (nk) keys.push(`__n:${nk}`);
      const img = productImageKey(candidate);
      if (img) keys.push(`__img:${img}`);
      const fam = garmentVarietyFamily(candidate);
      if (fam) keys.push(`__fam:${fam}`);
      return keys.filter(Boolean);
    };
    const pickCandidate = (predicate: (candidate: any) => boolean, slotOffset = 0, ignoreBudget = false) => {
      const basePass = (candidate: any, uniqueName: boolean) => {
        const key = String(candidate.id || '');
        if (usedIds.has(key)) return false;
        if (uniqueName) {
          const nk = nameKeyOf(candidate);
          if (nk && (usedIds.has(`__n:${nk}`) || deckUsedIds?.has(`__n:${nk}`))) return false;
          const img = productImageKey(candidate);
          if (img && (usedIds.has(`__img:${img}`) || deckUsedIds?.has(`__img:${img}`))) return false;
          const fam = garmentVarietyFamily(candidate);
          if (fam.startsWith('shoe:') && (usedIds.has(`__fam:${fam}`) || deckUsedIds?.has(`__fam:${fam}`))) return false;
          if (ctx.avoidIds && ctx.avoidIds.size > 0) {
            if (ctx.avoidIds.has(key)) return false;
            if (nk && ctx.avoidIds.has(`__n:${nk}`)) return false;
            if (img && ctx.avoidIds.has(`__img:${img}`)) return false;
            if (fam.startsWith('shoe:') && ctx.avoidIds.has(`__fam:${fam}`)) return false;
          }
        }
        return predicate(candidate) && (ignoreBudget || fitsBudget(candidate));
      };
      const collect = (uniqueName: boolean) => ({
        incomingMatches: incoming.filter((c: any) => basePass(c, uniqueName)),
        poolMatches: pickPoolFixed.filter((c: any) => basePass(c, uniqueName)),
      });
      let { incomingMatches, poolMatches } = collect(true);
      if (incomingMatches.length === 0 && poolMatches.length === 0) {
        ({ incomingMatches, poolMatches } = collect(false));
      }
      if (incomingMatches.length === 0 && poolMatches.length === 0) return undefined;
      const seed = (tapSeed ?? ctx.generationNonceRef?.current ?? 0) || 0;
      const ordered = orderSlotCandidates(incomingMatches, poolMatches, {
        seed: ((seed * 2654435761 + slotOffset * 97) >>> 0),
        deckUsedIds,
        keysOf: deckKeysOf,
      });
      const found = selectSlotPick(ordered, { rotationOffset, slotOffset, deckUsedIds, keysOf: deckKeysOf });
      if (!found) return undefined;
      usedIds.add(String(found.id || ''));
      deckUsedIds?.add(String(found.id || ''));
      const foundNameKey = nameKeyOf(found);
      if (foundNameKey) {
        usedIds.add(`__n:${foundNameKey}`);
        deckUsedIds?.add(`__n:${foundNameKey}`);
      }
      const foundImg = productImageKey(found);
      if (foundImg) {
        usedIds.add(`__img:${foundImg}`);
        deckUsedIds?.add(`__img:${foundImg}`);
      }
      const foundFam = garmentVarietyFamily(found);
      if (foundFam) {
        usedIds.add(`__fam:${foundFam}`);
        deckUsedIds?.add(`__fam:${foundFam}`);
      }
      const priced = withListedPrice({ ...found, image: resolveImage(found) || found.image || '' });
      pickedForBudget.push(priced);
      return priced;
    };

    // Anchor-first pick: when this slot IS the anchor's macro, the locked
    // item wins over AI order. Marks usedIds so it can't be double-picked.
    const pickAnchorForSlot = (slotMacro: string) => {
      if (!anchorany || !anchorMacro || anchorMacro !== slotMacro) return undefined;
      const key = String(anchorany.id || '');
      if (usedIds.has(key)) {
        // Already placed (e.g. outerwear anchor also matching top fallback) —
        // return the same object so caller keeps a single instance.
        return { ...anchorany, image: resolveImage(anchorany) || anchorany.image || '' };
      }
      usedIds.add(key);
      const priced = withListedPrice({ ...anchorany, image: resolveImage(anchorany) || anchorany.image || '' });
      pickedForBudget.push(priced);
      return priced;
    };

    // Prefer candidates whose image is a real URI (http / file / data / asset
    // / content). This lets a real wardrobe or shop_catalog item override
    // any server-side placeholder item that shares the same macroCategory
    // but only carries a legacy `basic_clothing_*` / empty string.
    // Uses the precomputed _hasImage flag (pool + incoming) — falls back to
    // the regex only for ad-hoc items (anchor/placeholders).
    const hasRealImage = (candidate: any): boolean => {
      const raw = typeof candidate?.image === 'string' && candidate.image
        ? candidate.image
        : candidate?.imageUrl;
      if (isTrustedProductImage(raw)) return true;
      if (typeof candidate?._hasImage === 'boolean') return candidate._hasImage && REAL_IMAGE_RE.test(String(raw || ''));
      return typeof raw === 'string' && REAL_IMAGE_RE.test(raw);
    };

    // Prefer NON-shorts bottoms. Never promote a random non-shoe/top item
    // to the pants slot — that was producing the "flowy shirt labeled
    // SHORTS" bug. Fall back to PLACEHOLDER_PANTS when no real pants exist.
    // Each slot gets its own offset (0/3/6/9) so outfit_0/1/2 rotate.
    // Formal occasions (work/date/formal) try occasion-clean picks FIRST
    // (no beach shoes, printed tops, puffer layers, sloppy bottoms), then
    // fall back to the unconstrained pool so thin pools still complete.
    const occasionOk = (candidate: any) =>
      !formalOccasion || rejectForStyleOccasion(candidate, styleId, occasionId) === null;
    const styleOk = (candidate: any) =>
      rejectForStyleOccasion(candidate, styleId, occasionId) === null
      && rejectForClimate(candidate, ctx.weather) === null;
    const silhouetteOk = (candidate: any) => matchesOccasionSilhouette(candidate, occasionId);
    // Short-sleeve knits are base tops, never layers (live bug: short-sleeve
    // sweater picked as outerwear). Anchor locks bypass this by design.
    const isShortSleeve = (c: any) => SHORT_SLEEVE_RE.test(`${c?.name || ''} ${c?.type || ''}`);
    const isWearableBottom = (candidate: any) =>
      candidate.macroCategory === 'bottom' && looksLikeRealBottom(candidate) && !isShortsBottom(candidate);
    const slotPantsRaw =
      pickAnchorForSlot('bottom')
      || pickCandidate((candidate: any) => isWearableBottom(candidate) && hasRealImage(candidate) && styleOk(candidate) && silhouetteOk(candidate), 6)
      || (formalOccasion ? pickCandidate((candidate: any) => isWearableBottom(candidate) && hasRealImage(candidate) && occasionOk(candidate) && styleOk(candidate), 6) : undefined)
      || pickCandidate((candidate: any) => isWearableBottom(candidate) && hasRealImage(candidate) && styleOk(candidate), 6)
      || pickCandidate((candidate: any) => isWearableBottom(candidate) && styleOk(candidate), 6)
      || { ...PLACEHOLDER_PANTS };
    // Force macroCategory to 'bottom' so OutfitSlotGrid.classifyItem never drops it
    let slotPants = { ...slotPantsRaw, macroCategory: 'bottom' as const };
    const pantsIsShorts = isShortsBottom(slotPantsRaw);

    const slotOuterwearRaw = needsLayer
      ? (pickAnchorForSlot('outerwear')
        || (formalOccasion ? pickCandidate((candidate: any) => candidate.macroCategory === 'outerwear' && isFormalLayer(candidate) && !isShortSleeve(candidate) && hasRealImage(candidate) && styleOk(candidate) && silhouetteOk(candidate), 0, climateNeedsLayer) : undefined)
        || pickCandidate((candidate: any) => candidate.macroCategory === 'outerwear' && !isShortSleeve(candidate) && hasRealImage(candidate) && !(pantsIsShorts && isFormalLayer(candidate)) && styleOk(candidate) && silhouetteOk(candidate), 0, climateNeedsLayer)
        || (formalOccasion ? pickCandidate((candidate: any) => candidate.macroCategory === 'outerwear' && !isShortSleeve(candidate) && hasRealImage(candidate) && !(pantsIsShorts && isFormalLayer(candidate)) && occasionOk(candidate) && styleOk(candidate), 0, climateNeedsLayer) : undefined)
        || pickCandidate((candidate: any) => candidate.macroCategory === 'outerwear' && !isShortSleeve(candidate) && hasRealImage(candidate) && !(pantsIsShorts && isFormalLayer(candidate)) && styleOk(candidate), 0, climateNeedsLayer)
        || pickCandidate((candidate: any) => candidate.macroCategory === 'outerwear' && !isShortSleeve(candidate) && !(pantsIsShorts && isFormalLayer(candidate)) && styleOk(candidate) && silhouetteOk(candidate), 0, climateNeedsLayer)
        || (!pantsIsShorts ? pickCandidate((candidate: any) => candidate.macroCategory === 'outerwear' && !isShortSleeve(candidate) && silhouetteOk(candidate), 0, climateNeedsLayer) : undefined))
      : (anchorMacro === 'outerwear' ? pickAnchorForSlot('outerwear') : undefined);
    // Force macroCategory to 'outerwear' so OutfitSlotGrid.classifyItem never drops it
    let slotOuterwear = slotOuterwearRaw ? { ...slotOuterwearRaw, macroCategory: 'outerwear' as const } : undefined;

    // ── Universal formal+shorts guard (ALL styles, not just formal ones) ──
    // "Swimming trunks" evaded detection for so long that a belt-and-braces
    // repair belongs here: a suit jacket / blazer must never share an outfit
    // with shorts or swim bottoms. Prefer swapping the pants; if the pool
    // has no real trousers, drop the formal layer instead of showing the
    // absurd combination.
    if (pantsIsShorts && slotOuterwear && isFormalLayer(slotOuterwear)) {
      const replacement =
        pickCandidate((c: any) => c.macroCategory === 'bottom' && hasRealImage(c) && !isShortsBottom(c), 7)
        || pickCandidate((c: any) => c.macroCategory === 'bottom' && !isShortsBottom(c), 7);
      if (replacement) {
        if (__DEV__) console.log('[normalizeTo4Slots] formal+shorts repair: pants', slotPantsRaw.name, '->', replacement.name);
        slotPants = { ...replacement, macroCategory: 'bottom' as const };
      } else {
        if (__DEV__) console.log('[normalizeTo4Slots] formal+shorts repair: dropping formal layer', slotOuterwear.name);
        slotOuterwear = undefined;
      }
    }

    const comboOk = (candidate: any, extras: any[]) =>
      rejectOutfitIncoherence([...extras, candidate].filter(Boolean), styleId, occasionId) === null;

    const slotMainTopRaw =
      pickAnchorForSlot('top')
      || pickCandidate((candidate: any) => looksLikeRealTop(candidate) && candidate.macroCategory === 'top' && hasRealImage(candidate) && styleOk(candidate) && silhouetteOk(candidate) && comboOk(candidate, [slotOuterwear, slotPants]), 3)
      || (formalOccasion ? pickCandidate((candidate: any) => looksLikeRealTop(candidate) && candidate.macroCategory === 'top' && hasRealImage(candidate) && occasionOk(candidate) && styleOk(candidate) && comboOk(candidate, [slotOuterwear, slotPants]), 3) : undefined)
      || pickCandidate((candidate: any) => looksLikeRealTop(candidate) && candidate.macroCategory === 'top' && hasRealImage(candidate) && styleOk(candidate) && comboOk(candidate, [slotOuterwear, slotPants]), 3)
      || pickCandidate((candidate: any) => looksLikeRealTop(candidate) && candidate.macroCategory === 'top' && styleOk(candidate) && comboOk(candidate, [slotOuterwear, slotPants]), 3)
      || pickCandidate((candidate: any) => looksLikeRealTop(candidate) && candidate.macroCategory !== 'bottom' && candidate.macroCategory !== 'shoes' && candidate.macroCategory !== 'outerwear' && candidate.macroCategory !== 'other' && candidate.macroCategory !== 'accessory' && !looksLikeLayerGarment(candidate) && styleOk(candidate) && comboOk(candidate, [slotOuterwear, slotPants]), 3);
    // Force macroCategory to 'top' so OutfitSlotGrid.classifyItem never drops it
    const slotMainTop = slotMainTopRaw ? { ...slotMainTopRaw, macroCategory: 'top' as const } : undefined;

    // Pick shoes BEFORE second-top so the second-top fallback can't
    // accidentally consume a shoes item whose macroCategory resolved
    // to 'other' or an unrecognized alias.
    const slotShoesRaw =
      pickAnchorForSlot('shoes')
      || (formalOccasion ? pickCandidate((candidate: any) => candidate.macroCategory === 'shoes' && hasRealImage(candidate) && occasionOk(candidate) && styleOk(candidate) && comboOk(candidate, [slotOuterwear, slotMainTop, slotPants]), 9) : undefined)
      || pickCandidate((candidate: any) => candidate.macroCategory === 'shoes' && hasRealImage(candidate) && styleOk(candidate) && comboOk(candidate, [slotOuterwear, slotMainTop, slotPants]), 9)
      || pickCandidate((candidate: any) => candidate.macroCategory === 'shoes' && hasRealImage(candidate) && styleOk(candidate) && comboOk(candidate, [slotOuterwear, slotMainTop, slotPants]), 9, true)
      || pickCandidate((candidate: any) => candidate.macroCategory === 'shoes' && styleOk(candidate) && comboOk(candidate, [slotOuterwear, slotMainTop, slotPants]), 9)
      || { ...PLACEHOLDER_SHOES };
    // Force macroCategory to 'shoes' so OutfitSlotGrid.classifyItem never drops it
    const slotShoes = { ...slotShoesRaw, macroCategory: 'shoes' as const };

    const slotSecondTopRaw = needsLayer && slotOuterwear && !(slotMainTop && !looksLikeLayerGarment(slotMainTop))
      ? (pickCandidate((candidate: any) => looksLikeRealTop(candidate) && candidate.macroCategory === 'top' && candidate.id !== slotMainTop?.id && hasRealImage(candidate) && styleOk(candidate) && silhouetteOk(candidate) && !looksLikeLayerGarment(candidate))
        || (formalOccasion ? pickCandidate((candidate: any) => looksLikeRealTop(candidate) && candidate.macroCategory === 'top' && candidate.id !== slotMainTop?.id && hasRealImage(candidate) && occasionOk(candidate) && styleOk(candidate) && !looksLikeLayerGarment(candidate)) : undefined)
        || pickCandidate((candidate: any) => looksLikeRealTop(candidate) && candidate.macroCategory === 'top' && candidate.id !== slotMainTop?.id && hasRealImage(candidate) && styleOk(candidate) && !looksLikeLayerGarment(candidate))
        || pickCandidate((candidate: any) => looksLikeRealTop(candidate) && candidate.macroCategory === 'top' && candidate.id !== slotMainTop?.id && styleOk(candidate) && !looksLikeLayerGarment(candidate)))
      : undefined;
    // Force macroCategory to 'top' so OutfitSlotGrid.classifyItem never drops it
    const slotSecondTop = slotSecondTopRaw ? { ...slotSecondTopRaw, macroCategory: 'top' as const } : undefined;

    if (__DEV__) console.log('[normalizeTo4Slots] PICKS:', {
      slotOuterwear: slotOuterwear ? { id: slotOuterwear.id, name: slotOuterwear.name, macro: slotOuterwear.macroCategory, image: (slotOuterwear.image || '').substring(0, 40) } : null,
      slotMainTop: slotMainTop ? { id: slotMainTop.id, name: slotMainTop.name, macro: slotMainTop.macroCategory, image: (slotMainTop.image || '').substring(0, 40) } : null,
      slotSecondTop: slotSecondTop ? { id: slotSecondTop.id, name: slotSecondTop.name, macro: slotSecondTop.macroCategory, image: (slotSecondTop.image || '').substring(0, 40) } : null,
      slotPants: slotPants ? { id: slotPants.id, name: slotPants.name, macro: slotPants.macroCategory, image: (slotPants.image || '').substring(0, 40) } : null,
      slotShoes: slotShoes ? { id: slotShoes.id, name: slotShoes.name, macro: slotShoes.macroCategory, image: (slotShoes.image || '').substring(0, 40) } : null,
    });

    // ── Layer-vs-base repair: never show two same-category tops ──
    // If the layer and the base share a fabric family (knit-on-knit) or the
    // same dark color with no texture contrast, swap the base for the next
    // top of a DIFFERENT fabric from the pool.
    let repairedMainTop = slotMainTop;
    if (slotOuterwear && slotMainTop && isSameLayerBaseCategory(slotOuterwear, slotMainTop)) {
      const layerFam = fabricFamily(slotOuterwear);
      const altBase =
        pickCandidate((c: any) => c.macroCategory === 'top' && String(c.id) !== String(slotOuterwear.id) && String(c.id) !== String(slotMainTop?.id) && fabricFamily(c) !== layerFam && hasRealImage(c) && styleOk(c) && silhouetteOk(c), 5)
        || pickCandidate((c: any) => c.macroCategory === 'top' && String(c.id) !== String(slotOuterwear.id) && String(c.id) !== String(slotMainTop?.id) && fabricFamily(c) !== layerFam && hasRealImage(c) && styleOk(c), 5)
        || pickCandidate((c: any) => c.macroCategory === 'top' && String(c.id) !== String(slotOuterwear.id) && String(c.id) !== String(slotMainTop?.id) && fabricFamily(c) !== layerFam && styleOk(c), 5);
      if (altBase) {
        if (__DEV__) console.log('[normalizeTo4Slots] layer/base same-category repair:', { layer: slotOuterwear.name, was: slotMainTop.name, now: altBase.name });
        repairedMainTop = { ...altBase, macroCategory: 'top' as const };
      }
    }

    // Exactly 4 items for the 2x2 grid:
    // 1. outerwear = main-top (layer)   2. baseTop = second-top (shirt/tee)
    // 3. pants                        4. shoes
    // When no outerwear exists, mainTop fills slot 1 and secondTop fills slot 2.
    const slot1 = slotOuterwear || repairedMainTop;           // main-top / layer
    const slot2 = repairedMainTop && slotOuterwear             // second-top (base shirt)
      ? (String(repairedMainTop.id) !== String(slotOuterwear.id) && !looksLikeLayerGarment(repairedMainTop) ? repairedMainTop : slotSecondTop)
      : (slotOuterwear ? slotSecondTop : slotSecondTop || repairedMainTop);
    let result = [slot1, slot2, slotPants, slotShoes].filter(Boolean) as any[];
    // Never ship two layers / the same jacket twice. Prefer a real base top
    // already in `result`; otherwise drop the duplicate so the grid cannot
    // render "TOP / TOP" as identical jackets (screenshot incident).
    if (result.length >= 2 && result[0] && result[1]) {
      const samePiece = String(result[0].id) === String(result[1].id)
        || (result[0].image && result[1].image && result[0].image === result[1].image);
      const twoLayers = looksLikeLayerGarment(result[0]) && looksLikeLayerGarment(result[1]);
      if (samePiece || twoLayers) {
        const altTop = slotSecondTop && String(slotSecondTop.id) !== String(result[0].id) && !looksLikeLayerGarment(slotSecondTop)
          ? slotSecondTop
          : repairedMainTop && String(repairedMainTop.id) !== String(result[0].id) && !looksLikeLayerGarment(repairedMainTop)
            ? repairedMainTop
            : undefined;
        if (altTop) result[1] = { ...altTop, macroCategory: 'top' as const };
        else result = [result[0], slotPants, slotShoes].filter(Boolean) as any[];
      }
    }
    // ── Post-enforce: anchor MUST be in the final outfit ────────────────
    // First-match picks above can still miss when AI order puts another item
    // of the same macro first and anchor-first was bypassed (e.g. placeholder
    // fallback). Replace the same-macro slot so every slot mapping stays
    // correct: outerwear→slot1(main-top), top→slot2(second-top),
    // bottom→slot3, shoes→slot4.
    if (anchorany && anchorMacro) {
      const hasAnchor = result.some(i => String(i.id) === String(anchorany.id));
      if (!hasAnchor) {
        const targetIdx =
          anchorMacro === 'outerwear' ? 0
          : anchorMacro === 'top' ? 1
          : anchorMacro === 'bottom' ? 2
          : anchorMacro === 'shoes' ? 3
          : result.findIndex(i => canonicalizeMacroCategory(i.macroCategory || '') === anchorMacro);
        if (targetIdx >= 0 && targetIdx < result.length) {
          result[targetIdx] = { ...anchorany, macroCategory: anchorMacro as any };
        } else {
          const fallbackIdx = result.findIndex(i => canonicalizeMacroCategory(i.macroCategory || '') === anchorMacro);
          if (fallbackIdx >= 0) result[fallbackIdx] = { ...anchorany, macroCategory: anchorMacro as any };
          else result.unshift({ ...anchorany, macroCategory: anchorMacro as any });
        }
      }
    }
    if (__DEV__) console.log('[normalizeTo4Slots] RESULT (4 items):', result.map((it: any) => ({ id: it.id, name: it.name, macro: it.macroCategory, price: it.price })));
    if (budget && outfitShopSpend(result, pickPoolFixed) > budget.amount + 0.009) {
      const clamped = result.map(withListedPrice);
      for (let guard = 0; guard < 6 && outfitShopSpend(clamped, pickPoolFixed) > budget.amount + 0.009; guard++) {
        let worstIdx = -1;
        let worst = -1;
        clamped.forEach((p: any, i: number) => {
          const pr = priceOf(p);
          if (pr > worst) { worst = pr; worstIdx = i; }
        });
        if (worstIdx < 0) break;
        const slot = String(clamped[worstIdx]?.macroCategory || '').toLowerCase();
        const used = new Set(clamped.map((p: any) => String(p.id)));
        const room = budget.amount - (outfitShopSpend(clamped, pickPoolFixed) - priceOf(clamped[worstIdx]));
        const repl = pickPoolFixed
          .filter((c: any) => canonicalizeMacroCategory(c.macroCategory || '') === slot && !used.has(String(c.id)))
          .map(withListedPrice)
          .filter((c: any) => {
            const pr = priceOf(c);
            return pr > 0 && pr <= room && styleOk(c);
          })
          .sort((a: any, b: any) => priceOf(a) - priceOf(b))[0];
        if (!repl) {
          if (slot === 'outerwear' && clamped.length > 3 && !climateNeedsLayer) {
            clamped.splice(worstIdx, 1);
            continue;
          }
          break;
        }
        clamped[worstIdx] = { ...repl, macroCategory: slot };
      }
      if (outfitShopSpend(clamped, pickPoolFixed) > budget.amount + 0.009) {
        const withoutLayer = clamped.filter((p: any) => canonicalizeMacroCategory(p.macroCategory || '') !== 'outerwear');
        if (!climateNeedsLayer && withoutLayer.length >= 3 && outfitShopSpend(withoutLayer, pickPoolFixed) <= budget.amount + 0.009) {
          if (__DEV__) console.log('[normalizeTo4Slots] budget: dropped outerwear to fit cap', budget.amount);
          return withoutLayer.map(withListedPrice);
        }
      }
      if (__DEV__) console.log('[normalizeTo4Slots] budget clamp', {
        cap: budget.amount,
        before: outfitShopSpend(result, pickPoolFixed),
        after: outfitShopSpend(clamped, pickPoolFixed),
      });
      return clamped;
    }
    return result.map(withListedPrice);
    };
    return normalizeTo4Slots;
}

