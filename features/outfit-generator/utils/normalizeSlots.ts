import { fitsBudgetAmount } from '../../../backend/supabase/functions/_shared/budgetFx';
import readinessLogger from '../../../src/utils/logger';
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
import { rejectForActivity, occasionFormality, rejectForStyleOccasion, rejectForClimate, rejectOutfitClimate, rejectOutfitIncoherence, matchesOccasionSilhouette, detectSlotMismatch, isUnderwearItem, isAccessoryPropItem, ACCESSORY_PROP_RE, SHORT_SLEEVE_RE, STRIP_SHORT_SLEEVE_RE, humanizeProductName } from './occasionRules';
import { orderSlotCandidates, selectSlotPick } from './deckDedupe';
import { parseBudgetFromText, resolveShopQuote, resolveShopPrice, outfitShopSpend, type ParsedBudget } from '../../../src/utils/outfitBudget';
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
  if (/\b(blazer|suit|coat|jacket|trench|overcoat|peacoat|bomber|puffer|vest|outerwear|parka|anorak|gilet|windbreaker)\b/.test(blob)) return 'tailored-outer';
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
    // Accessory names beat alias types: a watch row ships as upper_body/tops.
    // Clothing words still win so "belt-waist pants" stays a bottom.
    const nameBlob = `${name || ''}`.toLowerCase();
    const nameHasClothing = /\b(pant|trouser|jeans?|shorts?|skirt|shirt|t-?shirts?|tees?|polos?|blouses?|jackets?|coats?|blazers?|hoodies?|cardigans?|sweaters?|pullovers?|vest|puffers?|trench|shoes?|sneakers?|boots?|loafers?|sandals?|dresses?|sweatshirts?|overshirts?|knitwear|jumpers?|fleece|activewear|athletic)\b/.test(nameBlob);
    if (!nameHasClothing && ACCESSORY_PROP_RE.test(nameBlob)) return 'other';
    const aliasHit = canonicalizeMacroCategory(type) !== 'other'
      ? canonicalizeMacroCategory(type)
      : canonicalizeMacroCategory(category || '');
    // Explicit non-clothing alias on BOTH slots means the row was
    // DELIBERATELY dropped upstream (mapShopCatalogRow writes
    // type=category='accessories' for macroCategory 'other' so keyword
    // fallbacks can never re-derive it — md-uz-62899803: lower_body zone +
    // trousers photo + stolen "suit blazer" name came back as outerwear
    // through this name fallback). Keep 'other'. Rows with an absent slot
    // (edge items ship empty type/category) still classify from the name.
    if (
      (aliasHit === 'other' || aliasHit === 'accessory')
      && String(type || '').trim() !== ''
      && String(category || '').trim() !== ''
    ) return 'other';
    // Structured zone beats a CONFLICTING name: shop row md-uz-62899803 is
    // garment_type=lower_body with a trousers photo but a corrupted name
    // ("100% cool wool suit blazer") — name-first put pants in the LAYER
    // tile. Blazer-stored-as-"tops" rows keep working: upper-zone names only
    // REFINE the upper zone, they never conflict with it.
    const nameImpliesUpperZone = /\b(jacket|coat|blazer|hoodie|cardigan|sweater|pullover|vest|puffer|outerwear|trench|peacoat|bomber|shirt|t-?shirt|tee|polo|blouse|knit)\b/.test(nameBlob);
    const nameImpliesLowerZone = /\b(pant|trouser|jeans?|shorts?|skirt)\b/.test(nameBlob);
    if (aliasHit === 'bottom' && nameImpliesUpperZone && !nameImpliesLowerZone) return 'bottom';
    if (aliasHit === 'shoes' && nameImpliesUpperZone && !/\b(shoe|sneaker|boot|loafer|sandal|trainer|derby|moccasin)\b/.test(nameBlob)) return 'shoes';
    if (SHORT_SLEEVE_RE.test(t) && !/\b(pant|trouser|jeans?|shorts?|skirt)\b/.test(t.replace(STRIP_SHORT_SLEEVE_RE, ' '))) return 'top';
    if (t.match(/\b(jacket|coat|blazer|hoodie|cardigan|sweater|pullover|vest|puffer|outerwear|trench|peacoat|bomber)\b/)) return 'outerwear';

    // Prefer canonical aliases (upper_body / lower_body / tops / footwear / …)
    // so items coming back from shop_catalog are slotted correctly.
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
      if (__DEV__) readinessLogger.debug(String('[normalizeTo4Slots] slot repair:'), [{ id: it.id, was: it.macroCategory, now: fixed, name: (it.name || '').slice(0, 50) }]);
      return { ...it, macroCategory: fixed };
    };
    const pickPoolMapped: any[] = collapseDuplicateProductImages(
      (pickPool || []).map(repairItemMacro).filter((it: any) => !isUnderwearItem(it) && !isAccessoryPropItem(it)),
    );
    const pickPoolCdn = pickPoolMapped.filter((it: any) => isBrandCatalogImage(it.imageUrl || it.image));
    const pickPoolTrusted = pickPoolMapped.filter((it: any) => isTrustedProductImage(it.imageUrl || it.image));
    // Image quality is a PREFERENCE, not a hard filter. The CDN subset can be
    // dominated by a single macro (observed: 99 trousers / 1 outerwear / 0
    // shoes / 0 tops), which made every shoe pick miss and forced placeholder
    // "Shoes" tiles and missing top cards. Union keeps CDN-first order while
    // guaranteeing every slot macro stays reachable.
    const pickPoolFixed: any[] = ctx.currentSource === 'shop'
      ? collapseDuplicateProductImages([...pickPoolCdn, ...pickPoolTrusted, ...pickPoolMapped])
      : pickPoolMapped;
    // Zone-vs-name conflict rows: upstream inferShopRowMacro already decided
    // these must never fill a clothing slot (macroCategory 'other' with no
    // accessory-prop name — md-uz-62899803: lower_body zone + trousers photo +
    // stolen "suit blazer" name). The pool keeps them dropped, but the EDGE
    // re-derives macros from the name alone and returns macro 'outerwear'.
    // Cross-check incoming items against this set BEFORE repairItemMacro so a
    // corrupt row can never occupy the LAYER tile (matched by id, bare id —
    // edge prefixes some ids with `shop_` — and normalized name key).
    const poolConflictIds = new Set<string>();
    const poolConflictNames = new Set<string>();
    {
      const nameTotals = new Map<string, number>();
      const nameConflicts = new Map<string, number>();
      for (const r of pickPoolFixed) {
        const nk = normalizeCatalogNameKey(r?.name || '');
        const conflict = canonicalizeMacroCategory(r?.macroCategory || '') === 'other'
          && !isAccessoryPropItem(r);
        if (!conflict) continue;
        const id = String(r?.id || '');
        if (id) poolConflictIds.add(id);
        if (nk) {
          nameConflicts.set(nk, (nameConflicts.get(nk) || 0) + 1);
        }
      }
      for (const r of pickPoolFixed) {
        const nk = normalizeCatalogNameKey(r?.name || '');
        if (nk) nameTotals.set(nk, (nameTotals.get(nk) || 0) + 1);
      }
      // Name keys only when the name is conflict-ONLY: the corrupted
      // trousers row shares "100% cool wool suit blazer" with 6 legit blazer
      // rows — keying on that name would drop every real blazer too.
      for (const [nk, conf] of nameConflicts) {
        if (conf === nameTotals.get(nk)) poolConflictNames.add(nk);
      }
    }
    const requestBudgetFx = ctx.maxShopBudget?.fx;
    const rawBudget: ParsedBudget | null =
      ctx.maxShopBudget
      || parseBudgetFromText(ctx.budgetText || '', requestBudgetFx)
      || null;
    // Keep the cap's original units; price comparisons use its pinned snapshot.
    const budget: ParsedBudget | null =
      rawBudget;
    const priceOf = (it: any) => resolveShopPrice(it, pickPoolFixed, budget?.currency, budget?.fx);
    const withListedPrice = (it: any) => {
      if (!it) return it;
      const quote = resolveShopQuote(it, pickPoolFixed);
      if (!quote) return it;
      return { ...it, price: quote.price, currency: quote.currency, isShopItem: it.isShopItem ?? (ctx.currentSource === 'shop') };
    };
    // Image repair: edge/AI items frequently arrive with a legacy placeholder
    // string (`placeholder_pants`, `basic_clothing_shoes`) or an empty image
    // while the pool row for the SAME product carries a real photo. Without
    // this, the tile renders the bundled fallback asset — a real name over
    // the wrong garment photo. Resolve by id first, then by exact name.
    const isRenderableImage = (raw: any): raw is string =>
      typeof raw === 'string' && REAL_IMAGE_RE.test(raw);
    const poolImageById = new Map<string, string>();
    const poolImageByName = new Map<string, string>();
    for (const r of pickPoolFixed) {
      const raw = typeof r?.image === 'string' && r.image ? r.image : r?.imageUrl;
      if (!isRenderableImage(raw) || String(r?.id || '').startsWith('placeholder_')) continue;
      const id = String(r.id || '');
      if (id && !poolImageById.has(id)) poolImageById.set(id, raw);
      const nk = normalizeCatalogNameKey(r?.name || '');
      if (nk && !poolImageByName.has(nk)) poolImageByName.set(nk, raw);
    }
    const repairImage = (it: any) => {
      if (!it || String(it?.id || '').startsWith('placeholder_')) return it;
      const raw = typeof it.image === 'string' && it.image ? it.image : it.imageUrl;
      if (isRenderableImage(raw)) return it;
      const id = String(it?.id || '');
      const bare = id.replace(/^shop_/i, '');
      const byId = poolImageById.get(id)
        || (bare ? (poolImageById.get(bare) || poolImageById.get(`shop_${bare}`)) : undefined);
      const nk = normalizeCatalogNameKey(it?.name || '');
      const found = byId || (nk ? poolImageByName.get(nk) : undefined);
      if (!found) return it;
      return { ...it, image: found, imageUrl: found, _hasImage: true };
    };
    // Buy URL repair: edge items never carry sourceUrl and only price/image
    // were merged from the pool — tiles with a repaired price but no shopUrl
    // rendered without a Buy button. Reattach the pool row's URL by id/name.
    const poolShopUrlById = new Map<string, string>();
    const poolShopUrlByName = new Map<string, string>();
    for (const r of pickPoolFixed) {
      const url = r?.shopUrl || r?.shop_url || r?.sourceUrl || r?.source_url;
      if (typeof url !== 'string' || !url) continue;
      const id = String(r.id || '');
      if (id && !poolShopUrlById.has(id)) poolShopUrlById.set(id, url);
      const nk = normalizeCatalogNameKey(r?.name || '');
      if (nk && !poolShopUrlByName.has(nk)) poolShopUrlByName.set(nk, url);
    }
    const repairShopUrl = (it: any) => {
      if (!it) return it;
      const ownUrl = it.shopUrl || it.shop_url || it.sourceUrl || it.source_url;
      if (ownUrl) return { ...it, shopUrl: ownUrl };
      const id = String(it?.id || '');
      const bare = id.replace(/^shop_/i, '');
      const nk = normalizeCatalogNameKey(it?.name || '');
      const found = poolShopUrlById.get(id)
        || (bare ? (poolShopUrlById.get(bare) || poolShopUrlById.get(`shop_${bare}`)) : undefined)
        || (nk ? poolShopUrlByName.get(nk) : undefined);
      if (!found) return it;
      return { ...it, shopUrl: found };
    };
    const incomingItems = items;
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
      // Edge re-derives macros from names alone and will happily return a
      // pool-dropped conflict row as macro 'outerwear' (md-uz-62899803).
      // Drop it here — BEFORE repairItemMacro, which would otherwise
      // "repair" a zone-typed copy straight into the LAYER slot.
      {
        const rawId = String(raw?.id || '');
        const bareId = rawId.replace(/^shop_/i, '');
        const rawKey = normalizeCatalogNameKey(raw?.name || '');
        if ((rawId && poolConflictIds.has(rawId))
          || (bareId && poolConflictIds.has(bareId))
          || (rawKey && poolConflictNames.has(rawKey))) {
          if (__DEV__) readinessLogger.debug(String('[normalizeTo4Slots] drop zone-name conflict:'), [{ id: rawId, name: String(raw?.name || '').slice(0, 60) }]);
          return null;
        }
      }
      const it = repairItemMacro(raw);
      if (isUnderwearItem(it) || isAccessoryPropItem(it)) return null;
      const image = it.image || it.imageUrl || '';
      return repairImage(withListedPrice(repairShopUrl({
        ...it,
        image,
        macroCategory: resolveMacroCategory(it),
        _hasImage: (it as any)._hasImage ?? isTrustedProductImage(image || (it as any).imageUrl || ''),
      })));
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

    if (__DEV__) readinessLogger.debug(String('[normalizeTo4Slots] incoming items:'), [incoming.map((it: any) => ({ id: it.id, name: it.name, macro: it.macroCategory, image: (it.image || '').substring(0, 40) }))]);
    if (__DEV__) readinessLogger.debug(String('[normalizeTo4Slots] pool size:'), [pickPool.length, 'sample:', pickPool.slice(0, 5).map((it: any) => ({ id: it.id, macro: it.macroCategory }))]);
    if (__DEV__) readinessLogger.debug(String('[normalizeTo4Slots] needsLayer:'), [needsLayer, 'weather:', ctx.weather, 'anchor:', anchorMacro, ctx.baseItemId]);

    const pickedForBudget: any[] = [];
    const remainingBudget = () =>
      budget ? budget.amount - outfitShopSpend(pickedForBudget, pickPoolFixed, budget?.currency, budget?.fx) : Number.POSITIVE_INFINITY;
    const fitsBudget = (candidate: any) => {
      if (!budget) return true;
      const p = priceOf(candidate);
      if (p <= 0) return false;
      return fitsBudgetAmount(p, budget.amount) && fitsBudgetAmount(p, remainingBudget());
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
        if (rejectForActivity(candidate, occasionId) !== null) return false;
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
      // Real bottom before placeholder (4-item rule): budget relaxes first,
      // style next — same priority order as the other slots.
      || pickCandidate((candidate: any) => isWearableBottom(candidate) && hasRealImage(candidate) && styleOk(candidate), 6, true)
      || pickCandidate((candidate: any) => isWearableBottom(candidate) && styleOk(candidate), 6, true)
      || pickCandidate((candidate: any) => isWearableBottom(candidate) && hasRealImage(candidate), 6, true)
      || pickCandidate((candidate: any) => isWearableBottom(candidate), 6, true)
      || { ...PLACEHOLDER_PANTS };
    // Force macroCategory to 'bottom' so OutfitSlotGrid.classifyItem never drops it
    let slotPants = { ...slotPantsRaw, macroCategory: 'bottom' as const };
    const pantsIsShorts = isShortsBottom(slotPantsRaw);

    // 4-item rule: every card is top-layer + top + pants + shoes. The layer
    // is picked unconditionally (it used to be gated behind needsLayer, which
    // silently produced 3-tile cards). Strict style tiers first, then
    // progressively looser ones — a real layer always beats a missing tile.
    // Budget is the FIRST thing relaxed (never style): an over-cap layer
    // still shows, while a style violation judge-drops the whole card. It
    // must NOT be ignored up front — an expensive layer here made the
    // remaining budget negative and cascaded into placeholder shoes.
    const slotOuterwearRaw =
      pickAnchorForSlot('outerwear')
      || (formalOccasion ? pickCandidate((candidate: any) => candidate.macroCategory === 'outerwear' && isFormalLayer(candidate) && !isShortSleeve(candidate) && hasRealImage(candidate) && styleOk(candidate) && silhouetteOk(candidate), 0) : undefined)
      || pickCandidate((candidate: any) => candidate.macroCategory === 'outerwear' && !isShortSleeve(candidate) && hasRealImage(candidate) && !(pantsIsShorts && isFormalLayer(candidate)) && styleOk(candidate) && silhouetteOk(candidate), 0)
      || (formalOccasion ? pickCandidate((candidate: any) => candidate.macroCategory === 'outerwear' && !isShortSleeve(candidate) && hasRealImage(candidate) && !(pantsIsShorts && isFormalLayer(candidate)) && occasionOk(candidate) && styleOk(candidate), 0) : undefined)
      || pickCandidate((candidate: any) => candidate.macroCategory === 'outerwear' && !isShortSleeve(candidate) && hasRealImage(candidate) && !(pantsIsShorts && isFormalLayer(candidate)) && styleOk(candidate), 0)
      || pickCandidate((candidate: any) => candidate.macroCategory === 'outerwear' && !isShortSleeve(candidate) && !(pantsIsShorts && isFormalLayer(candidate)) && styleOk(candidate) && silhouetteOk(candidate), 0)
      || pickCandidate((candidate: any) => candidate.macroCategory === 'outerwear' && !isShortSleeve(candidate) && !(pantsIsShorts && isFormalLayer(candidate)) && styleOk(candidate), 0, true)
      || pickCandidate((candidate: any) => candidate.macroCategory === 'outerwear' && !isShortSleeve(candidate) && silhouetteOk(candidate), 0, true)
      || pickCandidate((candidate: any) => candidate.macroCategory === 'outerwear' && !isShortSleeve(candidate), 0, true)
      || pickCandidate((candidate: any) => candidate.macroCategory === 'outerwear', 0, true);
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
        if (__DEV__) readinessLogger.debug(String('[normalizeTo4Slots] formal+shorts repair: pants'), [slotPantsRaw.name, '->', replacement.name]);
        slotPants = { ...replacement, macroCategory: 'bottom' as const };
      } else {
        // 4-item rule: never drop the layer — swap the formal one for a
        // casual layer that survives with shorts, keep the original if the
        // pool has no alternative (an odd pairing beats a missing tile).
        const casualLayer =
          pickCandidate((c: any) => c.macroCategory === 'outerwear' && !isFormalLayer(c) && !isShortSleeve(c) && hasRealImage(c) && styleOk(c), 0)
          || pickCandidate((c: any) => c.macroCategory === 'outerwear' && !isFormalLayer(c) && !isShortSleeve(c), 0, true);
        if (casualLayer) {
          if (__DEV__) readinessLogger.debug(String('[normalizeTo4Slots] formal+shorts repair: layer'), [slotOuterwear.name, '->', casualLayer.name]);
          slotOuterwear = { ...casualLayer, macroCategory: 'outerwear' as const };
        } else if (__DEV__) {
          readinessLogger.debug(String('[normalizeTo4Slots] formal+shorts repair: keeping formal layer with shorts'), [slotOuterwear.name]);
        }
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
      || pickCandidate((candidate: any) => looksLikeRealTop(candidate) && candidate.macroCategory !== 'bottom' && candidate.macroCategory !== 'shoes' && candidate.macroCategory !== 'outerwear' && candidate.macroCategory !== 'other' && candidate.macroCategory !== 'accessory' && !looksLikeLayerGarment(candidate) && styleOk(candidate) && comboOk(candidate, [slotOuterwear, slotPants]), 3)
      // Backstops relax the BUDGET first — style AND coherence stay intact.
      // A coherence break here (basic tee under a suit blazer) makes EVERY
      // shoe fail comboOk (placeholder-shoes cascade), and a style break
      // judge-drops the card. Only when neither can be honored do the
      // constraints loosen, style first (product rule: style > budget).
      || pickCandidate((candidate: any) => looksLikeRealTop(candidate) && candidate.macroCategory === 'top' && hasRealImage(candidate) && styleOk(candidate) && comboOk(candidate, [slotOuterwear, slotPants]), 3, true)
      || pickCandidate((candidate: any) => candidate.macroCategory === 'top' && styleOk(candidate) && comboOk(candidate, [slotOuterwear, slotPants]), 3, true)
      || pickCandidate((candidate: any) => candidate.macroCategory === 'top' && hasRealImage(candidate) && styleOk(candidate), 3, true)
      || pickCandidate((candidate: any) => candidate.macroCategory === 'top' && styleOk(candidate), 3, true)
      || pickCandidate((candidate: any) => candidate.macroCategory === 'top' && hasRealImage(candidate) && comboOk(candidate, [slotOuterwear, slotPants]), 3, true)
      || pickCandidate((candidate: any) => candidate.macroCategory === 'top' && comboOk(candidate, [slotOuterwear, slotPants]), 3, true)
      || pickCandidate((candidate: any) => candidate.macroCategory === 'top' && hasRealImage(candidate), 3, true)
      || pickCandidate((candidate: any) => candidate.macroCategory === 'top', 3, true);
    // Force macroCategory to 'top' so OutfitSlotGrid.classifyItem never drops it
    const slotMainTop = slotMainTopRaw ? { ...slotMainTopRaw, macroCategory: 'top' as const } : undefined;

    // Pick shoes BEFORE second-top so the second-top fallback can't
    // accidentally consume a shoes item whose macroCategory resolved
    // to 'other' or an unrecognized alias.
    const slotShoesRaw =
      pickAnchorForSlot('shoes')
      // Sport occasions prefer trainers when the pool has a coherent one;
      // falls through to the generic tiers when it does not.
      || (occasionId === 'sport'
        ? pickCandidate((candidate: any) => candidate.macroCategory === 'shoes' && hasRealImage(candidate) && styleOk(candidate) && silhouetteOk(candidate) && comboOk(candidate, [slotOuterwear, slotMainTop, slotPants]), 9)
        : undefined)
      || (formalOccasion ? pickCandidate((candidate: any) => candidate.macroCategory === 'shoes' && hasRealImage(candidate) && occasionOk(candidate) && styleOk(candidate) && comboOk(candidate, [slotOuterwear, slotMainTop, slotPants]), 9) : undefined)
      || pickCandidate((candidate: any) => candidate.macroCategory === 'shoes' && hasRealImage(candidate) && styleOk(candidate) && comboOk(candidate, [slotOuterwear, slotMainTop, slotPants]), 9)
      || pickCandidate((candidate: any) => candidate.macroCategory === 'shoes' && hasRealImage(candidate) && styleOk(candidate) && comboOk(candidate, [slotOuterwear, slotMainTop, slotPants]), 9, true)
      || pickCandidate((candidate: any) => candidate.macroCategory === 'shoes' && styleOk(candidate) && comboOk(candidate, [slotOuterwear, slotMainTop, slotPants]), 9, true)
      // Budget/style backstops — a REAL shoe always beats the placeholder
      // (4-item rule). The placeholder is only reachable when the pool has
      // no shoe at all: even a judge-dropped real shoe is no worse than a
      // placeholder tile, and a coherent style-clean shoe survives.
      || pickCandidate((candidate: any) => candidate.macroCategory === 'shoes' && hasRealImage(candidate) && comboOk(candidate, [slotOuterwear, slotMainTop, slotPants]), 9, true)
      || pickCandidate((candidate: any) => candidate.macroCategory === 'shoes' && comboOk(candidate, [slotOuterwear, slotMainTop, slotPants]), 9, true)
      || pickCandidate((candidate: any) => candidate.macroCategory === 'shoes' && hasRealImage(candidate) && styleOk(candidate), 9, true)
      || pickCandidate((candidate: any) => candidate.macroCategory === 'shoes' && styleOk(candidate), 9, true)
      || pickCandidate((candidate: any) => candidate.macroCategory === 'shoes' && hasRealImage(candidate), 9, true)
      || pickCandidate((candidate: any) => candidate.macroCategory === 'shoes', 9, true)
      || { ...PLACEHOLDER_SHOES };
    // Force macroCategory to 'shoes' so OutfitSlotGrid.classifyItem never drops it
    const slotShoes = { ...slotShoesRaw, macroCategory: 'shoes' as const };
    if (__DEV__ && slotShoes.id === 'placeholder_shoes') {
      const shoesPool = pickPoolFixed.filter((c: any) => c.macroCategory === 'shoes');
      const combo = (c: any) => comboOk(c, [slotOuterwear, slotMainTop, slotPants]);
      readinessLogger.debug(String('[normalizeTo4Slots] SHOES PLACEHOLDER'), [{
        poolShoes: shoesPool.length,
        withImage: shoesPool.filter(hasRealImage).length,
        styleOk: shoesPool.filter(styleOk).length,
        comboOk: shoesPool.filter(combo).length,
        styleAndCombo: shoesPool.filter((c: any) => styleOk(c) && combo(c)).length,
        reason: rejectOutfitIncoherence(
          [slotOuterwear, slotMainTop, slotPants, shoesPool[0]].filter(Boolean),
          styleId,
          occasionId,
        ),
        outfit: [slotOuterwear, slotMainTop, slotPants].map((c: any) => String(c?.name || '?').slice(0, 40)),
        sample: shoesPool.slice(0, 8).map((c: any) => `${String(c.name).slice(0, 26)}|img=${hasRealImage(c)}|st=${styleOk(c)}|combo=${combo(c)}`),
      }]);
    }

    const slotSecondTopRaw = slotOuterwear && !(slotMainTop && !looksLikeLayerGarment(slotMainTop))
      ? (pickCandidate((candidate: any) => looksLikeRealTop(candidate) && candidate.macroCategory === 'top' && candidate.id !== slotMainTop?.id && hasRealImage(candidate) && styleOk(candidate) && silhouetteOk(candidate) && !looksLikeLayerGarment(candidate))
        || (formalOccasion ? pickCandidate((candidate: any) => looksLikeRealTop(candidate) && candidate.macroCategory === 'top' && candidate.id !== slotMainTop?.id && hasRealImage(candidate) && occasionOk(candidate) && styleOk(candidate) && !looksLikeLayerGarment(candidate)) : undefined)
        || pickCandidate((candidate: any) => looksLikeRealTop(candidate) && candidate.macroCategory === 'top' && candidate.id !== slotMainTop?.id && hasRealImage(candidate) && styleOk(candidate) && !looksLikeLayerGarment(candidate))
        || pickCandidate((candidate: any) => looksLikeRealTop(candidate) && candidate.macroCategory === 'top' && candidate.id !== slotMainTop?.id && styleOk(candidate) && !looksLikeLayerGarment(candidate)))
      : undefined;
    // Force macroCategory to 'top' so OutfitSlotGrid.classifyItem never drops it
    const slotSecondTop = slotSecondTopRaw ? { ...slotSecondTopRaw, macroCategory: 'top' as const } : undefined;

    if (__DEV__) readinessLogger.debug(String('[normalizeTo4Slots] PICKS:'), [{
      slotOuterwear: slotOuterwear ? { id: slotOuterwear.id, name: slotOuterwear.name, macro: slotOuterwear.macroCategory, image: (slotOuterwear.image || '').substring(0, 40) } : null,
      slotMainTop: slotMainTop ? { id: slotMainTop.id, name: slotMainTop.name, macro: slotMainTop.macroCategory, image: (slotMainTop.image || '').substring(0, 40) } : null,
      slotSecondTop: slotSecondTop ? { id: slotSecondTop.id, name: slotSecondTop.name, macro: slotSecondTop.macroCategory, image: (slotSecondTop.image || '').substring(0, 40) } : null,
      slotPants: slotPants ? { id: slotPants.id, name: slotPants.name, macro: slotPants.macroCategory, image: (slotPants.image || '').substring(0, 40) } : null,
      slotShoes: slotShoes ? { id: slotShoes.id, name: slotShoes.name, macro: slotShoes.macroCategory, image: (slotShoes.image || '').substring(0, 40) } : null,
    }]);

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
        if (__DEV__) readinessLogger.debug(String('[normalizeTo4Slots] layer/base same-category repair:'), [{ layer: slotOuterwear.name, was: slotMainTop.name, now: altBase.name }]);
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
            : pickCandidate((c: any) => c.macroCategory === 'top' && String(c.id) !== String(result[0].id) && !looksLikeLayerGarment(c) && hasRealImage(c), 3)
              || pickCandidate((c: any) => c.macroCategory === 'top' && String(c.id) !== String(result[0].id), 3, true);
        if (altTop) {
          result[1] = { ...altTop, macroCategory: 'top' as const };
        } else if (samePiece) {
          // The SAME item twice is invalid — drop it rather than render it
          // in two tiles. Two different layers stay: the 4-item rule wins.
          result = [result[0], slotPants, slotShoes].filter(Boolean) as any[];
        }
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
    if (__DEV__) readinessLogger.debug(String('[normalizeTo4Slots] RESULT (4 items):'), [result.map((it: any) => ({ id: it.id, name: it.name, macro: it.macroCategory, price: it.price }))]);
    if (budget && !fitsBudgetAmount(outfitShopSpend(result, pickPoolFixed, budget?.currency, budget?.fx), budget.amount)) {
      // 4-item rule: never drop a slot to fit the cap. Cascade swaps of the
      // most expensive piece for a strictly cheaper same-slot item (style-clean
      // first, any cheaper one as last resort) until under the cap or no swap
      // helps. If the catalog genuinely cannot fit the cap, keep all 4 items —
      // the card reports the honest spend.
      const clamped = result.map(withListedPrice);
      const spendOf = () => outfitShopSpend(clamped, pickPoolFixed, budget?.currency, budget?.fx);
      for (let guard = 0; guard < 12 && !fitsBudgetAmount(spendOf(), budget.amount); guard++) {
        const used = new Set(clamped.map((p: any) => String(p.id)));
        const byPrice = clamped
          .map((p: any, i: number) => ({ i, pr: priceOf(p) }))
          .sort((a: any, b: any) => b.pr - a.pr);
        let swapped = false;
        for (const { i, pr } of byPrice) {
          if (fitsBudgetAmount(spendOf(), budget.amount)) break;
          const slot = canonicalizeMacroCategory(clamped[i]?.macroCategory || '');
          const cheaperAll = pickPoolFixed
            .filter((c: any) => canonicalizeMacroCategory(c.macroCategory || '') === slot && !used.has(String(c.id)))
            .map(withListedPrice)
            .filter((c: any) => {
              const p = priceOf(c);
              return p > 0 && p < pr;
            });
          // Prefer cheaper pieces NOT already shown in this deck: without
          // this the greedy cheapest-swap converges every card (and every
          // fill attempt) to the SAME lowest-price set, making the whole
          // deck exact duplicates of card #1.
          const deckFresh = (c: any) => {
            if (!deckUsedIds || deckUsedIds.size === 0) return true;
            if (deckUsedIds.has(String(c.id || ''))) return false;
            const nk = nameKeyOf(c);
            if (nk && deckUsedIds.has(`__n:${nk}`)) return false;
            const img = productImageKey(c);
            if (img && deckUsedIds.has(`__img:${img}`)) return false;
            return true;
          };
          const freshCheaper = cheaperAll.filter(deckFresh);
          const cheaper = freshCheaper.length > 0 ? freshCheaper : cheaperAll;
          if (cheaper.length === 0) continue;
          // Style beats budget: a single style-violating piece makes the
          // style judge drop the WHOLE deck (kept 0/5 → error screen), which
          // is strictly worse than an over-cap card. Only swap in style-clean
          // candidates; if a slot has none, leave it and try the next slot.
          const styleClean = cheaper.filter((c: any) => hasRealImage(c) && styleOk(c));
          const choicePool = styleClean.length > 0 ? styleClean : cheaper.filter((c: any) => styleOk(c));
          if (choicePool.length === 0) continue;
          choicePool.sort((a: any, b: any) => priceOf(a) - priceOf(b));
          clamped[i] = { ...choicePool[0], macroCategory: String(clamped[i]?.macroCategory || slot) };
          used.add(String(choicePool[0].id));
          swapped = true;
          break;
        }
        if (!swapped) break;
      }
      if (__DEV__) readinessLogger.debug(String('[normalizeTo4Slots] budget clamp'), [{
        cap: budget.amount,
        before: outfitShopSpend(result, pickPoolFixed, budget?.currency, budget?.fx),
        after: spendOf(),
        items: clamped.length,
      }]);
      return clamped;
    }
    return result.map(withListedPrice);
    };
    return normalizeTo4Slots;
}
