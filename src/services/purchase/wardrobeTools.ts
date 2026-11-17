import { catalogTitleColor } from '../../utils/catalogIdentity';
import type { ShopCatalogItem } from '../../types/shop';
import type {
  PurchaseContext,
  PurchaseAlternative,
  FindBetterMode,
  BuyCheckResult,
} from '../../types/purchase';
import { rankDiverseShoppingCandidates } from '../shoppingIntelligence';
import { buyCheck, duplicateRisk, usefulUnlocksV2, colorFamily } from './buyCheck';
import { purchaseItemEligible, purchaseLookValid } from '../shoppingAdvisor';
import { visualFamilyIdentityOf, normalizeIdentityText } from '../../utils/catalogIdentity';
import { inferShopRowMacro } from '../../utils/shopCatalogOrder';
import {
  inferItemAttributes,
  scoreItemForStyle,
  normalizeStyleId,
} from '../../../features/outfit-generator/utils/styleInference';
import { convertShopAmount } from '../../utils/outfitBudget';
export interface ToolContext extends PurchaseContext {
  lockedIds?: readonly string[];
  sourceMode?: 'wardrobe' | 'shop' | 'mixed';
  selectedSizes?: Record<string, string>;
}
const quote = (item: ShopCatalogItem, currency: string, context: PurchaseContext) =>
  item.price > 0 && item.currency
    ? convertShopAmount(item.price, item.currency, currency, context.budget?.fx)
    : Infinity;
const wordSimilarity = (a: ShopCatalogItem, b: ShopCatalogItem) => {
  const x = new Set(normalizeIdentityText(a.name).split(' ')),
    y = new Set(normalizeIdentityText(b.name).split(' '));
  return [...x].filter((w) => y.has(w)).length / Math.max(1, new Set([...x, ...y]).size);
};
export function findBetterV2(
  reference: ShopCatalogItem,
  candidates: readonly ShopCatalogItem[],
  owned: readonly ShopCatalogItem[],
  mode: FindBetterMode,
  context: ToolContext = {}
): PurchaseAlternative[] {
  if (context.sourceMode === 'wardrobe') return [];
  const baseline = buyCheck(reference, owned, context),
    currency = context.budget?.currency || reference.currency || '';
  const slot = inferShopRowMacro(reference.garmentType, reference.name);
  const eligible = rankDiverseShoppingCandidates(
    candidates
      .filter(
        (item) =>
          item.id !== reference.id &&
          inferShopRowMacro(item.garmentType, item.name) === slot &&
          purchaseItemEligible(item, context)
      )
      .map((item) => ({
        item,
        score: 0.9 + 0.1 * scoreItemForStyle(item, normalizeStyleId(context.style || 'casual')),
      })),
    { limit: 16, eligible: () => true }
  );
  const results = eligible.map((item) => ({
    item,
    mode,
    analysis: buyCheck(item, owned, context),
    why: [] as string[],
    tradeoffs: [] as string[],
  }));
  const improvement = (r: PurchaseAlternative) => {
    const a = r.analysis,
      b = baseline,
      saving = quote(reference, currency, context) - quote(r.item, currency, context);
    if (mode === 'cheaper')
      return Number.isFinite(saving) && saving > 0
        ? saving / Math.max(1, quote(reference, currency, context))
        : -Infinity;
    if (mode === 'closest-style')
      return (
        wordSimilarity(reference, r.item) +
        (visualFamilyIdentityOf(reference).visualFamilyKey ===
        visualFamilyIdentityOf(r.item).visualFamilyKey
          ? 1
          : 0)
      );
    if (mode === 'fit') {
      if (!a.fit.recommendedSize || a.fit.confidence === 'low') return -Infinity;
      const difference =
        { high: 3, medium: 2, low: 0 }[a.fit.confidence] -
        { high: 3, medium: 2, low: 0 }[b.fit.confidence];
      return difference || (Number.isFinite(saving) && saving > 0 ? 0.25 : 0);
    }
    if (mode === 'wardrobe')
      return a.wardrobe.usefulOutfitUnlocks.total - b.wardrobe.usefulOutfitUnlocks.total;
    if (mode === 'lowest-duplicate')
      return (
        { low: 2, medium: 1, high: 0 }[a.wardrobe.duplicateRisk.level] -
        { low: 2, medium: 1, high: 0 }[b.wardrobe.duplicateRisk.level]
      );
    if (mode === 'missing-piece')
      return a.wardrobe.missingPieceValue === 'high'
        ? a.wardrobe.usefulOutfitUnlocks.total + 1
        : -Infinity;
    const overall = (value: BuyCheckResult) =>
      ({ BUY: 3, CONSIDER: 2, SKIP: 0, INSUFFICIENT_EVIDENCE: 0 })[value.verdict] * 10 +
      value.wardrobe.usefulOutfitUnlocks.total +
      { high: 2, medium: 1, low: 0 }[value.fit.confidence] -
      { low: 0, medium: 1, high: 3 }[value.wardrobe.duplicateRisk.level];
    return (
      overall(a) -
      overall(b) +
      (Number.isFinite(saving) && saving > 0
        ? Math.min(0.5, saving / Math.max(1, quote(reference, currency, context)))
        : 0)
    );
  };
  return results
    .filter(
      (r) =>
        r.analysis.verdict !== 'SKIP' &&
        Number.isFinite(improvement(r)) &&
        (!['overall', 'fit', 'wardrobe', 'lowest-duplicate'].includes(mode) || improvement(r) > 0)
    )
    .sort((a, b) => improvement(b) - improvement(a) || a.item.id.localeCompare(b.item.id))
    .slice(0, 6)
    .map((result) => {
      const saving = quote(reference, currency, context) - quote(result.item, currency, context);
      result.why = [
        mode === 'cheaper'
          ? 'Lower observed item price'
          : mode === 'fit'
            ? `Fit confidence ${result.analysis.fit.confidence}; ${result.analysis.fit.evidenceCount} relevant observations`
            : mode === 'wardrobe'
              ? `${result.analysis.wardrobe.usefulOutfitUnlocks.total - baseline.wardrobe.usefulOutfitUnlocks.total} more validated sampled looks`
              : mode === 'lowest-duplicate'
                ? `Duplicate risk ${result.analysis.wardrobe.duplicateRisk.level} versus ${baseline.wardrobe.duplicateRisk.level}`
                : mode === 'missing-piece'
                  ? `Addresses an eligible ${slot} gap`
                  : mode === 'closest-style'
                    ? 'Closest observed garment-role and style match among eligible alternatives'
                    : 'Stronger combined purchase evidence; inspect each dimension below',
        ...(Number.isFinite(saving)
          ? [`Observed item-price saving: ${saving.toFixed(2)} ${currency}`]
          : ['Prices are not directly comparable']),
        `${result.analysis.wardrobe.usefulOutfitUnlocks.total} validated sampled looks`,
      ];
      result.tradeoffs = [
        ...(result.analysis.fit.confidence === 'low' ? ['Fit evidence is limited'] : []),
        ...({ high: 2, medium: 1, low: 0 }[result.analysis.fit.confidence] <
        { high: 2, medium: 1, low: 0 }[baseline.fit.confidence]
          ? [
              `Fit confidence is lower (${result.analysis.fit.confidence} vs ${baseline.fit.confidence})`,
            ]
          : []),
        ...(Number.isFinite(saving) && saving < 0
          ? [`Costs ${(-saving).toFixed(2)} ${currency} more before checkout`]
          : []),
        ...(visualFamilyIdentityOf(reference).visualFamilyKey !==
        visualFamilyIdentityOf(result.item).visualFamilyKey
          ? ['Different garment archetype']
          : []),
        ...(result.analysis.wardrobe.usefulOutfitUnlocks.total <
        baseline.wardrobe.usefulOutfitUnlocks.total
          ? ['Lower sampled wardrobe gain']
          : []),
      ];
      return result;
    });
}
export function maximumWardrobeGain(
  candidates: readonly ShopCatalogItem[],
  owned: readonly ShopCatalogItem[],
  context: ToolContext
) {
  if (!context.budget || context.sourceMode === 'wardrobe')
    return {
      candidates: [],
      evaluated: 0,
      truncated: false,
      evidence: ['A budget and shopping mode are required'],
    };
  const eligible = rankDiverseShoppingCandidates(
    candidates
      .filter((item) => purchaseItemEligible(item, context))
      .map((item) => ({
        item,
        score: 0.9 + 0.1 * scoreItemForStyle(item, normalizeStyleId(context.style || 'casual')),
      })),
    { limit: 16, eligible: () => true }
  );
  const ranked = eligible
    .map((item) => buyCheck(item, owned, context))
    .filter(
      (r) =>
        r.purchase.withinBudget === true &&
        r.verdict !== 'SKIP' &&
        r.wardrobe.usefulOutfitUnlocks.total > 0
    )
    .sort((a, b) => {
      const gain = (r: BuyCheckResult) =>
        r.wardrobe.usefulOutfitUnlocks.total +
        { high: 2, medium: 1, low: 0 }[r.wardrobe.missingPieceValue] -
        { high: 3, medium: 1, low: 0 }[r.wardrobe.duplicateRisk.level] -
        (r.fit.confidence === 'low' ? 0.5 : 0);
      return (
        gain(b) - gain(a) ||
        quote(a.candidate, context.budget!.currency, context) -
          quote(b.candidate, context.budget!.currency, context)
      );
    });
  return {
    candidates: ranked.slice(0, 6),
    evaluated: eligible.length,
    truncated: candidates.length > 16,
    evidence: [
      'Incremental validated sampled gains, gap value, duplicates, fit uncertainty and native budget considered; not a financial ROI estimate',
    ],
  };
}
export function sameLookLessMoney(
  outfit: readonly ShopCatalogItem[],
  candidates: readonly ShopCatalogItem[],
  owned: readonly ShopCatalogItem[],
  context: ToolContext = {},
  target?: { budget?: number; savings?: number }
) {
  const currency = context.budget?.currency || outfit.find((item) => item.currency)?.currency || '',
    ownedIds = new Set(owned.map((item) => item.id));
  const spend = (items: readonly ShopCatalogItem[]) =>
    items.reduce((n, item) => n + (ownedIds.has(item.id) ? 0 : quote(item, currency, context)), 0);
  const originalTotal = spend(outfit);
  const empty = {
    original: [...outfit],
    items: [...outfit],
    originalTotal,
    total: originalTotal,
    savings: 0,
    currency,
    replacements: [] as Array<{
      from: string;
      to: string;
      similarity: number;
      fitConfidence: string;
    }>,
    confidence: 'low' as 'low' | 'medium',
    pieceRoles: outfit.map((item) => {
      const attributes = inferItemAttributes(item),
        silhouette = /\b(slim|oversized|relaxed|wide)\b/i.test(item.name);
      const critical =
        context.lockedIds?.includes(item.id) ||
        attributes.patterns.length > 0 ||
        silhouette ||
        (!!catalogTitleColor(item) && !attributes.isNeutralPalette);
      return {
        id: item.id,
        role: critical ? 'style-critical' : 'basic',
        evidence: context.lockedIds?.includes(item.id)
          ? 'Pinned by you'
          : attributes.patterns.length
            ? 'Distinctive observed pattern'
            : silhouette
              ? 'Named silhouette'
              : critical
                ? 'Accent color'
                : 'Neutral basic with no distinctive pattern/cut evidence',
      };
    }),
    meetsTarget: false,
    evidence: ['No validated cheaper substitution found; original look retained'] as string[],
  };
  if (
    context.sourceMode === 'wardrobe' ||
    (context.sourceMode === 'shop' && outfit.some((item) => ownedIds.has(item.id))) ||
    !Number.isFinite(originalTotal) ||
    !purchaseLookValid(outfit, ownedIds, { ...context, budget: undefined })
  )
    return { ...empty, evidence: ['No comparable valid purchasable outfit supplied'] };
  let best = empty;
  const options: Array<{ items: ShopCatalogItem[]; replacements: typeof empty.replacements }> = [];
  for (let i = 0; i < outfit.length; i++) {
    const current = outfit[i];
    if (context.lockedIds?.includes(current.id) || ownedIds.has(current.id)) continue;
    const attributes = inferItemAttributes(current);
    const alternatives = candidates
      .filter(
        (item) =>
          item.id !== current.id &&
          visualFamilyIdentityOf(item).visualFamilyKey ===
            visualFamilyIdentityOf(current).visualFamilyKey &&
          colorFamily(item) === colorFamily(current) &&
          Math.abs(inferItemAttributes(item).formality - attributes.formality) <= 0.15 &&
          quote(item, currency, context) < quote(current, currency, context) &&
          purchaseItemEligible(item, context)
      )
      .slice(0, 6);
    for (const alternative of alternatives) {
      const other = inferItemAttributes(alternative);
      if (
        attributes.materials.length &&
        other.materials.length &&
        !attributes.materials.some((material) => other.materials.includes(material))
      )
        continue;
      const silhouette = (item: ShopCatalogItem) =>
        normalizeIdentityText(item.name)
          .split(' ')
          .filter((word) => ['slim', 'relaxed', 'oversized', 'wide', 'straight'].includes(word))
          .sort()
          .join(' ');
      if (silhouette(current) && silhouette(alternative) !== silhouette(current)) continue;
      if (
        context.selectedSizes?.[current.id] &&
        alternative.availableSizes &&
        !alternative.availableSizes.includes(context.selectedSizes?.[current.id])
      )
        continue;
      const fit = buyCheck(alternative, owned, context).fit,
        oldFit = buyCheck(current, owned, context).fit;
      if (oldFit.confidence === 'high' && fit.confidence === 'low') continue;
      const similarity = wordSimilarity(current, alternative);
      if (similarity < 0.15) continue;
      if (
        context.selectedSizes?.[current.id] &&
        fit.confidence === 'high' &&
        fit.recommendedSize &&
        fit.recommendedSize !== context.selectedSizes?.[current.id]
      )
        continue;
      const replacement = {
        from: current.id,
        to: alternative.id,
        similarity,
        fitConfidence: fit.confidence,
      };
      const changed = outfit.map((piece, index) => (index === i ? alternative : piece));
      if (purchaseLookValid(changed, ownedIds, { ...context, budget: undefined }))
        options.push({ items: changed, replacements: [replacement] });
    }
  }
  // Prefer a single substitution. Only explore a second when the explicit target
  // cannot be reached by one; never replace an anchor or every piece blindly.
  const meets = (amount: number) =>
    amount < originalTotal &&
    (target?.budget === undefined || amount <= target.budget) &&
    (target?.savings === undefined || originalTotal - amount >= target.savings);
  if (target && !options.some((option) => meets(spend(option.items))))
    for (const option of [...options].slice(0, 12))
      for (const second of [...options].slice(0, 12)) {
        if (option.replacements[0].from === second.replacements[0].from) continue;
        const changed = option.items.map((item) =>
          item.id === second.replacements[0].from
            ? second.items.find((piece) => piece.id === second.replacements[0].to)!
            : item
        );
        if (purchaseLookValid(changed, ownedIds, context))
          options.push({
            items: changed,
            replacements: [...option.replacements, ...second.replacements],
          });
      }
  const validOptions = options.filter(
    (option) =>
      purchaseLookValid(option.items, ownedIds, context) &&
      (target?.budget === undefined || spend(option.items) <= target.budget)
  );
  validOptions.sort(
    (a, b) =>
      Number(meets(spend(b.items))) - Number(meets(spend(a.items))) ||
      a.replacements.length - b.replacements.length ||
      spend(a.items) - spend(b.items)
  );
  const winner = validOptions[0];
  if (winner) {
    const total = spend(winner.items);
    best = {
      ...empty,
      ...winner,
      total,
      savings: originalTotal - total,
      meetsTarget: meets(total),
      confidence: winner.replacements.every((replacement) => replacement.fitConfidence !== 'low')
        ? 'medium'
        : 'low',
      evidence: [
        'Existing whole-look rules, color family, archetype, formality and budget passed',
        'Similarity is based on observed text/attributes, not image or quality prediction',
        ...(winner.replacements.some((replacement) => replacement.fitConfidence === 'low')
          ? ['Fit evidence is limited; selected sizes need confirmation']
          : []),
        `Changed ${winner.replacements.length} piece(s); shipping/tax not included`,
      ],
    };
  }
  return best;
}
export function upgradeThisLook(
  outfit: readonly ShopCatalogItem[],
  candidates: readonly ShopCatalogItem[],
  owned: readonly ShopCatalogItem[],
  context: ToolContext = {}
) {
  if (context.sourceMode === 'wardrobe') return [];
  const ids = new Set(owned.map((item) => item.id)),
    options: Array<{
      from: ShopCatalogItem;
      to: ShopCatalogItem;
      priceDelta?: number;
      analysis: BuyCheckResult;
      evidence: string[];
      leverage: number;
    }> = [];
  for (const from of outfit.slice(0, 4)) {
    if (context.lockedIds?.includes(from.id)) continue;
    const before = buyCheck(from, owned, context),
      baseStyle = scoreItemForStyle(from, normalizeStyleId(context.style || 'casual'));
    for (const to of candidates
      .filter(
        (item) =>
          inferShopRowMacro(item.garmentType, item.name) ===
            inferShopRowMacro(from.garmentType, from.name) &&
          item.id !== from.id &&
          purchaseItemEligible(item, context)
      )
      .slice(0, 4)) {
      if (
        !purchaseLookValid(
          outfit.map((item) => (item.id === from.id ? to : item)),
          ids,
          context
        )
      )
        continue;
      const after = buyCheck(to, owned, context),
        styleGain = scoreItemForStyle(to, normalizeStyleId(context.style || 'casual')) - baseStyle;
      const fitGain =
        { high: 2, medium: 1, low: 0 }[after.fit.confidence] -
        { high: 2, medium: 1, low: 0 }[before.fit.confidence];
      const futureGain =
        after.wardrobe.usefulOutfitUnlocks.total - before.wardrobe.usefulOutfitUnlocks.total;
      if (styleGain <= 0 && fitGain <= 0 && futureGain <= 0) continue;
      const currency = context.budget?.currency || from.currency || '',
        delta =
          quote(to, currency, context) - (ids.has(from.id) ? 0 : quote(from, currency, context));
      options.push({
        from,
        to,
        analysis: after,
        priceDelta: Number.isFinite(delta) ? delta : undefined,
        leverage: futureGain + fitGain * 2 + styleGain * 4,
        evidence: [
          styleGain > 0
            ? `Stronger match for your stated ${context.style || 'casual'} style`
            : 'Similar stated-style compatibility',
          `Fit confidence ${after.fit.confidence} versus ${before.fit.confidence}`,
          `Validated sampled future-look change ${futureGain}`,
        ],
      });
    }
  }
  return options
    .sort(
      (a, b) => b.leverage - a.leverage || (a.priceDelta ?? Infinity) - (b.priceDelta ?? Infinity)
    )
    .slice(0, 3);
}
export function detectMissingPieces(
  owned: readonly ShopCatalogItem[],
  candidates: readonly ShopCatalogItem[],
  context: ToolContext = {}
) {
  const slots = ['outerwear', 'top', 'bottom', 'shoes'];
  const countFor = (request: PurchaseContext) =>
    Object.fromEntries(
      slots.map((slot) => [
        slot,
        owned.filter(
          (item) =>
            inferShopRowMacro(item.garmentType, item.name) === slot &&
            purchaseItemEligible(item, request, true)
        ).length,
      ])
    );
  const counts = countFor(context);
  const uses = context.contexts?.length
    ? context.contexts.slice(0, 4)
    : [{ id: context.occasion || 'Everyday', occasion: context.occasion || 'Everyday' }];
  const contextCoverage = uses.map((use) => ({
    id: use.id,
    counts: countFor({ ...context, ...use }),
  }));
  const missingFrequency = Object.fromEntries(
    slots.map((slot) => [
      slot,
      contextCoverage.filter((coverage) => coverage.counts[slot] === 0).length,
    ])
  );
  const bottlenecks = slots.filter((slot) => missingFrequency[slot] > 0);
  const suggestions = candidates
    .filter(
      (item) =>
        bottlenecks.includes(inferShopRowMacro(item.garmentType, item.name)) &&
        uses.some((use) => purchaseItemEligible(item, { ...context, ...use }))
    )
    .slice(0, 12)
    .map((item) => ({
      item,
      unlocks: usefulUnlocksV2(item, owned, context),
      duplicates: duplicateRisk(item, owned),
    }))
    .sort(
      (a, b) =>
        b.unlocks.total - a.unlocks.total ||
        a.duplicates.substitutes.length - b.duplicates.substitutes.length
    );
  return {
    eligibleCounts: counts,
    contextCoverage,
    missingFrequency,
    bottlenecks,
    suggestions: suggestions.slice(0, 5),
    evidence: bottlenecks.length
      ? [`Missing eligible slots: ${bottlenecks.join(', ')}`]
      : ['No empty required slot; compare incremental gains rather than buying more duplicates'],
  };
}
