import type { ShopCatalogItem } from '../../types/shop';
import type {
  BuyCheckResult,
  PurchaseContext,
  DuplicateAssessment,
  UsefulUnlocksV2,
  MerchantOffer,
  PriceObservation,
} from '../../types/purchase';
import { inferItemAttributes } from '../../../features/outfit-generator/utils/styleInference';
import {
  catalogTitleColor,
  normalizeIdentityText,
  visualFamilyIdentityOf,
  physicalProductIdentityOf,
} from '../../utils/catalogIdentity';
import { inferShopRowMacro } from '../../utils/shopCatalogOrder';
import { estimateUsefulOutfitUnlocks } from '../shoppingEvaluation';
import { purchaseItemEligible, purchaseLookValid } from '../shoppingAdvisor';
import { convertShopAmount } from '../../utils/outfitBudget';
import { fitsBudgetAmount } from '../../../backend/supabase/functions/_shared/budgetFx';
import { recommendPurchaseSize } from './fitEvidence';
import { assessPrice, compareMerchantOffers, truePrice } from './priceIntelligence';
export function colorFamily(item: ShopCatalogItem): string | undefined {
  const color = catalogTitleColor(item);
  if (!color) return;
  if (['black', 'navy', 'grey', 'charcoal'].includes(color)) return 'dark-neutral';
  if (['white', 'cream', 'ecru'].includes(color)) return 'light-neutral';
  if (['brown', 'beige', 'tan', 'olive'].includes(color)) return 'earth';
  return color;
}
const cut = (item: ShopCatalogItem) =>
  normalizeIdentityText(item.name)
    .split(' ')
    .filter((t) => ['slim', 'regular', 'oversized', 'relaxed', 'wide', 'straight'].includes(t))
    .sort()
    .join(' ');
export function duplicateRisk(
  candidate: ShopCatalogItem,
  owned: readonly ShopCatalogItem[]
): DuplicateAssessment {
  const a = inferItemAttributes(candidate),
    role = visualFamilyIdentityOf(candidate).visualFamilyKey,
    identity = physicalProductIdentityOf(candidate);
  const substitutes = owned
    .slice(0, 300)
    .flatMap((item) => {
      if (
        inferShopRowMacro(item.garmentType, item.name) !==
        inferShopRowMacro(candidate.garmentType, candidate.name)
      )
        return [];
      const b = inferItemAttributes(item),
        evidence: string[] = [];
      let similarity = 0;
      const otherIdentity = physicalProductIdentityOf(item);
      const exact =
        (identity.idKey && identity.idKey === otherIdentity.idKey) ||
        (identity.nameKey &&
          identity.nameKey === otherIdentity.nameKey &&
          identity.imageKey &&
          identity.imageKey === otherIdentity.imageKey &&
          normalizeIdentityText(candidate.brand) === normalizeIdentityText(item.brand) &&
          (!colorFamily(candidate) ||
            !colorFamily(item) ||
            colorFamily(candidate) === colorFamily(item)));
      if (exact) {
        similarity = 1;
        evidence.push('Same existing physical product');
      } else {
        if (role === visualFamilyIdentityOf(item).visualFamilyKey && !role.includes('other')) {
          similarity += 0.4;
          evidence.push('Same garment role');
        }
        if (colorFamily(candidate) && colorFamily(candidate) === colorFamily(item)) {
          similarity += 0.2;
          evidence.push('Same color family');
        }
        if (a.materials.length && a.materials.some((m) => b.materials.includes(m))) {
          similarity += 0.1;
          evidence.push('Shared observed material');
        }
        if (cut(candidate) && cut(candidate) === cut(item)) {
          similarity += 0.1;
          evidence.push('Same named silhouette');
        }
        if (Math.abs(a.formality - b.formality) < 0.15) {
          similarity += 0.1;
          evidence.push('Similar inferred formality');
        }
        if (a.styleTags.some((tag) => b.styleTags.includes(tag))) {
          similarity += 0.1;
          evidence.push('Shared inferred style role');
        }
      }
      return similarity >= 0.55 ? [{ item, similarity: Math.min(1, similarity), evidence }] : [];
    })
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, 5);
  const best = substitutes[0]?.similarity || 0,
    level = best >= 0.8 ? 'high' : best >= 0.55 ? 'medium' : 'low';
  return {
    level,
    substitutes,
    confidence: best >= 0.95 ? 'high' : substitutes.length ? 'medium' : 'low',
    explanation: substitutes.length
      ? `${substitutes.length} owned substitute${substitutes.length === 1 ? '' : 's'} share this wardrobe role; owning multiples can be intentional`
      : 'No strong owned substitute found in the supplied wardrobe',
  };
}
export function usefulUnlocksV2(
  candidate: ShopCatalogItem,
  owned: readonly ShopCatalogItem[],
  context: PurchaseContext = {}
): UsefulUnlocksV2 {
  const ids = new Set(owned.map((item) => item.id)),
    contexts = context.contexts?.length
      ? context.contexts.slice(0, 4)
      : [
          {
            id: context.occasion || 'Everyday',
            occasion: context.occasion || 'Everyday',
            style: context.style,
            weather: context.weather,
          },
        ];
  const result: UsefulUnlocksV2 = {
    total: 0,
    looks: [],
    contextBreakdown: {},
    evaluations: 0,
    truncated: (context.contexts?.length || 0) > 4,
    confidence: owned.length < 3 ? 'insufficient' : 'sampled',
    evidence: [],
  };
  const seen = new Set<string>(),
    contextSeen = new Set<string>();
  for (const use of contexts) {
    const request = { ...context, ...use },
      slot = inferShopRowMacro(candidate.garmentType, candidate.name);
    const sampled = estimateUsefulOutfitUnlocks(candidate, owned.slice(0, 300), {
      contexts: [{ id: use.id, occasion: use.occasion }],
      maxEvaluations: 64,
      maxResults: 10,
      maxPerSlot: 4,
      eligible: (item) => purchaseItemEligible(item, request, ids.has(item.id)),
      validateOutfit: (items) => purchaseLookValid(items, ids, request),
    });
    result.evaluations += sampled.evaluations;
    result.truncated ||= sampled.truncated;
    for (const look of sampled.outfits) {
      const equivalents = owned.filter(
        (item) =>
          inferShopRowMacro(item.garmentType, item.name) === slot &&
          visualFamilyIdentityOf(item).visualFamilyKey ===
            visualFamilyIdentityOf(candidate).visualFamilyKey &&
          colorFamily(item) === colorFamily(candidate) &&
          cut(item) === cut(candidate) &&
          inferItemAttributes(item).materials.join('|') ===
            inferItemAttributes(candidate).materials.join('|')
      );
      if (equivalents.length > 4) {
        result.truncated = true;
        continue;
      }
      let oldLook = false;
      for (const substitute of equivalents) {
        result.evaluations++;
        if (
          purchaseLookValid(
            look.items.map((item) => (item.id === candidate.id ? substitute : item)),
            ids,
            request
          )
        ) {
          oldLook = true;
          break;
        }
      }
      const signature = `${look.items.map((item) => `${visualFamilyIdentityOf(item).visualFamilyKey}:${colorFamily(item) || item.id}:${cut(item)}`).join('|')}`;
      if (oldLook) continue;
      const contextual = `${use.id}:${signature}`;
      if (!contextSeen.has(contextual)) {
        contextSeen.add(contextual);
        result.contextBreakdown[use.id] = (result.contextBreakdown[use.id] || 0) + 1;
      }
      if (seen.has(signature)) continue;
      seen.add(signature);
      result.total++;
      if (result.looks.length < 6) result.looks.push({ items: look.items, context: use.id });
    }
  }
  result.evidence = [
    `${result.total} meaningfully distinct validated sampled looks`,
    `${result.evaluations} bounded evaluations`,
    ...(result.truncated ? ['Search was capped; this is not an exhaustive total'] : []),
  ];
  return result;
}
export function buyCheck(
  candidate: ShopCatalogItem,
  owned: readonly ShopCatalogItem[],
  context: PurchaseContext = {},
  market: { offers?: MerchantOffer[]; history?: PriceObservation[] } = {}
): BuyCheckResult {
  const rawOffers = market.offers?.filter((offer) => offer.variantId === candidate.id);
  const offers = rawOffers?.[0]
    ? compareMerchantOffers(rawOffers[0], rawOffers, {
        size: context.selectedSize,
        currency: context.budget?.currency || candidate.currency,
        fx: context.budget?.fx,
      }).map((result) => result.offer)
    : rawOffers;
  const chosen = offers?.[0],
    price = chosen ? assessPrice(chosen, market.history || []) : undefined,
    checkout = chosen ? truePrice(chosen) : undefined;
  const pricedCandidate = chosen
    ? { ...candidate, price: chosen.currentPrice, currency: chosen.currency }
    : candidate;
  let garmentContext = context;
  if (context.budget && checkout && chosen) {
    const extras = checkout.knownSubtotal - chosen.currentPrice;
    const converted =
      extras === 0
        ? 0
        : Math.sign(extras) *
          convertShopAmount(
            Math.abs(extras),
            chosen.currency,
            context.budget.currency,
            context.budget.fx
          );
    garmentContext = {
      ...context,
      budget: {
        ...context.budget,
        amount: Number.isFinite(converted) ? context.budget.amount - converted : -1,
      },
    };
  }
  const fit = recommendPurchaseSize(candidate, context.fitProfile, context.bodyProfile),
    duplicates = duplicateRisk(candidate, owned),
    unlocks = usefulUnlocksV2(pricedCandidate, owned, garmentContext);
  const missing = !owned.some(
    (item) =>
      inferShopRowMacro(item.garmentType, item.name) ===
      inferShopRowMacro(candidate.garmentType, candidate.name)
  );
  const wardrobeValue = unlocks.total >= 4 ? 'High' : unlocks.total >= 1 ? 'Medium' : 'Low';
  const budgetBasis = checkout?.total !== undefined ? ('checkout' as const) : ('item' as const);
  const amount = checkout?.total ?? pricedCandidate.price,
    quoteCurrency = checkout?.currency || pricedCandidate.currency;
  const validAmount =
    checkout?.total !== undefined ? amount >= 0 && (chosen?.currentPrice || 0) > 0 : amount > 0;
  let withinBudget = context.budget
    ? validAmount &&
      !!quoteCurrency &&
      fitsBudgetAmount(
        convertShopAmount(amount, quoteCurrency, context.budget.currency, context.budget.fx),
        context.budget.amount
      )
    : undefined;
  if (
    context.budget &&
    checkout &&
    !fitsBudgetAmount(
      convertShopAmount(
        checkout.knownSubtotal,
        checkout.currency,
        context.budget.currency,
        context.budget.fx
      ),
      context.budget.amount
    )
  )
    withinBudget = false;
  const notes: string[] = [],
    evidence = [
      ...fit.notes,
      ...duplicates.substitutes.flatMap((s) => s.evidence),
      ...unlocks.evidence,
      ...(price?.evidence || ['Price history is unknown']),
    ];
  if (context.budget && budgetBasis === 'item')
    notes.push('Only the item-price budget is confirmed; checkout extras remain unknown');
  let verdict: BuyCheckResult['verdict'] = 'CONSIDER';
  if (withinBudget === false) {
    verdict = 'SKIP';
    notes.push('The observed item price does not fit your budget or cannot be converted safely');
  } else if (
    !candidate.id ||
    !candidate.name ||
    (owned.length < 3 &&
      fit.confidence === 'low' &&
      (!price || price.state === 'INSUFFICIENT_HISTORY'))
  ) {
    verdict = 'INSUFFICIENT_EVIDENCE';
    notes.push('More product, fit or wardrobe evidence is needed');
  } else if (!purchaseItemEligible(candidate, { ...context, budget: undefined })) {
    verdict = 'SKIP';
    notes.push(
      'This product conflicts with the supplied style, occasion, activity, climate or color constraints'
    );
  } else if (duplicates.level === 'high' && unlocks.total <= 1 && !context.allowMultiples) {
    verdict = 'SKIP';
    notes.push('Strong owned substitutes and little validated incremental value');
  } else if (unlocks.total >= 3 && fit.confidence !== 'low' && duplicates.level === 'low') {
    verdict = 'BUY';
    notes.push('Relevant fit evidence and several validated new looks support this purchase');
  } else {
    notes.push(
      fit.confidence === 'low'
        ? 'Fit evidence is limited'
        : 'Relevant brand/category fit evidence is available'
    );
    notes.push(
      unlocks.total
        ? `${unlocks.total} validated sampled new looks`
        : 'No new complete looks were validated in the bounded sample'
    );
  }
  if (context.budget && budgetBasis === 'item' && verdict === 'BUY') {
    verdict = 'CONSIDER';
    notes.push(
      'Confirm shipping, tax and required fees before treating this as within the full purchase budget'
    );
  }
  if (context.budget && checkout?.confidence === 'low' && verdict === 'BUY') {
    verdict = 'CONSIDER';
    notes.push(
      'Checkout price evidence is weak; verify the merchant total before committing the budget'
    );
  }
  if (context.selectedSize && fit.recommendedSize && context.selectedSize !== fit.recommendedSize) {
    if (verdict === 'BUY') verdict = 'CONSIDER';
    notes.push(
      `Selected ${context.selectedSize}; evidence instead supports ${fit.recommendedSize}. Selection was not changed`
    );
  }
  if (price?.quoteFresh === false) {
    if (verdict === 'BUY') verdict = 'CONSIDER';
    notes.push('Verify the current merchant price; the supplied quote is stale or undated');
  }
  if (price && ['HIGH_PRICE', 'WAIT'].includes(price.state)) {
    if (verdict === 'BUY') verdict = 'CONSIDER';
    notes.push(
      'The current price is high relative to observed history; compare offers or consider waiting without assuming a future sale'
    );
  }
  if (
    offers?.length &&
    offers.every(
      (offer) =>
        (context.selectedSize ? offer.sizes?.[context.selectedSize] : offer.inStock) === false
    )
  ) {
    if (verdict === 'BUY') verdict = 'CONSIDER';
    notes.push(
      'The selected variant/size is currently reported out of stock in the supplied offers'
    );
  }
  return {
    verdict,
    overallConfidence:
      verdict === 'INSUFFICIENT_EVIDENCE'
        ? 'low'
        : fit.confidence === 'high' && unlocks.confidence === 'sampled'
          ? 'medium'
          : 'low',
    candidate,
    selectedVariantId: candidate.id,
    fit,
    wardrobe: {
      usefulOutfitUnlocks: unlocks,
      duplicateRisk: duplicates,
      wardrobeValue,
      missingPieceValue: missing && unlocks.total ? 'high' : missing ? 'medium' : 'low',
    },
    purchase: {
      withinBudget,
      budgetBasis,
      priceAssessment: price,
      offers: offers?.[0]
        ? compareMerchantOffers(offers[0], offers, {
            size: context.selectedSize,
            fx: context.budget?.fx,
          }).map(({ offer, truePrice }) => ({ offer, truePrice }))
        : undefined,
    },
    explanation: notes,
    evidence: [...new Set(evidence)],
  };
}
