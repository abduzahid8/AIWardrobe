import { productIdentityOf } from '../../backend/supabase/functions/_shared/outfitDiversity';
import type { ShopCatalogItem } from '../types/shop';
import type { BodyProfile } from '../types/bodyProfile';
import type { ParsedBudget } from '../utils/outfitBudget';
import { convertShopAmount } from '../utils/outfitBudget';
import { assessFit } from '../lib/fit/fitEngine';
import { resolveStylistWeather } from '../../backend/supabase/functions/_shared/outfitComposition';
import { fitsBudgetAmount } from '../../backend/supabase/functions/_shared/budgetFx';
import {
  rejectForStyleOccasion,
  rejectForClimate,
  rejectForActivity,
  rejectForRequestIntent,
  styleJudgeReason,
} from '../../backend/supabase/functions/_shared/occasionRules';
import {
  catalogTitleColor,
  catalogColorEvidence,
  normalizeIdentityText,
} from '../utils/catalogIdentity';
import { inferShopRowMacro, inferCatalogColor } from '../utils/shopCatalogOrder';
import {
  normalizeStyleId,
  scoreItemForStyle,
  inferItemAttributes,
} from '../../features/outfit-generator/utils/styleInference';
import {
  createFixtureCatalogRepository,
  createSupabaseCatalogRepository,
  type CatalogRepository,
} from './catalogRepository';
import {
  rankShoppingAlternatives,
  estimateUsefulOutfitUnlocks,
  evaluateWardrobeValue,
  type AlternativeGoal,
} from './shoppingEvaluation';

export interface ShoppingAdviceContext {
  style?: string;
  occasion?: string;
  prompt?: string;
  weather?: { temp: number; condition?: string };
  avoidedColors?: readonly string[];
  avoidedPatterns?: readonly string[];
  avoidedBrands?: readonly string[];
  budget?: ParsedBudget;
  bodyProfile?: BodyProfile;
}
function judgeItem(item: ShopCatalogItem) {
  return {
    ...item,
    macroCategory: inferShopRowMacro(item.garmentType, item.name),
    color: catalogTitleColor(item) || inferCatalogColor(item.name, item.description),
  };
}
function eligible(item: ShopCatalogItem, context: ShoppingAdviceContext, owned = false) {
  const row = judgeItem(item),
    style = normalizeStyleId(context.style || 'casual');
  const intent = [context.occasion, context.prompt].filter(Boolean).join(' ');
  const weather = resolveStylistWeather(context.prompt, context.weather);
  const colors = catalogColorEvidence(item);
  if (
    context.avoidedBrands?.some(
      (brand) => normalizeIdentityText(brand) === normalizeIdentityText(item.brand)
    )
  )
    return false;
  const patterns = inferItemAttributes(item).patterns;
  if (
    context.avoidedPatterns?.some((pattern) =>
      patterns.some(
        (value) => value.includes(pattern.toLowerCase()) || pattern.toLowerCase().includes(value)
      )
    )
  )
    return false;
  if (
    context.avoidedColors?.some((color) => {
      const normalized =
        catalogTitleColor({ ...item, colorName: color }) || normalizeIdentityText(color);
      return colors.includes(normalized) || (!colors.length && normalized === row.color);
    })
  )
    return false;
  if (
    rejectForStyleOccasion(row, style, context.occasion) ||
    rejectForActivity(row, intent) ||
    rejectForClimate(row, weather) ||
    rejectForRequestIntent(row, intent, weather)
  )
    return false;
  if (
    !owned &&
    context.budget &&
    (!Number.isFinite(item.price) || item.price <= 0 || !item.currency?.trim())
  )
    return false;
  if (
    !owned &&
    context.budget &&
    !fitsBudgetAmount(
      convertShopAmount(item.price, item.currency, context.budget.currency, context.budget.fx),
      context.budget.amount
    )
  )
    return false;
  return true;
}
/** Fit is evaluated only against supplied product measurements. No seed size chart,
 * brand guess or inferred body measurement is substituted for missing data. */
export function shoppingFitEvidence(item: ShopCatalogItem, body?: BodyProfile) {
  if (!body || body.status !== 'ready' || !item.physicalProfiles?.length) return undefined;
  const assessed = item.physicalProfiles
    .filter((profile) => profile.garmentId === item.id)
    .map((profile) => assessFit(body, profile))
    .filter((assessment) => assessment.overall !== 'unknown' && assessment.confidence !== 'low');
  const score = (overall: string) =>
    overall === 'good_fit'
      ? 1
      : overall === 'relaxed'
        ? 0.8
        : overall === 'tight' || overall === 'oversized'
          ? 0.4
          : 0.1;
  const best = assessed.sort((a, b) => score(b.overall) - score(a.overall))[0];
  return best
    ? {
        score: score(best.overall),
        confidence: best.confidence === 'high' ? 0.9 : 0.7,
        evidence: [
          `${best.engineVersion}: ${best.selectedSize} ${best.overall} (${best.confidence})`,
        ],
      }
    : undefined;
}
export function evaluateShoppingPurchase(
  candidate: ShopCatalogItem,
  owned: readonly ShopCatalogItem[],
  context: ShoppingAdviceContext = {}
) {
  const ownedIds = new Set(owned.map((item) => item.id));
  const contextId = JSON.stringify([context.style, context.occasion, context.weather]);
  const unlocks = estimateUsefulOutfitUnlocks(candidate, owned, {
    contexts: [{ id: contextId, occasion: context.occasion || 'Everyday' }],
    eligible: (item) => eligible(item, context, ownedIds.has(item.id)),
    validateOutfit: (items) => {
      if (new Set(items.map((item) => inferShopRowMacro(item.garmentType, item.name))).size !== 4)
        return false;
      if (
        styleJudgeReason(
          items.map(judgeItem),
          normalizeStyleId(context.style || 'casual'),
          context.occasion,
          resolveStylistWeather(context.prompt, context.weather)
        )
      )
        return false;
      if (context.budget) {
        const spend = items
          .filter((item) => !ownedIds.has(item.id))
          .reduce(
            (sum, item) =>
              sum +
              convertShopAmount(
                item.price,
                item.currency,
                context.budget!.currency,
                context.budget!.fx
              ),
            0
          );
        if (!fitsBudgetAmount(spend, context.budget.amount)) return false;
      }
      return true;
    },
  });
  return {
    candidate,
    unlocks,
    wardrobeValue: evaluateWardrobeValue(candidate, owned, unlocks),
    fit: shoppingFitEvidence(candidate, context.bodyProfile),
  };
}
/** Repository injection makes offline/fixture behavior explicit. Live callers can
 * supply the existing canonical shopping repository, including its lazy fallback. */
export async function findBetterShoppingProducts(
  reference: ShopCatalogItem,
  owned: readonly ShopCatalogItem[],
  options: ShoppingAdviceContext & {
    goal: AlternativeGoal;
    repository?: CatalogRepository;
    limit?: number;
    seed?: string;
  }
) {
  const repository =
    options.repository || createSupabaseCatalogRepository(createFixtureCatalogRepository(() => []));
  const slot = inferShopRowMacro(reference.garmentType, reference.name);
  const page = await repository.search({
    query: '',
    category: slot === 'top' ? 'tops' : slot === 'bottom' ? 'bottoms' : slot,
    limit: 100,
    seed: options.seed,
    ...(options.goal === 'cheaper' && reference.currency
      ? { currency: reference.currency, priceMax: reference.price }
      : {}),
    ...(options.budget && !options.budget.fx
      ? { currency: options.budget.currency, priceMax: options.budget.amount }
      : {}),
  });
  const evaluations = new Map<string, ReturnType<typeof evaluateShoppingPurchase>>();
  const wardrobeSample = new Set(
    page.items
      .filter((item) => eligible(item, options))
      .sort(
        (a, b) =>
          scoreItemForStyle(judgeItem(b), normalizeStyleId(options.style || 'casual')) -
          scoreItemForStyle(judgeItem(a), normalizeStyleId(options.style || 'casual'))
      )
      .slice(0, 24)
      .map((item) => item.id)
  );
  return {
    origin: page.origin,
    alternatives: rankShoppingAlternatives(reference, page.items, {
      goal: options.goal,
      limit: options.limit,
      eligible: (item) => eligible(item, options),
      signals: (item) => {
        if (
          (options.goal === 'wardrobe' || options.goal === 'overall') &&
          wardrobeSample.has(item.id) &&
          !evaluations.has(item.id)
        )
          evaluations.set(item.id, evaluateShoppingPurchase(item, owned, options));
        const evaluation = evaluations.get(item.id);
        return {
          wardrobeGain: evaluation?.unlocks.outfits.length,
          fit: evaluation?.fit || shoppingFitEvidence(item, options.bodyProfile),
          relevance: scoreItemForStyle(
            judgeItem(item),
            normalizeStyleId(options.style || 'casual')
          ),
        };
      },
    }),
  };
}

/** Reusable deterministic gates for purchase tools; existing generation behavior is unchanged. */
export { eligible as purchaseItemEligible, judgeItem as purchaseJudgeItem };
export function purchaseLookValid(
  items: readonly ShopCatalogItem[],
  ownedIds: ReadonlySet<string>,
  context: ShoppingAdviceContext
): boolean {
  if (
    items.length !== 4 ||
    new Set(items.map((item) => inferShopRowMacro(item.garmentType, item.name))).size !== 4
  )
    return false;
  const identities = new Set<string>();
  for (const item of items) {
    if (['dresses', 'outfit', 'accessory'].includes(item.garmentType)) return false;
    const keys = productIdentityOf(item).keys;
    if (keys.some((key) => identities.has(key))) return false;
    keys.forEach((key) => identities.add(key));
  }
  if (items.some((item) => !eligible(item, context, ownedIds.has(item.id)))) return false;
  if (
    styleJudgeReason(
      items.map(judgeItem),
      normalizeStyleId(context.style || 'casual'),
      context.occasion,
      resolveStylistWeather(context.prompt, context.weather)
    )
  )
    return false;
  if (context.budget) {
    const total = items
      .filter((item) => !ownedIds.has(item.id))
      .reduce(
        (n, item) =>
          n +
          convertShopAmount(
            item.price,
            item.currency,
            context.budget!.currency,
            context.budget!.fx
          ),
        0
      );
    if (!fitsBudgetAmount(total, context.budget.amount)) return false;
  }
  return true;
}
