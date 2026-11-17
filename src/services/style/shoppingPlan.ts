/**
 * shoppingPlan.ts — Budget, pipeline, post-mortem, shopping plan, no-buy
 * engine (Phase 3S–3W).
 *
 * Restraint-first: wishlist items are never spent money; the planner must
 * be able to return buy-nothing. Attaches to Phase 2 BuyCheck dimensions
 * via caller-supplied verdicts — never re-implements BuyCheck.
 */

import type {
  NoBuyDecision,
  NoBuyRecommendation,
  PipelineEntry,
  PipelineStage,
  PlannedPurchase,
  PurchasePostMortem,
  ShoppingBudget,
  ShoppingPlan,
  WardrobeGap,
} from '../../types/styleOS';

const STAGE_ORDER: PipelineStage[] = [
  'discovered', 'considering', 'compared', 'waiting_for_price',
  'planned', 'bought', 'skipped', 'returned', 'archived',
];

let pipeSeq = 0;

/** Advance (or regress) a pipeline entry; terminal states are explicit. */
export function transitionPipeline(
  entry: PipelineEntry,
  to: PipelineStage,
  reason?: string,
  now = Date.now(),
): PipelineEntry {
  void now;
  const fromIdx = STAGE_ORDER.indexOf(entry.stage);
  const toIdx = STAGE_ORDER.indexOf(to);
  if (toIdx < 0) return entry;
  // bought/skipped/returned/archived are sticky outcomes — require explicit reopen.
  if (['bought', 'skipped', 'returned'].includes(entry.stage) && !['archived', 'considering'].includes(to)) {
    return entry;
  }
  void fromIdx;
  return {
    ...entry,
    stage: to,
    decisionReason: reason ?? entry.decisionReason,
    updatedAt: new Date(now).toISOString(),
  };
}

export function createPipelineEntry(input: {
  userId: string;
  productId: string;
  targetPrice?: number | null;
  currency?: string | null;
  evidence?: string[];
}): PipelineEntry {
  return {
    id: `pipe_${Date.now().toString(36)}_${(pipeSeq++).toString(36)}`,
    userId: input.userId,
    productId: input.productId,
    stage: 'discovered',
    decisionReason: null,
    targetPrice: input.targetPrice ?? null,
    currency: input.currency ?? null,
    evidence: input.evidence || [],
    updatedAt: new Date().toISOString(),
  };
}

// ---------- Budget (3S) ----------

export function createShoppingBudget(input: {
  userId: string;
  period: string;
  amount: number;
  currency: string;
  categoryAllocations?: Record<string, number>;
  flexibility?: ShoppingBudget['flexibility'];
}): ShoppingBudget {
  return {
    id: `sbud_${Date.now().toString(36)}`,
    userId: input.userId,
    period: input.period,
    amount: input.amount,
    spent: 0,
    committed: 0,
    available: input.amount,
    currency: input.currency,
    categoryAllocations: input.categoryAllocations || {},
    flexibility: input.flexibility || 'flexible',
  };
}

/** Wishlist entries are NOT spent money — only bought counts as spent. */
export function applyBudgetOutcomes(
  budget: ShoppingBudget,
  pipeline: readonly PipelineEntry[],
  prices: Record<string, number> = {},
): ShoppingBudget {
  let spent = 0;
  let committed = 0;
  for (const p of pipeline) {
    const price = prices[p.productId] || 0;
    if (p.stage === 'bought') spent += price;
    else if (p.stage === 'planned') committed += price;
  }
  return { ...budget, spent, committed, available: Math.max(0, budget.amount - spent - committed) };
}

// ---------- Post-mortem (3U) ----------

export function postMortemDue(
  boughtAt: string,
  wearCount: number,
  answered: boolean,
  minDays = 30,
  now = Date.now(),
): boolean {
  if (answered) return false;
  if (!Number.isFinite(Date.parse(boughtAt))) return false;
  void wearCount;
  return now - Date.parse(boughtAt) >= minDays * 86_400_000;
}

export function recordPostMortem(
  productId: string,
  userId: string,
  answers: { worthIt?: boolean | null; fitsWell?: boolean | null; wearCount: number; wouldBuyAgain?: boolean | null },
  now = Date.now(),
): PurchasePostMortem {
  const calibration: string[] = [];
  if (answers.worthIt === false) calibration.push('Reduce BuyCheck optimism for similar price/utility');
  if (answers.fitsWell === false) calibration.push('Lower fit confidence prior for this brand/category');
  if (answers.wearCount <= 1) calibration.push('Flag low-utility pattern for duplicate-role buys');
  if (answers.wouldBuyAgain === true) calibration.push('Reinforce brand/category prior');
  return {
    productId,
    userId,
    worthIt: answers.worthIt ?? null,
    fitsWell: answers.fitsWell ?? null,
    wearCount: answers.wearCount,
    wouldBuyAgain: answers.wouldBuyAgain ?? null,
    askedAt: new Date(now).toISOString(),
    answeredAt: new Date(now).toISOString(),
    calibration,
  };
}

// ---------- Shopping plan (3V) ----------

export function buildShoppingPlan(
  userId: string,
  gaps: readonly WardrobeGap[],
  budget: ShoppingBudget | null,
  noBuy: readonly NoBuyDecision[],
  limit = 5,
  now = Date.now(),
): ShoppingPlan {
  const blocked = new Set(
    noBuy.filter((n) => n.recommendation === 'SKIP_DUPLICATE' || n.recommendation === 'NO_PURCHASE_NEEDED')
      .map((n) => n.productId || ''),
  );
  const items: PlannedPurchase[] = [];
  for (const gap of gaps.slice(0, 20)) {
    if (items.length >= limit) break;
    if (blocked.has(`gap-candidate:${gap.id}`)) continue;
    if (gap.confidence === 'low' && gap.urgency === 'low') continue;
    items.push({
      productId: `gap-candidate:${gap.id}`,
      label: gap.label,
      priority: items.length + 1,
      expectedUnlocks: gap.expectedUnlocks,
      confidence: gap.confidence,
      evidence: [
        `Gap blocks ${gap.blockedContexts.join(', ')}`,
        `Expected unlocks: ${gap.expectedUnlocks}`,
        ...(budget ? [`Budget available: ${budget.available} ${budget.currency}`] : ['No budget set — confirm affordability']),
      ],
      tradeoffs: [
        'Confirm fit evidence before buying',
        'Check owned substitutes in the same role first',
      ],
    });
  }
  const buyNothing = items.length === 0;
  return {
    userId,
    computedAt: new Date(now).toISOString(),
    items,
    buyNothing,
    rationale: buyNothing
      ? ['No high-confidence gaps with budget headroom — buy nothing right now', 'Wear what you already own']
      : [`${items.length} prioritized purchase(s); wardrobe-first order preserved`],
  };
}

// ---------- No-buy engine (3W) ----------

export interface NoBuyInput {
  productId?: string | null;
  duplicateLevel?: 'low' | 'medium' | 'high';
  unlocks?: number;
  withinBudget?: boolean;
  fitConfidence?: 'low' | 'medium' | 'high';
  priceState?: 'FAIR' | 'HIGH_PRICE' | 'WAIT' | 'INSUFFICIENT_HISTORY' | string;
  ownedCount?: number;
  unwornOwned?: number;
}

export function recommendNoBuy(input: NoBuyInput): NoBuyDecision {
  if ((input.ownedCount || 0) >= 4 || input.duplicateLevel === 'high') {
    return {
      recommendation: 'SKIP_DUPLICATE',
      productId: input.productId ?? null,
      confidence: 'high',
      evidence: [`${input.ownedCount || 'Several'} owned in role; duplicate level ${input.duplicateLevel || 'high'}`],
      explanation: 'Strong owned substitutes — skip this duplicate and wear what you own.',
    };
  }
  if (input.withinBudget === false) {
    return {
      recommendation: 'WAIT_FOR_PRICE',
      productId: input.productId ?? null,
      confidence: 'high',
      evidence: ['Outside budget'],
      explanation: 'Over budget — wait for a price drop or skip.',
    };
  }
  if (input.priceState === 'HIGH_PRICE' || input.priceState === 'WAIT') {
    return {
      recommendation: 'WAIT_FOR_PRICE',
      productId: input.productId ?? null,
      confidence: 'medium',
      evidence: [`Price state ${input.priceState}`],
      explanation: 'Price is high relative to history — waiting is the better move.',
    };
  }
  if (input.fitConfidence === 'low') {
    return {
      recommendation: 'WAIT_FOR_BETTER_FIT_EVIDENCE',
      productId: input.productId ?? null,
      confidence: 'medium',
      evidence: ['Fit confidence low'],
      explanation: 'Fit evidence is too weak — wait until size confidence improves.',
    };
  }
  if ((input.unwornOwned || 0) >= 2) {
    return {
      recommendation: 'NO_PURCHASE_NEEDED',
      productId: input.productId ?? null,
      confidence: 'medium',
      evidence: [`${input.unwornOwned} unworn owned in category`],
      explanation: 'Buy nothing — unworn owned pieces cover this need.',
    };
  }
  if ((input.unlocks || 0) === 0) {
    return {
      recommendation: 'SKIP_LOW_VALUE',
      productId: input.productId ?? null,
      confidence: 'medium',
      evidence: ['No validated incremental outfit value'],
      explanation: 'Adds no new outfits — skip and restyle owned pieces.',
    };
  }
  const rec: NoBuyRecommendation = (input.unlocks || 0) >= 3 ? 'BUY_NOW' : 'BUY_IF_NEEDED';
  return {
    recommendation: rec,
    productId: input.productId ?? null,
    confidence: rec === 'BUY_NOW' ? 'medium' : 'low',
    evidence: [`${input.unlocks || 0} expected unlocks`, `Fit confidence ${input.fitConfidence || 'unknown'}`],
    explanation: rec === 'BUY_NOW' ? 'Genuine improvement — proceed if budget allows.' : 'Only if a real gap is confirmed.',
  };
}
