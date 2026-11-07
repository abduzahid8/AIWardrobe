/**
 * analytics.ts — Product funnels, north-star metrics, retention signals,
 * outcome attribution, quality monitoring (5Z–5AB, 5BJ).
 *
 * Value metrics over vanity metrics: acceptance, repeats, avoided
 * duplicates, no-buy adherence. Attribution uses confidence + time windows
 * and never claims causality from correlation.
 */
export interface FunnelStep { event: string; count: number }

export function funnelConversion(steps: readonly FunnelStep[]): Array<{ event: string; count: number; rateFromFirst: number; rateFromPrev: number }> {
  const first = steps[0]?.count || 1;
  return steps.map((s, i) => ({
    event: s.event, count: s.count, rateFromFirst: s.count / first,
    rateFromPrev: i === 0 ? 1 : s.count / Math.max(1, steps[i - 1].count),
  }));
}

export const FUNNELS: Record<string, string[]> = {
  closetOnboarding: ['started', 'first_item', 'fifth_item', 'first_complete_outfit'],
  outfit: ['generated', 'viewed', 'saved', 'worn', 'disliked'],
  buycheck: ['started', 'completed', 'verdict', 'alternative_opened', 'purchase_confirmed'],
  recreate: ['reference_added', 'result_generated', 'saved', 'missing_piece_opened'],
  fit: ['recommendation_viewed', 'feedback_given', 'correction'],
  proactive: ['surfaced', 'opened', 'dismissed', 'acted_on'],
};

export function funnelSteps(name: string): string[] { return FUNNELS[name] || []; }

// ---------- North star + retention (5AA–5AB) ----------

export interface ValueMetrics {
  validatedOutfitsUsed: number;
  wardrobeItemsWorn: number;
  successfulRepeats: number;
  unlocksRealized: number;
  avoidedDuplicates: number;
  noBuyAccepted: number;
  purchaseSatisfaction: number;
  fitPredictionSuccess: number;
  closetActivation: number;
  weeklyPlanningCompletion: number;
}

export function valueScore(m: ValueMetrics): { score: number; breakdown: Record<string, number> } {
  const breakdown: Record<string, number> = {
    usage: m.validatedOutfitsUsed + m.wardrobeItemsWorn + m.successfulRepeats,
    restraint: m.avoidedDuplicates + m.noBuyAccepted,
    satisfaction: m.purchaseSatisfaction + m.fitPredictionSuccess,
    activation: m.closetActivation + m.weeklyPlanningCompletion + m.unlocksRealized,
  };
  return { score: Object.values(breakdown).reduce((a, b) => a + b, 0), breakdown };
}

export const RETENTION_SIGNALS = [
  'plan_outfit_return', 'log_wear', 'save_reference', 'check_purchase',
  'wardrobe_over_shopping', 'recommendation_worked',
] as const;

// ---------- Attribution (5AC) ----------

export interface AttributedOutcome {
  recommendationId: string;
  verdict: string;
  outcome: string;
  daysToOutcome: number;
  confidence: 'low' | 'medium' | 'high';
  note: string;
}

export function attributeOutcome(input: {
  recommendationId: string; verdict: string; outcome: string; daysToOutcome: number;
}): AttributedOutcome {
  const withinWindow = input.daysToOutcome <= 90;
  return {
    ...input,
    confidence: withinWindow && input.outcome !== 'unknown' ? 'medium' : 'low',
    note: withinWindow
      ? 'Associated within 90-day window; correlation only, not proven causality.'
      : 'Outside attribution window; treat as unrelated.',
  };
}

// ---------- Quality monitoring (5BJ) ----------

export interface QualityAggregate {
  hardRuleViolations: number;
  lowConfidenceRate: number;
  repeatRate: number;
  duplicateRate: number;
  correctionRate: number;
  fallbackRate: number;
  noBuyFrequency: number;
  acceptanceRate: number;
  total: number;
}

export function qualityReport(events: ReadonlyArray<{
  hardViolation: boolean; lowConfidence: boolean; repeated: boolean;
  duplicate: boolean; corrected: boolean; fallback: boolean; noBuy: boolean; accepted: boolean;
}>): QualityAggregate {
  const n = Math.max(1, events.length);
  const r = (f: (e: (typeof events)[number]) => boolean) => events.filter(f).length / n;
  return {
    hardRuleViolations: events.filter((e) => e.hardViolation).length,
    lowConfidenceRate: r((e) => e.lowConfidence), repeatRate: r((e) => e.repeated),
    duplicateRate: r((e) => e.duplicate), correctionRate: r((e) => e.corrected),
    fallbackRate: r((e) => e.fallback), noBuyFrequency: r((e) => e.noBuy),
    acceptanceRate: r((e) => e.accepted), total: events.length,
  };
}

// ---------- Data quality (5BI) ----------

export function dataQualityReport(products: ReadonlyArray<{
  id: string; imageUrl?: string | null; price?: number | null; currency?: string | null;
  provenance?: string | null; sizes?: unknown;
}>): string[] {
  const issues: string[] = [];
  const ids = new Set<string>();
  for (const p of products.slice(0, 2000)) {
    if (!p.imageUrl) issues.push(`${p.id}: missing image`);
    if (p.price !== null && p.price !== undefined && p.price <= 0) issues.push(`${p.id}: impossible price`);
    if (p.price && !p.currency) issues.push(`${p.id}: missing currency`);
    if (ids.has(p.id)) issues.push(`${p.id}: duplicate listing id`);
    ids.add(p.id);
    if (!p.provenance) issues.push(`${p.id}: missing provenance`);
  }
  return issues.slice(0, 100);
}
