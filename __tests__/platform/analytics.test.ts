/**
 * analytics.test.ts — funnels, value metrics, attribution windows,
 * quality and data-quality monitoring.
 */
import {
  funnelConversion, funnelSteps, valueScore, RETENTION_SIGNALS,
  attributeOutcome, qualityReport, dataQualityReport,
} from '../../src/services/platform/analytics';

test('funnel conversion tracks stepwise rates', () => {
  const f = funnelConversion([
    { event: 'generated', count: 100 }, { event: 'viewed', count: 60 }, { event: 'saved', count: 30 },
  ]);
  expect(f[2].rateFromFirst).toBe(0.3);
  expect(f[2].rateFromPrev).toBe(0.5);
  expect(funnelSteps('buycheck')).toContain('purchase_confirmed');
});

test('value score rewards restraint and usage, not vanity', () => {
  const { score, breakdown } = valueScore({
    validatedOutfitsUsed: 5, wardrobeItemsWorn: 10, successfulRepeats: 3,
    unlocksRealized: 2, avoidedDuplicates: 4, noBuyAccepted: 2,
    purchaseSatisfaction: 4, fitPredictionSuccess: 3, closetActivation: 1, weeklyPlanningCompletion: 2,
  });
  expect(score).toBe(36);
  expect(breakdown.restraint).toBe(6);
  expect(RETENTION_SIGNALS).toContain('wardrobe_over_shopping');
});

test('attribution uses windows and disclaims causality', () => {
  const a = attributeOutcome({ recommendationId: 'r1', verdict: 'BUY', outcome: 'worn-8x', daysToOutcome: 40 });
  expect(a.confidence).toBe('medium');
  expect(a.note).toMatch(/correlation only/);
  const stale = attributeOutcome({ recommendationId: 'r2', verdict: 'BUY', outcome: 'worn-1x', daysToOutcome: 400 });
  expect(stale.confidence).toBe('low');
  expect(stale.note).toMatch(/unrelated/);
});

test('quality aggregates surface violations and fallback load', () => {
  const q = qualityReport([
    { hardViolation: true, lowConfidence: true, repeated: false, duplicate: false, corrected: true, fallback: true, noBuy: false, accepted: false },
    { hardViolation: false, lowConfidence: false, repeated: true, duplicate: false, corrected: false, fallback: false, noBuy: true, accepted: true },
  ]);
  expect(q.hardRuleViolations).toBe(1);
  expect(q.total).toBe(2);
  expect(q.noBuyFrequency).toBe(0.5);
  expect(q.acceptanceRate).toBe(0.5);
});

test('data quality flags missing/stale/invalid catalog facts', () => {
  const issues = dataQualityReport([
    { id: 'p1' }, { id: 'p2', price: 0, currency: 'USD', imageUrl: 'u', provenance: 'm' },
    { id: 'p2', price: 10, currency: 'USD', imageUrl: 'u', provenance: 'm' },
    { id: 'p3', price: 10, imageUrl: 'u', provenance: 'm' },
  ]);
  expect(issues.join('|')).toMatch(/missing image/);
  expect(issues.join('|')).toMatch(/impossible price/);
  expect(issues.join('|')).toMatch(/duplicate listing/);
  expect(issues.join('|')).toMatch(/missing currency/);
});
