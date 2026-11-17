/**
 * budgets.test.ts — latency budgets, cost caps, cache policy, rate
 * limits, entitlements, stale-data labels.
 */
import {
  latencyBudget, withinBudget, costPolicy, consumeCall, resetDailyUse, setCostPolicy,
  cacheDescriptor, isExpired, rateLimitCheck, resetRateLimits,
  canUse, freshnessOf, staleLabel,
} from '../../src/services/platform/budgets';

beforeEach(() => { resetDailyUse(); resetRateLimits(); });

test('latency budgets flag p50/p95 breaches without hardcoding promises', () => {
  expect(latencyBudget('vton').maxMs).toBe(60000);
  expect(withinBudget('instant-ui', 100, 200).ok).toBe(true);
  const bad = withinBudget('instant-ui', 500, 200);
  expect(bad.ok).toBe(false);
  expect(bad.violations.join(' ')).toMatch(/p50/);
});

test('cost policy caps daily calls and gates premium', () => {
  const day = '2026-11-04';
  expect(costPolicy('tryon').maxCalls).toBe(1);
  expect(costPolicy('tryon').estimatedUnitCost).toBeNull(); // no invented pricing
  for (let i = 0; i < 10; i++) consumeCall('tryon', day, true);
  expect(consumeCall('tryon', day, true)).toBe(false); // daily cap hit -> fallback
  expect(consumeCall('tryon', day, false)).toBe(false); // premium gate
  setCostPolicy({ ...costPolicy('tryon'), maxDailyPerUser: 99 });
  expect(consumeCall('tryon', day, true)).toBe(true);
});

test('cache descriptors bound size with TTL and version', () => {
  const d = cacheDescriptor('price-history')!;
  expect(d.ttlMs).toBe(6 * 3600000);
  expect(d.sizeBound).toBeLessThanOrEqual(500);
  expect(isExpired(Date.now() - 1000, 60000)).toBe(false);
  expect(isExpired(Date.now() - 70000, 60000)).toBe(true);
  expect(cacheDescriptor('nope')).toBeNull();
});

test('rate limits differ per feature; entitlements keep core free', () => {
  for (let i = 0; i < 3; i++) expect(rateLimitCheck('vton')).toBe(true);
  expect(rateLimitCheck('vton')).toBe(false);
  expect(rateLimitCheck('buycheck')).toBe(true);
  expect(canUse('buycheck', { premium: false, quotas: {} }).allowed).toBe(true);
  expect(canUse('vton', { premium: false, quotas: {} }).allowed).toBe(false);
  expect(canUse('vton', { premium: true, quotas: { vton: 0 } }).reason).toMatch(/quota/);
  expect(canUse('vton', { premium: true, quotas: { vton: 2 } })).toEqual({ allowed: true, reason: '2 remaining' });
});

test('stale price is labeled, never presented as live', () => {
  expect(freshnessOf('price')).toBe(6 * 3600000);
  expect(staleLabel('price', Date.now())).toBeNull();
  expect(staleLabel('price', Date.now() - 7 * 3600000)!).toMatch(/stale/i);
});
