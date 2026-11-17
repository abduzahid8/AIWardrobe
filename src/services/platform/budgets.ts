/**
 * budgets.ts — Latency budgets, cost budgets, cache policy, rate limits,
 * entitlements, stale-data policy (5Q–5S, 5AJ–5AK, 5AV).
 *
 * No invented provider pricing: estimatedUnitCost stays null unless the
 * caller configures it. Budgets bound calls/images/resolution/tokens and
 * prefer cache; stale price is always labeled, never presented as live.
 */
import type { CacheDescriptor, CostPolicy, LatencyBudget } from '../../types/platform';

export const LATENCY_BUDGETS: LatencyBudget[] = [
  { feature: 'instant-ui', p50Ms: 150, p95Ms: 300, maxMs: 1000 },
  { feature: 'interactive-retrieval', p50Ms: 500, p95Ms: 1200, maxMs: 3000 },
  { feature: 'ai-analysis', p50Ms: 2000, p95Ms: 8000, maxMs: 20000 },
  { feature: 'vton', p50Ms: 8000, p95Ms: 30000, maxMs: 60000 },
  { feature: 'background-planning', p50Ms: 5000, p95Ms: 30000, maxMs: 120000 },
];

export function latencyBudget(feature: string): LatencyBudget {
  return LATENCY_BUDGETS.find((b) => b.feature === feature) || LATENCY_BUDGETS[1];
}

export function withinBudget(feature: string, p50: number, p95: number): { ok: boolean; violations: string[] } {
  const b = latencyBudget(feature);
  const violations: string[] = [];
  if (p50 > b.p50Ms) violations.push(`p50 ${p50}ms exceeds ${b.p50Ms}ms`);
  if (p95 > b.p95Ms) violations.push(`p95 ${p95}ms exceeds ${b.p95Ms}ms`);
  return { ok: violations.length === 0, violations };
}

// ---------- Cost budgets (5R) ----------

const COSTS = new Map<string, CostPolicy>([
  ['parse-garment', { feature: 'parse-garment', provider: 'default', estimatedUnitCost: null, maxCalls: 3, maxDailyPerUser: 50, cacheTtlMs: 7 * 86400000, premiumRequired: false }],
  ['parse-outfit', { feature: 'parse-outfit', provider: 'default', estimatedUnitCost: null, maxCalls: 2, maxDailyPerUser: 30, cacheTtlMs: 7 * 86400000, premiumRequired: false }],
  ['recreate', { feature: 'recreate', provider: 'default', estimatedUnitCost: null, maxCalls: 2, maxDailyPerUser: 20, cacheTtlMs: 86400000, premiumRequired: true }],
  ['tryon', { feature: 'tryon', provider: 'default', estimatedUnitCost: null, maxCalls: 1, maxDailyPerUser: 10, cacheTtlMs: 0, premiumRequired: true }],
]);

const dailyUse = new Map<string, number>();

export function costPolicy(feature: string): CostPolicy {
  return COSTS.get(feature) || { feature, provider: 'default', estimatedUnitCost: null, maxCalls: 1, maxDailyPerUser: 10, cacheTtlMs: 3600000, premiumRequired: false };
}

export function setCostPolicy(p: CostPolicy): void { COSTS.set(p.feature, p); }

/** Returns false when the daily cap is hit (caller must use fallback). */
export function consumeCall(feature: string, day: string, isPremium: boolean): boolean {
  const p = costPolicy(feature);
  if (p.premiumRequired && !isPremium) return false;
  const key = `${feature}|${day}`;
  const used = dailyUse.get(key) || 0;
  if (p.maxDailyPerUser !== null && p.maxDailyPerUser !== undefined && used >= p.maxDailyPerUser) return false;
  dailyUse.set(key, used + 1);
  return true;
}

export function resetDailyUse(): void { dailyUse.clear(); }

// ---------- Cache policy (5S) ----------

export const CACHES: CacheDescriptor[] = [
  { name: 'garment-parse', key: 'asset|task|model|schema', ttlMs: 7 * 86400000, version: 'v1', sizeBound: 300 },
  { name: 'outfit-parse', key: 'asset|task|model|schema', ttlMs: 7 * 86400000, version: 'v1', sizeBound: 200 },
  { name: 'embeddings', key: 'asset|model', ttlMs: 30 * 86400000, version: 'v1', sizeBound: 500 },
  { name: 'fit-profile', key: 'user|version', ttlMs: 86400000, version: 'v2', sizeBound: 5 },
  { name: 'style-profile', key: 'user|version', ttlMs: 86400000, version: 'v1', sizeBound: 5 },
  { name: 'catalog-discovery', key: 'query|version', ttlMs: 3600000, version: 'v1', sizeBound: 50 },
  { name: 'price-history', key: 'product|merchant', ttlMs: 6 * 3600000, version: 'v2', sizeBound: 500 },
  { name: 'wardrobe-graph', key: 'user|version', ttlMs: 3600000, version: 'v1', sizeBound: 5 },
];

export function cacheDescriptor(name: string): CacheDescriptor | null {
  return CACHES.find((c) => c.name === name) || null;
}

export function isExpired(cachedAt: number, ttlMs: number, now = Date.now()): boolean {
  return now - cachedAt > ttlMs;
}

// ---------- Rate limits (5AJ) ----------

const RATE_LIMITS: Record<string, { perMin: number }> = {
  'ai-task': { perMin: 20 }, 'image-analysis': { perMin: 10 }, vton: { perMin: 3 },
  buycheck: { perMin: 30 }, 'catalog-search': { perMin: 30 }, 'price-watch': { perMin: 12 }, 'share-ingest': { perMin: 10 },
};

const hits = new Map<string, number[]>();

export function rateLimitCheck(feature: string, now = Date.now()): boolean {
  const cap = (RATE_LIMITS[feature] || { perMin: 15 }).perMin;
  const window = hits.get(feature) || [];
  const fresh = window.filter((t) => now - t < 60000);
  hits.set(feature, fresh);
  if (fresh.length >= cap) return false;
  fresh.push(now);
  return true;
}

export function resetRateLimits(): void { hits.clear(); }

// ---------- Entitlements (5AK) ----------

export interface EntitlementState { premium: boolean; quotas: Record<string, number> }

export function canUse(feature: string, ent: EntitlementState): { allowed: boolean; reason: string } {
  const premiumOnly = new Set(['vton', 'recreate-high-frequency', 'price-watches', 'autonomous-weekly']);
  if (!premiumOnly.has(feature)) return { allowed: true, reason: 'core deterministic utility' };
  if (!ent.premium) return { allowed: false, reason: 'premium required' };
  const remaining = ent.quotas[feature] ?? 0;
  if (remaining <= 0) return { allowed: false, reason: 'quota exhausted' };
  return { allowed: true, reason: `${remaining} remaining` };
}

// ---------- Stale data (5AV) ----------

const FRESHNESS_MS: Record<string, number> = {
  catalog: 24 * 3600000, price: 6 * 3600000, 'fit-profile': 24 * 3600000,
  'style-profile': 24 * 3600000, 'wardrobe-graph': 3600000, weather: 3 * 3600000,
  events: 15 * 60000, recommendations: 24 * 3600000,
};

export function freshnessOf(key: string): number { return FRESHNESS_MS[key] || 3600000; }

export function staleLabel(key: string, fetchedAt: number, now = Date.now()): string | null {
  if (now - fetchedAt <= freshnessOf(key)) return null;
  if (key === 'price') return 'Price may be stale — verify before buying.';
  return `${key} data is stale; refresh recommended.`;
}
