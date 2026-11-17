/**
 * resilience.ts — Provider health, circuit breaker, retries,
 * backpressure, degraded mode, error taxonomy + UX (5N–5W).
 *
 * Circuit breakers never wrap deterministic local rules — only remote
 * AI/provider calls. Every fallback keeps core wardrobe usable.
 */
import type { ErrorClass, ProviderHealth, ProviderState } from '../../types/platform';

// ---------- Provider health (5N) ----------

const health = new Map<string, { ok: number; timeout: number; malformed: number; latencies: number[]; recentFailures: number; circuitOpen: boolean; openedAt: number }>();

function rec(provider: string) {
  let r = health.get(provider);
  if (!r) { r = { ok: 0, timeout: 0, malformed: 0, latencies: [], recentFailures: 0, circuitOpen: false, openedAt: 0 }; health.set(provider, r); }
  return r;
}

export function recordProviderOutcome(provider: string, outcome: 'ok' | 'timeout' | 'malformed' | 'error', latencyMs = 0): void {
  const r = rec(provider);
  if (outcome === 'ok') { r.ok++; r.recentFailures = 0; }
  else { r.recentFailures++; if (outcome === 'timeout') r.timeout++; if (outcome === 'malformed') r.malformed++; }
  r.latencies.push(latencyMs);
  if (r.latencies.length > 50) r.latencies.shift();
}

export function providerHealth(provider: string): ProviderHealth {
  const r = rec(provider);
  const total = Math.max(1, r.ok + r.timeout + r.malformed);
  const state: ProviderState = r.circuitOpen ? 'unavailable'
    : r.recentFailures >= 5 ? 'unavailable'
    : r.recentFailures >= 2 ? 'degraded' : 'healthy';
  const avg = r.latencies.length ? r.latencies.reduce((a, b) => a + b, 0) / r.latencies.length : 0;
  return {
    provider, state, successRate: r.ok / total, timeoutRate: r.timeout / total,
    malformedRate: r.malformed / total, avgLatencyMs: Math.round(avg),
    recentFailures: r.recentFailures, circuitOpen: r.circuitOpen, updatedAt: new Date().toISOString(),
  };
}

export function clearProviderHealth(): void { health.clear(); }

// ---------- Circuit breaker (5O) ----------

export interface BreakerOptions { failureThreshold: number; cooldownMs: number }

export function shouldAllow(provider: string, options: BreakerOptions, now = Date.now()): boolean {
  const r = rec(provider);
  if (!r.circuitOpen) {
    if (r.recentFailures >= options.failureThreshold) { r.circuitOpen = true; r.openedAt = now; }
    return !r.circuitOpen;
  }
  if (now - r.openedAt >= options.cooldownMs) { r.circuitOpen = false; r.recentFailures = 0; return true; }
  return false;
}

// ---------- Retry policy (5P) ----------

export type RetryableKind = 'network-timeout' | 'malformed-output' | 'rate-limit' | 'invalid-input' | 'validation-failure';

const MAX_ATTEMPTS: Record<RetryableKind, number> = {
  'network-timeout': 2, 'malformed-output': 2, 'rate-limit': 1, 'invalid-input': 1, 'validation-failure': 1,
};

export function maxAttempts(kind: RetryableKind): number { return MAX_ATTEMPTS[kind]; }
export function shouldRetry(kind: RetryableKind, attempt: number): boolean {
  if (kind === 'invalid-input' || kind === 'validation-failure') return false;
  return attempt < MAX_ATTEMPTS[kind];
}

// ---------- Backpressure (5T) ----------

export interface GateOptions { maxConcurrent: number; queueLimit: number }

const inflight = new Map<string, number>();
const queued = new Map<string, number>();

export function tryAcquire(feature: string, options: GateOptions): 'run' | 'dedupe' | 'reject' {
  const cur = inflight.get(feature) || 0;
  if (cur >= options.maxConcurrent) {
    const q = queued.get(feature) || 0;
    if (q >= options.queueLimit) return 'reject';
    queued.set(feature, q + 1);
    return 'dedupe'; // collapse into latest request
  }
  inflight.set(feature, cur + 1);
  return 'run';
}

export function release(feature: string): void {
  inflight.set(feature, Math.max(0, (inflight.get(feature) || 1) - 1));
  queued.set(feature, Math.max(0, (queued.get(feature) || 1) - 1));
}

export function resetGates(): void { inflight.clear(); queued.clear(); }

// ---------- Degraded mode (5U) ----------

export type DependencyState = 'ok' | 'down';
export type CapabilityMode = 'full' | 'degraded' | 'unavailable' | 'hidden';

export function capabilityMode(deps: Record<string, DependencyState>, feature: 'wardrobe-core' | 'catalog' | 'visual-ai' | 'vton' | 'price'): CapabilityMode {
  if (feature === 'wardrobe-core') return 'full'; // never degraded
  if (feature === 'catalog') return deps.network === 'ok' ? 'full' : 'unavailable';
  if (feature === 'visual-ai') return deps.aiProvider === 'ok' ? 'full' : 'degraded';
  if (feature === 'vton') return deps.aiProvider === 'ok' ? 'full' : 'unavailable';
  return deps.priceService === 'ok' ? 'full' : 'degraded'; // price: stale-safe
}

export function degradedLabel(mode: CapabilityMode, feature: string): string | null {
  if (mode === 'full') return null;
  if (mode === 'degraded') return `${feature} is limited right now. Saved local data is still fully usable.`;
  if (mode === 'unavailable') return `${feature} is temporarily unavailable. Your wardrobe remains usable.`;
  return null;
}

// ---------- Error taxonomy + UX (5V–5W) ----------

export function classifyError(err: unknown): ErrorClass {
  const m = err instanceof Error ? err.message.toLowerCase() : String(err).toLowerCase();
  if (/timeout|timed out|etimedout/.test(m)) return 'PROVIDER_TIMEOUT';
  if (/rate.?limit|429/.test(m)) return 'PROVIDER_RATE_LIMIT';
  if (/network|econn|offline|fetch failed/.test(m)) return 'NETWORK_UNAVAILABLE';
  if (/unauthor|401|403/.test(m)) return 'AUTH_REQUIRED';
  if (/malformed|unexpected token|invalid json|schema/.test(m)) return 'PROVIDER_MALFORMED';
  if (/insufficient|not enough/.test(m)) return 'INSUFFICIENT_EVIDENCE';
  if (/permission|forbidden/.test(m)) return 'PERMISSION_REQUIRED';
  if (/disabled|flag|kill/.test(m)) return 'FEATURE_DISABLED';
  return 'VALIDATION_FAILED';
}

const USER_COPY: Record<ErrorClass, { title: string; actions: Array<'retry' | 'fallback' | 'cancel'> }> = {
  AUTH_REQUIRED: { title: 'Please sign in to continue.', actions: ['cancel'] },
  NETWORK_UNAVAILABLE: { title: 'You appear to be offline. Saved wardrobe data still works.', actions: ['retry', 'cancel'] },
  PROVIDER_TIMEOUT: { title: 'The AI service is taking too long. You can retry or continue with saved data.', actions: ['retry', 'fallback', 'cancel'] },
  PROVIDER_RATE_LIMIT: { title: 'Too many requests right now. Please wait a moment.', actions: ['retry', 'cancel'] },
  PROVIDER_MALFORMED: { title: 'The AI service returned an unusable response. Local data is unaffected.', actions: ['retry', 'fallback', 'cancel'] },
  VALIDATION_FAILED: { title: 'This input could not be validated. Please adjust and try again.', actions: ['cancel'] },
  INSUFFICIENT_EVIDENCE: { title: 'Not enough information yet to give a confident answer.', actions: ['cancel'] },
  CATALOG_UNAVAILABLE: { title: 'Catalog search is unavailable. Your wardrobe still works.', actions: ['retry', 'cancel'] },
  FIT_DATA_INCOMPLETE: { title: 'Fit data is incomplete, so sizing stays uncertain.', actions: ['cancel'] },
  PRICE_HISTORY_UNAVAILABLE: { title: 'Price history is unavailable; treat the price as unverified.', actions: ['cancel'] },
  PERMISSION_REQUIRED: { title: 'This action needs a permission you have not granted.', actions: ['cancel'] },
  FEATURE_DISABLED: { title: 'This feature is currently turned off.', actions: ['cancel'] },
};

export function userCopy(errorClass: ErrorClass): { title: string; actions: Array<'retry' | 'fallback' | 'cancel'> } {
  return USER_COPY[errorClass];
}
