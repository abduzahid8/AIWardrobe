/**
 * experimentation.ts — Experiments, shadow mode, feature flags,
 * kill switches (5J–5M).
 *
 * Assignment is deterministic (FNV hash of unit+experiment id), stable,
 * and privacy-safe (no PII in the hash input beyond the caller's
 * pseudonymous id). Shadow runs never change user-visible output.
 * Kill switches fail to deterministic local behavior, never to a dead core.
 */
import type { Experiment, PlatformFlag } from '../../types/platform';

function fnv(input: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < input.length; i++) { h ^= input.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h >>> 0;
}

const experiments = new Map<string, Experiment>();

export function registerExperiment(e: Experiment): void { experiments.set(e.id, e); }
export function getExperiment(id: string): Experiment | null { return experiments.get(id) || null; }
export function clearExperiments(): void { experiments.clear(); }

export function assignVariant(experimentId: string, unitId: string): string {
  const e = experiments.get(experimentId);
  if (!e || e.status !== 'running') return 'control';
  if (killed(`experiment:${experimentId}`)) return 'control';
  const total = e.variants.reduce((n, v) => n + v.weight, 0) || 1;
  let point = fnv(`${experimentId}|${unitId}`) % 1000 / 1000;
  for (const v of e.variants) {
    point -= v.weight / total;
    if (point < 0) return v.name;
  }
  return e.variants[0]?.name || 'control';
}

// ---------- Shadow mode (5K) ----------

export interface ShadowResult<P, S> {
  production: P;
  shadow: S | null;
  shadowError?: string | null;
  comparedAt: string;
}

export function runShadow<P, S>(production: () => P, shadow: () => S): ShadowResult<P, S> {
  const prod = production();
  try {
    return { production: prod, shadow: shadow(), comparedAt: new Date().toISOString() };
  } catch (err) {
    return { production: prod, shadow: null, shadowError: err instanceof Error ? err.message : 'shadow failed', comparedAt: new Date().toISOString() };
  }
}

export interface ShadowComparison {
  candidateOverlap: number;
  rankingChanged: boolean;
  violationsDelta: number;
  latencyDeltaMs: number;
}

export function compareRankings<T>(production: readonly T[], candidate: readonly T[], latencyDeltaMs = 0, violationsDelta = 0): ShadowComparison {
  const pSet = new Set(production);
  const overlap = candidate.length ? candidate.filter((c) => pSet.has(c)).length / candidate.length : 1;
  const sameOrder = production.length === candidate.length && production.every((v, i) => v === candidate[i]);
  return { candidateOverlap: overlap, rankingChanged: !sameOrder, violationsDelta, latencyDeltaMs };
}

// ---------- Feature flags (5L) ----------

export const PLATFORM_FLAG_DEFAULTS: Record<PlatformFlag, boolean> = {
  fit_dna_v2: false, buycheck_v2: false, visual_parser: false, recreate_look: false,
  autonomous_stylist: false, proactive_recommendations: false, price_intelligence: false,
  wardrobe_graph: true, daily_stylist: false, new_retrieval: false, new_ranking: false,
};

const overrides = new Map<string, boolean>();
const cohorts = new Map<string, Set<string>>();

export function flagDefault(flag: PlatformFlag): boolean { return PLATFORM_FLAG_DEFAULTS[flag]; }
export function setFlagOverride(flag: string, on: boolean): void { overrides.set(flag, on); }
export function clearFlagOverrides(): void { overrides.clear(); }
export function setCohort(flag: string, unitIds: string[]): void { cohorts.set(flag, new Set(unitIds)); }

export function isEnabled(flag: PlatformFlag, unitId?: string, experimentId?: string): boolean {
  if (killed(`flag:${flag}`)) return false;
  if (overrides.has(flag)) return overrides.get(flag)!;
  if (experimentId && assignVariant(experimentId, unitId || 'anon') !== 'control') return true;
  if (unitId && cohorts.get(flag)?.has(unitId)) return true;
  return PLATFORM_FLAG_DEFAULTS[flag];
}

// ---------- Kill switches (5M) ----------

const kills = new Set<string>();

export function kill(target: string): void { kills.add(target); }
export function revive(target: string): void { kills.delete(target); }
export function killed(target: string): boolean { return kills.has(target); }
export function killAll(): void { kills.clear(); }

export type FallbackMode = 'deterministic-local' | 'previous-stable' | 'hidden';

export function fallbackFor(killTarget: string): FallbackMode {
  if (killTarget.startsWith('provider:') || killTarget.startsWith('remote:')) return 'deterministic-local';
  if (killTarget.startsWith('experiment:')) return 'previous-stable';
  return 'hidden';
}
