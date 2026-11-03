/**
 * release.ts — CI gates, test tiers, release checklist, rollback,
 * rollout, guardrails, SLOs, jobs, idempotency, retention, export,
 * profiling, review command (5AL, 5AO–5AS, 5AT–5AW, 5AX–5AZ, 5BA, 5BN).
 *
 * Everything here prepares safe activation; nothing activates remotely.
 */
import type { JobDefinition, ReleaseStage } from '../../types/platform';

// ---------- Test tiers (5AP) + CI gates (5AO) ----------

export const TEST_TIERS = {
  tier0: { label: 'fastest pure unit', command: 'npx jest __tests__/utils __tests__/styleOS', gate: true },
  tier1: { label: 'critical domain', command: 'npx jest __tests__/purchase __tests__/services/shoppingEvaluation.test.ts', gate: true },
  tier2: { label: 'integration', command: 'npx jest __tests__/services/catalogRepository.test.ts', gate: true },
  tier3: { label: 'full regression', command: 'npx jest', gate: false },
  tier4: { label: 'manual/native visual QA', command: 'maestro test .maestro/', gate: false },
} as const;

export const CI_GATES = [
  'typecheck', 'lint', 'tier0', 'tier1', 'tier2', 'edge-typecheck',
  'migration-lint', 'security-scan', 'golden-schema-tests', 'diff-check',
] as const;

// ---------- Release checklist (5AQ) ----------

export interface ChecklistItem { id: string; label: string; done: boolean }

export function releaseChecklist(): ChecklistItem[] {
  return [
    'migrations-reviewed', 'flags-default-safe', 'providers-optional',
    'rls-verified', 'rollback-documented', 'analytics-schema-valid',
    'no-secrets', 'critical-tests-pass', 'p95-reviewed', 'kill-switches-tested',
    'offline-fallback-tested', 'upgrade-path-tested',
  ].map((id) => ({ id, label: id.replace(/-/g, ' '), done: false }));
}

// ---------- Rollback playbook (5AR) ----------

export const ROLLBACK_PLAYBOOK: Array<{ action: string; steps: string[] }> = [
  { action: 'disable-feature-flag', steps: ['set flag override off', 'verify fallback mode', 'monitor guardrails'] },
  { action: 'revert-rpc-caller', steps: ['point caller at previous version', 'run tier1 tests', 'monitor errors'] },
  { action: 'rollback-migration', steps: ['apply down-migration (additive tables: drop only if empty)', 'verify RLS', 'replay seed checks'] },
  { action: 'disable-provider', steps: ['kill provider target', 'confirm deterministic fallback', 'monitor latency'] },
  { action: 'restore-ranking-version', steps: ['set rulesVersion to previous', 'run golden scenarios', 'monitor acceptance'] },
];

export function rollbackFor(change: string): string[] {
  const entry = ROLLBACK_PLAYBOOK.find((r) => change.includes(r.action.split('-')[0]));
  return entry ? entry.steps : ['revert change', 'run tier0+tier1', 'monitor guardrails'];
}

// ---------- Rollout (5AS) + guardrails (5AT) + SLO (5AU) ----------

export const ROLLOUT_STAGES: ReleaseStage[] = [
  { stage: 0, label: 'local only', successCriteria: ['tier0-2 green'], guardrails: ['none remote'], rollbackThreshold: 'any failure' },
  { stage: 1, label: 'developer flag', successCriteria: ['manual QA pass'], guardrails: ['flag override only'], rollbackThreshold: 'any failure' },
  { stage: 2, label: 'internal/test cohort', successCriteria: ['no guardrail breach 48h'], guardrails: ['cohort-limited'], rollbackThreshold: 'guardrail breach' },
  { stage: 3, label: '1-5%', successCriteria: ['primary metric neutral+', 'guardrails green'], guardrails: ['kill switch armed'], rollbackThreshold: '>1% error uplift' },
  { stage: 4, label: '25%', successCriteria: ['guardrails green 7d'], guardrails: ['kill switch armed'], rollbackThreshold: '>0.5% violation uplift' },
  { stage: 5, label: '50%', successCriteria: ['guardrails green 7d'], guardrails: ['kill switch armed'], rollbackThreshold: 'any guardrail breach' },
  { stage: 6, label: '100%', successCriteria: ['SLO met 14d'], guardrails: ['standard monitoring'], rollbackThreshold: 'SLO breach' },
];

export const GUARDRAIL_METRICS = [
  'crash_rate', 'provider_failure_rate', 'hard_rule_violation_rate', 'p95_latency',
  'unexpected_fallback_rate', 'api_cost', 'duplicate_recommendation_rate',
  'fit_confidence_correction_rate', 'buycheck_insufficient_evidence_rate',
] as const;

export const SLO_TARGETS: Array<{ capability: string; target: string; mode: string }> = [
  { capability: 'core wardrobe local', target: 'works offline, no network dependency', mode: 'high availability' },
  { capability: 'catalog retrieval', target: 'degrades to cached results', mode: 'graceful degradation' },
  { capability: 'AI visual features', target: 'optional, hidden when unavailable', mode: 'degradable' },
  { capability: 'VTON', target: 'best-effort async, cancellable', mode: 'best effort' },
  { capability: 'price intelligence', target: 'stale labeled, never live-faked', mode: 'stale-safe' },
];

// ---------- Jobs (5AW) + idempotency (5AX) + retention/export (5AY–5AZ) ----------

export const JOB_DEFINITIONS: JobDefinition[] = [
  { name: 'profile-aggregation', idempotent: true, rateCapPerMin: 60, maxRetries: 2, description: 'Aggregate fit/style profiles with cursor checkpoints' },
  { name: 'price-checks', idempotent: true, rateCapPerMin: 30, maxRetries: 2, description: 'Scheduled price observation with dedupe keys' },
  { name: 'catalog-refresh', idempotent: true, rateCapPerMin: 10, maxRetries: 1, description: 'Incremental catalog sync, never full reload' },
  { name: 'recommendation-precompute', idempotent: true, rateCapPerMin: 20, maxRetries: 1, description: 'Precompute proactive suggestions off-peak' },
  { name: 'cleanup', idempotent: true, rateCapPerMin: 30, maxRetries: 0, description: 'Ephemeral asset + telemetry compaction' },
  { name: 'analytics-compaction', idempotent: true, rateCapPerMin: 20, maxRetries: 1, description: 'Aggregate funnel counters from raw events' },
];

const idempotencyKeys = new Set<string>();

export function idempotentDo(key: string, fn: () => void): 'executed' | 'duplicate-skipped' {
  if (idempotencyKeys.has(key)) return 'duplicate-skipped';
  idempotencyKeys.add(key);
  fn();
  return 'executed';
}

export function resetIdempotency(): void { idempotencyKeys.clear(); }

export const RETENTION_CLASSES: Record<string, { ttlMs: number; deletable: boolean }> = {
  'raw-image': { ttlMs: 90 * 86400000, deletable: false },
  'visual-features': { ttlMs: 365 * 86400000, deletable: true },
  'analytics-event': { ttlMs: 365 * 86400000, deletable: true },
  'price-history': { ttlMs: 730 * 86400000, deletable: true },
  'recommendation-trace': { ttlMs: 180 * 86400000, deletable: true },
  'fit-evidence': { ttlMs: Infinity, deletable: false },
  'style-evidence': { ttlMs: Infinity, deletable: false },
};

export const OWNERSHIP_MAP: Record<string, string[]> = {
  wardrobe: ['clothing_items', 'wear_logs', 'outfits'],
  images: ['visual_assets'],
  fit: ['fit_evidence', 'fit_profiles'],
  style: ['style_evidence', 'style_profiles'],
  purchases: ['purchase_decisions', 'purchase_outcomes'],
  price: ['price_watches', 'price_history'],
  recommendations: ['recommendation_feedback', 'stylist_plans'],
  analytics: ['analytics_event_queue'],
};

// ---------- Profiling (5BA) + review command (5BN) ----------

export interface ProfileSample { feature: string; durationMs: number; candidates: number; cacheHit: boolean }

export function profileSummary(samples: readonly ProfileSample[]): Record<string, { p50: number; p95: number; max: number; n: number }> {
  const byFeature = new Map<string, number[]>();
  for (const s of samples) {
    const list = byFeature.get(s.feature) || [];
    list.push(s.durationMs);
    byFeature.set(s.feature, list);
  }
  const out: Record<string, { p50: number; p95: number; max: number; n: number }> = {};
  for (const [f, vals] of byFeature) {
    const sorted = [...vals].sort((a, b) => a - b);
    out[f] = {
      p50: sorted[Math.floor(sorted.length * 0.5)] || 0,
      p95: sorted[Math.floor(sorted.length * 0.95)] || 0,
      max: sorted[sorted.length - 1] || 0, n: sorted.length,
    };
  }
  return out;
}

export interface QualityReview {
  criticalTests: string;
  regressionSummary: string;
  scorecards: Record<string, number>;
  performance: Record<string, { p50: number; p95: number; max: number; n: number }>;
  flagDefaults: Record<string, boolean>;
  migrations: string[];
  remoteActivations: string[];
  risks: string[];
  warnings: string[];
  generatedAt: string;
}

export function buildQualityReview(input: Omit<QualityReview, 'generatedAt'>): QualityReview {
  return { ...input, generatedAt: new Date().toISOString() };
}
