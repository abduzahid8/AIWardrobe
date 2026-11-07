/**
 * release.test.ts — checklist, rollback, rollout, jobs/idempotency,
 * retention, profiling, review command.
 */
import {
  TEST_TIERS, CI_GATES, releaseChecklist, ROLLBACK_PLAYBOOK, rollbackFor,
  ROLLOUT_STAGES, GUARDRAIL_METRICS, SLO_TARGETS,
  JOB_DEFINITIONS, idempotentDo, resetIdempotency,
  RETENTION_CLASSES, OWNERSHIP_MAP, profileSummary, buildQualityReview,
} from '../../src/services/platform/release';
import {
  timeIt, startupImportViolations, bounded, MEMORY_CAPS,
  a11yReview, emptyDebugPanel, stripSecrets, inspectDecision,
  buildBugSnapshot, runLoadProbe,
} from '../../src/services/platform/devtools';

test('tiers and gates name the CI contract', () => {
  expect(TEST_TIERS.tier0.gate).toBe(true);
  expect(TEST_TIERS.tier3.gate).toBe(false);
  expect(CI_GATES).toContain('migration-lint');
  expect(releaseChecklist()).toHaveLength(12);
  expect(releaseChecklist()[0].done).toBe(false);
});

test('rollback playbook covers remote-capable changes', () => {
  expect(ROLLBACK_PLAYBOOK.length).toBeGreaterThanOrEqual(5);
  expect(rollbackFor('disable feature flag now')).toContain('verify fallback mode');
  expect(rollbackFor('something unknown')).toContain('monitor guardrails');
  expect(ROLLOUT_STAGES).toHaveLength(7);
  expect(ROLLOUT_STAGES[0].label).toBe('local only');
  expect(GUARDRAIL_METRICS).toContain('hard_rule_violation_rate');
  expect(SLO_TARGETS.find((s) => s.capability === 'VTON')?.mode).toBe('best effort');
});

test('jobs are idempotent with rate caps; retention/export mapped', () => {
  expect(JOB_DEFINITIONS.every((j) => j.idempotent)).toBe(true);
  resetIdempotency();
  let n = 0;
  expect(idempotentDo('price:obs:1', () => { n++; })).toBe('executed');
  expect(idempotentDo('price:obs:1', () => { n++; })).toBe('duplicate-skipped');
  expect(n).toBe(1);
  expect(RETENTION_CLASSES['fit-evidence'].deletable).toBe(false);
  expect(OWNERSHIP_MAP.images).toContain('visual_assets');
});

test('profiling summarizes p50/p95/max locally', () => {
  const samples = Array.from({ length: 100 }, (_, i) => ({ feature: 'buycheck', durationMs: i, candidates: 5, cacheHit: false }));
  const s = profileSummary(samples);
  expect(s.buycheck.p50).toBe(50);
  expect(s.buycheck.max).toBe(99);
  const t = timeIt('x', () => 42);
  expect(t.result).toBe(42);
  expect(t.sample.durationMs).toBeGreaterThanOrEqual(0);
});

test('devtools: bounds, a11y, secrets, inspector, snapshots, load', () => {
  expect(bounded([1, 2, 3], 2)).toEqual([1, 2]);
  expect(MEMORY_CAPS.eventQueue).toBe(200);
  expect(a11yReview({})).toContain('44pt minimum touch targets');
  expect(startupImportViolations(['openai', './local'])).toEqual(['openai']);
  expect(emptyDebugPanel().flags).toEqual({});
  expect(stripSecrets({ apiKey: 'sk-x', other: 'v' })).toEqual({ apiKey: '[redacted]', other: 'v' });
  const ins = inspectDecision({ question: 'why-skip', evidence: ['budget exceeded'], rules: ['budget-gate'] });
  expect(ins.answer.join(' ')).toMatch(/hard gate/);
  expect(JSON.stringify(ins)).not.toMatch(/chain-of-thought/);
  const snap = buildBugSnapshot({
    appVersion: '1.0', featureVersions: { buycheck: 'v2' }, requestIds: ['r1'],
    inputs: { note: 'ok', img: 'data:image/png;base64,xxxx' }, flags: {}, experiments: {},
    errors: [{ errorClass: 'PROVIDER_TIMEOUT', message: 't' }], environment: 'test',
  });
  expect(snap.sanitizedInputs.img).toBe('[withheld-sensitive]');
  const load = runLoadProbe({ name: 'buycheck-x100', n: 100, run: () => {} });
  expect(load.avgMs).toBeGreaterThanOrEqual(0);
  const review = buildQualityReview({
    criticalTests: 'tier0-2', regressionSummary: 'ok', scorecards: { hardValidityRate: 0.99 },
    performance: {}, flagDefaults: { buycheck_v2: false }, migrations: ['20261105120000'],
    remoteActivations: [], risks: [], warnings: [],
  });
  expect(review.generatedAt).toBeTruthy();
});
