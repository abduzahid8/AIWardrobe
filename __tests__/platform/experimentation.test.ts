/**
 * experimentation.test.ts — deterministic assignment, control fallback,
 * kill switches, shadow mode, flags.
 */
import {
  registerExperiment, assignVariant, clearExperiments,
  runShadow, compareRankings,
  flagDefault, isEnabled, setFlagOverride, clearFlagOverrides, setCohort,
  kill, revive, killed, fallbackFor,
} from '../../src/services/platform/experimentation';

beforeEach(() => { clearExperiments(); clearFlagOverrides(); ['experiment:x', 'flag:buycheck_v2', 'provider:vision'].forEach(revive); });

test('assignment is deterministic, stable, and privacy-safe', () => {
  registerExperiment({ id: 'x', description: 'd', status: 'running', assignmentUnit: 'user', variants: [{ name: 'control', weight: 1 }, { name: 'treatment', weight: 1 }], primaryMetric: 'm' });
  expect(assignVariant('x', 'user1')).toBe(assignVariant('x', 'user1'));
  const seen = new Set(['u1', 'u2', 'u3', 'u4'].map((u) => assignVariant('x', u)));
  expect(seen.size).toBeGreaterThan(1); // splits traffic
});

test('draft/paused/unknown experiments fall back to control', () => {
  registerExperiment({ id: 'd', description: 'd', status: 'draft', assignmentUnit: 'user', variants: [{ name: 'treatment', weight: 1 }], primaryMetric: 'm' });
  expect(assignVariant('d', 'u1')).toBe('control');
  expect(assignVariant('missing', 'u1')).toBe('control');
});

test('kill switch forces control and safe flag defaults', () => {
  registerExperiment({ id: 'x', description: 'd', status: 'running', assignmentUnit: 'user', variants: [{ name: 'treatment', weight: 1 }], primaryMetric: 'm' });
  kill('experiment:x');
  expect(killed('experiment:x')).toBe(true);
  expect(assignVariant('x', 'u1')).toBe('control');
  expect(flagDefault('buycheck_v2')).toBe(false);
  setFlagOverride('buycheck_v2', true);
  expect(isEnabled('buycheck_v2')).toBe(true);
  kill('flag:buycheck_v2');
  expect(isEnabled('buycheck_v2')).toBe(false);
  expect(fallbackFor('provider:vision')).toBe('deterministic-local');
  expect(fallbackFor('experiment:x')).toBe('previous-stable');
});

test('cohort and experiment linkage enable treatment', () => {
  setCohort('new_ranking', ['u9']);
  expect(isEnabled('new_ranking', 'u9')).toBe(true);
  expect(isEnabled('new_ranking', 'other')).toBe(false);
});

test('shadow mode never changes production output', () => {
  const r = runShadow(() => [1, 2, 3], () => { throw new Error('boom'); });
  expect(r.production).toEqual([1, 2, 3]);
  expect(r.shadow).toBeNull();
  expect(r.shadowError).toBe('boom');
  const cmp = compareRankings([1, 2, 3], [1, 3, 2], 5, 0);
  expect(cmp.candidateOverlap).toBe(1);
  expect(cmp.rankingChanged).toBe(true);
  expect(cmp.latencyDeltaMs).toBe(5);
});
