/**
 * evaluation.test.ts — versions, golden scenarios, gates, scorecards,
 * calibration, comparison harness.
 */
import {
  rulesVersion, previousVersion, runEvaluation, evaluationSummary,
  GOLDEN_SCENARIOS, runGates, outfitScorecard, buycheckScorecard,
  calibrateConfidence, compareImplementations,
} from '../../src/services/platform/evaluation';

test('versioned contracts expose current and previous', () => {
  expect(rulesVersion('buycheck')).toBe('v2');
  expect(previousVersion('buycheck')).toBe('v1');
  expect(previousVersion('styleDna')).toBe('v1');
});

test('golden scenarios test invariants and run through harness', () => {
  expect(GOLDEN_SCENARIOS.length).toBeGreaterThan(10);
  const results = runEvaluation(
    GOLDEN_SCENARIOS.filter((c) => c.domain === 'buycheck'),
    (c) => ({ pass: !c.invariants.includes('verdict is SKIP') ? true : c.id !== 'buycheck-over-budget' ? true : false }),
    'v2',
  );
  const summary = evaluationSummary(results);
  expect(summary.total).toBeGreaterThan(0);
  expect(summary.passed + summary.failed).toBe(summary.total);
});

test('regression gates catch violations, pass clean facts', () => {
  expect(runGates({ wardrobeOnly: true, usedCatalog: true })).toContain('no-wardrobe-only-leakage');
  expect(runGates({ total: 200, budget: 100 })).toContain('no-budget-violation');
  expect(runGates({ fitConfidence: 'low', sizeGuaranteed: true })).toContain('no-lowconf-fit-guarantee');
  expect(runGates({ userCorrected: true, visualOverrode: true })).toContain('no-visual-override-correction');
  expect(runGates({})).toHaveLength(0);
});

test('scorecards stay per-domain with named dimensions', () => {
  const o = outfitScorecard([
    { valid: true, duplicates: 0, total: 4, shoes: ['s1'], anchorsKept: true, styleMatch: true, weatherOk: true, accepted: true },
    { valid: false, duplicates: 2, total: 4, shoes: ['s1'], anchorsKept: false, styleMatch: false, weatherOk: false, accepted: false },
  ]);
  expect(o.hardValidityRate).toBe(0.5);
  expect(o.shoeDiversity).toBe(0.5);
  const b = buycheckScorecard([
    { evidenceComplete: true, confidence: 'high', correct: true, duplicatePrecision: true, unlocksStable: true, noBuyCorrect: true },
    { evidenceComplete: true, confidence: 'high', correct: false, duplicatePrecision: false, unlocksStable: false, noBuyCorrect: false },
  ]);
  expect(b.highConfidenceAccuracy).toBe(0.5);
});

test('calibration detects monotonic reliability and overconfidence', () => {
  const good = calibrateConfidence([
    ...Array.from({ length: 8 }, () => ({ band: 'high' as const, correct: true })),
    ...Array.from({ length: 2 }, () => ({ band: 'high' as const, correct: false })),
    { band: 'low' as const, correct: false },
  ]);
  expect(good.monotonic).toBe(true);
  expect(good.overconfidentBands).toHaveLength(0);
  const bad = calibrateConfidence(Array.from({ length: 6 }, () => ({ band: 'high' as const, correct: false })));
  expect(bad.overconfidentBands).toContain('high');
});

test('comparison harness measures agreement and violations', () => {
  const r = compareImplementations({
    cases: [1, 2, 3],
    runA: (c) => c, runB: (c) => c,
    invariantViolations: () => [], agree: (a, b) => a === b, latencyMs: () => 5,
  });
  expect(r.agreement).toBe(1);
  expect(r.avgLatencyA).toBe(5);
});
