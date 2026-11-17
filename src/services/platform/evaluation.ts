/**
 * evaluation.ts — Versioned contracts, offline evaluation, golden
 * scenarios, regression gates, scorecards, calibration, comparison (5D–5I, 5AD).
 *
 * Invariants over prose: golden scenarios assert hard rules, never exact
 * copy. Scorecards stay per-domain; no single "AI quality" number.
 */

export const INTELLIGENCE_VERSIONS = {
  buycheck: 'v2',
  fitDna: 'v2',
  styleDna: 'v1',
  outfitRanking: 'v2',
  usefulUnlocks: 'v2',
  duplicateRisk: 'v2',
  priceAssessment: 'v2',
  visualParseSchema: 'v1',
  visualCompatibility: 'v1',
  proactiveRecommendations: 'v1',
} as const;

export type IntelligenceDomain = keyof typeof INTELLIGENCE_VERSIONS;

export function rulesVersion(domain: IntelligenceDomain): string {
  return INTELLIGENCE_VERSIONS[domain];
}

export function previousVersion(domain: IntelligenceDomain): string {
  const v = INTELLIGENCE_VERSIONS[domain];
  const n = parseInt(v.slice(1), 10);
  return n > 1 ? `v${n - 1}` : v;
}

// ---------- Evaluation framework (5E) ----------

import type { EvaluationCase, EvaluationResult } from '../../types/platform';

export function runEvaluation(
  cases: readonly EvaluationCase[],
  check: (c: EvaluationCase) => { pass: boolean; violations?: string[]; scores?: Record<string, number> },
  rulesVersionTag: string,
): EvaluationResult[] {
  const t0 = Date.now();
  return cases.map((c) => {
    const r = check(c);
    return {
      caseId: c.id, pass: r.pass, violations: r.violations || [],
      scores: r.scores || null, latencyMs: Date.now() - t0,
      provider: null, fallbackUsed: null, rulesVersion: rulesVersionTag,
    };
  });
}

export function evaluationSummary(results: readonly EvaluationResult[]): {
  total: number; passed: number; failed: number; criticalFailures: string[];
} {
  const failed = results.filter((r) => !r.pass);
  return { total: results.length, passed: results.length - failed.length, failed: failed.length, criticalFailures: failed.map((r) => r.caseId) };
}

// ---------- Golden scenarios (5F, excerpt; full set in tests/fixtures) ----------

export const GOLDEN_SCENARIOS: EvaluationCase[] = [
  { id: 'outfit-hot-formal', domain: 'outfit', input: { tempC: 34, occasion: 'formal' }, invariants: ['no weather hard-rule violation', 'outer layer light or absent'], severity: 'critical', source: 'golden' },
  { id: 'outfit-rainy-casual', domain: 'outfit', input: { condition: 'rain', occasion: 'casual' }, invariants: ['water-compatible footwear or warning'], severity: 'major', source: 'golden' },
  { id: 'outfit-anchor-shoe', domain: 'outfit', input: { anchors: ['shoe1'] }, invariants: ['anchor preserved'], severity: 'critical', source: 'golden' },
  { id: 'outfit-insufficient-wardrobe', domain: 'outfit', input: { itemCount: 1 }, invariants: ['no fabricated items', 'insufficient-evidence surfaced'], severity: 'major', source: 'golden' },
  { id: 'fit-strong-reference', domain: 'fit', input: { referenceCount: 5 }, invariants: ['confidence at least medium'], severity: 'major', source: 'golden' },
  { id: 'fit-conflicting-brand', domain: 'fit', input: { conflicts: 3 }, invariants: ['confidence capped at medium', 'contradiction reported'], severity: 'major', source: 'golden' },
  { id: 'fit-unseen-brand', domain: 'fit', input: { priors: 0 }, invariants: ['confidence low', 'no size guarantee'], severity: 'critical', source: 'golden' },
  { id: 'buycheck-high-duplicate', domain: 'buycheck', input: { duplicateLevel: 'high', unlocks: 0 }, invariants: ['verdict is SKIP'], severity: 'critical', source: 'golden' },
  { id: 'buycheck-over-budget', domain: 'buycheck', input: { withinBudget: false }, invariants: ['verdict is SKIP or WAIT_FOR_PRICE'], severity: 'critical', source: 'golden' },
  { id: 'buycheck-no-buy', domain: 'buycheck', input: { unlocks: 0, unwornOwned: 3 }, invariants: ['restraint recommendation'], severity: 'major', source: 'golden' },
  { id: 'style-stable-minimalist', domain: 'style', input: { signals: 20 }, invariants: ['confidence not low when evidence strong'], severity: 'minor', source: 'golden' },
  { id: 'style-single-event', domain: 'style', input: { signals: 1 }, invariants: ['no permanent DNA change'], severity: 'critical', source: 'golden' },
  { id: 'price-no-history', domain: 'price', input: { observations: 0 }, invariants: ['no fabricated history', 'insufficient-history stated'], severity: 'critical', source: 'golden' },
  { id: 'price-fake-discount', domain: 'price', input: { claimedDiscount: 0.8, medianRatio: 1.0 }, invariants: ['fake discount flagged'], severity: 'major', source: 'golden' },
  { id: 'visual-ambiguous-garment', domain: 'visual', input: { quality: 'degraded' }, invariants: ['unknown fields stay unknown', 'confidence bounded'], severity: 'major', source: 'golden' },
  { id: 'visual-partial-recreate', domain: 'visual', input: { coverage: 0.6 }, invariants: ['missing role named', 'no SKU required'], severity: 'minor', source: 'golden' },
];

// ---------- Regression gates (5G) ----------

export type GateCheck = (facts: Record<string, unknown>) => string | null;

export const REGRESSION_GATES: Array<{ id: string; check: GateCheck }> = [
  { id: 'no-wardrobe-only-leakage', check: (f) => (f.wardrobeOnly && f.usedCatalog ? 'wardrobe-only flow touched catalog' : null) },
  { id: 'no-shop-only-leakage', check: (f) => (f.shopOnly && f.usedWardrobe ? 'shop-only flow touched wardrobe' : null) },
  { id: 'no-budget-violation', check: (f) => (typeof f.total === 'number' && typeof f.budget === 'number' && f.total > f.budget && !f.overBudgetFlagged ? 'budget exceeded silently' : null) },
  { id: 'no-activity-violation', check: (f) => (f.hardActivityMismatch ? 'hard activity gate bypassed' : null) },
  { id: 'no-weather-violation', check: (f) => (f.hardWeatherMismatch ? 'hard weather gate bypassed' : null) },
  { id: 'no-broken-variant', check: (f) => (f.selectedVariant && !f.variantInStock && !f.variantFlagged ? 'broken selected variant' : null) },
  { id: 'no-identity-collapse', check: (f) => (f.distinctProducts === 1 && (f.productCount as number) > 1 && !f.sameProduct ? 'physical identity collapse' : null) },
  { id: 'no-fabricated-price-history', check: (f) => (f.priceHistoryEmpty && f.priceVerdict !== 'insufficient' ? 'fabricated price history' : null) },
  { id: 'no-lowconf-fit-guarantee', check: (f) => (f.fitConfidence === 'low' && f.sizeGuaranteed ? 'fit guarantee from low confidence' : null) },
  { id: 'no-visual-override-correction', check: (f) => (f.userCorrected && f.visualOverrode ? 'visual model overrode explicit correction' : null) },
  { id: 'no-client-secret', check: (f) => (f.clientBundle && /sk-|api[_-]?key|secret/i.test(String(f.clientBundle)) ? 'provider secret in client' : null) },
];

export function runGates(facts: Record<string, unknown>): string[] {
  return REGRESSION_GATES.filter((g) => g.check(facts)).map((g) => g.id);
}

// ---------- Scorecards (5H) ----------

export function outfitScorecard(runs: ReadonlyArray<{
  valid: boolean; duplicates: number; total: number; shoes: string[];
  anchorsKept: boolean; styleMatch: boolean; weatherOk: boolean; accepted: boolean;
}>): Record<string, number> {
  const n = Math.max(1, runs.length);
  const sum = (f: (r: (typeof runs)[number]) => number) => runs.reduce((a, r) => a + f(r), 0) / n;
  return {
    hardValidityRate: sum((r) => (r.valid ? 1 : 0)),
    duplicateRate: sum((r) => r.duplicates / Math.max(1, r.total)),
    shoeDiversity: runs.length ? new Set(runs.flatMap((r) => r.shoes)).size / n : 0,
    anchorPreservation: sum((r) => (r.anchorsKept ? 1 : 0)),
    styleRelevance: sum((r) => (r.styleMatch ? 1 : 0)),
    weatherValidity: sum((r) => (r.weatherOk ? 1 : 0)),
    acceptanceProxy: sum((r) => (r.accepted ? 1 : 0)),
  };
}

export function buycheckScorecard(runs: ReadonlyArray<{
  evidenceComplete: boolean; confidence: 'low' | 'medium' | 'high'; correct: boolean;
  duplicatePrecision: boolean; unlocksStable: boolean; noBuyCorrect: boolean;
}>): Record<string, number> {
  const n = Math.max(1, runs.length);
  const sum = (f: (r: (typeof runs)[number]) => number) => runs.reduce((a, r) => a + f(r), 0) / n;
  return {
    evidenceCompleteness: sum((r) => (r.evidenceComplete ? 1 : 0)),
    highConfidenceAccuracy: runs.filter((r) => r.confidence === 'high').length
      ? runs.filter((r) => r.confidence === 'high' && r.correct).length / Math.max(1, runs.filter((r) => r.confidence === 'high').length) : 0,
    duplicatePrecision: sum((r) => (r.duplicatePrecision ? 1 : 0)),
    unlockStability: sum((r) => (r.unlocksStable ? 1 : 0)),
    noBuyCorrectness: sum((r) => (r.noBuyCorrect ? 1 : 0)),
  };
}

// ---------- Calibration (5I) ----------

export type ConfidenceBand = 'low' | 'medium' | 'high';

export function calibrateConfidence(
  samples: ReadonlyArray<{ band: ConfidenceBand; correct: boolean }>,
): { monotonic: boolean; overconfidentBands: ConfidenceBand[]; rates: Record<ConfidenceBand, number> } {
  const rates = { low: 0, medium: 0, high: 0 } as Record<ConfidenceBand, number>;
  for (const band of ['low', 'medium', 'high'] as const) {
    const s = samples.filter((x) => x.band === band);
    rates[band] = s.length ? s.filter((x) => x.correct).length / s.length : 0;
  }
  const monotonic = rates.low <= rates.medium && rates.medium <= rates.high;
  const over: ConfidenceBand[] = [];
  if (samples.filter((x) => x.band === 'high').length >= 5 && rates.high < 0.6) over.push('high');
  if (samples.filter((x) => x.band === 'medium').length >= 5 && rates.medium < 0.4) over.push('medium');
  return { monotonic, overconfidentBands: over, rates };
}

// ---------- Comparison harness (5AD) ----------

export interface ComparisonInput<A, B> {
  cases: unknown[];
  runA: (c: unknown) => A;
  runB: (c: unknown) => B;
  invariantViolations: (r: A | B) => string[];
  agree: (a: A, b: B) => boolean;
  latencyMs: (r: A | B) => number;
}

export function compareImplementations<A, B>(input: ComparisonInput<A, B>): {
  agreement: number; violationsA: number; violationsB: number; avgLatencyA: number; avgLatencyB: number;
} {
  let agree = 0, vA = 0, vB = 0, lA = 0, lB = 0;
  for (const c of input.cases) {
    const a = input.runA(c), b = input.runB(c);
    if (input.agree(a, b)) agree++;
    vA += input.invariantViolations(a).length;
    vB += input.invariantViolations(b).length;
    lA += input.latencyMs(a); lB += input.latencyMs(b);
  }
  const n = Math.max(1, input.cases.length);
  return { agreement: agree / n, violationsA: vA, violationsB: vB, avgLatencyA: lA / n, avgLatencyB: lB / n };
}
