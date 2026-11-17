import { test } from 'node:test';
import assert from 'node:assert/strict';
import { summarizeBeta } from './beta-review.mjs';
import { makePersonas } from './personas.mjs';
test('review deduplicates exports and distinguishes absent usage evidence from zero', () => {
  const row = {
    schemaVersion: 1,
    id: 'beta-one',
    feature: 'buycheck',
    helpful: true,
    issueType: 'wrong_fit',
    comment: 'private',
    diagnostics: { confidence: 'low' },
  };
  const summary = summarizeBeta([row, row]);
  assert.equal(summary.total, 1);
  assert.equal(summary.usageRank, null);
  assert.equal(summary.byFeature.find((r) => r.feature === 'outfit').helpfulRate, null);
  assert.ok(!JSON.stringify(summary).includes('private'));
});
test('five reproducible synthetic personas cover required ranges and sparse evidence', () => {
  const personas = makePersonas();
  assert.deepEqual(
    personas.map((p) => p.wardrobe.length),
    [32, 48, 20, 72, 3]
  );
  assert.deepEqual(makePersonas(), personas);
  assert.equal(personas[4].fitEvidence.length, 0);
});

test('a reverted correction remains reverted regardless of export file order', () => {
  const original = {
    schemaVersion: 1,
    id: 'r',
    feature: 'fit',
    helpful: false,
    issueType: 'wrong_fit',
    createdAt: '2026-11-03T00:00:00Z',
    correction: { field: 'fit', before: 'snug', after: 'relaxed' },
  };
  const undone = {
    ...original,
    correction: { ...original.correction, revertedAt: '2026-11-03T00:01:00Z' },
  };
  assert.equal(summarizeBeta([undone, original]).topCorrections.fit, 0);
});

test('usage, measured timing and explicit value evidence are summarized without inventing missing denominators', () => {
  const row = (id, event, more = {}) => ({
    schemaVersion: 1,
    id,
    kind: 'observation',
    feature: 'visual',
    event,
    ...more,
  });
  const report = summarizeBeta([
    row('e', 'feature_exposed', { usageInstrumented: true }),
    row('u', 'feature_used'),
    row('t1', 'timing', {
      latencyMs: 10,
      measurementSource: 'host',
      device: 'Node',
      buildType: 'host-test',
      boundary: 'client_overhead',
    }),
    row('t2', 'timing', {
      latencyMs: 30,
      measurementSource: 'host',
      device: 'Node',
      buildType: 'host-test',
      boundary: 'client_overhead',
    }),
    row('o', 'outcome', { outcome: 'aha_confirmed' }),
  ]);
  assert.deepEqual(report.usageRank, [{ feature: 'visual', uses: 1 }]);
  assert.equal(report.slowestFeatures[0].samples, 2);
  assert.equal(report.slowestFeatures[0].productionClaim, false);
  assert.equal(report.ahaMoments[0].outcome, 'aha_confirmed');
  assert.equal(report.leastUsedComplexFeatures[0].exposures, 1);
  assert.equal(report.providerCost, null);
  const unknown = summarizeBeta([
    {
      schemaVersion: 1,
      id: 'f',
      feature: 'fit',
      helpful: false,
      issueType: 'other',
      diagnostics: {},
    },
  ]);
  assert.equal(unknown.byFeature.find((r) => r.feature === 'fit').lowConfidenceRate, null);
  assert.equal(unknown.byFeature.find((r) => r.feature === 'fit').fallbackRate, null);
});

test('positive feedback with the default other reason is not counted as a failure or repeated complaint', () => {
  const rows = ['a', 'b'].map((id) => ({
    schemaVersion: 1,
    id,
    feature: 'outfit',
    helpful: true,
    issueType: 'other',
  }));
  const summary = summarizeBeta(rows);
  assert.equal(summary.topFailureClasses.other, 0);
  assert.deepEqual(summary.repeatedComplaintClasses, []);
  assert.equal(summary.byFeature.find((r) => r.feature === 'outfit').helpfulRate, 1);
});
test('p50 is the sample median for even timing samples rather than the upper middle sample', () => {
  const samples = [10, 30].map((latencyMs, i) => ({
    schemaVersion: 1,
    id: `sample-${i}`,
    kind: 'observation',
    feature: 'buycheck',
    event: 'timing',
    latencyMs,
    measurementSource: 'host',
    device: 'Node',
    buildType: 'host-test',
    boundary: 'client_overhead',
  }));
  assert.equal(summarizeBeta(samples).slowestFeatures[0].p50Ms, 20);
});
