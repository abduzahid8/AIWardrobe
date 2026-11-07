#!/usr/bin/env node
/** Read only local, voluntarily exported JSON. No raw comments or identities in summary. */
import { readFileSync } from 'node:fs';
export function summarizeBeta(records) {
  const latest = new Map();
  for (const row of records.filter((r) => r?.schemaVersion === 1 && typeof r.id === 'string')) {
    const existing = latest.get(row.id),
      stamp = (r) => Date.parse(r?.correction?.revertedAt || r?.createdAt || '') || 0;
    if (!existing || stamp(row) >= stamp(existing)) latest.set(row.id, row);
  }
  const all = [...latest.values()];
  const unique = all.filter((row) => !row.kind && typeof row.helpful === 'boolean');
  const safeLabel = (value) => typeof value === 'string' && /^[a-zA-Z0-9_. -]{1,96}$/.test(value);
  const observed = all.filter((row) => row.kind === 'observation' && safeLabel(row.feature));
  const usages = observed.filter((row) => row.event === 'feature_used');
  const exposures = observed.filter(
    (row) => row.event === 'feature_exposed' && row.usageInstrumented === true
  );
  const timed = observed.filter(
    (row) =>
      row.event === 'timing' &&
      Number.isFinite(row.latencyMs) &&
      row.latencyMs >= 0 &&
      ['device', 'simulator', 'host'].includes(row.measurementSource) &&
      safeLabel(row.device) &&
      safeLabel(row.buildType) &&
      safeLabel(row.boundary)
  );
  const groups = new Map();
  for (const row of timed) {
    const key = JSON.stringify([
      row.feature,
      row.measurementSource,
      row.device,
      row.buildType,
      row.boundary,
    ]);
    const group = groups.get(key) || {
      feature: row.feature,
      measurementSource: row.measurementSource,
      device: row.device,
      buildType: row.buildType,
      boundary: row.boundary,
      values: [],
    };
    group.values.push(row.latencyMs);
    groups.set(key, group);
  }
  const slowest = [...groups.values()]
    .map(({ values, ...group }) => {
      values.sort((a, b) => a - b);
      return {
        ...group,
        samples: values.length,
        p50Ms:
          values.length % 2
            ? values[Math.floor(values.length / 2)]
            : (values[values.length / 2 - 1] + values[values.length / 2]) / 2,
        rangeMs: [values[0], values.at(-1)],
        productionClaim: false,
      };
    })
    .sort((a, b) => b.p50Ms - a.p50Ms);
  const outcomes = observed.filter(
    (row) =>
      row.event === 'outcome' &&
      [
        'outfit_saved',
        'outfit_worn',
        'buycheck_helpful',
        'duplicate_avoided',
        'missing_piece_confirmed',
        'look_recreated_saved',
        'fit_confirmed',
        'return_to_plan',
        'aha_confirmed',
      ].includes(row.outcome)
  );
  const costs = observed.filter(
    (row) =>
      row.event === 'cost' &&
      Number.isSafeInteger(row.amountMicros) &&
      row.amountMicros >= 0 &&
      /^[A-Z]{3}$/.test(row.currency || '') &&
      safeLabel(row.rateVersion)
  );
  const features = ['general', 'outfit', 'buycheck', 'fit', 'shopping', 'visual', 'subscription'];
  const issues = [
    'wrong_outfit',
    'wrong_fit',
    'duplicate',
    'irrelevant_shopping',
    'price_issue',
    'visual_analysis_wrong',
    'slow',
    'bug',
    'confusing_ux',
    'privacy_concern',
    'other',
  ];
  const countBy = (values, key, allowed) =>
    Object.fromEntries(allowed.map((v) => [v, values.filter((r) => key(r) === v).length]));
  const complaints = unique.filter((row) => row.helpful === false || row.issueType !== 'other');
  const counts = countBy(unique, (r) => r.feature, features);
  const ratio = (feature, predicate, eligible = () => true) => {
    const subset = unique.filter((r) => r.feature === feature && eligible(r));
    return subset.length ? subset.filter(predicate).length / subset.length : null;
  };
  return {
    schemaVersion: 1,
    source: 'voluntary exports; selection-biased, not usage telemetry',
    total: unique.length,
    topFailureClasses: countBy(complaints, (r) => r.issueType, issues),
    repeatedComplaintClasses: issues.filter(
      (issue) => complaints.filter((r) => r.issueType === issue).length >= 2
    ),
    topCorrections: countBy(
      unique.filter((r) => r.correction && !r.correction.revertedAt),
      (r) => r.correction.field,
      [
        'garment_type',
        'color',
        'fit',
        'duplicate_group',
        'outfit_context',
        'product_family',
        'price_match',
      ]
    ),
    byFeature: features.map((feature) => ({
      feature,
      reports: counts[feature],
      helpfulRate: ratio(feature, (r) => r.helpful === true),
      lowConfidenceRate: ratio(
        feature,
        (r) => r.diagnostics?.confidence === 'low',
        (r) => ['low', 'medium', 'high'].includes(r.diagnostics?.confidence)
      ),
      fallbackRate: ratio(
        feature,
        (r) => ['local', 'cached', 'unavailable'].includes(r.diagnostics?.fallback),
        (r) => ['none', 'local', 'cached', 'unavailable'].includes(r.diagnostics?.fallback)
      ),
    })),
    usageRank: usages.length
      ? [...new Set(usages.map((r) => r.feature))]
          .map((feature) => ({ feature, uses: usages.filter((r) => r.feature === feature).length }))
          .sort((a, b) => b.uses - a.uses)
      : null,
    leastUsedComplexFeatures: exposures.length
      ? [...new Set(exposures.map((r) => r.feature))]
          .filter(
            (feature) =>
              !['general', 'outfit', 'buycheck', 'shopping', 'subscription'].includes(feature)
          )
          .map((feature) => ({
            feature,
            exposures: exposures.filter((r) => r.feature === feature).length,
            uses: usages.filter((r) => r.feature === feature).length,
          }))
          .sort((a, b) => a.uses - b.uses)
      : null,
    slowestFeatures: slowest.length ? slowest : null,
    ahaMoments: outcomes.length
      ? outcomes.map(({ feature, outcome }) => ({ feature, outcome }))
      : null,
    providerCost: costs.length
      ? costs.map(({ feature, amountMicros, currency, rateVersion }) => ({
          feature,
          amountMicros,
          currency,
          rateVersion,
          estimateOnly: true,
        }))
      : null,
    missingEvidence: [
      'usage requires instrumented denominators, not report counts',
      'timing rankings retain measurement source/device/build/boundary; host is not device evidence',
      'aha moments and purchase avoidance require explicit user outcomes',
      'cost requires provider usage and versioned rates',
    ],
    reviewQuestions: [
      'Which feature drove first saved/worn outfit or helpful purchase decision?',
      'Which value action repeated on another day?',
      'Which impressive feature is unused with verified exposure?',
      'Where is confusion repeated?',
      'Which feature has the highest observed provider cost?',
      'Which premium value do users explicitly want to pay for?',
      'Which AI mistakes cause harm?',
      'Where does deterministic behavior beat AI on the same task?',
    ],
  };
}
if (process.argv[1]?.endsWith('beta-review.mjs')) {
  const files = process.argv.slice(2);
  if (!files.length) {
    console.error('Usage: npm run beta:review -- /absolute/path/to/export.json [more.json]');
    process.exitCode = 1;
  } else {
    const records = files.flatMap((file) => {
      const value = JSON.parse(readFileSync(file, 'utf8'));
      return Array.isArray(value) ? value : [value];
    });
    console.log(JSON.stringify(summarizeBeta(records), null, 2));
  }
}
