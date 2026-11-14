// Analyze redacted private hosted receipts. Self-reported visibility never
// becomes a ground truth label. Adjudicated flags are supplied separately.
import fs from 'node:fs';

const [directory, annotationsPath] = process.argv.slice(2);
const read = (name) => JSON.parse(fs.readFileSync(name, 'utf8'));
const adjudicated = annotationsPath ? read(annotationsPath) : {};
const scores = new Map();
const weights = { visualCohesion: .30, referenceFidelity: .30, styleFit: .15, occasionFit: .15, formalityFit: .10 };
const mandatory = ['receipt.json', 'follow-up/receipt.json', 'coverage-check/receipt.json'];
const followups = ['structured-evidence/receipt.json', 'structured-json-mode/receipt.json', 'reverse-order-single-wave/receipt.json'];
const grounding = fs.readdirSync(directory).filter((file) => /^grounding-[a-f0-9]{64}-(?:control|grounded)\.receipt\.json$/.test(file)).sort();
const files = [...mandatory, ...followups.filter((file) => fs.existsSync(directory + '/' + file)), ...grounding];
function timing(events, field) {
  const samples = events.map((event) => event[field]).filter(Number.isFinite).sort((a, b) => a - b);
  return samples.length ? {
    n: samples.length,
    p50: samples[Math.ceil(samples.length * .50) - 1],
    p95: samples[Math.ceil(samples.length * .95) - 1],
  } : null;
}
const runs = files.map((file) => {
  const run = read(directory + '/' + file);
  for (const row of run.verdicts) {
    const score = Object.entries(weights).reduce((sum, [key, weight]) => sum + row[key] * weight, 0);
    if (!scores.has(row.candidateId)) scores.set(row.candidateId, []);
    scores.get(row.candidateId).push({ file, score, reject: row.reject, confidence: row.confidence });
  }
  const falseMissing = adjudicated[file]?.falseMissing || [];
  for (const { candidateId, slot } of falseMissing) {
    if (!run.verdicts.some((row) => row.candidateId === candidateId) ||
        !['top', 'outerwear', 'bottom', 'shoes'].includes(slot)) throw new Error('invalid_review_annotation');
  }
  const classes = {};
  for (const event of run.events) {
    const label = event.reason?.includes('deadline') ? 'timeout' : event.outcome;
    (classes[label] ??= []).push(event);
  }
  const outcomes = Object.fromEntries(Object.entries(classes).map(([label, events]) => [label, {
    n: events.length, tokenReports: events.filter((event) => event.usage?.total_tokens !== undefined).length,
    reportedTokens: events.reduce((sum, event) => sum + (event.usage?.total_tokens || 0), 0),
    queueMs: timing(events, 'queueMs'), mediaMs: timing(events, 'mediaMs'),
    providerMs: timing(events, 'providerMs'), parseMs: timing(events, 'parseMs'),
    validationMs: timing(events, 'validationMs'),
  }]));
  return { file, arm: run.arm, criticVersion: run.criticVersion, calls: run.calls, wallMs: run.wallMs, valid: run.verdicts.length,
    http200: run.events.filter(({ httpStatus }) => httpStatus === 200).length,
    abstained: run.diagnostics?.visualAbstained ?? run.events.filter(({ outcome }) => outcome === 'abstain').length,
    rejected: run.diagnostics?.visualRejected ?? run.verdicts.filter(({ reject }) => reject).length,
    selected: run.winner ?? null, falseMissing, outcomes,
    verifiedSharedSlots: run.verdicts.reduce((sum, row) => sum + (row.verifiedSharedSlots?.length ?? 0), 0), abstentionReasons: run.events.filter((event) => event.outcome === 'abstain').map((event) => event.reason),
    reportedTokens: run.events.reduce((sum, event) => sum + (event.usage?.total_tokens || 0), 0),
  };
});
console.log(JSON.stringify({ runs, qualityByCandidate: Object.fromEntries(scores),
  independentFashionJudgments: 0, falseRejections: null, falseApprovals: null }, null, 2));
