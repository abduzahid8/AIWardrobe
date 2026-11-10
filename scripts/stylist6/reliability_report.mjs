// Analyze redacted private hosted receipts. Self-reported visibility never
// becomes a ground truth label. Adjudicated flags are supplied separately.
import fs from 'node:fs';

const [directory, annotationsPath] = process.argv.slice(2);
const read = (name) => JSON.parse(fs.readFileSync(name, 'utf8'));
const adjudicated = annotationsPath ? read(annotationsPath) : {};
const scores = new Map();
const weights = { visualCohesion: .30, referenceFidelity: .30, styleFit: .15, occasionFit: .15, formalityFit: .10 };
const runs = ['receipt.json', 'follow-up/receipt.json', 'coverage-check/receipt.json'].map((file) => {
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
  return { file, calls: run.calls, wallMs: run.wallMs, valid: run.verdicts.length,
    abstained: run.diagnostics.visualAbstained, rejected: run.diagnostics.visualRejected,
    selected: run.winner, falseMissing, abstentionReasons: run.events.filter((event) => event.outcome === 'abstain').map((event) => event.reason),
    reportedTokens: run.events.reduce((sum, event) => sum + (event.usage?.total_tokens || 0), 0),
  };
});
console.log(JSON.stringify({ runs, qualityByCandidate: Object.fromEntries(scores),
  independentFashionJudgments: 0, falseRejections: null, falseApprovals: null }, null, 2));
