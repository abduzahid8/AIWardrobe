/**
 * decisions.test.ts — trace/provenance: stable unique ids, parent/child
 * linkage, sanitized events, registry governance.
 */
import {
  newRequestId, newTraceId, startTrace, childStep, summarizeTrace,
  provenance, buildDecisionEvent, validateAnalyticsEvent, registryNames,
} from '../../src/services/platform/decisions';

test('request ids are unique; parent/child trace links', () => {
  const a = newRequestId(), b = newRequestId();
  expect(a).not.toBe(b);
  const t = startTrace('buycheck');
  expect(t.traceId).toBeTruthy();
  expect(t.parentRequestId).toBeNull();
  const child = startTrace('fit-eval', t.requestId);
  expect(child.parentRequestId).toBe(t.requestId);
  expect(child.traceId).not.toBe(t.traceId);
  const step = childStep(t, 'fit.evaluate');
  expect(step.stepId).toContain('fit.evaluate');
});

test('trace summary exposes tools/rules/counts, never reasoning', () => {
  const t = startTrace('outfit');
  const s = summarizeTrace(t, [
    { ...childStep(t, 'wardrobe.query'), rulesInvoked: ['weather-gate'], candidateCount: 24, provider: 'local' },
    { ...childStep(t, 'outfit.validate'), rulesInvoked: ['budget'], candidateCount: 7, fallbackUsed: 'cache' },
  ]);
  expect(s.toolsUsed).toEqual(['wardrobe.query', 'outfit.validate']);
  expect(s.totalCandidates).toBe(31);
  expect(s.provider).toBe('local');
  expect(s.fallbackPath).toBe('cache');
  expect(JSON.stringify(s)).not.toMatch(/chain-of-thought|reasoning/i);
});

test('sensitive payloads are dropped, safe events pass with schema version', () => {
  const bad = buildDecisionEvent({
    eventName: 'buycheck_completed', userId: 'u', requestId: 'r', feature: 'shopping',
    decisionType: 'verdict', inputEvidenceSummary: ['data:image/jpeg;base64,xxxx'],
    rulesVersion: 'v2',
  });
  expect(bad).toBeNull();
  const good = buildDecisionEvent({
    eventName: 'buycheck_completed', userId: 'u', requestId: 'r', feature: 'shopping',
    decisionType: 'verdict', inputEvidenceSummary: ['duplicate high', 'unlocks 0'],
    rulesVersion: 'v2', confidence: 0.8, candidateCount: 5,
  });
  expect(good?.schemaVersion).toBe(1);
  expect(good?.timestamp).toBeTruthy();
});

test('provenance envelope carries versions for reproducibility', () => {
  const p = provenance({ rulesVersion: 'v2', modelProvider: 'edge', experimentId: 'exp1' });
  expect(p.rulesVersion).toBe('v2');
  expect(p.catalogVersion).toBeNull();
  expect(p.createdAt).toBeTruthy();
});

test('analytics registry validates fields, rejects prohibited and unknown', () => {
  expect(validateAnalyticsEvent('buycheck_completed', { requestId: 'r', verdict: 'SKIP' }).ok).toBe(true);
  const bad = validateAnalyticsEvent('buycheck_completed', { requestId: 'r', images: ['x'] });
  expect(bad.ok).toBe(false);
  expect(validateAnalyticsEvent('nope', {}).ok).toBe(false);
  expect(registryNames()).toContain('fit_feedback_received');
});

test('short private image URLs and undeclared diagnostic fields cannot escape the event boundary', () => {
 const base = { eventName: 'buycheck_completed' as const, userId: 'u', requestId: 'r', feature: 'shopping', decisionType: 'verdict', inputEvidenceSummary: ['https://x.io/p.jpg'], rulesVersion: 'v1' };
 expect(buildDecisionEvent(base)).toBeNull();
 const clean = buildDecisionEvent({ ...base, inputEvidenceSummary: ['duplicate high'], images: ['private'], bodyProfile: { secret: true } } as any);
 expect(clean).not.toHaveProperty('images');
 expect(clean).not.toHaveProperty('bodyProfile');
});
