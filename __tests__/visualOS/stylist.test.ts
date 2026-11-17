/**
 * stylist.test.ts — intent routing, plan, orchestration, daily/weekly,
 * safety (flags, telemetry, shadow, sanitization, cost policy).
 */
import { classifyIntent, buildStylistPlan, orchestrateLocal, dailyStylist, weeklyPlan } from '../../src/services/vision/stylist';
import { isVisualEnabled, setVisualFlag, resetVisualFlags, recordTelemetry, telemetryLog, isLogSafe, shadowCompare, sanitizeProviderRequest, taskPolicy, metadataAlreadySufficient } from '../../src/services/vision/safety';
import type { CompatItem } from '../../src/services/vision/compatibility';

const items = (ids: string[]): CompatItem[] =>
  ids.map((id) => ({ id, category: id.startsWith('shoe') ? 'shoes' : 'top', primaryColor: 'navy', pattern: 'solid', formality01: 0.7, silhouette: 'regular', visualWeight: 'medium' as const }));
const okValidate = () => ({ ok: true, reasons: [] as string[] });

test('intent routing uses only required tools', () => {
  const daily = classifyIntent('Give me tomorrow\u2019s outfit');
  expect(daily.kind).toBe('daily_outfit');
  expect(daily.tools).toContain('wardrobe.query');
  expect(daily.tools).not.toContain('shopping.buyCheck');
  expect(classifyIntent('why does this look bad').kind).toBe('diagnose');
  expect(classifyIntent('recreate this pinterest look').kind).toBe('recreate');
  expect(classifyIntent('should I buy this').kind).toBe('shopping_advice');
  const plan = buildStylistPlan('tomorrow outfit', 24);
  expect(plan.candidateCounts.retrieved).toBeLessThanOrEqual(60);
  expect(plan.fallbackPath).toBe('structured-metadata-only');
});

test('orchestrator: validators authoritative; visual never overrides', () => {
  const plan = buildStylistPlan('x', 2);
  const fail = () => ({ ok: false, reasons: ['Too casual for formal.'] });
  const res = orchestrateLocal({ plan, candidates: [items(['a']), items(['b'])], hardValidate: fail });
  expect(res.outfitItemIds).toBeNull();
  expect(res.evidence).toEqual(['Too casual for formal.']);
  const good = orchestrateLocal({ plan, candidates: [items(['a', 'shoe1']), items(['b', 'shoe2'])], hardValidate: okValidate });
  expect(good.outfitItemIds).toHaveLength(2);
  expect(good.plan.evidenceSummary.length).toBeGreaterThan(0);
});

test('provider unavailable fallback keeps local function', () => {
  const plan = buildStylistPlan('x', 0);
  const res = orchestrateLocal({ plan, candidates: [], hardValidate: okValidate });
  expect(res.fallbacksUsed).toContain('structured-metadata-only');
  expect(res.confidence).toBe('low');
});

test('daily + weekly planning honor repetition and events', () => {
  const plan = buildStylistPlan('tomorrow', 4);
  const cands = [items(['a', 'shoe1']), items(['b', 'shoe2']), items(['a', 'shoe1'])];
  const daily = dailyStylist({ plan, candidates: cands, hardValidate: okValidate, context: { occasion: 'office', eventLabel: 'Standup' }, recentlyWornIds: ['a'] });
  expect(daily.outfitItemIds).not.toContain('a');
  expect(daily.preparationNotes.join(' ')).toMatch(/Standup/);
  const week = weeklyPlan({
    days: [{ label: 'Mon', occasion: 'office' }, { label: 'Tue', occasion: 'office' }],
    candidates: cands, hardValidate: () => ({ ok: true, reasons: [] }), maxRepeats: 1,
  });
  expect(week).toHaveLength(2);
  expect(week.every((d) => d.itemIds)).toBe(true);
});

test('flags default safely off; no remote config', () => {
  resetVisualFlags();
  expect(isVisualEnabled('stylist_orchestration')).toBe(false);
  setVisualFlag('outfit_diagnostics', true);
  expect(isVisualEnabled('outfit_diagnostics')).toBe(true);
  resetVisualFlags();
});

test('telemetry sanitized: no images, body data, or private URLs', () => {
  const e = recordTelemetry({ task: 'parse-garment', provider: 'deterministic-fallback', cacheHit: true, confidence: 0.8 });
  expect(telemetryLog()).toContain(e);
  expect(isLogSafe({ task: 'x', assetId: 'a1' })).toBe(true);
  expect(isLogSafe({ image: 'data:image/jpeg;base64,' + 'x'.repeat(5000) })).toBe(false);
  expect(isLogSafe({ waist: 72 })).toBe(false);
  expect(isLogSafe({ url: 'https://cdn.example.com/' + 'p'.repeat(60) })).toBe(false);
});

test('shadow mode keeps production authoritative', () => {
  const s = shadowCompare([1, 2], [1, 3], (a, b) => JSON.stringify(a) === JSON.stringify(b), 'compat-rank');
  expect(s.production).toEqual([1, 2]);
  expect(s.agree).toBe(false);
  expect(s.metric).toBe('compat-rank');
});

test('provider request sanitization strips sensitive metadata', () => {
  const r = sanitizeProviderRequest({ assetId: 'a', imageRef: 'file://x.jpg', task: 'parse-garment', maxResolution: 4000, metadata: { category: 'top', face: 'yes', location: 'home' } });
  expect(r?.maxResolution).toBe(1600);
  expect(r?.includeMetadata).toEqual({ category: 'top' });
  expect(taskPolicy('tryon').maxCalls).toBe(1);
  expect(metadataAlreadySufficient(['color', 'category'], ['color'])).toBe(true);
  expect(metadataAlreadySufficient(['color'], ['color', 'size'])).toBe(false);
});
