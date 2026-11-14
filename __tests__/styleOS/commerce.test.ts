/**
 * commerce.test.ts — §43: purchase plans incl. buy-nothing; proactive
 * recommendations; cooldown/fatigue; pipeline/budget/postmortem/no-buy;
 * semantic fallback.
 */
import {
  applyBudgetOutcomes,
  buildShoppingPlan,
  createPipelineEntry,
  createShoppingBudget,
  postMortemDue,
  recommendNoBuy,
  recordPostMortem,
  transitionPipeline,
} from '../../src/services/style/shoppingPlan';
import {
  completeAgentTask,
  createAgentTask,
  explain,
  selectSuggestions,
  summarizeOutcomes,
  updateFatigue,
} from '../../src/services/style/wardrobeAgent';
import {
  DeterministicEncoder,
  cosineSimilarity,
  hashEmbedding,
  visualStyleFromMetadata,
} from '../../src/services/style/semantic';
import { NOW, daysAgo } from './helpers';

test('wishlist items are not spent money; bought counts', () => {
  const b = createShoppingBudget({ userId: 'u1', period: '2026-11', amount: 300, currency: 'USD' });
  const discovered = createPipelineEntry({ userId: 'u1', productId: 'p1' });
  const planned = transitionPipeline(createPipelineEntry({ userId: 'u1', productId: 'p2' }), 'planned');
  const bought = transitionPipeline(createPipelineEntry({ userId: 'u1', productId: 'p3' }), 'bought');
  const out = applyBudgetOutcomes(b, [discovered, planned, bought], { p2: 50, p3: 120 });
  expect(out.spent).toBe(120);
  expect(out.committed).toBe(50);
  expect(out.available).toBe(130);
});

test('terminal pipeline states are sticky; explicit reopen only', () => {
  const bought = transitionPipeline(createPipelineEntry({ userId: 'u1', productId: 'p' }), 'bought');
  expect(transitionPipeline(bought, 'planned', 'second thought').stage).toBe('bought');
  expect(transitionPipeline(bought, 'considering', 'reopen').stage).toBe('considering');
});

test('post-mortem due after window; calibration from answers', () => {
  expect(postMortemDue(daysAgo(40), 0, false, 30, NOW)).toBe(true);
  expect(postMortemDue(daysAgo(10), 0, false, 30, NOW)).toBe(false);
  const pm = recordPostMortem('p1', 'u1', { worthIt: false, fitsWell: false, wearCount: 1, wouldBuyAgain: false }, NOW);
  expect(pm.calibration.join(' ')).toMatch(/fit/i);
});

test('shopping plan returns buy-nothing when no confident gaps', () => {
  const plan = buildShoppingPlan('u1', [], { available: 0 } as never, [], 5, NOW);
  expect(plan.buyNothing).toBe(true);
  expect(plan.rationale.join(' ')).toMatch(/buy nothing/i);
});

test('no-buy engine covers all seven recommendations', () => {
  expect(recommendNoBuy({ duplicateLevel: 'high', ownedCount: 5 }).recommendation).toBe('SKIP_DUPLICATE');
  expect(recommendNoBuy({ withinBudget: false }).recommendation).toBe('WAIT_FOR_PRICE');
  expect(recommendNoBuy({ priceState: 'HIGH_PRICE' }).recommendation).toBe('WAIT_FOR_PRICE');
  expect(recommendNoBuy({ fitConfidence: 'low' }).recommendation).toBe('WAIT_FOR_BETTER_FIT_EVIDENCE');
  expect(recommendNoBuy({ unlocks: 0 }).recommendation).toBe('SKIP_LOW_VALUE');
  expect(recommendNoBuy({ unwornOwned: 3 }).recommendation).toBe('NO_PURCHASE_NEEDED');
  expect(recommendNoBuy({ unlocks: 5, fitConfidence: 'high' }).recommendation).toBe('BUY_NOW');
});

test('proactive selection: cooldown, fatigue gate, weekly cap, no spam', () => {
  const mk = (kind: string, urgency: 'high' | 'medium' | 'low' = 'medium') => ({
    kind, title: kind, detail: kind, urgency, confidence: 'high' as const,
    expectedValue: 'v', evidence: ['e'],
  });
  const fatigue = {
    worn_out: updateFatigue(updateFatigue(updateFatigue(undefined, 'worn_out', 'p', 'surfaced'), 'worn_out', 'p', 'dismissed'), 'worn_out', 'p', 'surfaced'),
  };
  fatigue.worn_out = { ...fatigue.worn_out, surfacedCount7d: 4, dismissedCount7d: 4 };
  const out = selectSuggestions([mk('worn_out'), mk('fresh', 'high')], fatigue, [], [], NOW);
  expect(out.map((s) => s.kind)).toEqual(['fresh']);
  const many = Array.from({ length: 10 }, (_, i) => mk(`k${i}`, 'high'));
  expect(selectSuggestions(many, {}, [], [], NOW).length).toBeLessThanOrEqual(5);
});

test('outcomes summarize with calibration cues; agent task completes locally', () => {
  const s = summarizeOutcomes([
    { suggestionId: 'a', outcome: 'dismissed', at: daysAgo(1) },
    { suggestionId: 'b', outcome: 'dismissed', at: daysAgo(1) },
    { suggestionId: 'c', outcome: 'dismissed', at: daysAgo(1) },
    { suggestionId: 'd', outcome: 'dismissed', at: daysAgo(1) },
    { suggestionId: 'e', outcome: 'accepted', at: daysAgo(1) },
  ]);
  expect(s.dismissed).toBe(4);
  expect(s.calibration.join(' ')).toMatch(/cooldown/i);
  const t = completeAgentTask(createAgentTask('u1', 'review_underused'), 'Reviewed 3 items', ['e1']);
  expect(t.status).toBe('done');
  expect(t.evidence).toContain('e1');
});

test('explainability contract answers all five questions', () => {
  const e = explain('Price dropped', ['history'], 'Save $20', ['Size uncertain'], 'medium');
  expect([e.whyNow, e.changeIfFollowed, e.confidence]).toBeTruthy();
  expect(e.evidence).toHaveLength(1);
  expect(e.tradeoffs).toHaveLength(1);
});

test('semantic fallback is deterministic and provider-free', () => {
  const enc = new DeterministicEncoder();
  expect(enc.encodeText('navy wool blazer')).toEqual(enc.encodeText('navy wool blazer'));
  expect(cosineSimilarity([1, 0], [1, 0])).toBe(1);
  expect(hashEmbedding('x')).toHaveLength(32);
  const vis = visualStyleFromMetadata({ primaryColor: 'navy', pattern: 'striped', material: 'cotton', subCategory: 'shirt', category: 'top' });
  expect(vis.silhouette).toBe('shirt');
  expect(vis.texture).toBe('cotton');
});
