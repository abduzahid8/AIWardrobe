/**
 * evidence.test.ts — §43: StyleEvidence positive/negative/context/decay/
 * override/conflict; DNA stability/evolution/context/novelty.
 */
import {
  createStyleEvidence,
  dedupeStyleEvidence,
  evidenceWeight,
  isPermanentCandidate,
  layerStyleEvidence,
} from '../../src/services/style/styleEvidence';
import { aggregateStyleDNA, detectStyleEvolution } from '../../src/services/style/styleDna';
import {
  buildContextualProfile,
  contextConstraints,
} from '../../src/services/style/contextualProfiles';
import {
  manualOverrideEvidence,
  noveltyFeedbackEvidence,
} from '../../src/services/style/wardrobeAgent';
import { NOW, daysAgo } from './helpers';

const base = {
  userId: 'u1',
  entityType: 'color' as const,
  entityKey: 'color:navy',
  provenance: 'test',
  // Anchor every fixture row to the frozen test clock. Production rejects
  // future-dated evidence (observedAt > now + 1 day), so the createStyleEvidence
  // wall-clock default would rot as real time moves past NOW.
  observedAt: new Date(NOW).toISOString(),
};

test('positive and negative evidence carry opposite polarity with bounded weight', () => {
  const pos = createStyleEvidence({ ...base, sourceType: 'outfit_worn', polarity: 'positive' });
  const neg = createStyleEvidence({ ...base, sourceType: 'outfit_disliked', polarity: 'negative' });
  expect(pos.polarity).toBe('positive');
  expect(neg.polarity).toBe('negative');
  expect(evidenceWeight(pos, NOW)).toBeGreaterThan(0);
  expect(evidenceWeight(neg, NOW)).toBeGreaterThan(0);
});

test('expiry kills weight; one-off clicks are not permanent', () => {
  const expired = createStyleEvidence({
    ...base, sourceType: 'outfit_worn', observedAt: daysAgo(10),
    expiresAt: daysAgo(1),
  });
  expect(evidenceWeight(expired, NOW)).toBe(0);
  const click = createStyleEvidence({ ...base, sourceType: 'product_saved' });
  expect(isPermanentCandidate(click)).toBe(false);
  const declared = createStyleEvidence({
    ...base, sourceType: 'declared_preference', confidence: 0.9,
  });
  expect(isPermanentCandidate(declared)).toBe(true);
});

test('decay: old observed evidence weighs less than fresh evidence', () => {
  const fresh = createStyleEvidence({ ...base, sourceType: 'outfit_worn', observedAt: daysAgo(2) });
  const old = createStyleEvidence({ ...base, sourceType: 'outfit_worn', observedAt: daysAgo(300) });
  expect(evidenceWeight(fresh, NOW)).toBeGreaterThan(evidenceWeight(old, NOW));
});

test('dedupe keeps newest per key and is bounded', () => {
  const a = createStyleEvidence({ ...base, sourceType: 'outfit_worn', observedAt: daysAgo(5) });
  const b = createStyleEvidence({ ...base, sourceType: 'outfit_worn', observedAt: daysAgo(1) });
  const out = dedupeStyleEvidence([a, b], NOW);
  expect(out).toHaveLength(1);
  expect(out[0].observedAt).toBe(b.observedAt);
});

test('layers separate baseline / recent / override / contextual', () => {
  const rows = [
    createStyleEvidence({ ...base, sourceType: 'manual_feedback' }),
    createStyleEvidence({ ...base, sourceType: 'outfit_worn', observedAt: daysAgo(100) }),
    createStyleEvidence({ ...base, sourceType: 'outfit_worn', observedAt: daysAgo(2) }),
    createStyleEvidence({
      ...base, sourceType: 'outfit_worn', observedAt: daysAgo(2),
      context: { contextId: 'gym' },
    }),
  ];
  const layers = layerStyleEvidence(rows, { contextId: 'gym' });
  expect(layers.override).toHaveLength(1);
  expect(layers.baseline).toHaveLength(1);
  expect(layers.recent).toHaveLength(1);
  expect(layers.contextual).toHaveLength(1);
});

test('DNA aggregates with confidence, provenance, recency, contradiction', () => {
  const rows = [
    createStyleEvidence({ ...base, sourceType: 'declared_preference', confidence: 0.9, strength: 0.9, observedAt: daysAgo(3) }),
    createStyleEvidence({ ...base, sourceType: 'outfit_worn', confidence: 0.8, strength: 0.8, observedAt: daysAgo(2) }),
    createStyleEvidence({ ...base, sourceType: 'outfit_worn', confidence: 0.8, strength: 0.8, observedAt: daysAgo(1) }),
    createStyleEvidence({ ...base, sourceType: 'outfit_worn', confidence: 0.8, strength: 0.8, entityKey: 'color:red', observedAt: daysAgo(1) }),
  ];
  const dna = aggregateStyleDNA('u1', rows, NOW);
  const dim = dna.dimensions.preferred_colors;
  expect(dim.value).toBe('navy');
  expect(dim.confidence).not.toBe('low');
  expect(dim.provenance).toContain('declared_preference');
  expect(dim.contradiction).toBeGreaterThan(0);
  expect(dim.recency).toBeTruthy();
});

test('conflicting evidence raises contradiction without flipping silently', () => {
  const rows = [
    createStyleEvidence({ ...base, sourceType: 'outfit_worn', polarity: 'positive', observedAt: daysAgo(1) }),
    createStyleEvidence({ ...base, sourceType: 'outfit_disliked', polarity: 'negative', observedAt: daysAgo(1) }),
  ];
  const dna = aggregateStyleDNA('u1', rows, NOW);
  expect(dna.dimensions.preferred_colors.contradiction).toBeGreaterThan(0.3);
});

test('evolution requires sufficient recent weight; one signal is not change', () => {
  const rows = [
    createStyleEvidence({ ...base, sourceType: 'outfit_worn', entityKey: 'color:navy', observedAt: daysAgo(200) }),
    createStyleEvidence({ ...base, sourceType: 'outfit_worn', entityKey: 'color:navy', observedAt: daysAgo(190) }),
    createStyleEvidence({ ...base, sourceType: 'outfit_worn', entityKey: 'color:red', observedAt: daysAgo(2) }),
  ];
  // single recent red signal — insufficient weight
  expect(detectStyleEvolution('u1', rows, {}, NOW)).toHaveLength(0);
  const strong = [
    ...rows,
    ...Array.from({ length: 6 }, (_, i) =>
      createStyleEvidence({
        userId: 'u1', sourceType: 'outfit_worn', entityType: 'color',
        entityKey: 'color:red', confidence: 0.9, strength: 0.9,
        observedAt: daysAgo(i + 1), provenance: 'test',
      })),
  ];
  const signals = detectStyleEvolution('u1', strong, {}, NOW);
  expect(signals.length).toBeGreaterThan(0);
  expect(signals[0].emergingValue).toBe('red');
  expect(signals[0].evidenceWindow.count).toBeGreaterThan(1);
});

test('contextual profile uses scoped evidence; strong global survives weak local', () => {
  const global = aggregateStyleDNA('u1', [
    createStyleEvidence({ ...base, sourceType: 'declared_preference', confidence: 0.95, strength: 1, observedAt: daysAgo(2) }),
    createStyleEvidence({ ...base, sourceType: 'declared_preference', confidence: 0.95, strength: 1, observedAt: daysAgo(1) }),
  ], NOW);
  const scoped = buildContextualProfile('gym', undefined, global, [
    createStyleEvidence({
      userId: 'u1', sourceType: 'outfit_worn', entityType: 'color',
      entityKey: 'color:neon', confidence: 0.4, strength: 0.4,
      context: { contextId: 'gym' }, observedAt: daysAgo(1), provenance: 't',
    }),
  ], NOW);
  // weak local signal must not override strong declared global
  expect(scoped.dimensions.preferred_colors.value).toBe('navy');
  expect(contextConstraints('gym')).toContain('activity:gym');
});

test('experimental dislike is quarantined; manual override outranks inference', () => {
  const [q] = noveltyFeedbackEvidence('u1', 'o1', false, 'experimental', 'gym');
  expect(q.expiresAt).toBeTruthy();
  expect(q.strength).toBeLessThan(0.5);
  const manual = manualOverrideEvidence('u1', 'stop_recommending', 'style_tag:neon');
  expect(manual.confidence).toBe(0.95);
  expect(manual.polarity).toBe('negative');
});
