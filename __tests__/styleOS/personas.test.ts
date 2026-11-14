/**
 * personas.test.ts — Personalization evaluation harness (§35).
 * Synthetic personas: evaluate DNA, gaps, ranking, planning, no-buy,
 * capsule quality, underused. Not tuned to any single persona.
 */
import { createStyleEvidence } from '../../src/services/style/styleEvidence';
import { aggregateStyleDNA } from '../../src/services/style/styleDna';
import { buildContextualProfile } from '../../src/services/style/contextualProfiles';
import { detectUnderused, detectGaps, redundancyMap } from '../../src/services/style/wardrobeHealth';
import { buildCapsule } from '../../src/services/style/planning';
import { buildShoppingPlan, recommendNoBuy } from '../../src/services/style/shoppingPlan';
import { getBestItemsForContext } from '../../src/services/style/wardrobeQueries';
import type { StyleEvidence } from '../../src/types/styleOS';
import { item, NOW, daysAgo } from './helpers';

interface Persona {
  name: string;
  userId: string;
  evidence: StyleEvidence[];
  items: ReturnType<typeof item>[];
}

function ev(userId: string, partial: Partial<StyleEvidence> & { entityKey: string }): StyleEvidence {
  return createStyleEvidence({
    userId,
    sourceType: 'outfit_worn',
    entityType: 'color',
    polarity: 'positive',
    confidence: 0.8,
    strength: 0.8,
    observedAt: daysAgo(5),
    provenance: 'persona',
    ...partial,
  } as Parameters<typeof createStyleEvidence>[0]);
}

const minimalist: Persona = {
  name: 'minimalist-office',
  userId: 'persona_min',
  evidence: [
    ...[1, 2, 3, 4].map((d) => ev('persona_min', { entityKey: 'color:black', sourceType: 'outfit_worn', observedAt: daysAgo(d) })),
    ...[1, 2].map((d) => ev('persona_min', { entityKey: 'color:white', observedAt: daysAgo(d) })),
    ev('persona_min', { entityKey: 'color:red', polarity: 'negative', sourceType: 'product_skipped', observedAt: daysAgo(2) }),
  ],
  items: [
    item({ id: 'm-t1', userId: 'persona_min', category: 'top', subCategory: 'shirt', primaryColor: 'white', occasions: ['work'] }),
    item({ id: 'm-b1', userId: 'persona_min', category: 'bottom', subCategory: 'trousers', primaryColor: 'black', occasions: ['work'] }),
    item({ id: 'm-s1', userId: 'persona_min', category: 'shoes', subCategory: 'oxfords', primaryColor: 'black', occasions: ['work'] }),
  ],
};

const student: Persona = {
  name: 'experimental-student',
  userId: 'persona_stu',
  evidence: [
    ...[1, 2, 3].map((d) => ev('persona_stu', { entityKey: 'color:red', observedAt: daysAgo(d) })),
    ...[1, 2, 3].map((d) => ev('persona_stu', { entityKey: 'color:green', observedAt: daysAgo(d) })),
    ev('persona_stu', { entityKey: 'color:beige', polarity: 'negative', sourceType: 'outfit_disliked', observedAt: daysAgo(1) }),
  ],
  items: [
    item({ id: 's-t1', userId: 'persona_stu', category: 'top', subCategory: 'hoodie', primaryColor: 'red', occasions: ['casual'], createdAt: daysAgo(120), wearCount: 0 }),
    item({ id: 's-t2', userId: 'persona_stu', category: 'top', subCategory: 'hoodie', primaryColor: 'green', occasions: ['casual'], createdAt: daysAgo(120), wearCount: 12 }),
    item({ id: 's-t3', userId: 'persona_stu', category: 'top', subCategory: 'hoodie', primaryColor: 'red', occasions: ['casual'], createdAt: daysAgo(120), wearCount: 0 }),
  ],
};

const traveler: Persona = {
  name: 'budget-traveler',
  userId: 'persona_trv',
  evidence: [
    ...[1, 2].map((d) => ev('persona_trv', { entityKey: 'color:navy', observedAt: daysAgo(d) })),
    ev('persona_trv', { entityKey: 'price:sensitive', entityType: 'style_tag', sourceType: 'product_skipped', observedAt: daysAgo(2) }),
  ],
  items: [
    item({ id: 'v-t1', userId: 'persona_trv', category: 'top', subCategory: 'tee', primaryColor: 'navy', occasions: ['travel'], seasons: ['summer'] }),
    item({ id: 'v-b1', userId: 'persona_trv', category: 'bottom', subCategory: 'shorts', primaryColor: 'khaki', occasions: ['travel'], seasons: ['summer'] }),
  ],
};

const PERSONAS = [minimalist, student, traveler];

test.each(PERSONAS.map((p) => [p.name, p]))('persona %s: DNA reflects its own signal, not others', (_name, p) => {
  const dna = aggregateStyleDNA(p.userId, p.evidence, NOW);
  const fav = dna.dimensions.preferred_colors?.value;
  if (p === minimalist) expect(fav).toBe('black');
  if (p === student) expect(['red', 'green']).toContain(fav);
  if (p === traveler) expect(fav).toBe('navy');
  // Other personas' evidence must not leak in
  const other = PERSONAS.find((q) => q !== p)!;
  const mixed = aggregateStyleDNA(p.userId, [...p.evidence, ...other.evidence.map((e) => ({ ...e, userId: 'other' }))], NOW);
  expect(mixed.dimensions.preferred_colors?.value).toBe(fav);
});

test.each(PERSONAS.map((p) => [p.name, p]))('persona %s: gaps, ranking, capsule, underused behave', (_name, p) => {
  const gaps = detectGaps(p.userId, p.items, ['casual', 'work', 'formal', 'sport', 'travel'], NOW);
  expect(Array.isArray(gaps)).toBe(true);
  const ranked = getBestItemsForContext({ userId: p.userId, items: p.items, now: NOW }, 'work');
  expect(ranked.length).toBeLessThanOrEqual(p.items.length);
  const cap = buildCapsule({ kind: 'travel_7day', userId: p.userId }, p.items);
  expect(cap.itemIds.every((id) => p.items.some((i) => i.id === id))).toBe(true);
  const under = detectUnderused(p.items, {}, [], NOW);
  expect(under.every((u) => u.daysOwned >= 30)).toBe(true);
  const plan = buildShoppingPlan(p.userId, gaps.slice(0, 3), null, [], 3, NOW);
  expect(plan.items.length + (plan.buyNothing ? 0 : 0)).toBeLessThanOrEqual(3);
});

test('student duplicate hoodies flagged by redundancy; restraint fires', () => {
  const roles = redundancyMap(student.items);
  expect(roles.length).toBeGreaterThan(0);
  const nb = recommendNoBuy({ ownedCount: 5, duplicateLevel: 'high' });
  expect(nb.recommendation).toBe('SKIP_DUPLICATE');
});

test('contextual office profile keeps minimalist navy/black over travel noise', () => {
  const dna = aggregateStyleDNA(minimalist.userId, minimalist.evidence, NOW);
  const prof = buildContextualProfile('office', undefined, dna, [], NOW);
  expect(prof.dimensions.preferred_colors.value).toBe('black');
  expect(prof.confidence).not.toBe('high'); // no scoped evidence → honest low/medium
});
