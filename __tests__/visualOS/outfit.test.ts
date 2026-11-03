/**
 * outfit.test.ts — compatibility, diagnostics, transform, comparison,
 * color/proportion, missing piece, accessories.
 */
import { scoreCompatibility, diagnoseOutfit, type CompatItem } from '../../src/services/vision/compatibility';
import { colorFamily, colorRelationship, proportionNotes, compareOutfits, explainChoice, transformOutfit, missingPiece, accessorySuggestion } from '../../src/services/vision/styleLanguage';

const formal: CompatItem[] = [
  { id: 'blazer', category: 'top', primaryColor: 'navy', pattern: 'solid', formality01: 0.8, silhouette: 'regular', visualWeight: 'medium' },
  { id: 'trousers', category: 'bottom', primaryColor: 'charcoal', pattern: 'solid', formality01: 0.8, silhouette: 'slim', visualWeight: 'medium' },
  { id: 'oxfords', category: 'shoes', primaryColor: 'black', pattern: 'solid', formality01: 0.85, silhouette: 'sleek', visualWeight: 'medium' },
];
const clash: CompatItem[] = [
  { id: 'top', category: 'top', primaryColor: 'red', pattern: 'floral', formality01: 0.7, silhouette: 'oversized', visualWeight: 'light' },
  { id: 'bottom', category: 'bottom', primaryColor: 'green', pattern: 'striped', formality01: 0.7, silhouette: 'wide', visualWeight: 'heavy' },
  { id: 'sneakers', category: 'shoes', primaryColor: 'white', pattern: 'solid', formality01: 0.2, silhouette: 'chunky', visualWeight: 'heavy' },
];

test('compatible outfit: strengths, no hard conflicts', () => {
  const c = scoreCompatibility(formal, { occasion: 'office', tempC: 18 });
  expect(c.hardConflicts).toHaveLength(0);
  expect(c.strengths.length).toBeGreaterThan(0);
  expect(c.dimensions.length).toBeGreaterThan(5);
  expect(c.dimensions.every((d) => d.score >= 0 && d.score <= 1)).toBe(true);
});

test('valid outfit returns no forced criticism', () => {
  expect(diagnoseOutfit(formal, { occasion: 'office', tempC: 18 })).toHaveLength(0);
});

test('diagnostics: context, formality, shoe mismatch named actionably', () => {
  const d = diagnoseOutfit(clash, { occasion: 'formal', tempC: 20 });
  const issues = d.map((x) => x.issue);
  expect(issues).toContain('context mismatch');
  expect(issues).toContain('shoe formality mismatch');
  expect(d.every((x) => x.actionableFix && x.actionableFix.length > 10)).toBe(true);
  expect(d.some((x) => x.issue === 'palette conflict')).toBe(true);
});

test('cold without outerwear is a hard conflict', () => {
  const c = scoreCompatibility(formal, { tempC: 2 });
  expect(c.hardConflicts.join(' ')).toMatch(/outer layer/);
});

test('color relationships without pseudo-science', () => {
  expect(colorFamily('Navy')).toBe('blue');
  expect(colorRelationship('black', 'navy')).toBe('neutral-bridged');
  expect(colorRelationship('navy', 'navy')).toBe('monochrome');
  expect(colorRelationship('red', 'green')).toBe('contrasting');
  expect(colorRelationship(undefined, 'red')).toBe('unknown');
});

test('proportion notes are garment-to-garment, never body judgment', () => {
  const notes = proportionNotes([
    { id: 'coat', category: 'outerwear', silhouette: 'long' },
    { id: 'trousers', category: 'bottom', silhouette: 'wide' },
  ]);
  expect(notes.join(' ').toLowerCase()).not.toMatch(/body|weight-loss|diet|figure/);
  expect(notes.length).toBeGreaterThan(0);
});

test('comparison says better-for-X per category', () => {
  const cmp = compareOutfits({ id: 'A', items: formal }, { id: 'B', items: clash }, { occasion: 'office' });
  const formality = cmp.find((c) => c.category === 'formality_coherence');
  expect(formality?.winner).toBe('A');
  expect(formality?.note).toMatch(/better for/);
});

test('transformation: minimal one-swap, anchors preserved, budget kept', () => {
  const wardrobe: CompatItem[] = [
    ...formal,
    { id: 'loafers', category: 'shoes', primaryColor: 'brown', formality01: 0.55, silhouette: 'sleek', price: 120 } as CompatItem & { price?: number },
    { id: 'derby', category: 'shoes', primaryColor: 'black', formality01: 0.75, silhouette: 'sleek', price: 900 } as CompatItem & { price?: number },
  ];
  const t = transformOutfit(
    formal.map((f) => ({ ...f, price: 100 })),
    wardrobe.map((w) => ({ ...w, price: (w as { price?: number }).price ?? 100 })),
    'more_casual', { anchors: ['blazer'], budget: 200, wardrobeOnly: true },
  );
  expect(t.swappedOut).toHaveLength(1);
  expect(t.preservedAnchors).toContain('blazer');
  expect(t.swappedIn).toEqual(['loafers']); // derby over budget
  expect(t.rationale.join(' ')).toMatch(/Wardrobe-only/);
  const none = transformOutfit(formal, [], 'more_formal');
  expect(none.swappedOut).toHaveLength(0);
});

test('missing piece: shoes flagged; complete outfits need nothing', () => {
  expect(missingPiece([{ id: 't', category: 'top' }]).suggestion).toBe('shoes');
  const full = missingPiece(formal);
  expect(full.suggestion).toBeNull();
  expect(full.reason).toMatch(/Nothing needed/);
});

test('accessories: wardrobe-first, quiet when outfit is busy', () => {
  const acc = [{ id: 'belt', category: 'accessory', pattern: 'solid' }, { id: 'scarf', category: 'accessory', pattern: 'paisley' }];
  const pick = accessorySuggestion(clash, acc);
  expect(pick.accessoryId).toBe('belt');
  expect(accessorySuggestion(formal, []).accessoryId).toBeNull();
  expect(explainChoice({ id: 'x', category: 'outerwear', silhouette: 'cropped' }, formal).join(' ')).toMatch(/proportion/);
});
