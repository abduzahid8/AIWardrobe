/**
 * inspiration.test.ts — reference looks, recreate, visual search,
 * moodboard, build-style, shopping match.
 */
import { saveReferenceLook, recreateLook, visualSearch, moodboardDirection, moodboardCoverage, buildThisStyle, visualShoppingMatch } from '../../src/services/vision/inspiration';

const wardrobe = [
  { id: 'trousers', role: 'trousers', category: 'bottom', color: 'charcoal', silhouette: 'slim', formality01: 0.8, available: true },
  { id: 'oxford', role: 'shirt', category: 'top', color: 'white', silhouette: 'regular', formality01: 0.8, available: true },
  { id: 'loafers', role: 'shoes', category: 'shoes', color: 'black', silhouette: 'sleek', formality01: 0.8, available: true },
  { id: 'tee', role: 'tee', category: 'top', color: 'white', silhouette: 'regular', formality01: 0.2, available: true },
];
const roles = [
  { role: 'trousers', category: 'bottom', color: 'charcoal', silhouette: 'slim', formality01: 0.8 },
  { role: 'shirt', category: 'top', color: 'white', silhouette: 'regular', formality01: 0.8 },
  { role: 'shoes', category: 'shoes', color: 'black', silhouette: 'sleek', formality01: 0.8 },
  { role: 'overshirt', category: 'outerwear', color: 'olive', silhouette: 'relaxed', formality01: 0.5 },
];

test('reference look saved without touching Style DNA', () => {
  const ref = saveReferenceLook({ userId: 'u', assetId: 'a1', sourceLabel: 'pinterest', contexts: ['office'] });
  expect(ref.id).toMatch(/^ref_/);
  expect(ref.recreatedOutfitIds).toEqual([]);
  expect(ref.styleRepresentation).toEqual({});
});

test('recreate: exact wardrobe match, partial, one missing piece, anchors', () => {
  const full = recreateLook('r1', roles.slice(0, 3), wardrobe);
  expect(full.coverage).toBe(1);
  expect(full.missingPieceSuggestion).toBeNull();
  expect(full.confidence).toBe('high');
  const partial = recreateLook('r1', roles, wardrobe);
  expect(partial.missingRoles).toEqual(['overshirt']);
  expect(partial.missingPieceSuggestion).toMatch(/only.*missing role|missing role/i);
  expect(partial.coverage).toBeCloseTo(0.75);
  const noMatch = recreateLook('r1', [{ role: 'gown', category: 'dress', color: 'red' }], wardrobe);
  expect(noMatch.closestWardrobeOutfit).toBeNull();
  expect(noMatch.confidence).toBe('low');
  const anchored = recreateLook('r1', roles.slice(0, 3), wardrobe, { anchors: ['tee'] });
  expect(anchored.closestWardrobeOutfit).toContain('tee');
});

test('visual search: bounded, family-diverse, price-gated', () => {
  const pool = [
    { id: 'j1', category: 'outerwear', silhouette: 'relaxed', color: 'olive', price: 150, familyKey: 'f1' },
    { id: 'j2', category: 'outerwear', silhouette: 'relaxed', color: 'olive', price: 90, familyKey: 'f1' },
    { id: 'j3', category: 'outerwear', silhouette: 'relaxed', color: 'tan', price: 80, familyKey: 'f2' },
    { id: 'x', category: 'shoes', price: 50 },
  ];
  const res = visualSearch({ category: 'outerwear', silhouette: 'relaxed', maxPrice: 100 }, pool);
  expect(res.map((r) => r.id)).toEqual(['j2', 'j3']);
  expect(res[0].why).toContain('Same silhouette');
});

test('moodboard direction + coverage; build-style caps leverage pieces', () => {
  const { direction } = moodboardDirection([
    { styleRepresentation: { colors: ['navy', 'grey'], footwear: ['loafers'] } },
    { styleRepresentation: { colors: ['navy', 'white'], footwear: ['loafers'] } },
  ]);
  expect(direction.colors?.[0]).toBe('navy');
  expect(direction.footwear).toEqual(['loafers']);
  const cov = moodboardCoverage(direction, [{ role: 'loafers', covered: true }, { role: 'overshirt', covered: false }]);
  expect(cov.covered).toEqual(['loafers']);
  expect(cov.missing).toEqual(['overshirt']);
  const plan = buildThisStyle([['a', 'b']], ['overshirt', 'belt', 'bag', 'hat']);
  expect(plan.leveragePieces).toHaveLength(3);
  expect(buildThisStyle([['a']], []).noBuyPath).toBe(true);
});

test('visual shopping match: duplicates skipped, gaps supported', () => {
  const dup = visualShoppingMatch({ productAttrs: {}, closestOwnedId: 'i1', unlocks: 0, dnaFit: 'high', duplicateLevel: 'high' });
  expect(dup.recommendation).toMatch(/Skip/);
  const good = visualShoppingMatch({ productAttrs: {}, unlocks: 4, dnaFit: 'high', duplicateLevel: 'low' });
  expect(good.recommendation).toMatch(/Consider/);
  expect(good.utility).toMatch(/Unlocks/);
});
