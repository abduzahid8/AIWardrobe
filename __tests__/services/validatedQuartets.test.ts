import {
  NVIDIA_RANK_UTILITY_WEIGHT,
  selectedPlanDiagnostics,
  validatedQuartets,
} from '../../src/domain/validatedQuartets';
import { stylingContext } from '../../src/domain/stylingContext';
import { fourSlotOutfitReason } from '../../src/domain/outfitStructure';
const pool = ['Cotton shirt', 'Fine cardigan', 'Tailored trousers', 'Leather loafers'].flatMap(
  (name, s) =>
    Array.from({ length: 6 }, (_, i) => ({
      id: `${s}-${i}`,
      name: `${name} model ${i}`,
      macroCategory: ['top', 'outerwear', 'bottom', 'shoes'][s],
    }))
);
it('searches valid quartets and preserves all required anchors', () => {
  for (const slot of [0, 1, 2, 3]) {
    const id = `${slot}-5`;
    const result = validatedQuartets(
      pool,
      stylingContext({ style: 'minimalist', occasion: 'Date night' }),
      (p) => !fourSlotOutfitReason(p),
      { requiredIds: [id] }
    );
    expect(result.length).toBeGreaterThan(2);
    for (const r of result) {
      expect(r.items).toHaveLength(4);
      expect(r.items.some((i) => i.id === id)).toBe(true);
    }
  }
});
it('bounds hostile validator work and never fills missing shoes', () => {
  const accepts = jest.fn(() => false);
  expect(validatedQuartets(pool, stylingContext({}), accepts, { searchLimit: 25 })).toEqual([]);
  expect(accepts.mock.calls.length).toBeLessThanOrEqual(25);
  expect(
    validatedQuartets(
      pool.filter((i) => i.macroCategory !== 'shoes'),
      stylingContext({}),
      () => true
    )
  ).toEqual([]);
});
it('stops the Edge reserve at its admitted size and caps rejected attempts', () => {
  const accepts = jest.fn(() => true);
  const result = validatedQuartets(pool, stylingContext({}), accepts,
    {maximum: 3, reserveMultiplier: 1, searchLimit: 12});
  expect(result).toHaveLength(3);
  expect(accepts.mock.calls.length).toBe(3);
  const rejects = jest.fn(() => false);
  expect(validatedQuartets(pool, stylingContext({}), rejects,
    {maximum: 24, reserveMultiplier: 1, searchLimit: 12})).toEqual([]);
  expect(rejects.mock.calls.length).toBeLessThanOrEqual(12);
});
it('does not override caller budget/source validation', () => {
  const result = validatedQuartets(pool, stylingContext({}), (p) =>
    p.every((i) => !i.id.endsWith('-0'))
  );
  expect(result.length).toBeGreaterThan(0);
  expect(result.every((r) => r.items.every((i) => !i.id.endsWith('-0')))).toBe(true);
});
it('replays deterministically and rejects physical aliases across slots', () => {
  const c = stylingContext({});
  expect(validatedQuartets(pool, c, () => true)).toEqual(validatedQuartets(pool, c, () => true));
  const aliases = pool.map((i) => ({
    ...i,
    name: 'Same actual product',
    imageUrl: 'https://cdn.test/same.jpg',
  }));
  expect(validatedQuartets(aliases, c, () => true)).toEqual([]);
});
it('enforces explicit source mode before constructing quartets', () => {
  const shop = pool.map((i) => ({ ...i, id: 'shop_' + i.id, isShopItem: true }));
  expect(validatedQuartets(shop, stylingContext({ sourceMode: 'wardrobe' }), () => true)).toEqual(
    []
  );
  expect(validatedQuartets(pool, stylingContext({ sourceMode: 'catalog' }), () => true)).toEqual(
    []
  );
  expect(
    validatedQuartets(
      [...pool, ...shop],
      stylingContext({ sourceMode: 'catalog' }),
      () => true
    ).every((c) => c.items.every((i) => i.id.startsWith('shop_')))
  ).toBe(true);
});

it('reports diagnostics from the selected deck rather than the planner seed', () => {
  const looks = [
    { concept: { id: 'minimalist' }, identity: 'clean-separates' },
    { concept: { id: 'classic' }, identity: 'tailored-contrast' },
  ];
  expect(selectedPlanDiagnostics(looks)).toEqual({
    conceptsAchieved: 2,
    editorialIdentities: 2,
  });
  expect(selectedPlanDiagnostics([looks[0], looks[0]])).toEqual({
    conceptsAchieved: 1,
    editorialIdentities: 1,
  });
});

it('keeps provider rank as a bounded quality tie-breaker', () => {
  expect(NVIDIA_RANK_UTILITY_WEIGHT).toBeGreaterThan(0);
  expect(NVIDIA_RANK_UTILITY_WEIGHT).toBeLessThan(0.035);
});
