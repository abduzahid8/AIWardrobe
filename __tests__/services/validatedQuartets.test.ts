import { validatedQuartets } from '../../src/domain/validatedQuartets';
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
