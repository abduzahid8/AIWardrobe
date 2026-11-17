import {
  rankDiverseShoppingCandidates,
  compareShoppingAlternative,
  duplicateRiskEvidence,
  wardrobeValueInputs,
} from '../../src/services/shoppingIntelligence';
import type { ShopCatalogItem } from '../../src/types/shop';
const item = (
  id: string,
  name: string,
  garmentType: ShopCatalogItem['garmentType'] = 'shoes'
): ShopCatalogItem => ({
  id,
  name,
  garmentType,
  brand: 'Brand',
  price: 100,
  currency: 'USD',
  imageUrl: `https://img.test/${id}.jpg`,
});
test('top-heavy pools leave room for shoes and different shoe roles; models cannot occupy the first pass', () => {
  const tops = Array.from({ length: 30 }, (_, i) =>
    item('t' + i, 'Macharles Shirt ' + (i % 2 ? 'Brown' : 'Black'), 'upper_body')
  );
  const shoes = [
    item('l', 'Penny Loafer Black'),
    item('s', 'Trainer White'),
    item('b', 'Chelsea Boot Brown'),
  ];
  const result = rankDiverseShoppingCandidates(
    [...tops, ...shoes].map((item) => ({ item, score: 1 })),
    { limit: 4, eligible: () => true }
  );
  expect(result.filter((i) => i.garmentType === 'shoes')).toHaveLength(3);
});
test('eligibility is hard, recent rotation soft, anchor and budget representatives survive', () => {
  const candidates = ['anchor', 'cheap', 'avoided', 'wrong-style', 'old'].map((id) => ({
    item: item(id, 'Penny Loafer Black'),
    score: id === 'old' ? 100 : 1,
    anchor: id === 'anchor',
    budgetRepresentative: id === 'cheap',
  }));
  const result = rankDiverseShoppingCandidates(candidates, {
    limit: 3,
    eligible: (i) => !['avoided', 'wrong-style'].includes(i.id),
    recentIds: new Set(['old']),
  });
  expect(result.map((i) => i.id).sort()).toEqual(['anchor', 'cheap', 'old']);
  expect(() =>
    rankDiverseShoppingCandidates(candidates, { limit: 1, eligible: () => true })
  ).toThrow();
});
test('comparison requires observed currencies and known colors, with no invented price history or ROI', () => {
  const a = item('a', 'Penny Loafer Black'),
    b = { ...item('b', 'Penny Loafer Black'), price: 80 };
  expect(compareShoppingAlternative(a, b).priceSaving).toBe(20);
  expect(compareShoppingAlternative(a, { ...b, currency: 'EUR' }).priceSaving).toBeUndefined();
  expect(duplicateRiskEvidence(a, [b])).toHaveLength(1);
  expect(duplicateRiskEvidence(item('x', 'Loafer'), [b])).toEqual([]);
  expect(
    wardrobeValueInputs(['one', 'one', 'two'], ['work', 'work'], true).validatedDistinctOutfits
  ).toBe(2);
});
