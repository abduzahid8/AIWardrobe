import {
  rankShoppingAlternatives,
  estimateUsefulOutfitUnlocks,
  evaluateWardrobeValue,
} from '../../src/services/shoppingEvaluation';
import { rankDiverseShoppingCandidates } from '../../src/services/shoppingIntelligence';
import type { ShopCatalogItem } from '../../src/types/shop';
const row = (
  id: string,
  name: string,
  garmentType: ShopCatalogItem['garmentType'] = 'shoes',
  price = 100
): ShopCatalogItem => ({
  id,
  name,
  brand: 'Brand ' + id,
  price,
  currency: 'USD',
  imageUrl: `https://img.test/${id}.jpg`,
  garmentType,
});
const reference = row('reference', 'Brown Leather Penny Loafer');
const owned = [
  row('layer', 'Navy Wool Blazer', 'upper_body'),
  row('shirt', 'White Cotton Shirt', 'upper_body'),
  row('pants', 'Navy Tailored Trousers', 'lower_body'),
];
const roles = (items: ShopCatalogItem[]) => items.length === 4;
const contexts = [{ id: 'work', occasion: 'Work' }];
test('goal ranking requires evidence and known currencies; unsupported fit and wardrobe claims stay absent', () => {
  const near = row('near', 'Brown Leather Loafer', 'shoes', 80),
    other = row('other', 'Black Chelsea Boot', 'shoes', 40),
    fx = { ...near, id: 'foreign', currency: 'EUR', price: 10 };
  const rank = (goal: Parameters<typeof rankShoppingAlternatives>[2]['goal']) =>
    rankShoppingAlternatives(reference, [other, near, fx], { goal, eligible: () => true });
  expect(rank('cheaper').map((result) => result.item.id)).not.toContain('foreign');
  expect(rank('closest-style')[0].item.id).toBe('near');
  expect(rank('fit')).toEqual([]);
  expect(rank('wardrobe')).toEqual([]);
  const fit = rankShoppingAlternatives(reference, [near, other], {
    goal: 'fit',
    eligible: () => true,
    signals: (item) => ({
      fit: {
        score: item.id === 'near' ? 0.95 : 0.2,
        confidence: 0.9,
        evidence: ['measured size chart'],
      },
    }),
  });
  expect(fit[0].item.id).toBe('near');
  expect(fit[0].evidence).toContain('measured size chart');
});
test('brand, model, shoe and color diversity stays inside the relevance quality band', () => {
  const candidates = [
    row('a', 'Brown Leather Loafer'),
    row('b', 'Brown Leather Loafer'),
    row('c', 'White Sneaker'),
    row('d', 'Black Boot'),
  ].map((item) => ({ item, score: 0.9 }));
  candidates[0].item.brand = candidates[1].item.brand = 'Dominant';
  const result = rankDiverseShoppingCandidates(candidates, { limit: 3, eligible: () => true });
  expect(new Set(result.map((item) => item.brand)).size).toBe(3);
  const high = [
    { item: row('high', 'Leather Loafer'), score: 1 },
    { item: row('high2', 'Leather Loafer Model Two'), score: 0.95 },
    { item: row('bad', 'Sneaker'), score: 0.1 },
  ];
  expect(
    rankDiverseShoppingCandidates(high, { limit: 2, eligible: () => true }).map((item) => item.id)
  ).toEqual(['high', 'high2']);
});
test('useful unlocks validate complete outfits and deduplicate near-identical owned partners', () => {
  const wardrobe = [...owned, row('shirt2', 'Another White Cotton Shirt', 'upper_body')];
  const validate = jest.fn(roles);
  const unlocks = estimateUsefulOutfitUnlocks(reference, wardrobe, {
    contexts,
    eligible: () => true,
    validateOutfit: validate,
  });
  expect(unlocks.outfits).toHaveLength(1);
  expect(validate).toHaveBeenCalled();
  expect(unlocks.outfits[0].items.map((item) => item.id)).toContain('reference');
  expect(evaluateWardrobeValue(reference, wardrobe, unlocks)).toMatchObject({
    fillsGap: true,
    sampledNewOutfits: 1,
    confidence: 'low',
  });
  const denied = estimateUsefulOutfitUnlocks(reference, wardrobe, {
    contexts,
    eligible: () => true,
    validateOutfit: () => false,
  });
  expect(denied.outfits).toEqual([]);
  expect(evaluateWardrobeValue(reference, wardrobe, denied).level).toBe('Low');
});
test('an equivalent owned role does not count as a new look, and missing slots never multiply into invented value', () => {
  const duplicate = row('duplicate', 'Model Brown Leather Loafer');
  expect(
    estimateUsefulOutfitUnlocks(reference, [...owned, duplicate], {
      contexts,
      eligible: () => true,
      validateOutfit: roles,
    }).outfits
  ).toEqual([]);
  expect(
    estimateUsefulOutfitUnlocks(reference, owned.slice(0, 2), {
      contexts,
      eligible: () => true,
      validateOutfit: roles,
    }).outfits
  ).toEqual([]);
  const many = Array.from({ length: 12 }, (_, i) => ({ id: 'context-' + i, occasion: 'Work' }));
  const capped = estimateUsefulOutfitUnlocks(reference, owned, {
    contexts: many,
    eligible: () => true,
    validateOutfit: roles,
    maxEvaluations: 2,
  });
  expect(capped.evaluations).toBeLessThanOrEqual(2);
  expect(capped.truncated).toBe(true);
});

test('wardrobe goal requires positive validated gains and look signatures retain observed patterns',()=>{
 const zero=rankShoppingAlternatives(reference,[row('other','White Sneaker')],{goal:'wardrobe',eligible:()=>true,signals:()=>({wardrobeGain:0})});
 expect(zero).toEqual([]);
 const {shoppingLookKey}=require('../../src/services/shoppingEvaluation');
 expect(shoppingLookKey(row('plain','White Cotton Shirt','upper_body'))).not.toBe(shoppingLookKey(row('striped','White Striped Cotton Shirt','upper_body')));
});

test('four-slot wardrobe estimates do not manufacture looks from unsupported dress or accessory slots',()=>{
 for(const type of ['dresses','outfit','accessory'] as const)
   expect(estimateUsefulOutfitUnlocks(row(type,'Black Garment',type),owned,{contexts,eligible:()=>true,validateOutfit:roles}).outfits).toEqual([]);
});
