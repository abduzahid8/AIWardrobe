/** @jest-environment node */
jest.mock('../../lib/supabase', () => ({ supabase: { from: jest.fn() } }));
import {
  evaluateShoppingPurchase,
  findBetterShoppingProducts,
  shoppingFitEvidence,
} from '../../src/services/shoppingAdvisor';
import { createFixtureCatalogRepository } from '../../src/services/catalogRepository';
import type { ShopCatalogItem } from '../../src/types/shop';
import type { BodyProfile } from '../../src/types/bodyProfile';
const row = (
  id: string,
  name: string,
  type: ShopCatalogItem['garmentType'] = 'shoes',
  price = 100
): ShopCatalogItem => ({
  id,
  name,
  brand: 'Brand',
  price,
  currency: 'USD',
  imageUrl: `https://img.test/${id}.jpg`,
  garmentType: type,
});
const candidate = row('candidate', 'Brown leather penny loafers');
const owned = [
  row('jacket', 'Navy lightweight cotton blazer', 'upper_body'),
  row('shirt', 'White cotton oxford shirt', 'upper_body'),
  row('pants', 'Navy chino trousers', 'lower_body'),
];
test('integrated wardrobe value uses existing occasion, climate, activity and budget gates', () => {
  const context = {
    style: 'classic',
    occasion: 'Work',
    weather: { temp: 20 },
    budget: { amount: 100, currency: 'USD' },
  };
  const valid = evaluateShoppingPurchase(candidate, owned, context);
  expect(valid.unlocks.outfits.length).toBeGreaterThan(0);
  expect(
    evaluateShoppingPurchase(candidate, owned, {
      ...context,
      budget: { amount: 1, currency: 'USD' },
    }).unlocks.outfits
  ).toEqual([]);
  expect(
    evaluateShoppingPurchase(candidate, owned, { ...context, avoidedColors: ['brown'] }).unlocks
      .outfits
  ).toEqual([]);
  expect(
    evaluateShoppingPurchase(candidate, owned, { ...context, occasion: 'Workout' }).unlocks.outfits
  ).toEqual([]);
});
test('Find Better runs real contextual filtering and goal ranking through the injected offline repository', async () => {
  const cheaper = row('cheaper', 'Brown leather loafers', 'shoes', 70),
    wrong = row('wrong', 'Neon athletic running sneakers', 'shoes', 20),
    expensive = row('expensive', 'Brown leather loafers', 'shoes', 300);
  const result = await findBetterShoppingProducts(candidate, owned, {
    goal: 'cheaper',
    style: 'classic',
    occasion: 'Work',
    budget: { amount: 90, currency: 'USD' },
    repository: createFixtureCatalogRepository(() => [cheaper, wrong, expensive]),
  });
  expect(result.origin).toBe('fixture');
  expect(result.alternatives.map((result) => result.item.id)).toEqual(['cheaper']);
  expect(result.alternatives[0].item.id).toBe('cheaper');
});
test('predicted fit delegates to existing measurement engine and stays unknown without profiles', () => {
  const body = {
    id: 'body',
    userId: 'user',
    status: 'ready',
    isActive: true,
    bodyType: 'average',
    height: { valueCm: 175, confidence: 'high', source: 'manual' },
    measurements: { footLength: { valueCm: 26.5, confidence: 'high', source: 'manual' } },
    privacy: { retainMesh: false, retainSourcePhoto: false },
    version: 1,
    createdAt: '2026-01-01',
    updatedAt: '2026-01-01',
  } as BodyProfile;
  expect(shoppingFitEvidence(candidate, body)).toBeUndefined();
  const measured = (length: number): ShopCatalogItem => ({
    ...candidate,
    physicalProfiles: [
      {
        id: 'profile',
        garmentId: candidate.id,
        sizeLabel: '42',
        category: 'shoes',
        source: 'manufacturer_size_chart',
        measurements: { shoeLength: { valueCm: length, confidence: 'high' } },
      },
    ],
  });
  expect(shoppingFitEvidence(measured(27.5), body)!.score).toBeGreaterThan(
    shoppingFitEvidence(measured(24), body)!.score
  );
  expect(shoppingFitEvidence(measured(27.5), body)!.evidence[0]).toContain('fit-engine/');
});

test('activity and destination temperature in free text use the shared deterministic gates',()=>{
 const base={style:'classic',occasion:'Work',weather:{temp:20},budget:{amount:100,currency:'USD'}};
 expect(evaluateShoppingPurchase(candidate,owned,{...base,prompt:'Workout at the gym'}).unlocks.outfits).toEqual([]);
 expect(evaluateShoppingPurchase(candidate,owned,{...base,prompt:'Winter trip at -10°C'}).unlocks.outfits).toEqual([]);
});

test('avoided colors inspect all title colors before styling descriptions',()=>{
 const white={...candidate,name:'White Leather Loafers',description:'Pair with black trousers'};
 const multicolor={...candidate,name:'White Blue Leather Loafers'};
 const context={style:'classic',occasion:'Work',weather:{temp:20}};
 expect(evaluateShoppingPurchase(white,owned,{...context,avoidedColors:['white']}).unlocks.outfits).toEqual([]);
 expect(evaluateShoppingPurchase(multicolor,owned,{...context,avoidedColors:['blue']}).unlocks.outfits).toEqual([]);
});

test('unknown or zero listing prices never become free purchases under a budget',()=>{
 const context={style:'classic',occasion:'Work',weather:{temp:20},budget:{amount:100,currency:'USD'}};
 for(const invalid of [{...candidate,price:0},{...candidate,price:NaN},{...candidate,currency:undefined}])
   expect(evaluateShoppingPurchase(invalid,owned,context).unlocks.outfits).toEqual([]);
});
