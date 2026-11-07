import { selectEssentialShoppingMix } from '../../src/services/homeShoppingSelection';
import { buildCatalogFamilyViews } from '../../src/services/catalogFamilyView';
import type { ShopCatalogItem } from '../../src/types/shop';
const row = (
  id: string,
  name: string,
  type: ShopCatalogItem['garmentType'] = 'shoes'
): ShopCatalogItem => ({
  id,
  name,
  brand: 'Brand ' + id,
  price: 100,
  currency: 'USD',
  imageUrl: `https://img.test/${id}.jpg`,
  garmentType: type,
});
test('Home essentials collapse supported families while retaining selectable original variants', () => {
  const variants = [row('black', 'Black Penny Loafers'), row('brown', 'Brown Penny Loafers')].map(
    (item) => ({ ...item, brand: 'Same', source: 'product-only', sourceProductGroupId: 'model' })
  );
  const rows = [...variants, row('sneaker', 'White Sneakers'), row('boot', 'Brown Boots')];
  const selection = selectEssentialShoppingMix(rows, '2026-11-02');
  expect(selection).toHaveLength(3);
  expect(selection.filter((item) => item.brand === 'Same')).toHaveLength(1);
  const family = buildCatalogFamilyViews(rows).families.find((family) => family.brand === 'Same')!;
  expect(family.variants.map((variant) => variant.originalCatalogId)).toEqual(['black', 'brown']);
  const preferred = buildCatalogFamilyViews(rows, { selectedVariantIds: ['brown'] }).families.map(
    (family) =>
      family.variants.find((variant) => variant.variantId === family.preferredVariantId)!.item
  );
  expect(
    selectEssentialShoppingMix(preferred, '2026-11-02').some((item) => item.id === 'brown')
  ).toBe(true);
});
test('Home quotas and deterministic daily order survive brand and shoe diversity selection', () => {
  const rows = [
    ...Array.from({ length: 20 }, (_, i) => ({
      ...row('l' + i, 'Black Penny Loafers Model ' + i),
      brand: 'Dominant',
    })),
    row('s', 'White Leather Sneakers'),
    row('b', 'Brown Chelsea Boots'),
    row('p1', 'Navy Tailored Trousers', 'lower_body'),
    row('p2', 'Beige Cotton Chinos', 'lower_body'),
    row('p3', 'Blue Jeans', 'lower_body'),
    row('j1', 'Navy Wool Blazer', 'upper_body'),
    row('j2', 'Brown Leather Jacket', 'upper_body'),
    row('t1', 'White Oxford Shirt', 'upper_body'),
    row('t2', 'Blue Poplin Shirt', 'upper_body'),
  ];
  const a = selectEssentialShoppingMix(rows, 'daily-seed');
  expect(a.filter((item) => item.garmentType === 'shoes')).toHaveLength(3);
  expect(a.some((item) => item.id === 's')).toBe(true);
  expect(a.some((item) => item.id === 'b')).toBe(true);
  expect(a.filter((item) => item.garmentType === 'lower_body')).toHaveLength(3);
  expect(selectEssentialShoppingMix(rows, 'daily-seed').map((item) => item.id)).toEqual(
    a.map((item) => item.id)
  );
});
