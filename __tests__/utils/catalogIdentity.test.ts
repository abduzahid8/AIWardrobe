import {
  familyIdentityCandidate,
  listingIdentityOf,
  physicalProductIdentityOf,
  normalizeIdentityText,
} from '../../src/utils/catalogIdentity';
import {
  buildCatalogFamilyViews,
  selectCatalogVariant,
} from '../../src/services/catalogFamilyView';
import type { ShopCatalogItem } from '../../src/types/shop';
const row = (id: string, changes: Partial<ShopCatalogItem> = {}): ShopCatalogItem => ({
  id,
  brand: 'Brand',
  name: 'Macharles Blazer Black',
  imageUrl: `https://img.test/${id}.jpg`,
  garmentType: 'upper_body',
  price: 100,
  currency: 'USD',
  ...changes,
});

test('listing and physical equivalence stay separate from family evidence', () => {
  const a = row('a');
  expect(listingIdentityOf(a).rowId).toBe('a');
  const same = row('b', { name: 'MACHARLES BLAZER BLACK', imageUrl: a.imageUrl + '?width=300' });
  expect(physicalProductIdentityOf(a).canonicalKey).toBe(
    physicalProductIdentityOf(same).canonicalKey
  );
  expect(physicalProductIdentityOf(a).imageKey).toBe(physicalProductIdentityOf(same).imageKey);
  expect(buildCatalogFamilyViews([a, { ...a }]).diagnostics.variants).toBe(1);
});
test('different colors group while conserving every concrete row and price', () => {
  const a = row('a'),
    b = row('b', { name: 'Macharles Blazer Brown', price: 120 });
  const view = buildCatalogFamilyViews([a, b], {
    selectedVariantIds: ['b'],
    avoidedColors: ['brown'],
  });
  expect(view.diagnostics).toMatchObject({ families: 1, variants: 2 });
  expect(view.families[0].groupingConfidence).toBe('medium');
  expect(selectCatalogVariant(view.families[0])).toBe(b);
  expect(selectCatalogVariant(view.families[0], 'a')).toBe(a);
  expect(view.families[0].variants.map((v) => v.item.price)).toEqual([100, 120]);
});
test('generic names, cross-brand titles, fuzzy titles and conflicting references stay separate', () => {
  for (const pair of [
    [row('a', { name: 'Cotton Shirt Black' }), row('b', { name: 'Cotton Shirt Brown' })],
    [row('a'), row('b', { brand: 'Other', name: 'Macharles Blazer Brown' })],
    [row('a'), row('b', { name: 'Macharles Slim Blazer Brown' })],
    [
      row('a', { modelReference: 'one', source: 'feed' }),
      row('b', { modelReference: 'two', source: 'feed' }),
    ],
    [row('a'), row('b', { name: 'Macharles Blazer Brown', price: 500 })],
  ])
    expect(buildCatalogFamilyViews(pair).families).toHaveLength(2);
});
test('trusted source group and canonical product URLs provide high confidence', () => {
  expect(
    buildCatalogFamilyViews([
      row('a', { source: 'feed', sourceProductGroupId: '123' }),
      row('b', { source: 'feed', sourceProductGroupId: '123' }),
    ]).families[0].groupingConfidence
  ).toBe('high');
  const urls = [
    'https://shop.test/products/model?color=black&utm_source=x',
    'https://shop.test/products/model?color=brown',
  ];
  expect(familyIdentityCandidate(row('a', { sourceUrl: urls[0] })).familyKey).toBe(
    familyIdentityCandidate(row('b', { sourceUrl: urls[1] })).familyKey
  );
  expect(
    buildCatalogFamilyViews([
      row('a', { sourceUrl: 'https://shop.test/' }),
      row('b', { sourceUrl: 'https://shop.test/' }),
    ]).families
  ).toHaveLength(2);
});
test('Unicode and actual images remain distinct; external SKU alone is insufficient', () => {
  expect(normalizeIdentityText(' École  Рубашка 日本 ')).toBe('école рубашка 日本');
  expect(normalizeIdentityText('Рубашка')).not.toBe(normalizeIdentityText('Брюки'));
  expect(
    buildCatalogFamilyViews([
      row('a', { name: 'Shirt', externalId: '123-black' }),
      row('b', { name: 'Shirt', externalId: '123-brown' }),
    ]).families
  ).toHaveLength(2);
});
test('append merges variants without replacing an explicit selection or dropping inventory', () => {
  const a = row('a'),
    b = row('b', { name: 'Macharles Blazer Brown' }),
    c = row('c', { name: 'Different Shirt' });
  const result = buildCatalogFamilyViews([a, c, b], { selectedVariantIds: ['a'] });
  expect(result.families).toHaveLength(2);
  expect(selectCatalogVariant(result.families[0])).toBe(a);
  expect(result.families.flatMap((f) => f.variants)).toHaveLength(3);
  expect(selectCatalogVariant(result.families[0], 'missing')).toBeUndefined();
});

test('generic tailoring descriptors are not model evidence; overlapping pages retain first-card order', () => {
  const a = row('a', { name: 'White cotton gabardine double-pleated pants' }),
    b = row('b', { name: 'Beige cotton gabardine double-pleated pants' });
  expect(buildCatalogFamilyViews([a, b]).families).toHaveLength(2);
  const model = row('m'),
    different = row('d', { name: 'Different Shirt' }),
    color = row('c', { name: 'Macharles Blazer Brown' });
  const view = buildCatalogFamilyViews([model, different, model, color]);
  expect(view.families[0].variants[0].variantId).toBe('m');
  expect(view.diagnostics.variants).toBe(3);
});

test('multilingual colors preserve base model and direct variant traceability',()=>{
 const {catalogBaseName,catalogTitleColor,normalizeCatalogBrand}=require('../../src/utils/catalogIdentity');
 const black=row('ru-black',{name:'Macharles Пиджак Чёрный'}),brown=row('ru-brown',{name:'Macharles Пиджак Коричневый'});
 expect(catalogTitleColor(black)).toBe('black');expect(catalogBaseName(black)).toBe(catalogBaseName(brown));
 expect(normalizeCatalogBrand('ÉCOLE 日本')).toBe('école 日本');
 const family=buildCatalogFamilyViews([black,brown]).families[0];
 expect(family.variants).toHaveLength(2);expect(family.variants[1].originalRow).toBe(brown);
 expect(family.variants[1].originalCatalogId).toBe('ru-brown');
});

test('official merchant article references group generic titles without merging different articles or brands',()=>{
 const a=row('jj-brown',{brand:'JACK & JONES',name:'Regular Fit Chino trousers',sourceUrl:'https://www.jackjones.com/en-gb/product/12306276_2048/regular-fit-chinos'});
 const b={...a,id:'jj-other',imageUrl:'other.jpg',sourceUrl:'https://www.jackjones.com/en-gb/product/12306276_5104/regular-fit-chinos'};
 expect(buildCatalogFamilyViews([a,b]).families[0].groupingConfidence).toBe('high');
 expect(buildCatalogFamilyViews([a,b]).families[0].variants).toHaveLength(2);
 expect(buildCatalogFamilyViews([a,{...b,sourceUrl:b.sourceUrl.replace('12306276','12306277')}]).families).toHaveLength(2);
 expect(buildCatalogFamilyViews([a,{...b,brand:'Other brand'}]).families).toHaveLength(2);
 expect(buildCatalogFamilyViews([a,{...b,sourceUrl:b.sourceUrl.replace('jackjones.com','unknown.test')}]).families).toHaveLength(2);
 expect(buildCatalogFamilyViews([a,{...b,sourceUrl:b.sourceUrl.replace('https:','javascript:')}]).families).toHaveLength(2);
 const jcrew=[row('jc-black',{sourceUrl:'https://www.jcrew.com/p/product/CH145?fit=Slim&color_name=black'}),row('jc-brown',{sourceUrl:'https://www.jcrew.com/p/product/CH145?fit=Slim&color_name=brown'})];
 expect(buildCatalogFamilyViews(jcrew).families).toHaveLength(1);
 expect(jcrew[1].sourceUrl).toContain('color_name=brown');
});
