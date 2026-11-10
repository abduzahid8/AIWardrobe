import test from 'node:test';
import assert from 'node:assert/strict';
import {
  projectCatalogItem,
  adaptCatalog,
  dedupeVariants,
  baseProductId,
  inferColor,
  inferTexture,
  CATALOG_ADAPTER_VERSION,
} from './catalog_adapter.mjs';

const row = (over = {}) => ({
  id: 'sku-1',
  name: 'Brand Ivory Mockneck Knit Sweater',
  description: 'merino knit',
  brand: 'Brand',
  category: 'tops',
  garment_type: 'upper_body',
  image_url: 'https://example.com/i.jpg',
  source_url: 'https://example.com/p',
  price: 29,
  currency: 'EUR',
  is_active: true,
  ...over,
});

test('knit mockneck in tops projects to top/fine-knit as a base layer', () => {
  const item = projectCatalogItem(row());
  assert.equal(item.macroCategory, 'top');
  assert.equal(item.subtype, 'fine-knit');
  assert.equal(item.projection.kind, 'knit_base_layer');
  assert.ok(item.projection.evidence.length > 0);
});

test('cardigan never takes the base-layer path', () => {
  const item = projectCatalogItem(row({ name: 'Grey Shawl Knit Cardigan' }));
  assert.equal(item.macroCategory, 'outerwear');
  assert.equal(item.subtype, 'cardigan');
  assert.equal(item.projection.kind, 'outerwear_family');
});

test('tailored outerwear in tops projects to the outerwear slot', () => {
  for (const [name, subtype] of [
    ['Camel Double-Breasted Overcoat', 'coat'],
    ['Olive Tailored Blazer', 'blazer'],
    ['Cognac Suede Jacket', 'jacket'],
    ['Grey Wool Overshirt', 'overshirt'],
    ['Chocolate Suede Bomber', 'bomber'],
  ]) {
    const item = projectCatalogItem(row({ name }));
    assert.equal(item.macroCategory, 'outerwear', name);
    assert.equal(item.subtype, subtype, name);
  }
});

test('shirts, polos and tees stay native tops', () => {
  assert.equal(projectCatalogItem(row({ name: 'White Collared Shirt', description: 'cotton' })).subtype, 'shirt');
  assert.equal(projectCatalogItem(row({ name: 'Navy Polo Shirt', description: 'cotton pique' })).subtype, 'polo');
  assert.equal(projectCatalogItem(row({ name: 'Cream Crewneck T-Shirt', description: 'cotton' })).subtype, 't-shirt');
});

test('bottoms and shoes map natively', () => {
  const trouser = projectCatalogItem(row({
    id: 'b1', name: 'Brown Tailored Trousers', category: 'bottoms', garment_type: 'lower_body',
  }));
  assert.equal(trouser.macroCategory, 'bottom');
  assert.equal(trouser.subtype, 'tailored-trouser');
  assert.equal(trouser.projection.kind, 'native_category');

  const loafer = projectCatalogItem(row({
    id: 's1', name: 'Brown Suede Penny Loafer', category: 'shoes', garment_type: 'shoes',
  }));
  assert.equal(loafer.macroCategory, 'shoes');
  assert.equal(loafer.subtype, 'loafer');

  const derby = projectCatalogItem(row({
    id: 's2', name: 'Black Leather Derby', category: 'shoes', garment_type: 'shoes',
  }));
  assert.equal(derby.subtype, 'derby');
});

test('unshoppable rows are counted, never admitted', () => {
  const adapted = adaptCatalog([
    row({ id: 'ok' }),
    row({ id: 'off', is_active: false }),
    row({ id: 'free', price: 0 }),
    row({ id: 'noimg', image_url: 'http://insecure/x.jpg' }),
    row({ id: 'noshop', source_url: '' }),
    row({ id: 'badcur', currency: 'usd' }),
    row({ id: 'acc', category: 'accessories', garment_type: 'accessory' }),
  ]);
  assert.equal(adapted.items.length, 1);
  assert.equal(adapted.items[0].id, 'ok');
  assert.deepEqual(adapted.skipped, {
    inactive: 1, unpriced: 1, non_https_media_or_shop: 2, bad_currency: 1, non_outfit_category: 1,
  });
});

test('variant duplicates collapse for diversity counts', () => {
  assert.equal(baseProductId('lefties-merino-sweater-grey-v35'), 'lefties-merino-sweater-grey');
  const items = [row({ id: 'a-v1' }), row({ id: 'a-v2' }), row({ id: 'b' })];
  const { kept, duplicates } = dedupeVariants(items);
  assert.equal(kept.length, 2);
  assert.equal(duplicates, 1);
});

test('color and texture inference covers reference palette words', () => {
  assert.equal(inferColor('Camel Double-Breasted Overcoat'), 'camel');
  assert.equal(inferColor('Charcoal Zip Blouson'), 'charcoal');
  assert.equal(inferColor('Olive Quilted Barn Jacket'), 'olive');
  assert.equal(inferTexture('Cognac Suede Jacket'), 'suede');
  assert.equal(inferTexture('Cream Crewneck Knit'), 'knit');
});

test('adapter version is pinned for pilot provenance', () => {
  assert.equal(CATALOG_ADAPTER_VERSION, 'catalog-adapter-v1');
});
