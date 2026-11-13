import test from 'node:test';
import assert from 'node:assert/strict';
import { assessCatalogCategory, explicitGarmentRole, merchantPathRole } from './catalog_category_integrity.mjs';
import { adaptCatalog, projectCatalogItem } from './catalog_adapter.mjs';

const row = (props = {}) => ({
  id: 'example-1', name: 'Cotton Oxford Shirt', category: 'tops',
  garment_type: 'upper_body', is_active: true, price: 110, currency: 'USD',
  source_url: 'https://merchant.example/men/oxford-shirt-123',
  image_url: 'https://merchant.example/image.jpg', ...props,
});

test('Oxford shirts and bootcut jeans never become shoes from isolated keywords', () => {
  assert.equal(explicitGarmentRole('Regular Fit Oxford Shirt'), 'upper_body');
  assert.equal(explicitGarmentRole('Regular Fit Bootcut Jeans'), 'lower_body');
  assert.equal(explicitGarmentRole('Black Oxford Shoes'), 'shoes');
  assert.equal(explicitGarmentRole('Trench Coat Khaki Performance Chino'), null);
  assert.equal(explicitGarmentRole('Tommy Jeans Denim Jacket'), null);
  assert.equal(explicitGarmentRole('Shirt and Shorts Set'), null);
});

test('merchant URL path, not query or homepage metadata, supplies independent evidence', () => {
  assert.equal(merchantPathRole('https://merchant.example/men/clothing/button-down-shirts/CH1917-51.html?color=F6Z'), 'upper_body');
  assert.equal(merchantPathRole('https://merchant.example/uz/en/bleach-washed-tapered-fit-jeans-l00401121?pelement=61729790'), 'lower_body');
  assert.equal(merchantPathRole('https://merchant.example/?q=jeans'), null);
  assert.equal(merchantPathRole('http://merchant.example/jeans'), null);
  assert.equal(merchantPathRole('not-a-url'), null);
});

test('confirmed wrong-category rows cannot be styled as their false native slot', () => {
  const badShirt = row({ id: 'lacoste-ch1917_f6z', name: "Men's Regular Fit Short Sleeve Oxford Shirt",
    category: 'bottoms', garment_type: 'lower_body',
    source_url: 'https://www.lacoste.com/us/lacoste/men/clothing/button-down-shirts/CH1917-51.html?color=F6Z',
  });
  const badJeans = row({ id: 'md-uz-61729790', name: 'Bleach washed tapered fit jeans',
    source_url: 'https://www.massimodutti.com/uz/en/bleach-washed-tapered-fit-jeans-l00401121?pelement=61729790',
  });
  const badBlazer = row({ id: 'md-uz-62899803', name: '100% cool wool suit blazer', category: 'bottoms', garment_type: 'lower_body',
    source_url: 'https://www.massimodutti.com/uz/en/100-cool-wool-suit-blazer-l03623522?pelement=62899803',
  });
  for (const x of [badShirt, badJeans, badBlazer]) {
    assert.equal(assessCatalogCategory(x).status, 'confirmed_category_conflict', x.id);
    assert.deepEqual(projectCatalogItem(x), { skipped: 'catalog_category_conflict', id: x.id });
  }
  assert.deepEqual(adaptCatalog([badShirt, badJeans, badBlazer, row()]).skipped, { catalog_category_conflict: 3 });
});

test('unverified or ambiguous hints do not auto-reclassify or ban valid offers', () => {
  const colorwayChino = row({ name: 'Performance Chino | Trench Coat Khaki | Slim',
    category: 'bottoms', garment_type: 'lower_body',
    source_url: 'https://merchant.example/chino-performance-987',
  });
  assert.notEqual(assessCatalogCategory(colorwayChino).status, 'confirmed_category_conflict');
  assert.equal(projectCatalogItem(colorwayChino).macroCategory, 'bottom');
  const normalOxford = row();
  assert.equal(projectCatalogItem(normalOxford).macroCategory, 'top');
  const unverified = row({ name: 'Tailored Slim Trousers', source_url: 'https://merchant.example/product/19821' });
  assert.equal(assessCatalogCategory(unverified).status, 'needs_manual_review');
  assert.equal(projectCatalogItem(unverified).macroCategory, 'top');
  const misleadingBrand = row({ name: 'Armani Jeans Denim Jacket',
    source_url: 'https://merchant.example/denim-jacket', });
  assert.equal(projectCatalogItem(misleadingBrand).macroCategory, 'outerwear');
  const nativeConflict = row({ garment_type: 'shoes' });
  assert.equal(projectCatalogItem(nativeConflict).skipped, 'catalog_category_conflict');
});
