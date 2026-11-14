import test from 'node:test';
import assert from 'node:assert/strict';
import {
  projectCatalogItem,
  adaptCatalog,
  dedupeVariants,
  baseProductId,
  inferColor,
  inferTexture,
  devLookFeasibility,
  priceAgeDays,
  priceFreshness,
  priceImplausibilityReason,
  PRICE_PLAUSIBILITY_UZS,
  PRICE_FRESHNESS_FRESH_MAX_DAYS,
  PRICE_FRESHNESS_RECENT_MAX_DAYS,
  CATALOG_ADAPTER_VERSION,
  inferCatalogColor,
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


test('source bottoms and shoes cannot become outerwear because of colorway words in live taxonomy', () => {
  // Verified 2026-11-12 catalog regression: Everlane Performance Chino /
  // "Trench Coat Khaki" was falsely classified as outerwear/coat by
  // catalog_garment_intelligence despite BOTH native fields saying bottoms.
  const mislabeledChino = row({
    id: 'everlane-7539353845846',
    name: 'The Performance Chino | Uniform | Trench Coat Khaki | Slim',
    category: 'bottoms', garment_type: 'lower_body',
    fashion_slot: 'outerwear', fashion_subtype: 'coat',
  });
  const chino = projectCatalogItem(mislabeledChino);
  assert.equal(chino.macroCategory, 'bottom');
  assert.equal(chino.subtype, 'chino');
  assert.equal(chino.projection.kind, 'native_category');
  const adapted = adaptCatalog([mislabeledChino]);
  assert.equal(adapted.bySlotSubtype['bottom/chino'], 1);
  assert.equal(adapted.bySlotSubtype['outerwear/coat'], undefined);

  const shoe = projectCatalogItem(row({
    id: 'shoe-colorway-conflict', name: 'Black Penny Loafers | Trench Coat Beige',
    category: 'shoes', garment_type: 'footwear',
    fashion_slot: 'outerwear', fashion_subtype: 'coat',
  }));
  assert.equal(shoe.macroCategory, 'shoes');
  assert.equal(shoe.subtype, 'loafer');

  // A valid matching live hint remains authoritative.
  const correct = projectCatalogItem(row({
    id: 'correct-chino', name: 'Cotton chino trousers',
    category: 'bottoms', garment_type: 'lower_body',
    fashion_slot: 'bottom', fashion_subtype: 'tailored-trouser',
  }));
  assert.equal(correct.macroCategory, 'bottom');
  assert.equal(correct.subtype, 'tailored-trouser');
  assert.equal(correct.projection.kind, 'live_intelligence');

  // Real blazers are stored as source tops: do not ban their live outerwear role.
  const blazer = projectCatalogItem(row({
    id: 'legit-blazer', name: 'Single Breasted Tailored Blazer',
    category: 'tops', garment_type: 'upper_body',
    fashion_slot: 'outerwear', fashion_subtype: 'blazer',
  }));
  assert.equal(blazer.macroCategory, 'outerwear');
  assert.equal(blazer.subtype, 'blazer');
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
  assert.equal(CATALOG_ADAPTER_VERSION, 'catalog-adapter-v3-merchant-color');
});

test('live intelligence hints are honored with provenance, never remapped', async () => {
  const { liveHint } = await import('./catalog_adapter.mjs');
  assert.deepEqual(liveHint({ fashion_slot: 'outerwear', fashion_subtype: 'pullover' }),
    { macroCategory: 'outerwear', subtype: 'pullover' });
  assert.deepEqual(liveHint({ fashion_slot: 'TOP', fashion_subtype: 'fine_knit' }),
    { macroCategory: 'top', subtype: 'fine-knit' });
  assert.equal(liveHint({ fashion_slot: 'top', fashion_subtype: 'spacesuit' }), null);
  assert.equal(liveHint({}), null);
  const item = projectCatalogItem(row({
    id: 'live-1', name: 'Ripstop Stadium Anorak',
    description: 'water-resistant nylon',
    fashion_slot: 'outerwear', fashion_subtype: 'pullover',
  }));
  assert.equal(item.macroCategory, 'outerwear');
  assert.equal(item.subtype, 'pullover');
  assert.equal(item.projection.kind, 'live_intelligence');
});

test('knit pullovers keep their top role under live hints; cardigans do not', async () => {
  const { pulloverBaseLayer } = await import('./catalog_adapter.mjs');
  const hint = { macroCategory: 'outerwear', subtype: 'pullover' };
  const knit = pulloverBaseLayer(
    { name: 'Brown Geometric Knit Pullover', description: 'merino wool' }, hint);
  assert.deepEqual([knit.macroCategory, knit.subtype], ['top', 'fine-knit']);
  assert.equal(knit.via, 'live_intelligence_role_projection');
  assert.equal(pulloverBaseLayer({ name: 'Nylon Stadium Pullover' }, hint), null);
  assert.equal(pulloverBaseLayer({ name: 'Knit Cardigan Pullover' }, hint), null);
  assert.equal(pulloverBaseLayer({ name: 'Knit Pullover' },
    { macroCategory: 'top', subtype: 'fine-knit' }), null);
  const item = projectCatalogItem(row({
    id: 'live-knit-1', name: 'Cream Mockneck Knit Pullover',
    fashion_slot: 'outerwear', fashion_subtype: 'pullover',
  }));
  assert.equal(item.macroCategory, 'top');
  assert.equal(item.subtype, 'fine-knit');
  assert.equal(item.projection.kind, 'live_intelligence_role_projection');
});

// UZS fixture prices live near the observed cheap end of the frozen snapshot
// (Zara tops from 219,900 UZS); EUR fixtures are untouched by the UZS-only gate.
const uzsRow = (over = {}) => row({ price: 649_900, currency: 'UZS', ...over });

test('UZS prices outside the slot plausibility band are excluded as price_implausible, never altered', () => {
  // x100 minor-unit magnification on a GANT polo (real snapshot corruption).
  const magnified = projectCatalogItem(uzsRow({
    id: 'gant-polo-x100', name: 'Striped Waffle Texture Polo Shirt',
    price: 24_252_000, fashion_slot: 'top', fashion_subtype: 'polo',
  }));
  assert.deepEqual(magnified, { skipped: 'price_implausible', id: 'gant-polo-x100' });

  // Currency/unit misparse landing below any plausible retail floor.
  const misparse = projectCatalogItem(uzsRow({
    id: 'shoes-misparse', name: 'Brown Suede Penny Loafer', category: 'shoes',
    garment_type: 'shoes', price: 2490,
  }));
  assert.deepEqual(misparse, { skipped: 'price_implausible', id: 'shoes-misparse' });

  const adapted = adaptCatalog([
    uzsRow({ id: 'ok-top', name: 'White Collared Shirt', description: 'cotton' }),
    uzsRow({ id: 'bad-top', name: 'Striped Polo Shirt', price: 24_252_000 }),
  ]);
  assert.equal(adapted.items.length, 1);
  assert.equal(adapted.items[0].id, 'ok-top');
  assert.equal(adapted.skipped.price_implausible, 1);
});

test('plausibility bands pass legit premium items and boundary prices', () => {
  // Legit premium: 2.5M UZS outerwear and a 28.5M Canali cashmere cardigan.
  assert.equal(projectCatalogItem(uzsRow({
    id: 'premium-1', name: 'Camel Double-Breasted Overcoat', price: 2_500_000,
  })).price, 2_500_000);
  assert.equal(projectCatalogItem(uzsRow({
    id: 'premium-2', name: 'Burgundy Cashmere Shawl Cardigan', price: 28_545_957,
    fashion_slot: 'outerwear', fashion_subtype: 'cardigan',
  })).price, 28_545_957);
  // Exact boundary prices on every slot pass (floor and ceiling inclusive).
  for (const [slot, { floor, ceiling }] of Object.entries(PRICE_PLAUSIBILITY_UZS)) {
    assert.equal(priceImplausibilityReason(slot, 'UZS', floor), null, `${slot} floor`);
    assert.equal(priceImplausibilityReason(slot, 'UZS', ceiling), null, `${slot} ceiling`);
    assert.equal(priceImplausibilityReason(slot, 'UZS', floor - 1), 'price_implausible');
    assert.equal(priceImplausibilityReason(slot, 'UZS', ceiling + 1), 'price_implausible');
  }
  // Committed provenance: bands derive from the frozen snapshot percentiles.
  assert.deepEqual(PRICE_PLAUSIBILITY_UZS.top, { floor: 100_000, ceiling: 20_000_000 });
  assert.deepEqual(PRICE_PLAUSIBILITY_UZS.outerwear, { floor: 100_000, ceiling: 30_000_000 });
  assert.deepEqual(PRICE_PLAUSIBILITY_UZS.bottom, { floor: 100_000, ceiling: 20_000_000 });
  assert.deepEqual(PRICE_PLAUSIBILITY_UZS.shoes, { floor: 150_000, ceiling: 15_000_000 });
});

test('plausibility gate abstains outside its evidence base (non-UZS, unknown slots)', () => {
  // No committed evidence for EUR prices — no invented FX bounds.
  assert.equal(priceImplausibilityReason('top', 'EUR', 29), null);
  assert.equal(priceImplausibilityReason('top', 'EUR', 0.01), null);
  assert.equal(projectCatalogItem(row({ price: 29, currency: 'EUR' })).price, 29);
  // Accessories never resolve to an outfit slot, so no band applies.
  assert.equal(priceImplausibilityReason('accessory', 'UZS', 1), null);
});

test('devLookFeasibility rejects UZS offers outside the band with price_implausible', () => {
  const base = {
    id: 'sku-uzs', macroCategory: 'top', subtype: 'shirt',
    imageUrl: 'https://example.com/i.jpg', shopUrl: 'https://example.com/p',
    isShopItem: true, currency: 'UZS',
  };
  assert.equal(devLookFeasibility([{ ...base, price: 649_900 }]).ok, true);
  assert.equal(devLookFeasibility([{ ...base, price: 24_252_000 }]).ok, false);
  assert.equal(devLookFeasibility([{ ...base, price: 24_252_000 }]).reason, 'price_implausible');
  assert.equal(devLookFeasibility([{ ...base, price: 99_999 }]).reason, 'price_implausible');
  // Non-UZS offers abstain; wardrobe pieces (no currency) abstain.
  assert.equal(devLookFeasibility([{ ...base, currency: 'EUR', price: 29 }]).ok, true);
  assert.equal(devLookFeasibility([{
    id: 'w1', macroCategory: 'top', subtype: 'shirt',
    imageUrl: 'https://example.com/w.jpg', isShopItem: false, price: null, currency: null,
  }]).ok, true);
});

test('priceAgeDays derives from last_seen_at and stays unknown without a timestamp', () => {
  const now = Date.parse('2026-11-13T12:00:00.000Z');
  assert.equal(priceAgeDays({ last_seen_at: '2026-11-13T00:00:00.000Z' }, now), 0);
  assert.equal(priceAgeDays({ last_seen_at: '2026-11-06T00:00:00.000Z' }, now), 7);
  assert.equal(priceAgeDays({ last_seen_at: '2026-11-05T00:00:00.000Z' }, now), 8);
  assert.equal(priceAgeDays({ last_seen_at: '2026-09-29T00:00:00.000Z' }, now), 45);
  assert.equal(priceAgeDays({ last_seen_at: '2026-09-28T00:00:00.000Z' }, now), 46);
  // Future timestamps never report negative age.
  assert.equal(priceAgeDays({ last_seen_at: '2026-12-01T00:00:00.000Z' }, now), null);
  // ~4.4k live rows lack the column entirely: honest unknown, never guessed.
  assert.equal(priceAgeDays({}), null);
  assert.equal(priceAgeDays({ last_seen_at: 'not-a-date' }), null);
  assert.equal(priceAgeDays(null), null);
});

test('priceFreshness tiers are bounded: fresh ≤7d, recent ≤45d, stale beyond, unknown absent', () => {
  assert.equal(PRICE_FRESHNESS_FRESH_MAX_DAYS, 7);
  assert.equal(PRICE_FRESHNESS_RECENT_MAX_DAYS, 45);
  assert.equal(priceFreshness(0), 'fresh');
  assert.equal(priceFreshness(7), 'fresh');
  assert.equal(priceFreshness(8), 'recent');
  assert.equal(priceFreshness(45), 'recent');
  assert.equal(priceFreshness(46), 'stale');
  assert.equal(priceFreshness(400), 'stale');
  assert.equal(priceFreshness(null), 'unknown');
  assert.equal(priceFreshness(undefined), 'unknown');
  // Projected items carry the tier on the payload from the row timestamp.
  const fresh = projectCatalogItem(uzsRow({
    id: 'fresh-1', last_seen_at: new Date(Date.now() - 2 * 86_400_000).toISOString(),
  }));
  assert.equal(fresh.priceFreshness, 'fresh');
  assert.ok(Number.isFinite(fresh.priceAgeDays));
  const stale = projectCatalogItem(uzsRow({
    id: 'stale-1', last_seen_at: new Date(Date.now() - 60 * 86_400_000).toISOString(),
  }));
  assert.equal(stale.priceFreshness, 'stale');
  const unknown = projectCatalogItem(uzsRow({ id: 'unknown-1' }));
  assert.equal(unknown.priceAgeDays, null);
  assert.equal(unknown.priceFreshness, 'unknown');
});
