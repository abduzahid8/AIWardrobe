import test from 'node:test';
import assert from 'node:assert/strict';
import {
  catalogObservationBatches, hydrateCatalogObservations, CATALOG_OBSERVATION_MAX_IDS_PER_GET,
  CATALOG_OBSERVATION_MAX_QUERY_ID_BYTES,
} from './catalog_observation.mjs';
import { adaptCatalog, priceAgeDays } from './catalog_adapter.mjs';

const sku = (id, extras = {}) => ({
  id, is_active: true, name: 'Blue Oxford Shirt', category: 'tops',
  garment_type: 'upper_body', price: 175000, currency: 'UZS',
  image_url: 'https://shop.example/blue-shirt.jpg',
  source_url: 'https://shop.example/products/blue-oxford-shirt',
  ...extras,
});
const current = (row, observedAt) => ({ ...row, last_seen_at: observedAt });
const params = (url) => new URL(url).searchParams;
const requestIds = (url) => params(url).get('id').replace(/^in\.\(/, '').replace(/\)$/, '').split(',');

function mockGetFromCatalog(rows, requests = []) {
  const map = new Map(rows.map(row => [row.id, row]));
  return async (url, options) => {
    requests.push({ url, options });
    assert.equal(new URL(url).pathname, '/rest/v1/shop_catalog');
    assert.equal(options.method, 'GET');
    assert.equal(options.cache, 'no-store');
    assert.equal(options.redirect, 'error');
    assert.equal(params(url).get('select'),
      'id,is_active,last_seen_at,name,category,garment_type,price,currency,image_url,source_url');
    assert.ok(options.headers.Authorization.startsWith('Bearer '));
    return Response.json(requestIds(url).flatMap(id => map.has(id) ? [map.get(id)] : []));
  };
}

const input = (rows, fetchImpl) => ({
  rows, url: 'https://project.supabase.co', serviceKey: 'internal-test-only-key', fetchImpl,
});

test('last_seen_at is hydrated from live catalog, but cannot assert stock or create a date', async () => {
  const s = sku('blue-shirt');
  const observation = '2026-10-26T00:00:00Z';
  const requests = [];
  const rows = await hydrateCatalogObservations(input([s],
    mockGetFromCatalog([current(s, observation)], requests)));
  assert.equal(rows.length, 1);
  assert.equal(rows[0].last_seen_at, observation);
  const found = adaptCatalog(rows).items[0];
  assert.equal(found.priceFreshness, 'recent'); // as of Nov 13; feed observation only
  assert.ok(found.priceAgeDays >= 17);
  assert.equal(found.isShopItem, true);
  assert.equal(found.stock, undefined);
  assert.equal(requests.length, 1);
});

test('null catalog observation remains unknown and a future observation is never fresh', async () => {
  const unknown = sku('unknown');
  const future = sku('future');
  const observed = [current(unknown, null), current(future, '2099-01-01T00:00:00Z')];
  const rows = await hydrateCatalogObservations(input([unknown, future],
    mockGetFromCatalog(observed)));
  assert.equal(rows.length, 2);
  assert.deepEqual(rows.map(r => adaptCatalog([r]).items[0].priceFreshness), ['unknown', 'unknown']);
  assert.equal(priceAgeDays(rows[1]), null);
});

test('deactivated, deleted or repriced rows are removed; never silently rewritten', async () => {
  const good = sku('good'), inactive = sku('inactive'), deleted = sku('deleted');
  const changedPrice = sku('price-changed'), changedImage = sku('photo-changed');
  const rows = [good, inactive, deleted, changedPrice, changedImage];
  const records = [
    current(good, '2026-10-26T00:00:00Z'),
    current(inactive, '2026-10-26T00:00:00Z'),
    current(changedPrice, '2026-10-26T00:00:00Z'),
    current(changedImage, '2026-10-26T00:00:00Z'),
  ];
  records[1].is_active = false;
  records[2].price += 1;
  records[3].image_url = 'https://shop.example/changed.jpg';
  const result = await hydrateCatalogObservations(input(rows, mockGetFromCatalog(records)));
  assert.deepEqual(result.map(x => x.id), ['good']);
  assert.equal(good.last_seen_at, undefined, 'caller input untouched');
});

test('a failed metadata page fails open to UNKNOWN for the complete window, never partially recent', async () => {
  const rows = [sku('first'), sku('second')];
  const broken = async () => Response.json({ message: 'temporarily unavailable' }, { status: 503 });
  const result = await hydrateCatalogObservations(input(rows, broken));
  assert.deepEqual(result, rows);
  assert.equal(result[0].last_seen_at, undefined);
});

test('a single failed page invalidates all observations in multi-page discovery', async () => {
  const rows = Array.from({ length: 170 }, (_, i) => sku('valid-' + i));
  let pages = 0;
  const fetchImpl = async (url) => {
    pages++;
    const ids = requestIds(url);
    if (pages === 2) return Response.json({ error: 'feed not available' }, { status: 503 });
    return Response.json(rows.filter(row => ids.includes(row.id))
      .map(row => current(row, '2026-10-26T00:00:00Z')));
  };
  const got = await hydrateCatalogObservations(input(rows, fetchImpl));
  assert.ok(pages >= 2);
  assert.deepEqual(got, rows, 'partial freshness must not leak from successful pages');
});

test('malformed data and mismatched identities cannot add trusted observations', async () => {
  const row = sku('blue-shirt');
  const rogue = async () => Response.json([current(sku('unrequested'), '2026-10-26T00:00:00Z')]);
  assert.deepEqual(await hydrateCatalogObservations(input([row], rogue)), [row]);
  const duplicate = async () => Response.json([current(row, null), current(row, null)]);
  assert.deepEqual(await hydrateCatalogObservations(input([row], duplicate)), [row]);
});

test('future RPC projection with observation date needs no additional metadata query', async () => {
  const row = sku('self-describing', { last_seen_at: '2026-10-05T00:00:00Z' });
  const fetchImpl = async () => { throw new Error('metadata must not be refetched'); };
  assert.deepEqual(await hydrateCatalogObservations(input([row], fetchImpl)), [row]);
});

test('only safe IDs are selected, with bounded pages and query bytes', async () => {
  const ids = Array.from({ length: 400 }, (_, i) => ('sku-' + i + '-' + 'a'.repeat(87)));
  const batches = catalogObservationBatches(ids.map(id => sku(id)));
  assert.ok(batches.length >= 4);
  assert.equal(batches.flat().length, 400);
  for (const batch of batches) {
    assert.ok(batch.length <= CATALOG_OBSERVATION_MAX_IDS_PER_GET);
    assert.ok(batch.reduce((sum, id) => sum + encodeURIComponent(id).length + 1, 0) <=
      CATALOG_OBSERVATION_MAX_QUERY_ID_BYTES);
  }
  const bad = sku('other),in.(evil');
  const normal = sku('normal');
  const result = await hydrateCatalogObservations(input([bad, normal],
    mockGetFromCatalog([current(normal, '2026-10-05T00:00:00Z')])));
  assert.equal(result[0], bad);
  assert.equal(result[1].last_seen_at, '2026-10-05T00:00:00Z');
  assert.throws(() => catalogObservationBatches(Array.from({ length: 401 }, (_, i) => sku('row-' + i))),
    /catalog_observation_limit_exceeded/);
});
