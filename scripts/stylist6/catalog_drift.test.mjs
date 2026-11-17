import test from 'node:test';
import assert from 'node:assert/strict';
import { driftReport, DRIFT_VERSION } from './catalog_drift.mjs';

const row = (over = {}) => ({
  id: 'a', name: 'N', brand: 'B', description: 'D', category: 'tops',
  garment_type: 'upper_body', image_url: 'https://example.com/i.jpg',
  source_url: 'https://example.com/p', price: 100, currency: 'UZS',
  is_active: true, ...over,
});

test('identical universes report fully stable', () => {
  const r = driftReport([row({ id: 'a' }), row({ id: 'b' })], [row({ id: 'a' }), row({ id: 'b' })]);
  assert.equal(r.inBoth, 2);
  assert.equal(r.stable, 2);
  assert.equal(r.onlyOffline, 0);
  assert.equal(r.onlyLive, 0);
});

test('price, deactivation and dead media are listed, never averaged', () => {
  const r = driftReport(
    [row({ id: 'a', price: 100 }), row({ id: 'gone' })],
    [row({ id: 'a', price: 120 }), row({ id: 'b', is_active: false }),
      row({ id: 'c', image_url: 'http://insecure/x.jpg' })],
  );
  assert.equal(r.onlyOffline, 1);
  assert.equal(r.onlyLive, 2);
  assert.equal(r.changed, 1);
  assert.ok(r.changedSample[0].diffs.some((d) => d.startsWith('price:')));
  assert.ok(r.onlyLiveSample.length === 2 && r.onlyOfflineSample.length === 1);
});

test('variant bases collapse for stale-product counts', () => {
  const r = driftReport([row({ id: 'base-v1' }), row({ id: 'base-v2' })], [row({ id: 'other' })]);
  assert.equal(r.distinctBaseOffline, 1);
  assert.equal(r.staleVariantBases, 1);
});

test('version pinned for provenance', () => {
  assert.equal(DRIFT_VERSION, 'catalog-drift-v1');
});
