import test from 'node:test';
import assert from 'node:assert/strict';
import { auditRows, readLocalCatalog } from './catalog_category_audit.mjs';

test('offline audit reports conflicts without rewriting or guessing merchant inventory', () => {
  const rows = [
    { file: 'x.json', row: { id: 'bad', name: 'Regular Fit Oxford Shirt',
      category: 'bottoms', garment_type: 'lower_body', source: 'lacoste',
      source_url: 'https://shop.example/men/button-down-shirts/234', is_active: true } },
    { file: 'x.json', row: { id: 'ambiguous', name: 'Tommy Jeans denim jacket',
      category: 'tops', garment_type: 'upper_body', source: 'lacoste',
      source_url: 'https://shop.example/products/567', is_active: true } },
    { file: 'x.json', row: { id: 'inactive', is_active: false } },
  ];
  const original = structuredClone(rows);
  const audit = auditRows(rows);
  assert.deepEqual(audit.counts, {
    total: 3, active: 2, confirmed: 1, manualReview: 0,
    nativeRoleConflict: 0, missingSnapshotObservation: 2,
    uniqueConfirmedSkuIds: 1,
  });
  assert.equal(audit.findings[0].suggested.garment_type, 'upper_body');
  assert.deepEqual(rows, original);
});

test('committed catalog snapshots are auditable without network or credentials', () => {
  const catalog = readLocalCatalog();
  assert.ok(catalog.files >= 100);
  assert.ok(catalog.entries.length > 1000);
  const audit = auditRows(catalog.entries);
  assert.ok(audit.counts.confirmed > 0, 'known wrong-category data must surface in report');
  assert.ok(audit.findings.some(x => x.id === 'lacoste-ch1917_f6z' &&
    x.status === 'confirmed_category_conflict'));
  assert.ok(audit.findings.some(x => x.id === 'md-uz-61729790' &&
    x.status === 'confirmed_category_conflict'));
});
