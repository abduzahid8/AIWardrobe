import test from 'node:test';
import assert from 'node:assert/strict';
import { trialImageDecision, purchaseEvidence, outfitPurchaseEvidence } from './retail_evidence.mjs';

const now = Date.parse('2026-11-10T12:00:00Z');
const photo = { kind: 'product', id: 'shop-a', sha256: 'a'.repeat(64), sourceUrl: 'https://merchant.example/a.jpg' };
const grant = {
  ...photo, rightsHolder: 'Merchant', reviewedBy: 'Owner', evidence: { reference: 'contract-1', sha256: 'b'.repeat(64) },
  status: 'active', validFrom: '2026-11-01', expiresAt: null, display: true,
  nvidiaTrial: { submit: true, terms: 'nvidia-api-trial', territory: 'global',
    providerServiceImprovement: true, purposes: ['internal_outfit_evaluation'] },
};
const decision = (assets, asset = photo) => trialImageDecision({ version: 1, assets }, asset, now);

test('retail photo must match the approved bytes, source, purpose and active grant', () => {
  assert.equal(decision([grant]), 'cleared');
  assert.equal(decision([{ ...grant, nvidiaTrial: undefined }]), 'trial_scope_denied');
  assert.equal(decision([{ ...grant, nvidiaTrial: { ...grant.nvidiaTrial, providerServiceImprovement: false } }]), 'trial_scope_denied');
  assert.equal(decision([{ ...grant, status: 'revoked' }]), 'inactive_grant');
  assert.equal(decision([{ ...grant, expiresAt: '2026-11-10' }]), 'inactive_grant');
  assert.equal(decision([grant, grant]), 'ambiguous_grant');
  assert.equal(decision([grant], { ...photo, sha256: 'c'.repeat(64) }), 'missing_exact_asset_grant');
  assert.equal(decision([grant], { ...photo, sourceUrl: 'https://merchant.example/revised.jpg' }), 'source_mismatch');
});

test('merchant offer needs exact SKU, size, destination, fresh evidence and landed budget', () => {
  const item = { id: 'shop-a', macroCategory: 'top', shopUrl: 'https://merchant.example/a', isActive: true };
  const offer = {
    sku: item.id, variantId: 'merchant-a-m', url: item.shopUrl, observedAt: '2026-11-10T10:00:00Z',
    source: 'merchant_feed', evidenceRef: 'private-export', evidenceSha256: 'a'.repeat(64), size: 'M', sizeSystem: 'merchant-alpha',
    country: 'UZ', shippingConfirmed: true, stock: 'in_stock', currency: 'UZS', price: 100, landedPrice: 120,
  };
  const size = { size: 'M', sizeSystem: 'merchant-alpha' };
  const req = { country: 'UZ', currency: 'UZS', budget: 120 };
  const items = [item, ...['bottom', 'shoes'].map((macroCategory) => ({
    ...item, id: macroCategory, macroCategory, shopUrl: 'https://merchant.example/' + macroCategory,
  }))];
  const otherOffers = items.slice(1).map(({ id, shopUrl }) => ({ ...offer, sku: id, variantId: id, url: shopUrl }));
  const request = { ...req, sizes: { top: size, bottom: size, shoes: size }, budget: 360 };
  const check = (offers, requirement = request) => outfitPurchaseEvidence(items, [...offers, ...otherOffers], requirement, now);
  assert.equal(check([offer]).status, 'verified');
  assert.equal(check([offer], { ...request, budget: 359 }).status, 'over_budget_or_invalid_budget');
  assert.equal(outfitPurchaseEvidence([], [], request, now).status, 'unverified');
  assert.equal(purchaseEvidence(item, offer, { ...req, ...size }, now), 'verified');
  for (const changed of [
    { stock: null }, { observedAt: '2026-11-01' }, { shippingConfirmed: false }, { size: 'S' }, { sizeSystem: 'EU' },
    { evidenceRef: '' }, { url: 'https://merchant.example/b' }, { currency: 'USD' },
  ]) assert.equal(check([{ ...offer, ...changed }]).status, 'unverified');
  assert.equal(check([offer, offer]).products[0].status, 'ambiguous_offer');
  assert.equal(check([]).status, 'unverified');
});
