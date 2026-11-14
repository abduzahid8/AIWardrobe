import test from 'node:test';
import assert from 'node:assert/strict';
import { classifyFrFootwearFamily, normalizeFrShoeOffer } from './collect_uz_footwear_supply.mjs';

test('classifyFrFootwearFamily maps UZ footwear categories accurately', () => {
  assert.equal(classifyFrFootwearFamily('Лоферы Leather Penny Loafer', 'tufli'), 'loafer');
  assert.equal(classifyFrFootwearFamily('Туфли Derby Classic', 'tufli'), 'derby');
  assert.equal(classifyFrFootwearFamily('Ботинки Chelsea Boot', 'botinki-i-sapogi'), 'chelsea');
  assert.equal(classifyFrFootwearFamily('Ботинки Winter Boot', 'botinki-i-sapogi'), 'boot');
  assert.equal(classifyFrFootwearFamily('Кроссовки Minimal Leather', 'krossovki'), 'sneaker');
});

test('normalizeFrShoeOffer blocks disallowed brands and preserves unverified checkout/rights fields', () => {
  const baseProduct = {
    '@type': 'Product',
    brand: { name: 'Calvin Klein' },
    name: 'Кроссовки Minimal Leather Cup',
    sku: 'YM0YM00999',
    image: ['https://media.frgroup.uz/images/catalog-product-desktop/ck-shoe.jpg'],
    offers: {
      url: 'https://frgroup.uz/ru/store/ck-minimal-leather-cup',
      price: '1890000',
      priceCurrency: 'UZS',
      availability: 'https://schema.org/InStock',
      shippingDetails: { shippingDestination: { addressCountry: 'UZ' } },
    },
  };
  const disallowed = normalizeFrShoeOffer(
    { ...baseProduct, brand: { name: 'ECCO' } },
    { categorySlug: 'krossovki', observedAt: '2026-11-14T00:00:00Z', evidenceHash: 'a'.repeat(64) },
  );
  assert.equal(disallowed, null);

  const normalized = normalizeFrShoeOffer(baseProduct, {
    categorySlug: 'krossovki',
    observedAt: '2026-11-14T00:00:00Z',
    evidenceHash: 'a'.repeat(64),
    variantEvidence: {
      availableSizes: [{ variantId: 901, euSize: '42', merchantReportedQty: 3 }],
      totalMerchantReportedUnits: 3,
    },
  });
  assert.ok(normalized);
  assert.equal(normalized.brand, 'Calvin Klein');
  assert.equal(normalized.is_active, false);
  assert.equal(normalized.sizeStockStatus, 'merchant_reported_by_size');
  assert.deepEqual(normalized.availableEuSizes, ['42']);
  assert.equal(normalized.imageNvidiaRights, 'not_authorized');
  assert.equal(normalized.exactCheckoutAvailability, 'unverified');
  assert.equal(normalized.deliveredTotalStatus, 'unverified');
  assert.equal(normalized.offerVerification.deliveryEligible, true);
  assert.equal(normalized.offerVerification.imagePermissions.externalInferenceAllowed, false);
  assert.equal(normalized.offerVerification.imagePermissions.technicalInferenceAllowlisted, true);
  assert.equal(normalized.offerVerification.checkoutAvailability, 'unverified');
  assert.equal(normalized.imageNvidiaRights, 'not_authorized');
});
