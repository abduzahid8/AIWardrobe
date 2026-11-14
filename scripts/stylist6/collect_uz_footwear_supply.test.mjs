import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { classifyFrFootwearFamily, normalizeFrShoeOffer, runCollector } from './collect_uz_footwear_supply.mjs';

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

test('runCollector separates product leads from verified purchasable offers and breaks down dress vs sneaker supply', async () => {
  const htmlList = (slug) => `
    <script type="application/ld+json">${JSON.stringify({
      '@type': 'Product',
      name: slug === 'tufli' ? 'Лоферы Classic Penny' : 'Кроссовки Runner',
      sku: slug === 'tufli' ? 'SKU-LOAFER-1' : 'SKU-SNEAK-1',
      brand: { name: 'Tommy Hilfiger' },
      image: [`https://media.frgroup.uz/images/${slug}.jpg`],
      offers: {
        url: `https://frgroup.uz/ru/store/${slug}-1`,
        price: '1900000',
        priceCurrency: 'UZS',
        availability: 'https://schema.org/InStock',
        shippingDetails: { shippingDestination: { addressCountry: 'UZ' } },
      },
    })}</script>
  `;
  const detailHtml = (slug) => `${htmlList(slug)}<script>var productId = 12345;</script>`;
  const fakeFetch = async (url) => {
    const s = String(url);
    if (s.includes('?page=')) return new Response('', { status: 404 });
    if (s.includes('/tufli') && !s.includes('tufli-1')) return new Response(htmlList('tufli'), { status: 200 });
    if (s.includes('/botinki-i-sapogi')) return new Response('', { status: 200 });
    if (s.includes('/krossovki') && !s.includes('krossovki-1')) return new Response(htmlList('krossovki'), { status: 200 });
    if (s.includes('tufli-1')) return new Response(detailHtml('tufli'), { status: 200 });
    if (s.includes('krossovki-1')) return new Response(detailHtml('krossovki'), { status: 200 });
    if (s.includes('get-product-info')) {
      return new Response(JSON.stringify({
        modelId: 12345, salePrice: 1900000, canBePurchased: true, isOutOfStock: false,
        sizes: [{ euSize: '42', variantId: 1, quantity: 2 }],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } });
    }
    return new Response('', { status: 404 });
  };
  const summary = await runCollector({ fetchImpl: fakeFetch, writeOutput: false });
  assert.equal(summary.liveFrGroupSupply.netNewDistinctOptions, 2);
  assert.equal(summary.styleFootwearBreakdown.dressAndSmartLeads, 1);
  assert.equal(summary.styleFootwearBreakdown.sneakerAndCasualLeads, 1);
  assert.equal(summary.combinedSupplyTotals.verifiedPurchasableOffers, 0);
  assert.equal(summary.combinedSupplyTotals.meets200InitialSupplyObjective, false);
});
