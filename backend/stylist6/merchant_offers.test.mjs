import test from 'node:test';
import assert from 'node:assert/strict';
import { classifyItemImagePermissions, summarizeDeckOffers, verifyMerchantOffer, MERCHANT_OFFER_VERSION } from './merchant_offers.mjs';

test('verifyMerchantOffer never fabricates checkout, stock, or delivery from catalog listing alone', () => {
  const now = Date.parse('2026-11-14T00:00:00Z');
  const item = {
    id: 'geox-u66h9a00022c6005',
    brand: 'Geox',
    isShopItem: true,
    price: 1_299_900,
    currency: 'UZS',
    priceObservedAt: '2026-11-12T10:00:00Z',
    priceFreshness: 'fresh',
    shopUrl: 'https://www.geox.com/product/u66h9a',
    imageUrl: 'https://geox-cdn.thron.com/delivery/public/image/geox/abc.jpg',
  };
  const v = verifyMerchantOffer(item, { now });
  assert.equal(v.version, MERCHANT_OFFER_VERSION);
  assert.equal(v.offerStatus, 'unverified_catalog_listing');
  assert.equal(v.sizeStockStatus, 'unverified');
  assert.equal(v.deliveryEligible, 'unverified');
  assert.equal(v.checkoutAvailability, 'unverified');
  assert.equal(v.purchaseUrlValid, true);
  assert.equal(v.priceFreshness, 'fresh');
  assert.equal(v.imagePermissions.externalInferenceAllowed, false);
  assert.equal(v.imagePermissions.technicalInferenceAllowlisted, true);
  assert.equal(v.imagePermissions.rightsEvidenceStatus, 'unverified');
});

test('verifyMerchantOffer promotes fresh live variant snapshot with UZ delivery without faking checkout', () => {
  const now = Date.parse('2026-11-14T00:10:00Z');
  const item = {
    id: 'fruz-th-123',
    brand: 'Calvin Klein',
    isShopItem: true,
    price: 1_791_300,
    currency: 'UZS',
    shopUrl: 'https://frgroup.uz/ru/store/lofery-hybrid-clean-ck',
    imageUrl: 'https://media.frgroup.uz/images/ck-loafer.jpg',
  };
  const registry = new Map([[item.id, {
    merchant: 'FR Group',
    canonicalProductId: 'fruz-th-123',
    variantId: 90123,
    targetMarket: 'UZ',
    deliveryEligible: true,
    sizeStockStatus: 'merchant_reported_by_size',
    observedAt: '2026-11-14T00:00:00Z',
    refreshBefore: '2026-11-14T00:30:00Z',
    price: 1_791_300,
    currency: 'UZS',
    availableSizes: [{ euSize: '42', variantId: 90123, merchantReportedQty: 2 }],
  }]]);
  const v = verifyMerchantOffer(item, { offerRegistry: registry, requestedSize: '42', now });
  assert.equal(v.offerStatus, 'merchant_reported_live');
  assert.equal(v.requestedSizeStatus, 'merchant_reported_available_not_reserved');
  assert.equal(v.deliveryEligible, true);
  assert.equal(v.checkoutAvailability, 'unverified');

  // Expired snapshot must never stay marked live.
  const expired = verifyMerchantOffer(item, { offerRegistry: registry, requestedSize: '42', now: now + 31 * 60_000 });
  assert.equal(expired.offerStatus, 'unverified_catalog_listing');
  assert.equal(expired.deliveryEligible, 'unverified');
});

test('unlisted third-party image CDN is blocked from external inference', () => {
  const perms = classifyItemImagePermissions('https://admin.etln.uz/storage/photo.webp');
  assert.equal(perms.status, 'unlicensed_external_host');
  assert.equal(perms.externalInferenceAllowed, false);
});

test('summarizeDeckOffers aggregates unique shop items accurately', () => {
  const s = summarizeDeckOffers([{
    items: [
      { id: 'w-1', isShopItem: false },
      { id: 's-1', isShopItem: true, brand: 'Geox', price: 1000, currency: 'UZS', priceFreshness: 'fresh', shopUrl: 'https://geox.com/p', imageUrl: 'https://geox-cdn.thron.com/a.jpg' },
    ],
  }]);
  assert.equal(s.uniqueShopItems, 1);
  assert.equal(s.checkoutGuaranteed, false);
  assert.equal(s.imageInferencePermitted, 0);
});

test('strict 24-hour merchant evidence must bind SKU, price, delivery, available size and direct link',()=>{
  const now=Date.parse('2026-11-14T18:00:00Z');
  const item={id:'verified-1',isShopItem:true,shopUrl:'https://frgroup.uz/ru/store/verified-1',
    imageUrl:'https://media.frgroup.uz/images/x.jpg',price:1500000,currency:'UZS'};
  const base={canonicalProductId:item.id,targetMarket:'UZ',observedAt:'2026-11-14T17:00:00Z',
    deliveryEligible:true,sizeStockStatus:'merchant_reported_by_size',
    availableSizes:[{euSize:'42',merchantReportedQty:1}],price:item.price,currency:'UZS'};
  const read=row=>verifyMerchantOffer(item,{now,requestedSize:'42',offerRegistry:new Map([[item.id,row]])});
  assert.equal(read(base).offerStatus,'merchant_reported_live');
  assert.equal(read({...base,observedAt:'2026-11-12T17:00:00Z'}).offerStatus,'unverified_catalog_listing');
  assert.equal(read({...base,canonicalProductId:'other-item'}).offerStatus,'unverified_catalog_listing');
  assert.equal(read({...base,price:1}).offerStatus,'unverified_catalog_listing');
  assert.equal(read({...base,deliveryEligible:false}).offerStatus,'unverified_catalog_listing');
  assert.equal(read({...base,availableSizes:[{euSize:'42',merchantReportedQty:0}]}).offerStatus,'unverified_catalog_listing');
  assert.equal(read({...base,availableSizes:[]}).offerStatus,'unverified_catalog_listing');
  assert.equal(read({...base,refreshBefore:'invalid-date'}).offerStatus,'unverified_catalog_listing');
  assert.equal(read({...base,purchaseUrl:'https://frgroup.uz/ru/store/other'}).offerStatus,'unverified_catalog_listing');
});
