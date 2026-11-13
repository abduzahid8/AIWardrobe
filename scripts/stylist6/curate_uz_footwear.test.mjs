import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {
  ALLOWED_BRANDS, DISALLOWED_BRANDS, parseFrStructuredProducts,
  frProductFamily, projectFrCandidate, manualReviewCandidates,
  validateReviewBatch, reconcileExistingCatalog, projectInactiveCatalogPreview,
} from './curate_uz_footwear.mjs';
import { adaptCatalog } from '../../backend/stylist6/catalog_adapter.mjs';

const root = new URL('../../benchmarks/stylist6/', import.meta.url);
const staged = JSON.parse(fs.readFileSync(new URL('FOOTWEAR_UZ_CANDIDATES_20261113.json',root),'utf8'));
const archived = JSON.parse(fs.readFileSync(new URL('FOOTWEAR_UZ_EXISTING_INACTIVE_20261113.json',root),'utf8'));
const preview = JSON.parse(fs.readFileSync(new URL('FOOTWEAR_UZ_INACTIVE_PREVIEW_20261113.json',root),'utf8'));

test('only owner-approved labels, exactly the four brands, zero excluded brands', () => {
  assert.deepEqual(ALLOWED_BRANDS,['Tommy Hilfiger','Zara','MANGO','Massimo Dutti']);
  assert.deepEqual(DISALLOWED_BRANDS,['ECCO','Cabani','Basconi']);
  assert.equal(staged.items.length,16);
  assert.deepEqual(staged.byBrand,{'Tommy Hilfiger':7,Zara:5,MANGO:3,'Massimo Dutti':1});
  assert.ok(staged.items.every(x => ALLOWED_BRANDS.includes(x.brand) && !DISALLOWED_BRANDS.includes(x.brand)));
  assert.equal(validateReviewBatch(staged.items).checked,16);
  assert.equal(staged.readyToPublish,0);
  assert.equal(staged.items.filter(x=>x.imageNvidiaRights==='not_authorized').length,16);
});

test('JSON-LD parses independently of page text, with source-locked price and SKU', () => {
  const item={
    '@type':'Product',brand:{name:'Tommy Hilfiger'},name:'Лоферы HILFIGER BLACK LOAFER',
    sku:'FM0FM12345',
    offers:{
      url:'https://frgroup.uz/ru/store/lofery-hilfiger-black-tommy-hilfiger-fm0fm12345-chernyj',
      price:'2050300.00',priceCurrency:'UZS',availability:'https://schema.org/InStock',
      shippingDetails:{shippingDestination:{addressCountry:'UZ'}},
    },
    image:['https://media.frgroup.uz/images/catalog-product-desktop/fake.jpg'],
  };
  const html='<script type="application/ld+json">'+JSON.stringify({'@type':'ItemList',itemListElement:[{item}]})+'</script>';
  const parsed=parseFrStructuredProducts(html);
  assert.equal(parsed.length,1);
  const candidate=projectFrCandidate(parsed[0],{observedAt:'2026-11-13T10:00:00Z',evidenceHash:'f'.repeat(64)});
  assert.equal(candidate.family,'loafer');
  assert.equal(candidate.price,2050300);
  assert.equal(candidate.merchantSku,'FM0FM12345');
  assert.equal(candidate.shippingToUz,true);
  assert.equal(candidate.listingInStock,true);
  assert.equal(candidate.sizeStockStatus,'unverified');
  assert.equal(candidate.deliveredTotalStatus,'unverified');
  assert.equal(candidate.imageNvidiaRights,'not_authorized');
  assert.equal(projectFrCandidate({...item,brand:{name:'Cabani'}},{observedAt:'2026-11-13',evidenceHash:'f'.repeat(64)}),null);
  assert.equal(projectFrCandidate({...item,offers:{...item.offers,priceCurrency:'GBP'}},{observedAt:'2026-11-13',evidenceHash:'f'.repeat(64)}),null);
  assert.equal(projectFrCandidate({...item,offers:{...item.offers,url:'https://evil.example/shop'}} ,{observedAt:'2026-11-13',evidenceHash:'f'.repeat(64)}),null);
});

test('chukka runners and espadrilles do not masquerade as dress boots/loafers', () => {
  assert.equal(frProductFamily('Хайтопы HILFIGER WP MIX CHUKKA RUNNER'),null);
  assert.equal(frProductFamily('Лоферы FLEXIBLE HILFIGER SDE ESPADRILLE'),null);
  assert.equal(frProductFamily('Лоферы FLEX HILFIGER SDE TASSEL LOAFER'),'loafer');
  assert.equal(frProductFamily('Ботинки HILFIGER SDE WARM HYBRID CHUKKA'),'chukka');
});

test('archived source URL matches are reconciled; never copied or reactivated', () => {
  assert.equal(archived.matches.length,3);
  const matched=staged.items.filter(x=>x.inventoryAction==='review_archived_existing_row');
  assert.equal(matched.length,3);
  assert.ok(matched.every(x=>x.existingCatalogId && Number.isInteger(x.oldPriceUZS)));
  assert.equal(staged.netNewCandidates,13);
  assert.equal(staged.previouslyArchived,3);
  assert.deepEqual(reconcileExistingCatalog(
    staged.items.map(({inventoryAction,existingCatalogId,...x})=>x),
    archived.matches).map(x=>x.existingCatalogId).filter(Boolean).sort(),
    archived.matches.map(x=>x.id).sort());
});

test('every merchant photo and listing is staged, NEVER active or paid-inference-authorized', () => {
  assert.equal(preview.rows.length,4);
  assert.equal(staged.eligibleInactiveShopPreview,4);
  assert.ok(preview.rows.every(x=>x.is_active===false && x.garment_type==='shoes' &&
    x.category==='shoes' && x.currency==='UZS' && x.external_id));
  assert.deepEqual(projectInactiveCatalogPreview(staged.items),preview.rows);
  const mapped=adaptCatalog(preview.rows);
  assert.equal(mapped.items.length,0);
  assert.equal(mapped.skipped.inactive,4);
  const hypothetical=adaptCatalog(preview.rows.map(x=>({...x,is_active:true})));
  assert.equal(hypothetical.items.length,4);
  assert.ok(hypothetical.items.every(x=>x.macroCategory==='shoes'));
  // Hypothetical in-memory shoe projection is not an availability/rights grant.
});

test('review-only records cannot claim merchant stock, image rights or guessed availability', () => {
  const manual=manualReviewCandidates();
  assert.equal(manual.length,9);
  assert.ok(manual.every(x=>x.observedAt===null && x.imageUrl===null &&
    x.listingInStock===null && x.imageDisplayRights==='unverified'));
  assert.throws(()=>validateReviewBatch([{...staged.items[0], imageNvidiaRights:'cleared'},...staged.items.slice(1)]),/inaccurate_purchase_or_image_rights/);
  assert.throws(()=>validateReviewBatch([{...staged.items[0], brand:'ECCO'},...staged.items.slice(1)]),/brand_not_allowed/);
  const suspicious={...staged.items[0],productUrl:'https://frgroup.uz.evil.example/ru/store/steal'};
  assert.throws(()=>validateReviewBatch([suspicious,...staged.items.slice(1)]),/untrusted_url/);
});

test('all candidate URLs are unique and evidence has not been converted into stock guarantees', () => {
  assert.equal(new Set(staged.items.map(x=>x.id)).size,16);
  assert.equal(new Set(staged.items.map(x=>x.productUrl)).size,16);
  assert.equal(staged.items.filter(x=>x.priceEvidence==='direct_merchant_product_jsonld').length,7);
  assert.equal(staged.items.filter(x=>x.priceEvidence==='official_search_result_needs_direct_recheck').length,9);
  assert.ok(staged.items.every(x=>x.sizeStockStatus==='unverified' && x.deliveredTotalStatus==='unverified'));
});
