import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { validateCollectionLeads } from './collection_leads.mjs';
import { adaptCatalog } from '../../backend/stylist6/catalog_adapter.mjs';

const file = new URL('../../benchmarks/stylist6/MANGO_UZ_COLLECTION_LEADS_20261113.json', import.meta.url);
const batch = JSON.parse(fs.readFileSync(file, 'utf8'));

test('eight official catalog references include derby, oxford and monk strap, with no active SKUs', () => {
  const result = validateCollectionLeads(batch);
  assert.equal(result.listedVariantReferences,8);
  assert.equal(result.uniqueMerchantArticles,5);
  assert.deepEqual(
    Object.fromEntries([...new Set(result.leads.map(x => x.family))]
      .map(f => [f, result.leads.filter(x => x.family===f).length])),
    { 'formal-shoes':2, 'monk-strap':2, 'derby':2, oxford:2 });
  assert.ok(result.leads.every(x=>x.isActive===false && x.productUrl===null &&
    x.imageUrl===null && x.observedAt===null && x.merchantVariantId===null &&
    x.sizeStockStatus==='not_checked' && x.checkoutStockStatus==='unverified'));
  assert.equal(result.readyToPublish,0);
  const direct = adaptCatalog(result.leads);
  assert.equal(direct.items.length,0, 'collection leads cannot be accepted as shop inventory');
});

test('duplicate article+color, wrong market/host, fabricated active status and untrusted prices fail', () => {
  assert.throws(()=>validateCollectionLeads({...batch,country:'GB'}),/scope/);
  assert.throws(()=>validateCollectionLeads({...batch,sourceCollection:'https://example.com/catalog'}),/scope/);
  assert.throws(()=>validateCollectionLeads({...batch,leads:[...batch.leads,batch.leads[0]]}),/invalid_collection/);
  assert.throws(()=>validateCollectionLeads({...batch,leads:batch.leads.map((x,i)=>i===0?{...x,indexedPriceUZS:1}:x)}),/invalid_collection/);
});

test('leads cannot silently become sellable items without missing exact offer data',()=>{
  const result=validateCollectionLeads(batch);
  assert.equal(result.leads.filter(x=>x.stage==='collection_reference_only').length,8);
  const unsafeCatalogProjection=result.leads.map(x=>({
    id:x.leadId,name:x.name, brand:x.brand,price:x.indexedPriceUZS,
    currency:'UZS',category:'shoes',garment_type:'shoes',
    source_url:x.productUrl,image_url:x.imageUrl,is_active:false,
  }));
  assert.equal(adaptCatalog(unsafeCatalogProjection).items.length,0);
});
