import test from 'node:test';
import assert from 'node:assert/strict';
import {
  FR_PRODUCT_INFO_ENDPOINT, FR_VARIANT_SNAPSHOT_MAX_AGE_MS,
  frModelIdFromHtml, parseFrSizeResponse, fetchFrSizeEvidence, assessFrVariantSize,
} from './fr_variant_stock.mjs';

const observedAt = '2026-11-13T10:50:00Z';
const example = {
  modelId: 164544, canBePurchased: true, isOutOfStock: false,
  salePrice: 2050300, article: 158,
  skuSortedList: [{ id: 438597 }, { id: 437165 }, { id: 437190 }],
  sizes: [
    { quantity: 2, value: 438597, size: '43', sizeKey: '43', available: 1 },
    { quantity: 1, value: 437165, size: '44', sizeKey: '44', available: 1 },
    { quantity: 1, value: 437190, size: '45', sizeKey: '45', available: 1 },
  ],
};
const args = { modelId: 164544, expectedPrice: 2050300,
  collectedAt: observedAt, responseSha256: 'c'.repeat(64) };

test('numeric model ID parsed from server-assigned script variable only', () => {
  assert.equal(frModelIdFromHtml('var productId = "164544";'),164544);
  assert.equal(frModelIdFromHtml("  var productId = '171923';"),171923);
  assert.equal(frModelIdFromHtml("productId = '164544';"),null);
  assert.equal(frModelIdFromHtml('var productId = "../../api";'),null);
  assert.equal(frModelIdFromHtml('var productId = "0";'),null);
  assert.equal(frModelIdFromHtml(null),null);
});

test('exact merchant model, price, variant and reported size quantities are preserved', () => {
  const evidence = parseFrSizeResponse(example,args);
  assert.equal(evidence.status,'merchant_reported_by_size');
  assert.equal(evidence.modelId,164544);
  assert.deepEqual(evidence.availableSizes,[
    {euSize:'43',variantId:438597,merchantReportedQty:2},
    {euSize:'44',variantId:437165,merchantReportedQty:1},
    {euSize:'45',variantId:437190,merchantReportedQty:1},
  ]);
  assert.equal(evidence.totalMerchantReportedUnits,4);
  assert.equal(evidence.exactCheckoutAvailability,'unverified');
  assert.equal(evidence.deliveryCharge,'unverified');
  assert.equal(Date.parse(evidence.refreshBefore)-Date.parse(evidence.checkedAt),FR_VARIANT_SNAPSHOT_MAX_AGE_MS);
});

test('invalid inventory, bad size, duplicate variant or price mismatch cannot claim availability', () => {
  const corrupted = [
    { ...example, modelId: 999 },
    { ...example, salePrice: 2050301 },
    { ...example, canBePurchased: false },
    { ...example, isOutOfStock: true },
    { ...example, sizes: [] },
    { ...example, sizes: example.sizes.map((x,i)=>i===1?{...x,value:438597}:x) },
    { ...example, sizes: example.sizes.map((x,i)=>i===1?{...x,size:'43'}:x) },
    { ...example, sizes: example.sizes.map((x,i)=>i===1?{...x,quantity:0}:x) },
    { ...example, sizes: example.sizes.map((x,i)=>i===1?{...x,available:0}:x) },
    { ...example, sizes: example.sizes.map((x,i)=>i===1?{...x,size:'100'}:x) },
    { ...example, skuSortedList: [{id:438597},{id:437165}] },
    { ...example, skuSortedList: [{id:438597},{id:437165},{id:438597}] },
  ];
  for(const value of corrupted) assert.equal(parseFrSizeResponse(value,args),null,
    'must reject untrusted variant metadata: '+JSON.stringify(value).slice(0,130));
  assert.equal(parseFrSizeResponse(example,{...args,modelId:-1}),null);
  assert.equal(parseFrSizeResponse(example,{...args,expectedPrice:1.2}),null);
  assert.equal(parseFrSizeResponse(example,{...args,responseSha256:'bad'}),null);
});

test('same-size evidence expires after 30 minutes and never claims checkout', () => {
  const variantEvidence = parseFrSizeResponse(example,args);
  const row = {brand:'Tommy Hilfiger',sizeStockStatus:'merchant_reported_by_size',variantEvidence};
  assert.deepEqual(assessFrVariantSize(row,'44',Date.parse(observedAt)+1000),{
    status:'merchant_reported_available_not_reserved',
    merchantVariantId:437165, merchantReportedQty:1,
    checkedAt:new Date(observedAt).toISOString(),checkoutVerified:false,
  });
  assert.deepEqual(assessFrVariantSize(row,'40',Date.parse(observedAt)+1000),{
    status:'size_not_listed_in_snapshot',checkoutVerified:false,
  });
  assert.equal(assessFrVariantSize(row,'44',Date.parse(observedAt)+FR_VARIANT_SNAPSHOT_MAX_AGE_MS).status,
    'unknown_or_expired');
  assert.equal(assessFrVariantSize(row,'44',Date.parse(observedAt)-1).status,
    'unknown_or_expired');
  assert.equal(assessFrVariantSize({...row,brand:'ECCO'},'44',Date.parse(observedAt)+1000).status,
    'unknown_or_expired');
});

test('read-only GET uses fixed allowlisted endpoint and no credentials or writes', async () => {
  const captured=[];
  const fetchImpl = async (url, options)=>{
    captured.push({url,options});
    return Response.json(example);
  };
  const evidence=await fetchFrSizeEvidence({
    html:'var productId = "164544";',
    expectedPrice:2050300,fetchImpl,collectedAt:observedAt,
  });
  assert.equal(evidence.availableSizes.length,3);
  assert.equal(captured.length,1);
  assert.equal(captured[0].url,FR_PRODUCT_INFO_ENDPOINT+'?productId=164544');
  assert.equal(captured[0].options.method,'GET');
  assert.equal(captured[0].options.redirect,'error');
  assert.equal(captured[0].options.cache,'no-store');
  assert.equal(captured[0].options.headers.Accept,'application/json');
  assert.equal(captured[0].options.headers.Authorization,undefined);
});

test('non-json, invalid content, out-of-stock or wrong model ID return no size claims', async () => {
  const invalid = [
    new Response('No', {status:502}),
    new Response('<html>failed</html>',{headers:{'content-type':'text/html'}}),
    Response.json({...example,modelId:171923}),
    Response.json({...example,isOutOfStock:true}),
    Response.json({...example, salePrice:2050400}),
  ];
  for(const response of invalid){
    const result=await fetchFrSizeEvidence({
      html:'var productId = "164544";',expectedPrice:2050300,
      fetchImpl: async()=>response,collectedAt:observedAt,
    });
    assert.equal(result,null);
  }
  const noModel=await fetchFrSizeEvidence({html:'empty',expectedPrice:2050300,
    fetchImpl:async()=>{throw new Error('must not fetch without modelId')}});
  assert.equal(noModel,null);
});
