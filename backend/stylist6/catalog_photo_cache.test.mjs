import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {productPhotoSourceHash,verifiedCatalogPhotoCache,
  resolveCatalogCachedPhoto} from './catalog_photo_cache.mjs';

const now=Date.parse('2026-11-17T16:00:00Z');
const original={id:'zara-uz-548984761',
  imageUrl:'https://static.zara.net/assets/public/source/123-f1.jpg'};
const raw=Buffer.concat([Buffer.from([0xff,0xd8,0xff,0xe0]),
  Buffer.alloc(150,0x3f),Buffer.from([0xff,0xd9])]);
const hash=createHash('sha256').update(raw).digest('hex');
const sourceHash=productPhotoSourceHash(original.id,original.imageUrl);
const record={
  product_id:original.id,source_image_url:original.imageUrl,
  source_hash:sourceHash,status:'ready',storage_bucket:'shop-catalog',
  storage_path:'licensed-stylist6/'+hash.slice(0,2)+'/'+hash+'.jpg',
  image_sha256:hash,media_type:'image/jpeg',rights_basis:'merchant_contract',
  rights_reference:'contract-verified-20261117-001',
  cache_hosting_granted:true,external_ai_inference_granted:true,
  rights_granted_at:new Date(now-86400000).toISOString(),
  last_verified_at:new Date(now-86400000).toISOString(),
  rights_expires_at:null,
};
const supabaseOrigin='https://fyqpifmrsftsfqibhwhy.supabase.co';
const opts={supabaseOrigin,now};
test('source hash is deterministic for current product + URL, not a loose filename',()=>{
  assert.match(sourceHash,/^[0-9a-f]{64}$/);
  assert.notEqual(sourceHash,productPhotoSourceHash(original.id,original.imageUrl+'?changed'));
  assert.equal(productPhotoSourceHash('x','http://vendor.example/x'),null);
  assert.equal(productPhotoSourceHash('../etc','https://vendor.example/a'),null);
  assert.equal(productPhotoSourceHash('abc','https://user@vendor.example/a'),null);
});
test('only explicit licenced current-source approved first-party image can be resolved',()=>{
  const value=verifiedCatalogPhotoCache(original,record,opts);
  assert.ok(value);
  assert.match(value.cachedUrl,/^https:\/\/fyqpifmrsftsfqibhwhy\.supabase\.co\/storage\/v1\/object\/public\/shop-catalog\/licensed-stylist6\//);
  for(const [field,value] of [
    ['product_id','OTHER_SKU'],['source_image_url','https://vendor.example/other'],
    ['source_hash','0'.repeat(64)],['status','pending'],
    ['cache_hosting_granted',false],['external_ai_inference_granted',false],
    ['rights_basis','unknown'],['rights_reference',''],['storage_bucket','private'],
    ['storage_path','../user_uploads/secret.jpg'],['image_sha256','bad'],
    ['media_type','image/png'],['last_verified_at','2025-11-17T00:00:00Z'],
    ['rights_expires_at','2026-11-01T00:00:00Z'],
  ]){
    assert.equal(verifiedCatalogPhotoCache(original,{...record,[field]:value},opts),null,
      field+' must fail closed');
  }
  assert.equal(verifiedCatalogPhotoCache({...original,imageUrl:original.imageUrl+'?v=2'},
    record,opts),null,'image source drift invalidates cached source');
  assert.equal(verifiedCatalogPhotoCache(original,record,{
    supabaseOrigin:'https://attacker.example',now}),null);
});
test('exact downloaded bytes hash is revalidated before transmitting to AI',async()=>{
  let requested='',originalCalls=0;
  const output=await resolveCatalogCachedPhoto({
    item:original,record,supabaseOrigin,now,
    originalResolver:async()=>{originalCalls++;throw Error('unexpected original');},
    decodeImage:async bytes=>bytes,
    fetchImpl:async url=>{
      requested=url;
      return new Response(raw,{headers:{'Content-Type':'image/jpeg'}});
    },
  });
  assert.ok(requested.includes('/licensed-stylist6/'));
  assert.equal(originalCalls,0);
  assert.deepEqual(output.bytes,raw);
  assert.equal(output.sha256,hash);
});
test('tampered bytes or revoked permissions safely fall back to original source, NEVER fake approval',async()=>{
  let fallbacks=0;
  const originalResolver=async item=>{
    fallbacks++;
    assert.equal(item.imageUrl,original.imageUrl);
    return {bytes:raw,original:true};
  };
  const tampered=Buffer.from(raw);tampered[10]^=0x20;
  const a=await resolveCatalogCachedPhoto({
    item:original,record,supabaseOrigin,now,originalResolver,
    decodeImage:async bytes=>bytes,
    fetchImpl:async()=>new Response(tampered),
  });
  assert.equal(a.original,true);
  const b=await resolveCatalogCachedPhoto({
    item:original,record:{...record,rights_basis:'unreviewed'},supabaseOrigin,now,
    originalResolver,decodeImage:async bytes=>bytes,
    fetchImpl:async()=>{throw Error('no fetch allowed');},
  });
  assert.equal(b.original,true);
  assert.equal(fallbacks,2);
});
test('oversized or non-JPEG content never serves stale cached product',async()=>{
  let fallbacks=0;
  for(const data of [Buffer.alloc(600000,0xff),Buffer.alloc(150,0xaa)]){
    const answer=await resolveCatalogCachedPhoto({
      item:original,record,supabaseOrigin,now,
      originalResolver:async()=>{fallbacks++;return {fallback:true}},
      decodeImage:async x=>x,fetchImpl:async()=>new Response(data),
    });
    assert.equal(answer.fallback,true);
  }
  assert.equal(fallbacks,2);
});
