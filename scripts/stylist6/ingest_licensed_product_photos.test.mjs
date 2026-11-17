import test from 'node:test';
import assert from 'node:assert/strict';
import sharp from 'sharp';
import {checkPhotoGrant,ingestLicensedProductPhoto} from './ingest_licensed_product_photos.mjs';
import {verifiedCatalogPhotoCache} from '../../backend/stylist6/catalog_photo_cache.mjs';

const now=Date.parse('2026-11-17T16:00:00Z');
const grant={
  productId:'zara-uz-548984761',
  sourceImageUrl:'https://static.zara.net/assets/public/sample.jpg',
  rightsBasis:'merchant_contract',
  rightsReference:'contract-20261117-original-image-usage-verified',
  rightsGrantedAt:'2026-11-16T10:00:00Z',
  rightsExpiresAt:'2027-11-17T10:00:00Z',
  hostingPermission:true,externalAiInferencePermission:true,
};
const row={id:grant.productId,image_url:grant.sourceImageUrl,
  source_url:'https://www.zara.com/uz/en/example',is_active:true,price:129,currency:'USD'};
const photo=await sharp({create:{width:440,height:480,
  channels:3,background:'#b2afa8'}}).jpeg({quality:80}).toBuffer();
test('admin cannot ingest photographed merchandise without EXPLICIT hosting AND external AI rights',()=>{
  assert.equal(checkPhotoGrant({...grant,hostingPermission:false},now).ok,false);
  assert.equal(checkPhotoGrant({...grant,externalAiInferencePermission:false},now).ok,false);
  assert.equal(checkPhotoGrant({...grant,rightsBasis:'public_website'},now).ok,false);
  assert.equal(checkPhotoGrant({...grant,rightsReference:''},now).ok,false);
  assert.equal(checkPhotoGrant({...grant,sourceImageUrl:'https://evil.example/x'},now).ok,false);
  assert.equal(checkPhotoGrant({...grant,rightsExpiresAt:'2026-01-01T00:00:00Z'},now).ok,false);
});
test('dry-run never fetches, uploads, updates or mutates merchant records',async()=>{
  let uploads=0,network=0,writes=0;
  const result=await ingestLicensedProductPhoto(grant,{now,
    loadProduct:async()=>row,
    fetchImpl:async()=>{network++;throw Error('unexpected');},
    uploadObject:async()=>{uploads++;return true;},
    saveProvenance:async()=>{writes++;},
  });
  assert.equal(result.status,'dry_run');
  assert.deepEqual([uploads,network,writes],[0,0,0]);
});
test('a current-source exact-SKU licensed JPEG is uploaded and recorded with checkable hash',async()=>{
  const writes=[];let stored=null,loads=0;
  const outcome=await ingestLicensedProductPhoto(grant,{now,apply:true,
    loadProduct:async()=>{loads++;return row;},
    fetchImpl:async()=>new Response(photo,{headers:{'Content-Type':'image/jpeg'}}),
    uploadObject:async(path,bytes)=>{
      stored={path,bytes};
      const info=await sharp(bytes).metadata();
      assert.equal(info.format,'jpeg');
      return true;
    },
    saveProvenance:async data=>writes.push(data),
  });
  assert.equal(outcome.status,'ingested');
  assert.equal(loads,2,'source must be rechecked after upload');
  assert.equal(writes.length,1);
  assert.equal(writes[0].status,'ready');
  assert.equal(writes[0].source_image_url,row.image_url);
  assert.equal(writes[0].storage_path,stored.path);
  assert.equal(writes[0].cache_hosting_granted,true);
  assert.equal(writes[0].external_ai_inference_granted,true);
  assert.ok(verifiedCatalogPhotoCache({id:row.id,imageUrl:row.image_url},
    writes[0],{now,supabaseOrigin:'https://fyqpifmrsftsfqibhwhy.supabase.co'}));
});
test('source drift during ingest never publishes a READY cache record',async()=>{
  let reads=0,commits=0;
  await assert.rejects(ingestLicensedProductPhoto(grant,{
    now,apply:true,
    loadProduct:async()=>{reads++;return reads===1?row:
      {...row,image_url:row.image_url+'?changed'};},
    fetchImpl:async()=>new Response(photo),
    uploadObject:async()=>true,
    saveProvenance:async()=>{commits++;},
  }),/source_changed_during_ingest/);
  assert.equal(commits,0);
});
test('active catalog item and actual seller link are required before fetching media',async()=>{
  let downloads=0;
  const output=await ingestLicensedProductPhoto(grant,{now,apply:true,
    loadProduct:async()=>({...row,is_active:false}),
    fetchImpl:async()=>{downloads++;throw Error('must not fetch');},
    uploadObject:async()=>true,saveProvenance:async()=>{},
  });
  assert.equal(output.status,'rejected');
  assert.equal(downloads,0);
});
