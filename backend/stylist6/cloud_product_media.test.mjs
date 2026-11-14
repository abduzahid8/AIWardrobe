import test from 'node:test';
import assert from 'node:assert/strict';
import sharp from 'sharp';
import { readBoundedProductImage, resolveProductAsset } from './cloud_handler.mjs';

const url='https://geox-cdn.thron.com/allowed-garment-photo.jpg';
const syntheticJpeg=()=>sharp({create:{
  width:400,height:500,channels:3,background:{r:66,g:99,b:125},
}}).jpeg({quality:80}).toBuffer();

test('same merchant photo reused by concurrent looks produces exactly one network fetch',async()=>{
  const bytes=await syntheticJpeg();
  let releases;
  let calls=0;
  const gate=new Promise(resolve=>{releases=resolve});
  const fetchStub=async(src,opts)=>{
    calls++;
    assert.equal(src,url);
    assert.equal(opts.redirect,'error');
    assert.ok(opts.signal instanceof AbortSignal);
    await gate;
    return new Response(bytes,{status:200,headers:{'Content-Type':'image/jpeg'}});
  };
  const cache=new Map();
  const pending=Array.from({length:4},()=>resolveProductAsset(url,cache,fetchStub));
  assert.equal(calls,1,'in-flight same URL must be coalesced');
  releases();
  const assets=await Promise.all(pending);
  assert.equal(calls,1);
  assert.ok(assets[0].bytes.length>=128);
  assert.equal(new Set(assets.map(x=>x.sha256)).size,1);
  assert.equal(await resolveProductAsset(url,cache,fetchStub),assets[0]);
  assert.equal(calls,1,'success must also stay cached within request');
});

test('one failing merchant URL is not repeatedly re-requested on other looks',async()=>{
  let calls=0;
  const cache=new Map();
  const fetchStub=async()=>{calls++;return new Response('Forbidden',{status:403})};
  const settled=await Promise.allSettled(Array.from({length:4},
    ()=>resolveProductAsset(url,cache,fetchStub)));
  assert.equal(calls,1);
  assert.ok(settled.every(x=>x.status==='rejected'&&x.reason.mediaFailure==='http_403'));
  assert.equal(cache.size,1);
  await assert.rejects(resolveProductAsset(url,cache,fetchStub),
    x=>x.mediaFailure==='http_403');
  assert.equal(calls,1);
});

test('CDN 429 and 5xx remain explicit merchant failures, never client/user errors',async()=>{
  for(const [status,reason] of [[429,'http_429'],[502,'http_5xx'],[404,'http_404']]){
    const result=resolveProductAsset(url,new Map(),async()=>new Response('',{status}));
    await assert.rejects(result,err=>
      err.message==='product_image_unresolved'&&err.mediaFailure===reason);
  }
});

test('unknown/blocked seller host never triggers HTTP requests',async()=>{
  let calls=0;
  const fake=async()=>{calls++;throw new Error('must_not_fetch')};
  await assert.rejects(resolveProductAsset('https://geox-cdn.thron.com.evil.invalid/image.jpg',new Map(),fake),
    err=>err.mediaFailure==='host_not_allowlisted');
  await assert.rejects(resolveProductAsset('http://geox-cdn.thron.com/image.jpg',new Map(),fake),
    err=>err.mediaFailure==='host_not_allowlisted');
  assert.equal(calls,0);
});

test('streaming body enforces four MiB even when Content-Length lies or is absent',async()=>{
  const oversized=new ReadableStream({
    start(controller){
      controller.enqueue(new Uint8Array(3*1024*1024));
      controller.enqueue(new Uint8Array(2*1024*1024));
      controller.close();
    },
  });
  const response=new Response(oversized,{headers:{'Content-Type':'image/jpeg'}});
  await assert.rejects(readBoundedProductImage(response),x=>x.mediaFailure==='oversize');
  const fake=new Response('x',{headers:{'Content-Length':'5000000'}});
  await assert.rejects(readBoundedProductImage(fake),x=>x.mediaFailure==='oversize');
});

test('valid WebP bytes are normalized to real JPEG without new provider calls',async()=>{
  const webp=await sharp({create:{
    width:480,height:600,channels:3,background:'#776644',
  }}).webp().toBuffer();
  let calls=0;
  const normalized=await resolveProductAsset(url,new Map(),async()=>{
    calls++;
    return new Response(webp,{headers:{'Content-Type':'image/webp'}});
  });
  assert.equal(calls,1);
  assert.equal(normalized.bytes[0],0xff);
  assert.equal(normalized.bytes[1],0xd8);
  const m=await sharp(normalized.bytes).metadata();
  assert.equal(m.format,'jpeg');
});
