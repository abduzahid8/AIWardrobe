import test from 'node:test';
import assert from 'node:assert/strict';
import {createRemoteMediaProbe,measureRemoteMedia} from './remote-media-probe.mjs';
const config={projectUrl:'https://fixture.supabase.co',owner:'11111111-1111-4111-8111-111111111111',objectId:'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',anonKey:'public-fixture',bytes:4,sha256:'a'.repeat(64),expiresAt:100000};
const post=()=>new Request('https://fixture.supabase.co/functions/v1/probe',{method:'POST',headers:{Authorization:'Bearer fixture-token'},body:JSON.stringify({url:'https://attacker.invalid',rps:10000})});
test('missing auth, expired requests and wrong method never trigger remote work',async()=>{
 let calls=0;const handler=createRemoteMediaProbe(config,{now:()=>1000,fetch:()=>{calls++;throw new Error('Unexpected request');}});
 assert.equal((await handler(new Request('https://fixture.invalid',{method:'POST'}))).status,401);
 assert.equal((await handler(new Request('https://fixture.invalid'))).status,405);
 const expired=createRemoteMediaProbe({...config,expiresAt:999},{now:()=>1000,fetch:()=>{calls++;}});
 assert.equal((await expired(post())).status,410);assert.equal(calls,0);
});
test('only verified fixture owner may execute and body cannot configure target or rate',async()=>{
 let measured=0,identity='other';const handler=createRemoteMediaProbe(config,{now:()=>1000,fetch:async url=>{assert.equal(url,'https://fixture.supabase.co/auth/v1/user');return Response.json({id:identity});},measure:async(actual,auth)=>{assert.deepEqual(actual,config);assert.equal(auth,'Bearer fixture-token');measured++;return {passed:true};}});
 assert.equal((await handler(post())).status,403);identity=config.owner;
 const response=await handler(post());assert.equal(response.status,200);assert.equal((await response.json()).passed,true);
 assert.equal((await handler(post())).status,409);assert.equal(measured,1);
});
test('identity API outage fails closed without starting traffic',async()=>{
 const handler=createRemoteMediaProbe(config,{now:()=>1000,fetch:async()=>Response.json({}, {status:503}),measure:()=>{throw new Error('Must not run');}});
 assert.equal((await handler(post())).status,503);
});
test('unsafe or excessive deployment configuration is rejected',()=>{
 for(const patch of [{owner:'../other'},{objectId:'bad'},{bytes:2097153},{expiresAt:700001},{sha256:'bad'},{projectUrl:'https://user@fixture.supabase.co'}])assert.throws(()=>createRemoteMediaProbe({...config,...patch},{now:()=>1000}),/Invalid probe/);
});
test('fixed ramp is capped at 170 reads and never uses request-supplied URLs',async()=>{
 let clock=0,calls=0;const result=await measureRemoteMedia(config,'Bearer fixture-token',{now:()=>clock,sleep:async ms=>{clock+=ms;for(let i=0;i<8;i++)await Promise.resolve();},digest:async()=>config.sha256,fetch:async(url,options)=>{assert.equal(url,`https://fixture.supabase.co/storage/v1/object/authenticated/wardrobe-images/${config.owner}/${config.objectId}/thumbnail.png`);assert.equal(options.headers.Authorization,'Bearer fixture-token');calls++;return new Response(new Uint8Array(4));}});
 assert.equal(calls,170);assert.equal(result.passed,true);assert.deepEqual(result.stages.map(x=>x.planned),[20,50,100]);
});
test('wrong image bytes fail the diagnostic rather than reporting success',async()=>{
 let clock=0;const result=await measureRemoteMedia(config,'Bearer fixture-token',{now:()=>clock,sleep:async ms=>{clock+=ms;for(let i=0;i<8;i++)await Promise.resolve();},digest:async()=> 'b'.repeat(64),fetch:async()=>new Response(new Uint8Array(4))});
 assert.equal(result.passed,false);assert.equal(result.stages.length,1);assert.equal(result.stages[0].failures,20);
});

test('oversized response streams are cancelled at the configured byte limit',async()=>{
 let clock=0,cancelled=0;
 const result=await measureRemoteMedia(config,'Bearer fixture-token',{now:()=>clock,sleep:async ms=>{clock+=ms;for(let i=0;i<12;i++)await Promise.resolve();},fetch:async()=>new Response(new ReadableStream({pull(controller){controller.enqueue(new Uint8Array(8));},cancel(){cancelled++;}}))});
 assert.equal(result.passed,false);assert.equal(cancelled,20);assert.equal(result.stages[0].failures,20);
});

test('expiry stops scheduling reads instead of extending the probe lifetime',async()=>{
 let clock=0,calls=0;
 const result=await measureRemoteMedia({...config,expiresAt:1000},'Bearer fixture-token',{now:()=>clock,sleep:async ms=>{clock+=ms;for(let i=0;i<12;i++)await Promise.resolve();},digest:async()=>config.sha256,fetch:async()=>{calls++;return new Response(new Uint8Array(4));}});
 assert.equal(calls,2);assert.equal(result.passed,false);assert.equal(result.stages.length,1);
});

for(const cap of [20,40])test(`slow reads obey declared ${cap}-read cap, report its peak and retain explicit drops`,async()=>{
 let clock=0,calls=0,inFlight=0,maximum=0;const pending=[];
 const release=()=>{for(const resolve of pending.splice(0))resolve();};
 const result=await measureRemoteMedia({...config,concurrencyCap:cap},'Bearer fixture-token',{now:()=>clock,digest:async()=>config.sha256,
  sleep:async ms=>{clock+=ms;if(clock>=19300)release();for(let i=0;i<20;i++)await Promise.resolve();},
  fetch:async()=>{calls++;inFlight++;maximum=Math.max(maximum,inFlight);await new Promise(resolve=>{pending.push(resolve);if(calls===20||clock>=19300)release();});inFlight--;return new Response(new Uint8Array(4));}});
 assert.equal(maximum,cap);assert.equal(result.concurrencyCap,cap);assert.equal(Math.max(...result.stages.map(s=>s.peakActiveReads)),cap);
 assert.equal(result.passed,false);assert.ok(result.stages.at(-1).dropped>0);assert.ok(calls<=cap+21);
});
test('unsupported concurrency caps cannot deploy or issue traffic',async()=>{
 for(const concurrencyCap of [0,1,39,41,200,'40',null]){
  assert.throws(()=>createRemoteMediaProbe({...config,concurrencyCap},{now:()=>1000}),/Invalid probe/);
  await assert.rejects(measureRemoteMedia({...config,concurrencyCap},'Bearer fixture-token'),/Invalid probe/);
 }
});

test('distinct cohort never reuses an object across the complete ramp and records cache evidence',async()=>{
 const ids=Array.from({length:170},(_,i)=>`aaaaaaaa-aaaa-4aaa-8aaa-${i.toString(16).padStart(12,'0')}`),seen=new Set();let clock=0;
 const result=await measureRemoteMedia({...config,objectIds:ids},'Bearer fixture-token',{now:()=>clock,sleep:async ms=>{clock+=ms;for(let i=0;i<12;i++)await Promise.resolve();},digest:async()=>config.sha256,fetch:async url=>{
  assert.ok(!seen.has(url));seen.add(url);return new Response(new Uint8Array(4),{headers:{'cf-cache-status':'MISS'}});
 }});
 assert.equal(result.passed,true);assert.equal(result.distinctIssuedObjects,170);assert.equal(seen.size,170);
 assert.equal(result.stages[0].results[0].cache['cf-cache-status'],'MISS');
});
test('malformed, duplicated or oversized distinct cohorts cannot deploy or measure',async()=>{
 for(const ids of [[],Array(170).fill(config.objectId),Array(171).fill(config.objectId),['../escape']]){
  assert.throws(()=>createRemoteMediaProbe({...config,objectIds:ids},{now:()=>1000}),/Invalid probe/);
  await assert.rejects(measureRemoteMedia({...config,objectIds:ids},'Bearer fixture-token'),/Invalid probe/);
 }
});


test('native JPEG fixture uses the JPEG object path without changing the fixed ramp',async()=>{
 let clock=0,calls=0;const jpeg={...config,thumbnailExtension:'jpg'};
 const result=await measureRemoteMedia(jpeg,'Bearer fixture-token',{now:()=>clock,sleep:async ms=>{clock+=ms;for(let i=0;i<8;i++)await Promise.resolve();},digest:async()=>config.sha256,fetch:async url=>{assert.ok(url.endsWith('/thumbnail.jpg'));calls++;return new Response(new Uint8Array(4));}});
 assert.equal(calls,170);assert.equal(result.passed,true);
});
test('unsupported or path-like extensions fail before any traffic',async()=>{
 for(const thumbnailExtension of ['jpeg','../jpg','jpg?token=x','',null]){
  assert.throws(()=>createRemoteMediaProbe({...config,thumbnailExtension},{now:()=>1000}),/Invalid probe/);
  await assert.rejects(measureRemoteMedia({...config,thumbnailExtension},'Bearer fixture-token'),/Invalid probe/);
 }
});
