import test from 'node:test';
import assert from 'node:assert/strict';
import { verifyStylistTransport } from '../transport/signature.mjs';

const secret = 'testing-stylist6-isolated-secret-123456789';
const bypass = 'ABCD1234abcd5678EFGH9012efgh3456';
let originOverride;
let handler;
const oldDeno = globalThis.Deno;
globalThis.Deno = {
  serve(fn) { handler = fn; },
  env: { get(key) {
    if (key === 'STYLIST6_NODE_SHARED_SECRET') return secret;
    if (key === 'STYLIST_VERCEL_BYPASS_SECRET') return bypass;
    if (key === 'STYLIST6_NODE_ORIGIN') return originOverride;
    return undefined;
  } },
};
await import('../../stylist-proxy/stylist6-staging.ts');

test('isolated Stylist 6 staging GET never calls Vercel', async()=>{
  const oldFetch=globalThis.fetch;
  let called=0;
  globalThis.fetch=async()=>{called++;throw new Error('no upstream calls');};
  try {
    const response=await handler(new Request('https://supabase.example/functions/v1/aiw-stylist6-node-staging'));
    assert.equal(response.status,405);
    assert.equal((await response.json()).code,'method_not_allowed');
    assert.equal(called,0);
  }finally {globalThis.fetch=oldFetch;}
});

test('isolated Stylist 6 staging rejects missing bearer before any upstream calls', async()=>{
  const oldFetch=globalThis.fetch;
  let called=0;
  globalThis.fetch=async()=>{called++;throw new Error('no upstream calls');};
  try {
    const response=await handler(new Request('https://supabase.example/functions/v1/aiw-stylist6-node-staging',{
      method:'POST',body:'{}'
    }));
    assert.equal(response.status,401);
    assert.equal((await response.json()).code,'unauthorized');
    assert.equal(called,0);
  }finally {globalThis.fetch=oldFetch;}
});

test('Stylist 6 staging forwards HMAC-signed JWT and nonce ONLY to pinned READY preview', async()=>{
  const oldFetch=globalThis.fetch;
  const authorization='Bearer synthetic-jwt-no-entitlement';
  const body=JSON.stringify({request:{style:'classic',occasion:'Work',requiredIds:['non-existent-anchor']}});
  let called=0;
  globalThis.fetch=async(url, init)=>{
    called++;
    assert.equal(url,'https://aiwardrobe-stylist-compute-9gwqkmv1c-abduzahid8s-projects.vercel.app/api/stylist6-generate-outfits');
    assert.equal(init.method,'POST');
    assert.equal(init.redirect,'error');
    assert.equal(init.body,body);
    assert.equal(init.headers.authorization,authorization);
    assert.equal(init.headers['x-vercel-protection-bypass'],bypass);
    assert.equal(await verifyStylistTransport({
      secret,body,authorization,
      timestamp:init.headers['x-aiw-timestamp'],
      nonce:init.headers['x-aiw-nonce'],
      signature:init.headers['x-aiw-signature'],
    }),true);
    // No paid work: a synthetic 422 application payload, not a generation.
    return Response.json({
      success:false,code:'anchor_not_found',generationVersion:'stylist-6-cloud-staging-1'
    },{status:422,headers:{'server-timing':'aiw_node_cpu;dur=11.2, aiw_node_wall;dur=23.5'}});
  };
  try {
    const response=await handler(new Request('https://supabase.example/functions/v1/aiw-stylist6-node-staging',{
      method:'POST',headers:{authorization,apikey:'public-anon-fixture'},body
    }));
    assert.equal(called,1);
    assert.equal(response.status,422);
    const json=await response.json();
    assert.equal(json.code,'anchor_not_found');
    assert.equal(json.generationVersion,'stylist-6-cloud-staging-1');
    assert.equal(response.headers.get('server-timing'),'aiw_node_cpu;dur=11.2, aiw_node_wall;dur=23.5');
  }finally{globalThis.fetch=oldFetch;}
});

test('oversize isolated staging body fails before signing or transport',async()=>{
  const oldFetch=globalThis.fetch;
  let called=0;
  globalThis.fetch=async()=>{called++;throw new Error('must not call');};
  try {
    const response=await handler(new Request('https://supabase.example/functions/v1/aiw-stylist6-node-staging',{
      method:'POST',headers:{authorization:'Bearer synthetic','content-length':'2097152'},body:'{}'
    }));
    assert.equal(response.status,413);
    assert.equal(called,0);
  }finally{globalThis.fetch=oldFetch;}
});

test('staging proxy validates JSON payloads, rejects >2MB or truncated streams, classifies 504 timeouts, and validates optional STYLIST6_NODE_ORIGIN', async()=>{
  const oldFetch=globalThis.fetch;
  const req=()=>new Request('https://supabase.example/functions/v1/aiw-stylist6-node-staging',{
    method:'POST',headers:{authorization:'Bearer synthetic'},body:'{}'
  });
  try {
    globalThis.fetch=async()=>new Response('{broken',{status:200,headers:{'content-type':'application/json'}});
    const invalid=await handler(req());
    assert.equal(invalid.status,502);
    assert.equal((await invalid.json()).code,'staging_invalid_json');

    globalThis.fetch=async()=>new Response('{}',{status:200,headers:{'content-type':'application/json','content-length':'3000000'}});
    const large=await handler(req());
    assert.equal(large.status,502);
    assert.equal((await large.json()).code,'staging_response_too_large');

    globalThis.fetch=async()=>{const err=new Error('deadline');err.name='TimeoutError';throw err;};
    const timedOut=await handler(req());
    assert.equal(timedOut.status,504);
    assert.equal((await timedOut.json()).code,'staging_compute_timeout');

    originOverride='http://evil.example.com';
    const badHost=await handler(req());
    assert.equal(badHost.status,503);
    assert.equal((await badHost.json()).code,'staging_transport_unconfigured');
  }finally{originOverride=undefined;globalThis.fetch=oldFetch;}
});
test.after(()=>{globalThis.Deno=oldDeno;});
