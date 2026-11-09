import test from 'node:test';
import assert from 'node:assert/strict';
import { verifyStylistTransport } from '../transport/signature.mjs';
const secret='this-is-a-strong-test-only-shared-secret-123456';
const bypass='ABCD1234abcd5678EFGH9012efgh3456';
let handler;
const prevDeno=globalThis.Deno;
globalThis.Deno={
  serve(fn){handler=fn;},
  env:{get(k){if(k==='STYLIST_NODE_SHARED_SECRET')return secret;if(k==='STYLIST_VERCEL_BYPASS_SECRET')return bypass;return undefined;}}
};
await import('../../stylist-proxy/staging.ts');
test('staging health GET no longer makes any signed upstream call', async()=>{
  const prevFetch=globalThis.fetch;let calls=0;
  globalThis.fetch=async()=>{calls++;throw new Error('GET must never call Node')};
  try{
    const out=await handler(new Request('https://supabase.local/functions/v1/aiw-stylist-node-staging/health'));
    assert.equal(out.status,405);
    assert.equal((await out.json()).code,'method_not_allowed');
    assert.equal(calls,0);
  }finally{globalThis.fetch=prevFetch}
});
test('staging rejects unauthorized generation before hitting Vercel',async()=>{
  const out=await handler(new Request('https://supabase.local/functions/v1/aiw-stylist-node-staging',{
    method:'POST',body:'{}'
  }));
  assert.equal(out.status,401);
  assert.equal((await out.json()).code,'unauthorized');
});

test('staging signs the live request, forwards optional preview bypass, and preserves upstream status/timing', async()=>{
  const previous=globalThis.fetch;
  const bearer='Bearer test-user-token-not-production';
  const body=JSON.stringify({stylePreferences:'minimalist',occasion:'Work',catalogOnly:true});
  let calls=0;
  globalThis.fetch=async(url,init)=>{
    calls++;
    assert.equal(url,'https://aiwardrobe-stylist-compute-5vy45vooo-abduzahid8s-projects.vercel.app/api/generate-outfits');
    assert.equal(init.method,'POST');
    assert.equal(init.redirect,'error');
    assert.equal(init.body,body);
    assert.equal(init.headers['x-vercel-protection-bypass'],bypass);
    assert.equal(init.headers.authorization,bearer);
    assert.equal(init.headers.apikey,'test-anon-public');
    assert.equal(await verifyStylistTransport({
      secret,body,authorization:bearer,
      timestamp:init.headers['x-aiw-timestamp'],
      nonce:init.headers['x-aiw-nonce'],
      signature:init.headers['x-aiw-signature'],
    }),true);
    // The dummy JWT must still be rejected by the original Node handler.
    return Response.json({success:false,outfits:[],error:'unauthorized'},{
      status:401,
      headers:{'server-timing':'aiw_node_cpu;dur=12.3, aiw_node_wall;dur=24.6'},
    });
  };
  try{
    const response=await handler(new Request('https://supabase.local/functions/v1/aiw-stylist-node-staging',{
      method:'POST',headers:{authorization:bearer,apikey:'test-anon-public','content-type':'application/json'},body,
    }));
    assert.equal(calls,1);
    assert.equal(response.status,401);
    assert.equal((await response.json()).error,'unauthorized');
    assert.equal(response.headers.get('server-timing'),'aiw_node_cpu;dur=12.3, aiw_node_wall;dur=24.6');
    assert.equal(response.headers.get('access-control-expose-headers'),'server-timing');
  }finally{globalThis.fetch=previous}
});
test('staging rejects oversized body before contacting the compute endpoint',async()=>{
  const prev=globalThis.fetch;
  let calls=0;
  globalThis.fetch=async()=>{calls++;throw new Error('must not call upstream')};
  try{
    const response=await handler(new Request('https://supabase.local/functions/v1/aiw-stylist-node-staging',{
      method:'POST',headers:{authorization:'Bearer dummy','content-length':'2097152'},body:'{}',
    }));
    assert.equal(response.status,413);
    assert.equal(calls,0);
  }finally{globalThis.fetch=prev}
});
test.after(()=>{globalThis.Deno=prevDeno});
