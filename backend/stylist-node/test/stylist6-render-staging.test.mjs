import test from 'node:test';
import assert from 'node:assert/strict';
import {verifyStylistTransport} from '../transport/signature.mjs';
const secret='render-canary-hmac-test-123456789-123456789';
let handler;
const oldDeno=globalThis.Deno;
globalThis.Deno={serve(fn){handler=fn},env:{get(name){return name==='STYLIST6_RENDER_NODE_SHARED_SECRET'?secret:undefined}}};
await import('../../stylist-proxy/stylist6-render-staging.ts');
test('separate Render canary rejects GET, absent bearer and oversized POST without contacting Render',async()=>{
  const prior=globalThis.fetch;let calls=0;
  globalThis.fetch=async()=>{calls++;throw Error('network must not execute')};
  try{
    assert.equal((await handler(new Request('https://x.example/functions/v1/aiw-stylist6-node-render-staging'))).status,405);
    assert.equal((await handler(new Request('https://x.example/functions/v1/aiw-stylist6-node-render-staging',{method:'POST',body:'{}'}))).status,401);
    assert.equal((await handler(new Request('https://x.example/functions/v1/aiw-stylist6-node-render-staging',{
      method:'POST',headers:{authorization:'Bearer dummy','content-length':'2000000'},body:'{}'}))).status,413);
    assert.equal(calls,0);
  }finally{globalThis.fetch=prior}
});
test('Render canary signs JWT/body and forwards only to isolated Render route',async()=>{
  const prior=globalThis.fetch;
  const body='{"request":{"style":"old_money","occasion":"Casual","requiredIds":["not-found"]}}';
  const authorization='Bearer synthetic-not-real-user';let calls=0;
  globalThis.fetch=async(url,init)=>{
    calls++;
    assert.equal(url,'https://aiwardrobe-stylist6-staging-compute.onrender.com/api/stylist6-generate-outfits');
    assert.equal(init.method,'POST');assert.equal(init.redirect,'error');
    assert.equal(init.body,body);assert.equal(init.headers.authorization,authorization);
    assert.equal(init.headers['x-vercel-protection-bypass'],undefined);
    assert.equal(await verifyStylistTransport({secret,body,authorization,
      timestamp:init.headers['x-aiw-timestamp'],nonce:init.headers['x-aiw-nonce'],
      signature:init.headers['x-aiw-signature']}),true);
    return Response.json({success:false,code:'anchor_not_found'},{status:422,
      headers:{'server-timing':'aiw_node_cpu;dur=11.2, aiw_node_wall;dur=23.5'}});
  };
  try{
    const r=await handler(new Request('https://x.example/functions/v1/aiw-stylist6-node-render-staging',{
      method:'POST',headers:{authorization,apikey:'anon-fixture'},body}));
    assert.equal(r.status,422);assert.equal((await r.json()).code,'anchor_not_found');
    assert.equal(r.headers.get('server-timing'),'aiw_node_cpu;dur=11.2, aiw_node_wall;dur=23.5');
    assert.equal(calls,1);
  }finally{globalThis.fetch=prior}
});
test.after(()=>{globalThis.Deno=oldDeno});
