import test from 'node:test';
import assert from 'node:assert/strict';
import { verifyStylistTransport } from '../transport/signature.mjs';
const secret='this-is-a-strong-test-only-shared-secret-123456';
let handler;
const prevDeno=globalThis.Deno;
globalThis.Deno={
  serve(fn){handler=fn;},
  env:{get(k){if(k==='STYLIST_NODE_SHARED_SECRET')return secret;return undefined;}}
};
await import('../../stylist-proxy/staging.ts');
test('staging signed transport probe is a non-billable HMAC-only call', async()=>{
  const prevFetch=globalThis.fetch;let calls=0;
  globalThis.fetch=async(url,init)=>{
    calls++;
    assert.match(String(url),/\/api\/transport-health$/);
    assert.equal(init.redirect,'error');
    assert.equal(await verifyStylistTransport({
      secret, body:init.body,authorization:init.headers.authorization,
      timestamp:init.headers['x-aiw-timestamp'],
      nonce:init.headers['x-aiw-nonce'],
      signature:init.headers['x-aiw-signature'],
    }),true);
    return Response.json({ok:true,transport:'signed'});
  };
  try{
    const out=await handler(new Request('https://supabase.local/functions/v1/aiw-stylist-node-staging/health'));
    assert.equal(out.status,200);
    assert.equal((await out.json()).signedTransport,true);
    assert.equal(calls,1);
  }finally{globalThis.fetch=prevFetch}
});
test('staging rejects unauthorized generation before hitting Vercel',async()=>{
  const out=await handler(new Request('https://supabase.local/functions/v1/aiw-stylist-node-staging',{
    method:'POST',body:'{}'
  }));
  assert.equal(out.status,401);
  assert.equal((await out.json()).code,'unauthorized');
});
test.after(()=>{globalThis.Deno=prevDeno});
