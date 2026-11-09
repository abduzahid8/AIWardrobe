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
test.after(()=>{globalThis.Deno=prevDeno});
