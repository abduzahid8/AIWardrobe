import test from 'node:test';
import assert from 'node:assert/strict';
import { POST, GET, stylist6CreativeServerConfig, stylist6ComputeBudgetMs, STYLIST6_RENDER_CANARY_BUDGET_MS } from '../api/stylist6-generate-outfits.ts';
import { STYLIST6_CLOUD_VERSION } from '../../stylist6/cloud_handler.mjs';

test('route and handler report the identical engine version', async () => {
  const get = GET();
  assert.equal(get.status, 405);
  const payload = await get.json();
  assert.equal(payload.generationVersion, STYLIST6_CLOUD_VERSION);
  assert.match(payload.generationVersion, /^stylist-6-/);
});

test('stylist6 route fails closed without a transport secret', async () => {
  const saved = process.env.STYLIST6_NODE_SHARED_SECRET;
  delete process.env.STYLIST6_NODE_SHARED_SECRET;
  try {
    const response = await POST(new Request('https://node.example/api/stylist6-generate-outfits', {
      method: 'POST', headers: { authorization: 'Bearer x' }, body: '{}',
    }));
    assert.equal(response.status, 503);
    const payload = await response.json();
    assert.equal(payload.code, 'stylist_compute_unconfigured');
    assert.equal(payload.generationVersion, STYLIST6_CLOUD_VERSION);
  } finally {
    if (saved === undefined) delete process.env.STYLIST6_NODE_SHARED_SECRET;
    else process.env.STYLIST6_NODE_SHARED_SECRET = saved;
  }
});

test('stylist6 route rejects unsigned callers and wrong methods', async () => {
  process.env.STYLIST6_NODE_SHARED_SECRET = 'route-test-shared-secret-0123456789abcdef';
  const unsigned = await POST(new Request('https://node.example/api/stylist6-generate-outfits', {
    method: 'POST', headers: { authorization: 'Bearer test-jwt-value' }, body: '{}',
  }));
  assert.equal(unsigned.status, 401);
  const get = GET();
  assert.equal(get.status, 405);
  const oversize = await POST(new Request('https://node.example/api/stylist6-generate-outfits', {
    method: 'POST',
    headers: { authorization: 'Bearer test-jwt-value', 'content-length': String(1024 * 1024 + 1) },
    body: '{}',
  }));
  assert.equal(oversize.status, 413);
});


test('both server-owned AI director switches reach the cloud handler without client control', () => {
  const names=['STYLIST6_CREATIVE_DIRECTOR_ENABLED','STYLIST6_AI_SLATE_ENABLED'];
  const originals=Object.fromEntries(names.map(name=>[name,process.env[name]]));
  try {
    process.env.STYLIST6_CREATIVE_DIRECTOR_ENABLED='1';
    process.env.STYLIST6_AI_SLATE_ENABLED='1';
    assert.deepEqual(stylist6CreativeServerConfig(), {
      STYLIST6_CREATIVE_DIRECTOR_ENABLED:'1',STYLIST6_AI_SLATE_ENABLED:'1',
    },'enabled AI fashion slate cannot silently disappear from the route');
    process.env.STYLIST6_CREATIVE_DIRECTOR_ENABLED='';
    process.env.STYLIST6_AI_SLATE_ENABLED='';
    assert.deepEqual(stylist6CreativeServerConfig(), {
      STYLIST6_CREATIVE_DIRECTOR_ENABLED:'',STYLIST6_AI_SLATE_ENABLED:'',
    },'server kill-switch must remain effective when environment is disabled');
    assert.equal(Object.keys(stylist6CreativeServerConfig()).length,2,
      'never forward arbitrary secret-containing process.env to the model');
  } finally {
    for(const name of names){
      if(originals[name]===undefined)delete process.env[name];
      else process.env[name]=originals[name];
    }
  }
});


test('Render-only Stylist 6 compute budget cannot be forged by clients or Vercel',()=>{
  assert.equal(STYLIST6_RENDER_CANARY_BUDGET_MS,72000);
  assert.equal(stylist6ComputeBudgetMs(),undefined);
  assert.equal(stylist6ComputeBudgetMs({}),undefined);
  assert.equal(stylist6ComputeBudgetMs({renderCanary:false}),undefined);
  assert.equal(stylist6ComputeBudgetMs({renderCanary:true}),72000);
  assert.ok(STYLIST6_RENDER_CANARY_BUDGET_MS < 85000,
    'signed Supabase Render proxy must still have time to read complete JSON');
});
