import test from 'node:test';
import assert from 'node:assert/strict';
import { capacityPlan, runLoad } from './readiness.mjs';
const users = [{ id: 'user-a', accessToken: 'token-a' }, { id: 'user-b', accessToken: 'token-b' }];
const settings = { baseUrl: 'https://test.invalid', apiKey: 'anon', users, rps: 40, durationSec: 0.2 };
const validFetch = async (url, init) => {
  const owner = init.headers.Authorization === 'Bearer token-a' ? 'user-a' : 'user-b';
  return new Response(JSON.stringify(url.pathname.endsWith('/shop_catalog') ? [{ id: 'item' }] : [{ id: url.pathname.endsWith('/profiles') ? owner : 'item', user_id: owner }]), { status: 200 });
};
test('10k DAU assumptions produce 240k daily requests and a 40 RPS target', () => {
  assert.equal(capacityPlan().dailyRequests, 240000);
  assert.equal(capacityPlan().targetRps, 40);
  assert.throws(() => capacityPlan({ dau: -1 }));
});
test('passes valid read paths and records distinct authenticated identities', async () => {
  const result = await runLoad({ ...settings, fetchImpl: validFetch });
  assert.equal(result.passed, true);
  assert.equal(result.completed, 8);
  assert.equal(result.distinctAuthenticatedUsers, 2);
});
test('fails if HTTP 200 contains rows belonging to another user', async () => {
  const result = await runLoad({ ...settings, fetchImpl: async () => new Response(JSON.stringify([{ id: 'other-user', user_id: 'other-user' }])) });
  assert.equal(result.passed, false);
  assert.ok(result.failures.wardrobe_ownership_violation > 0);
});
test('fails HTTP errors and never accepts rate limiting as successful capacity', async () => {
  const result = await runLoad({ ...settings, fetchImpl: async () => new Response('{}', { status: 429 }) });
  assert.equal(result.passed, false);
  assert.equal(result.statuses[429], 8);
  assert.equal(result.errorRate, 1);
});
test('fails empty inventory and malformed success payloads', async () => {
  const result = await runLoad({ ...settings, fetchImpl: async url => new Response(url.pathname.endsWith('/shop_catalog') ? '[]' : '{}') });
  assert.equal(result.passed, false);
  assert.equal(result.failures.catalog_empty, 2);
});

test('multiple semantic failures on one response count as one failed request', async () => {
  const result = await runLoad({ ...settings, fetchImpl: async () => new Response('[{}]') });
  assert.ok(result.errorRate <= 1);
  assert.equal(result.failedRequests, 8);
});

test('transport diagnostics retain safe codes without credential-bearing messages', async () => {
  const result = await runLoad({ ...settings, fetchImpl: async () => {
    throw new TypeError('Request failed with Bearer token-a', { cause: { code: 'ECONNRESET' } });
  } });
  assert.equal(result.failedRequests, 8);
  assert.equal(result.failureDetails[0].errorType, 'TypeError');
  assert.equal(result.failureDetails[0].causeCode, 'ECONNRESET');
  assert.equal(JSON.stringify(result).includes('token-a'), false);
});

test('representative app reads include ownership filters and reject missing fixture rows', async () => {
  const expectedUsers = users.map(user => ({ ...user, expectedWardrobeItems: 1, expectedSavedOutfits: 1 }));
  const result = await runLoad({ ...settings, users: expectedUsers, workload: 'app', fetchImpl: async (url, init) => {
    if (!url.pathname.endsWith('/shop_catalog')) {
      const owner = init.headers.Authorization === 'Bearer token-a' ? 'user-a' : 'user-b';
      assert.equal(url.searchParams.get(url.pathname.endsWith('/profiles') ? 'id' : 'user_id'), `eq.${owner}`);
    }
    return validFetch(url, init);
  } });
  assert.equal(result.passed, true);
  const missing = await runLoad({ ...settings, users: expectedUsers, workload: 'app', fetchImpl: async url => new Response(JSON.stringify(url.pathname.endsWith('/shop_catalog') ? [{id:'catalog'}] : [])) });
  assert.ok(missing.failures.wardrobe_row_count > 0);
  assert.ok(missing.failures.outfits_row_count > 0);
  assert.equal(missing.passed, false);
});

test('invalid JSON diagnostics distinguish response decoding from transport failure', async () => {
  const result = await runLoad({ ...settings, fetchImpl: async () => new Response('invalid JSON') });
  assert.equal(result.failureDetails[0].stage, 'decode_json');
  assert.equal(result.failureDetails[0].errorType, 'SyntaxError');
});

test('an aborted run stops scheduling and cannot pass capacity', async () => {
  const controller = new AbortController(); controller.abort();
  let calls = 0;
  const result = await runLoad({ ...settings, signal: controller.signal, fetchImpl: async () => { calls++; return new Response('[]'); } });
  assert.equal(calls, 0);
  assert.equal(result.aborted, true);
  assert.equal(result.passed, false);
});

test('HTTP 502 retains route, time and validated correlation IDs without response contents',async()=>{
 const result=await runLoad({...settings,fetchImpl:async()=>new Response('secret error body',{status:502,headers:{'sb-request-id':'12345678-1234-1234-1234-123456789abc','cf-ray':'1234567890abcdef-FRA','x-envoy-upstream-service-time':'7'}})});
 const first=result.failureDetails[0];assert.equal(first.status,502);assert.equal(first.route,'catalog');assert.equal(first.requestIndex,0);
 assert.ok(Number.isFinite(Date.parse(first.startedAt)));assert.equal(first.supabaseRequestId,'12345678-1234-1234-1234-123456789abc');
 assert.equal(result.timingMs.headers.samples,8);assert.equal(result.timingMs.body.samples,8);assert.equal(result.passed,false);
 assert.equal(JSON.stringify(result).includes('secret error body'),false);assert.equal(JSON.stringify(result).includes('token-a'),false);
});
