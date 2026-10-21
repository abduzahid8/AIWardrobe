/** @jest-environment node */
const fs = require('fs');
const path = require('path');
const ts = require('typescript');
const snapshots = require('../../backend/supabase/functions/_shared/revenuecatSnapshot');
const source = fs.readFileSync(path.resolve(__dirname, '../../backend/supabase/functions/sync-subscription/index.ts'), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
const owner = '11111111-1111-4111-8111-111111111111';
function load({ denied = null, authenticated = true, apiStatus = 200, rpcError = null } = {}) {
  let handler;
  const mockRate = jest.fn(async () => denied);
  const mockRpc = jest.fn(async () => ({ data: rpcError ? null : { duplicate: false }, error: rpcError }));
  const mockGetUser = jest.fn(async () => ({ data: { user: authenticated ? { id: owner } : null }, error: null }));
  const ownership = [];
  const admin = { auth: { getUser: mockGetUser }, rpc: mockRpc,
    from: () => ({ select: () => ({ eq: (key, value) => { ownership.push({ key, value }); return { single: async () => ({ data: { subscription_tier: 'free', subscription_expires_at: null }, error: null }) }; } }) }),
  };
  const mockFetch = jest.fn(async () => new Response(JSON.stringify({ request_date_ms: Date.parse('2026-10-15T00:00:00Z'), subscriber: { entitlements: {} } }), { status: apiStatus }));
  new Function('require', 'exports', 'Deno', 'fetch', compiled)(name => {
    if (name.includes('rateLimit')) return { enforceRateLimit: mockRate };
    if (name.includes('revenuecatSnapshot')) return snapshots;
    if (name.includes('server.ts')) return { serve: fn => { handler = fn; } };
    return { createClient: () => admin };
  }, {}, { env: { get: name => name === 'REVENUECAT_ENVIRONMENT' ? 'PRODUCTION' : 'configuration' } }, mockFetch);
  return { handler, mockRate, mockRpc, mockGetUser, mockFetch, ownership };
}
const request = () => new Request('https://example.test', { method: 'POST', headers: { Authorization: 'Bearer caller-jwt' }, body: JSON.stringify({ userId: 'victim', tier: 'vip' }) });
test('receipt sync authenticates and throttles free users without requiring an existing paid entitlement', async () => {
  const { handler, mockRate, mockRpc, mockGetUser, mockFetch, ownership } = load();
  const response = await handler(request());
  expect(response.status).toBe(200);
  expect(mockRate.mock.calls[0][1]).toMatchObject({ feature: 'sync-subscription', meterPaidWork: false });
  expect(mockGetUser).toHaveBeenCalledWith('caller-jwt');
  expect(mockFetch.mock.calls[0][0]).toBe(`https://api.revenuecat.com/v1/subscribers/${owner}`);
  expect(mockRpc.mock.calls[0][1].p_accounts[0].user_id).toBe(owner);
  expect(ownership).toEqual([{ key: 'id', value: owner }]);
  expect(await response.json()).toMatchObject({ verified: true, tier: 'free' });
});
test('rate denial stops receipt verification work', async () => {
  const denied = new Response('{}', { status: 429 });
  const { handler, mockFetch, mockRpc } = load({ denied });
  expect(await handler(request())).toBe(denied);
  expect(mockFetch).not.toHaveBeenCalled();
  expect(mockRpc).not.toHaveBeenCalled();
});
test('a missing identity never queries RevenueCat or reconciles caller-supplied IDs', async () => {
  const { handler, mockFetch, mockRpc } = load({ authenticated: false });
  expect((await handler(request())).status).toBe(401);
  expect(mockFetch).not.toHaveBeenCalled();
  expect(mockRpc).not.toHaveBeenCalled();
});
test.each([{ apiStatus: 503 }, { rpcError: { message: 'unavailable' } }])('verification failure is explicit: %j', async config => {
  const { handler } = load(config);
  expect((await handler(request())).status).toBe(503);
});
