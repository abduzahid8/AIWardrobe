/** @jest-environment node */
const fs = require('fs');
const path = require('path');
const ts = require('typescript');
const snapshots = require('../../backend/supabase/functions/_shared/revenuecatSnapshot');
const source = fs.readFileSync(path.resolve(__dirname, '../../backend/supabase/functions/revenuecat-webhook/index.ts'), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
const owner = '11111111-1111-4111-8111-111111111111';
const other = '22222222-2222-4222-8222-222222222222';
const event = { id: 'event-1', type: 'INITIAL_PURCHASE', app_user_id: owner, environment: 'PRODUCTION' };
const customer = { request_date_ms: Date.parse('2026-10-15T00:00:00Z'), subscriber: { entitlements: {}, subscriptions: {} } };
function load({ profiles = [{ id: owner }], apiStatus = 200, rpcError = null, secret = 'webhook-secret' } = {}) {
  let handler;
  const queries = [];
  const rpc = jest.fn(async () => ({ data: rpcError ? null : { duplicate: false }, error: rpcError }));
  const admin = { rpc, from: table => { queries.push(table); return { select: () => ({ in: async () => ({ data: profiles, error: null }) }) }; } };
  const fetchMock = jest.fn(async () => new Response(JSON.stringify(customer), { status: apiStatus }));
  const env = { REVENUECAT_WEBHOOK_SECRET: secret, REVENUECAT_SECRET_API_KEY: 'server-api-key', REVENUECAT_ENVIRONMENT: 'PRODUCTION', SUPABASE_URL: 'https://example.test', SUPABASE_SERVICE_ROLE_KEY: 'server-key' };
  new Function('require', 'exports', 'Deno', 'fetch', compiled)(name => {
    if (name.includes('revenuecatSnapshot')) return snapshots;
    if (name.includes('server.ts')) return { serve: fn => { handler = fn; } };
    return { createClient: () => admin };
  }, {}, { env: { get: name => env[name] } }, fetchMock);
  return { handler, rpc, fetchMock, queries };
}
const request = (value = event, authorization = 'Bearer webhook-secret') => new Request('https://example.test', { method: 'POST', headers: { authorization }, body: JSON.stringify({ event: value }) });

test('requires a dedicated configured webhook secret and rejects forged requests before API work', async () => {
  const unavailable = load({ secret: '' });
  expect((await unavailable.handler(request())).status).toBe(503);
  const denied = load();
  expect((await denied.handler(request(event, 'Bearer server-api-key'))).status).toBe(401);
  expect(denied.fetchMock).not.toHaveBeenCalled();
});
test('fetches current customer info and performs exactly one atomic sync RPC', async () => {
  const { handler, rpc, fetchMock, queries } = load();
  expect((await handler(request())).status).toBe(200);
  expect(queries).toEqual(['profiles']);
  expect(fetchMock.mock.calls[0][0]).toBe(`https://api.revenuecat.com/v1/subscribers/${owner}`);
  expect(rpc).toHaveBeenCalledTimes(1);
  expect(rpc.mock.calls[0][1]).toMatchObject({ p_event: { id: 'event-1', resolved_user_id: owner }, p_accounts: [{ user_id: owner, entitlements: [] }] });
});
test('provider failure cannot acknowledge or mark an event processed', async () => {
  const { handler, rpc } = load({ apiStatus: 503 });
  expect((await handler(request())).status).toBe(503);
  expect(rpc).not.toHaveBeenCalled();
});
test('database failure is retryable rather than a successful acknowledgement', async () => {
  const { handler } = load({ rpcError: { message: 'database unavailable' } });
  expect((await handler(request())).status).toBe(503);
});
test('anonymous identity never falls back to a recent purchaser of the same product', async () => {
  const { handler, queries, rpc, fetchMock } = load({ profiles: [] });
  expect((await handler(request({ ...event, app_user_id: '$RCAnonymousUserId:123', product_id: 'com.aiwardrobe.premium.monthly' }))).status).toBe(503);
  expect(queries).toEqual([]);
  expect(rpc).not.toHaveBeenCalled();
  expect(fetchMock).not.toHaveBeenCalled();
});
test('ambiguous aliases do not assign a receipt to an arbitrary account', async () => {
  const { handler, fetchMock } = load({ profiles: [{ id: owner }, { id: other }] });
  expect((await handler(request({ ...event, app_user_id: '$RCAnonymousUserId:123', aliases: [owner, other] }))).status).toBe(503);
  expect(fetchMock).not.toHaveBeenCalled();
});
test('reconciles both sides of a transfer in the same atomic batch', async () => {
  const { handler, rpc, fetchMock } = load({ profiles: [{ id: owner }, { id: other }] });
  const transfer = { id: 'transfer-1', type: 'TRANSFER', transferred_from: [owner], transferred_to: [other] };
  expect((await handler(request(transfer))).status).toBe(200);
  expect(fetchMock).toHaveBeenCalledTimes(2);
  expect(rpc.mock.calls[0][1].p_accounts.map(account => account.user_id)).toEqual([owner, other]);
});
test('sandbox webhook cannot modify production entitlements', async () => {
  const { handler, rpc, fetchMock } = load();
  const response = await handler(request({ ...event, environment: 'SANDBOX' }));
  expect(response.status).toBe(200);
  expect((await response.json()).ignored).toBe('different_environment');
  expect(rpc).not.toHaveBeenCalled();
  expect(fetchMock).not.toHaveBeenCalled();
});
