/** @jest-environment node */
const fs = require('fs');
const path = require('path');
const ts = require('typescript');

// Execute the actual Deno module with only its remote SDK import substituted.
// No provider, Auth, or database requests leave the test process.
const source = fs.readFileSync(path.resolve(__dirname, '../../backend/supabase/functions/_shared/rateLimit.ts'), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
const mockRpc = jest.fn();
const mockGetUser = jest.fn();
const env = new Map();
const moduleExports = {};
new Function('require', 'exports', 'Deno', compiled)(
  () => ({ createClient: () => ({ auth: { getUser: mockGetUser }, rpc: mockRpc }) }),
  moduleExports,
  { env: { get: key => env.get(key) } },
);
const { enforceRateLimit } = moduleExports;
const opts = { feature: 'test', limit: 2, headers: { 'Access-Control-Allow-Origin': '*' } };
const request = token => new Request('https://example.test', { headers: token ? { authorization: token } : {} });

beforeEach(() => {
  jest.restoreAllMocks();
  mockRpc.mockReset().mockImplementation(async name => ({ data: name === 'consume_edge_quota' ? true : 1, error: null }));
  mockGetUser.mockReset().mockResolvedValue({ data: { user: { id: 'user-1' } }, error: null });
  env.clear();
  env.set('SUPABASE_URL', 'https://example.test');
  env.set('SUPABASE_SERVICE_ROLE_KEY', 'test-key');
});

test.each([undefined, 'invalid', 'Bearer ', 'Bearer a b'])('rejects malformed authorization %s before DB work', async token => {
  const result = await enforceRateLimit(request(token), opts);
  expect(result.status).toBe(401);
  expect(result.headers.get('Access-Control-Allow-Origin')).toBe('*');
  expect(mockRpc).not.toHaveBeenCalled();
});

test('allows authenticated callers within their counter', async () => {
  expect(await enforceRateLimit(request('Bearer jwt'), opts)).toBeNull();
  expect(mockGetUser).toHaveBeenCalledWith('jwt');
  expect(mockRpc).toHaveBeenCalledWith('increment_rate_bucket', { p_user: 'user-1', p_feature: expect.stringMatching(/^rl:test:\d+$/) });
});

test('returns the actual time to the next window on denial', async () => {
  jest.spyOn(Date, 'now').mockReturnValue(119_100);
  mockRpc.mockResolvedValue({ data: 3, error: null });
  const result = await enforceRateLimit(request('Bearer jwt'), opts);
  expect(result.status).toBe(429);
  expect(result.headers.get('retry-after')).toBe('1');
});

test.each([null, '1', 0, NaN, 1.5])('denies invalid counter %s', async data => {
  mockRpc.mockResolvedValue({ data, error: null });
  expect((await enforceRateLimit(request('Bearer jwt'), opts)).status).toBe(503);
});

test('denies DB outages, thrown network failures, and missing configuration', async () => {
  mockRpc.mockResolvedValue({ data: null, error: { message: 'unavailable' } });
  expect((await enforceRateLimit(request('Bearer jwt'), opts)).status).toBe(503);
  mockRpc.mockRejectedValue(new Error('network'));
  expect((await enforceRateLimit(request('Bearer jwt'), opts)).status).toBe(503);
  env.clear();
  expect((await enforceRateLimit(request('Bearer jwt'), opts)).status).toBe(503);
});

test('rejects invalid users without incrementing and distinguishes Auth outages', async () => {
  mockGetUser.mockResolvedValue({ data: { user: null }, error: null });
  expect((await enforceRateLimit(request('Bearer jwt'), opts)).status).toBe(401);
  mockGetUser.mockResolvedValue({ data: null, error: { status: 503 } });
  expect((await enforceRateLimit(request('Bearer jwt'), opts)).status).toBe(503);
  expect(mockRpc).not.toHaveBeenCalled();
});

test.each([0, -1, 1.2, NaN])('rejects invalid limit %s', async limit => {
  expect((await enforceRateLimit(request('Bearer jwt'), { ...opts, limit })).status).toBe(503);
  expect(mockRpc).not.toHaveBeenCalled();
});

test('10,000 independent users are keyed to their authenticated identities', async () => {
  jest.spyOn(Date, 'now').mockReturnValue(120_000);
  mockGetUser.mockImplementation(async jwt => ({ data: { user: { id: jwt } }, error: null }));
  const counters = new Map();
  mockRpc.mockImplementation(async (name, args) => {
    if (name === 'consume_edge_quota') return { data: true, error: null };
    const key = `${args.p_user}:${args.p_feature}`;
    const used = (counters.get(key) || 0) + 1;
    counters.set(key, used);
    return { data: used, error: null };
  });
  for (let round = 0; round < 3; round++) {
    const results = await Promise.all(Array.from({ length: 10_000 }, (_, i) => enforceRateLimit(request(`Bearer user-${i}`), opts)));
    expect(results.every(result => round < 2 ? result === null : result?.status === 429)).toBe(true);
  }
  expect(counters.size).toBe(10_000);
}, 30_000);

 test('daily entitlement denial and quota-store failures cannot reach paid work', async () => {
  mockRpc.mockImplementation(async name => ({ data: name === 'consume_edge_quota' ? false : 1, error: null }));
  expect((await enforceRateLimit(request('Bearer jwt'), opts)).status).toBe(403);
  mockRpc.mockImplementation(async name => name === 'consume_edge_quota' ? { data: null, error: { message: 'offline' } } : { data: 1, error: null });
  expect((await enforceRateLimit(request('Bearer jwt'), opts)).status).toBe(503);
});

test('rate-only admission does not consume a paid quota', async () => {
  expect(await enforceRateLimit(request('Bearer jwt'), { ...opts, meterPaidWork: false })).toBeNull();
  expect(mockRpc).toHaveBeenCalledTimes(1);
  expect(mockRpc.mock.calls[0][0]).toBe('increment_rate_bucket');
});
