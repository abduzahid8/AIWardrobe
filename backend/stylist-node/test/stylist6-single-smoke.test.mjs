import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import vm from 'node:vm';
import { canonicalRequestFingerprint } from '../../stylist6/cloud_handler.mjs';
const source = (await readFile(new URL('../scripts/stylist6-staging-smoke.mjs', import.meta.url), 'utf8'))
  .replace(/^import .*;\n/gm, '');
function fixture({ timeout = false, preflight = 422, existing = false } = {}) {
  const ledgers = new Map();
  let used = 3, paid = 0, request;
  const userId = 'dedicated-user';
  const client = {
    auth: { getUser: async () => ({ data: { user: { id: userId } }, error: null }) },
    rpc: async () => ({ data: 'premium', error: null }),
    from(table) {
      return { select() { return this; }, eq() { return this; },
        async maybeSingle() { return { data: table === 'usage_counters' ? { used } : existing ? { request_id: 'existing' } : null, error: null }; },
        async single() { return { data: { request_fingerprint: canonicalRequestFingerprint(request), status_code: 200 }, error: null }; },
      };
    },
  };
  const env = { SUPABASE_URL: 'https://fyqpifmrsftsfqibhwhy.supabase.co', SUPABASE_ANON_KEY: 'public', AIW_TEST_SERVICE_ROLE_KEY: 'test',
    AIW_TEST_ACCESS_TOKEN: 'test-token', AIW_TEST_USER_ID: userId, AIW_S6_REQUEST_ID: 's6-smoke-fixed-request', AIW_S6_STAGING_SMOKE_ACK: 'I_ACCEPT_S6_QUOTA_DEBITS' };
  const fetch = async (_url, options) => {
    const body = JSON.parse(options.body);
    let status, payload;
    if (!options.headers.authorization) status = 401;
    else if (body.pad) status = 413;
    else if (body.request.requiredIds?.length) { status = preflight; payload = { code: 'anchor_not_found', generationVersion: 'stylist-6-cloud-staging-1' }; }
    else {
      paid++; used++; request = body.request;
      if (timeout) throw new Error('ambiguous timeout');
      status = 200; payload = { success: true, generationVersion: 'stylist-6-cloud-staging-1', source: 'stylist-6-cloud-staging', serverInventory: true, visualStatus: 'NVIDIA_NOT_RUN',
        outfits: [{ items: ['top', 'bottom', 'shoes'].map(id => ({ id, isShopItem: true, imageUrl: 'https://example.com/photo', shopUrl: 'https://example.com/product', currency: 'UZS', price: 100 })) }] };
    }
    return { status, ok: status === 200, json: async () => payload || {} };
  };
  async function run(argv = ['--single-request']) {
    const context = vm.createContext({ assert, createHash, canonicalRequestFingerprint, createClient: () => client, performance, URL, AbortSignal, fetch,
      process: { env, argv: ['node', 'smoke', ...argv] }, console: { log() {} },
      open: async path => { if (ledgers.has(path)) throw new Error('EEXIST'); ledgers.set(path, 'created'); return { writeFile: async data => ledgers.set(path, data), close: async () => {} }; },
      writeFile: async (path, data) => ledgers.set(path, data),
    });
    return new vm.Script('(async () => {' + source + '\n})()').runInContext(context);
  }
  return { run, paid: () => paid, used: () => used, ledgers };
}
test('single smoke refuses the former three-call/default invocation', async () => {
  const f = fixture(); await assert.rejects(f.run([]), /single-request/); assert.equal(f.paid(), 0);
});
test('successful single mode dispatches exactly one paid generation', async () => {
  const f = fixture(); await f.run(); assert.equal(f.paid(), 1); assert.equal(f.used(), 4);
  await assert.rejects(f.run(), /EEXIST/); assert.equal(f.paid(), 1);
});
test('ambiguous paid timeout retains the fuse and cannot dispatch on rerun', async () => {
  const f = fixture({ timeout: true }); await assert.rejects(f.run(), /no automatic retry/);
  await assert.rejects(f.run(), /EEXIST/); assert.equal(f.paid(), 1); assert.equal(f.ledgers.size, 1);
});
test('failed protected admission and pre-existing durable claims never dispatch', async () => {
  for (const options of [{ preflight: 503 }, { existing: true }]) {
    const f = fixture(options); await assert.rejects(f.run()); assert.equal(f.paid(), 0);
  }
});
