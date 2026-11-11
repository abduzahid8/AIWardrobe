import test from 'node:test';
import assert from 'node:assert/strict';
import { signStylistTransport } from '../stylist-node/transport/signature.mjs';
import {
  handleStylist6CloudGenerate,
  mapCloudVisualStatus,
  projectWardrobeItem,
  fitVisualBytesCloud,
  STYLIST6_CLOUD_VERSION,
  canonicalRequestFingerprint,
} from './cloud_handler.mjs';

const SECRET = 'cloud-test-shared-secret-0123456789abcdef';
const JWT = 'good-jwt-token';
const USER = 'user-1';

const shopRow = (id, slot, price = 100, currency = 'UZS') => ({
  id, name: `${slot} garment ${id}`, category: { top: 'tops', bottom: 'bottoms', shoes: 'shoes', outerwear: 'tops' }[slot],
  garment_type: slot === 'bottom' ? 'lower_body' : slot === 'shoes' ? 'shoes' : 'upper_body',
  price, currency,
  image_url: `https://example.com/images/${id}.jpg`,
  source_url: `https://example.com/shop/${id}`,
  is_active: true, brand: 'Test',
});
const SHOP = [
  shopRow('top-1', 'top', 100), shopRow('top-2', 'top', 150),
  shopRow('bottom-1', 'bottom', 200), shopRow('bottom-2', 'bottom', 120),
  shopRow('shoes-1', 'shoes', 300), shopRow('shoes-2', 'shoes', 180),
  shopRow('outer-1', 'outerwear', 500),
];
const WARDROBE = [
  { id: 'w-top-1', type: 'shirt', category: 'tops', color: 'white', brand: '', name: 'white cotton shirt', image_url: 'https://example.com/w/t1.jpg' },
  { id: 'w-bottom-1', type: 'jeans', category: 'bottoms', color: 'blue', brand: '', name: 'blue denim jeans', image_url: 'https://example.com/w/b1.jpg' },
  { id: 'w-shoes-1', type: 'sneakers', category: 'shoes', color: 'white', brand: '', name: 'white leather sneakers', image_url: 'https://example.com/w/s1.jpg' },
  { id: 'w-top-2', type: 't-shirt', category: 'tops', color: 'grey', brand: '', name: 'grey cotton t-shirt', image_url: 'https://example.com/w/t2.jpg' },
  { id: 'w-bottom-2', type: 'trousers', category: 'bottoms', color: 'black', brand: '', name: 'black tailored trousers', image_url: 'https://example.com/w/b2.jpg' },
  { id: 'w-shoes-2', type: 'loafers', category: 'shoes', color: 'brown', brand: '', name: 'brown leather loafers', image_url: 'https://example.com/w/s2.jpg' },
];

function makeIo(overrides = {}) {
  const calls = {
    verifyUser: 0, checkRate: 0, debitQuota: 0, claimRequest: 0, finishRequest: 0,
    fetchShopRows: 0, fetchWardrobeRows: 0, fetchVisualProfiles: 0,
  };
  const claims = new Map();
  return {
    calls,
    claims,
    io: {
      async verifyUser(jwt) {
        calls.verifyUser += 1;
        if (jwt !== JWT) throw new Error('stylist6_unauthorized');
        return USER;
      },
      async checkRate(userId) {
        assert.equal(userId, USER);
        if (overrides.rateDeny) {
          const denied = new Error('denied');
          denied.code = overrides.rateDeny;
          throw denied;
        }
      },
      async debitQuota(userId) {
        calls.debitQuota += 1;
        assert.equal(userId, USER);
        if (overrides.quotaDeny) {
          const denied = new Error('denied');
          denied.code = overrides.quotaDeny;
          throw denied;
        }
      },
      async claimRequest(userId, requestId, fingerprint) {
        calls.claimRequest += 1;
        assert.equal(userId, USER);
        if (claims.has(requestId)) {
          const prior = claims.get(requestId);
          return { duplicate: true, conflict: prior.fingerprint !== fingerprint, prior, durable: true };
        }
        claims.set(requestId, { status: null, code: null, fingerprint });
        return { duplicate: false, prior: null, durable: true };
      },
      async finishRequest(userId, requestId, status, code) {
        calls.finishRequest += 1;
        claims.set(requestId, { ...claims.get(requestId), status, code });
      },
      async fetchShopRows() {
        calls.fetchShopRows += 1;
        return overrides.shopRows ?? SHOP;
      },
      async fetchWardrobeRows(userId) {
        calls.fetchWardrobeRows += 1;
        assert.equal(userId, USER);
        return overrides.wardrobeRows ?? WARDROBE;
      },
      async fetchVisualProfiles() {
        calls.fetchVisualProfiles += 1;
        return [];
      },
    },
  };
}

const env = { STYLIST6_NODE_SHARED_SECRET: SECRET, SUPABASE_URL: 'https://example.supabase.co' };

async function signedCall(request, { secret = SECRET, jwt = JWT, method = 'POST' } = {}, ioOverride) {
  const { calls, claims, io } = makeIo(ioOverride);
  const body = JSON.stringify(request);
  const authorization = `Bearer ${jwt}`;
  const signature = await signStylistTransport({ secret, body, authorization });
  const headers = { authorization, ...signature };
  const result = await handleStylist6CloudGenerate({ method, headers, bodyText: body, env, io });
  return { ...result, calls, claims };
}

const baseRequest = {
  style: 'old_money', occasion: 'Work', brief: 'work, old money',
  sourceMode: 'catalog', preferences: {}, recentSignatures: [], excludeItemIds: [],
};

test('missing transport secret fails closed without touching auth or quota', async () => {
  const { calls, io } = makeIo();
  const result = await handleStylist6CloudGenerate({
    method: 'POST', headers: {}, bodyText: '{}',
    env: { STYLIST6_NODE_SHARED_SECRET: '' }, io,
  });
  assert.equal(result.status, 503);
  assert.equal(result.body.code, 'stylist_compute_unconfigured');
  assert.equal(calls.verifyUser, 0);
  assert.equal(calls.debitQuota, 0);
});

test('method, size and HMAC admission cost no quota', async () => {
  const probe = await makeIo();
  const badSecret = await signedCall({ request: baseRequest }, { secret: 'wrong-secret-000000000000000000000001' });
  assert.equal(badSecret.status, 401);
  assert.equal(badSecret.calls.debitQuota, 0);
  const get = await handleStylist6CloudGenerate({ method: 'GET', headers: {}, bodyText: '', env, io: probe.io });
  assert.equal(get.status, 405);
  const big = await handleStylist6CloudGenerate({
    method: 'POST', headers: {}, bodyText: 'x'.repeat(1024 * 1024 + 1), env, io: probe.io,
  });
  assert.equal(big.status, 413);
  assert.equal(probe.calls.debitQuota, 0);
});

test('invalid JWT fails before inventory and quota', async () => {
  const result = await signedCall({ request: baseRequest }, { jwt: 'bad-token' });
  assert.equal(result.status, 401);
  assert.equal(result.calls.verifyUser, 1);
  assert.equal(result.calls.debitQuota, 0);
  assert.equal(result.calls.fetchShopRows, 0);
});

test('rate denial is cheap and early; quota denial comes after admission', async () => {
  const limited = await signedCall({ request: baseRequest }, {}, { rateDeny: 'rate_limited' });
  assert.equal(limited.status, 429);
  assert.equal(limited.calls.fetchShopRows, 0);
  assert.equal(limited.calls.debitQuota, 0);
  const denied = await signedCall({ request: baseRequest }, {}, { quotaDeny: 'feature_limit_reached' });
  assert.equal(denied.status, 403);
  assert.equal(denied.calls.debitQuota, 1);
});

test('invalid anchors fail 422 with zero quota debit', async () => {
  const result = await signedCall({ request: { ...baseRequest, requiredIds: ['no-such-sku'] } });
  assert.equal(result.status, 422);
  assert.equal(result.body.code, 'anchor_not_found');
  assert.equal(result.calls.debitQuota, 0);
  assert.equal(result.calls.claimRequest, 0);
});

test('redelivered logical requests replay the recorded outcome without another debit', async () => {
  const harness = makeIo();
  const once = async () => {
    const body = JSON.stringify({ request: { ...baseRequest, generationRequestId: 'idem-1' } });
    const authorization = `Bearer ${JWT}`;
    const signature = await signStylistTransport({ secret: SECRET, body, authorization });
    return handleStylist6CloudGenerate({
      method: 'POST', headers: { authorization, ...signature }, bodyText: body, env, io: harness.io,
    });
  };
  const first = await once();
  assert.equal(first.status, 200);
  assert.equal(first.body.replayProtection, 'durable');
  assert.equal(harness.calls.debitQuota, 1);
  const second = await once();
  assert.equal(second.status, 409);
  assert.equal(second.body.code, 'duplicate_request');
  assert.equal(second.body.priorStatus, 200);
  assert.equal(second.body.replayProtection, 'durable');
  assert.equal(harness.calls.debitQuota, 1);
});

test('durable claim RPC wins over the local table and replays outcomes', async () => {
  const { createCloudIo } = await import('./cloud_handler.mjs');
  const store = new Map();
  const io = createCloudIo({
    url: 'https://example.supabase.co', anonKey: 'anon', serviceKey: 'service',
    supabaseJs: {
      createClient: () => ({
        auth: { getUser: async () => ({ data: { user: { id: USER } }, error: null }) },
        rpc: async (name, args) => {
          if (name === 'claim_stylist6_request') {
            const key = `${args.p_user}:${args.p_request_id}`;
            if (store.has(key)) {
              const prior = store.get(key);
              return { data: [{ duplicate: true, status_code: prior.status, outcome_code: prior.code }], error: null };
            }
            store.set(key, { status: 0, code: '' });
            return { data: [{ duplicate: false, status_code: 0, outcome_code: '' }], error: null };
          }
          if (name === 'finish_stylist6_request') {
            store.set(`${args.p_user}:${args.p_request_id}`, { status: args.p_status, code: args.p_code });
            return { data: null, error: null };
          }
          throw new Error('unexpected rpc ' + name);
        },
        from: () => { throw new Error('unused'); },
      }),
    },
  });
  const first = await io.claimRequest(USER, 'durable-1');
  assert.equal(first.duplicate, false);
  assert.equal(first.durable, true);
  await io.finishRequest(USER, 'durable-1', 200, 'complete');
  const second = await io.claimRequest(USER, 'durable-1');
  assert.equal(second.duplicate, true);
  assert.deepEqual(second.prior, { status: 200, code: 'complete' });
  assert.equal(second.durable, true);
});

test('missing claim storage fails closed before any debit', async () => {
  const { createCloudIo } = await import('./cloud_handler.mjs');
  const calls = { debit: 0 };
  const io = createCloudIo({
    url: 'https://example.supabase.co', anonKey: 'anon', serviceKey: 'service',
    supabaseJs: {
      createClient: () => ({
        auth: { getUser: async () => ({ data: { user: { id: USER } }, error: null }) },
        rpc: async (name) => {
          if (name === 'claim_stylist6_request') {
            return { data: null, error: { code: 'PGRST202', message: 'claim_stylist6_request does not exist' } };
          }
          if (name === 'increment_rate_bucket') return { data: 1, error: null };
          if (name === 'consume_edge_quota') { calls.debit += 1; return { data: true, error: null }; }
          throw new Error('unexpected rpc ' + name);
        },
        from: () => { throw new Error('unused'); },
      }),
    },
  });
  // Real claim path, stubbed inventory: the claim gate is what this test proves.
  io.fetchShopRows = async () => SHOP;
  io.fetchWardrobeRows = async () => [];
  io.fetchVisualProfiles = async () => [];
  await assert.rejects(io.claimRequest(USER, 'closed-1'), /stylist6_claim_unavailable/);
  const body = JSON.stringify({ request: { ...baseRequest, generationRequestId: 'closed-1' } });
  const authorization = `Bearer ${JWT}`;
  const signature = await signStylistTransport({ secret: SECRET, body, authorization });
  const result = await handleStylist6CloudGenerate({
    method: 'POST', headers: { authorization, ...signature }, bodyText: body, env, io,
  });
  assert.equal(result.status, 503);
  assert.equal(result.body.code, 'stylist6_claim_unavailable');
  assert.equal(calls.debit, 0);
});

test('concurrent duplicates admit exactly one debit', async () => {
  const harness = makeIo();
  const once = () => {
    const body = JSON.stringify({ request: { ...baseRequest, generationRequestId: 'race-1' } });
    const authorization = `Bearer ${JWT}`;
    return signStylistTransport({ secret: SECRET, body, authorization }).then((signature) =>
      handleStylist6CloudGenerate({
        method: 'POST', headers: { authorization, ...signature }, bodyText: body, env, io: harness.io,
      }));
  };
  const [first, second] = await Promise.all([once(), once()]);
  const statuses = [first.status, second.status].sort();
  assert.deepEqual(statuses, [200, 409]);
  assert.equal(harness.calls.debitQuota, 1);
});

test('a failed planning outcome replays without another debit', async () => {
  const harness = makeIo();
  const once = () => {
    const body = JSON.stringify({
      request: { ...baseRequest, generationRequestId: 'fail-1', budget: { amount: 10, currency: 'UZS' } },
    });
    const authorization = `Bearer ${JWT}`;
    return signStylistTransport({ secret: SECRET, body, authorization }).then((signature) =>
      handleStylist6CloudGenerate({
        method: 'POST', headers: { authorization, ...signature }, bodyText: body, env, io: harness.io,
      }));
  };
  const first = await once();
  assert.equal(first.status, 502);
  const second = await once();
  assert.equal(second.status, 409);
  assert.equal(second.body.priorStatus, 502);
  assert.equal(harness.calls.debitQuota, 1);
});

test('catalog deck returns truthful stylist-6 version with real SKUs', async () => {
  const result = await signedCall({ request: baseRequest });
  assert.equal(result.status, 200);
  assert.equal(result.body.success, true);
  assert.equal(result.body.generationVersion, STYLIST6_CLOUD_VERSION);
  assert.match(result.body.generationVersion, /^stylist-6-/);
  assert.equal(result.body.sourceMode, 'catalog');
  assert.equal(result.body.serverInventory, true);
  assert.equal(result.body.clientInventoryIgnored, false);
  assert.ok(result.body.outfits.length >= 1);
  for (const outfit of result.body.outfits) {
    assert.ok(outfit.items.length >= 3);
    assert.equal(outfit.engineVersion, STYLIST6_CLOUD_VERSION);
    for (const item of outfit.items) {
      assert.ok(item.id && typeof item.id === 'string');
      assert.equal(item.isShopItem, true);
      assert.ok(item.imageUrl?.startsWith('https://'));
      assert.ok(Number.isFinite(item.price) && item.price > 0);
    }
  }
  assert.equal(result.body.visualStatus, 'NVIDIA_NOT_RUN');
  assert.equal(result.calls.debitQuota, 1);
  // The deck echoes the logical request identity for feedback attribution,
  // and every shop SKU carries truthful owned-piece provenance.
  assert.equal(typeof result.body.generationRequestId, 'string');
  assert.ok(result.body.generationRequestId.length >= 1);
  for (const outfit of result.body.outfits) {
    for (const item of outfit.items) {
      assert.equal(item.isWardrobe, false);
    }
  }
});

test('client-supplied inventory is ignored, server SKUs rule', async () => {
  const result = await signedCall({
    request: baseRequest,
    catalogRows: [{ id: 'attacker-sku', price: 1, currency: 'UZS', image_url: 'https://evil.example/a.jpg' }],
  });
  assert.equal(result.status, 200);
  assert.equal(result.body.clientInventoryIgnored, true);
  const ids = result.body.outfits.flatMap((outfit) => outfit.items.map((item) => item.id));
  assert.ok(!ids.includes('attacker-sku'));
  assert.ok(ids.every((id) => SHOP.some((row) => row.id === id)));
});

test('wardrobe mode leaks no shop inventory', async () => {
  const result = await signedCall({ request: { ...baseRequest, sourceMode: 'wardrobe' } });
  assert.equal(result.status, 200);
  assert.equal(result.body.sourceMode, 'wardrobe');
  const items = result.body.outfits.flatMap((outfit) => outfit.items);
  assert.ok(items.length >= 3);
  for (const item of items) assert.equal(item.isShopItem, false);
  for (const item of items) {
    assert.equal(item.isWardrobe, true);
    assert.equal(item.shopUrl, null);
    assert.equal(item.price, null);
  }
  assert.equal(result.calls.fetchShopRows, 0);
});

test('mixed mode draws from both sources', async () => {
  const result = await signedCall({ request: { ...baseRequest, sourceMode: 'mixed' } });
  assert.equal(result.status, 200);
  const items = result.body.outfits.flatMap((outfit) => outfit.items);
  assert.ok(items.some((item) => item.isShopItem === true));
  assert.ok(items.some((item) => item.isShopItem === false));
});

test('present anchor survives in every outfit', async () => {
  const kept = await signedCall({ request: { ...baseRequest, requiredIds: ['top-1'] } });
  assert.equal(kept.status, 200);
  for (const outfit of kept.body.outfits) {
    assert.ok(outfit.items.some((item) => item.id === 'top-1'), 'locked anchor dropped');
  }
});

test('UZS budget keeps same-currency shop totals under the cap', async () => {
  const variance = [
    shopRow('top-1', 'top', 100, 'UZS'), shopRow('top-2', 'top', 50, 'USD'),
    shopRow('bottom-1', 'bottom', 200, 'UZS'), shopRow('bottom-2', 'bottom', 60, 'USD'),
    shopRow('shoes-1', 'shoes', 300, 'UZS'), shopRow('shoes-2', 'shoes', 70, 'USD'),
  ];
  const result = await signedCall(
    { request: { ...baseRequest, budget: { amount: 1000, currency: 'UZS' } } }, {}, { shopRows: variance });
  assert.equal(result.status, 200);
  for (const outfit of result.body.outfits) {
    const uzs = outfit.items.filter((item) => item.currency === 'UZS');
    const subtotal = uzs.reduce((sum, item) => sum + item.price, 0);
    assert.ok(subtotal <= 1000, `over budget: ${subtotal}`);
    assert.ok(!outfit.items.some((item) => item.isShopItem && item.currency && item.currency !== 'UZS'));
  }
});

test('requested visual review without admissible media stays NOT_RUN with zero provider calls', async () => {
  const result = await signedCall({ request: { ...baseRequest, enableVisual: true } });
  assert.equal(result.status, 200);
  assert.equal(result.body.visualStatus, 'NVIDIA_NOT_RUN');
  assert.equal(result.body.visualReceipt.requested, true);
  assert.equal(result.body.visualReceipt.providerCalls, 0);
});

test('retailer imagery without inference grants never reaches the provider', async () => {
  const seenHosts = new Set();
  const guardFetch = async (url) => {
    seenHosts.add(new URL(String(url)).hostname);
    throw new Error('must_not_fetch_here');
  };
  const harness = makeIo();
  const body = JSON.stringify({ request: { ...baseRequest, enableVisual: true } });
  const authorization = `Bearer ${JWT}`;
  const signature = await signStylistTransport({ secret: SECRET, body, authorization });
  const result = await handleStylist6CloudGenerate({
    method: 'POST', headers: { authorization, ...signature }, bodyText: body,
    env: { ...env, NVIDIA_API_KEY: 'nvapi-test-key-0123456789abcdef' },
    io: harness.io, fetchImpl: guardFetch,
  });
  assert.equal(result.status, 200);
  assert.equal(result.body.visualStatus, 'NVIDIA_NOT_RUN');
  assert.equal(result.body.visualReceipt.providerCalls, 0);
  assert.equal(seenHosts.size, 0);
});

test('authorized media flows end to end through the critic with strict mocks', async () => {
  const jpeg = Buffer.concat([
    Buffer.from([0xff, 0xd8]),
    Buffer.from([0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00]),
    Buffer.from([0xff, 0xc0, 0x00, 0x0b, 0x08, 0x00, 0x08, 0x00, 0x08, 0x01, 0x01, 0x11, 0x00]),
    Buffer.from([0xff, 0xfe, 0x00, 0xca, ...new Array(200).fill(0x2a)]),
    Buffer.from([0xff, 0xd9]),
  ]);
  const verdictRow = {
    candidateId: 'someone-else',
    slotObservations: {
      top: 'a light cotton shirt', outerwear: 'a dark wool blazer',
      bottom: 'grey tailored trousers', shoes: 'black leather shoes',
    },
    visualCohesion: 0.9, referenceFidelity: 0.9, styleFit: 0.9,
    occasionFit: 0.9, formalityFit: 0.9, confidence: 0.9, reject: false,
    relationships: [
      { kind: 'outfit', slots: ['top', 'bottom'], observation: 'the shirt tucks cleanly into the trousers' },
      { kind: 'outfit', slots: ['outerwear', 'shoes'], observation: 'dark layer echoes the dark shoes' },
      { kind: 'reference', slots: ['top', 'bottom', 'shoes'], observation: 'palette follows the reference' },
    ],
    evidence: ['mock cohesion evidence one', 'mock cohesion evidence two', 'mock cohesion evidence three'],
  };
  const seenHosts = new Set();
  const stubFetch = async (url, options = {}) => {
    const parsed = new URL(String(url));
    seenHosts.add(parsed.hostname);
    if (parsed.hostname === 'integrate.api.nvidia.com') {
      return new Response(JSON.stringify({
        choices: [{ finish_reason: 'stop', message: { content: JSON.stringify(verdictRow) } }],
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
      }), { status: 200, headers: { 'Content-Type': 'application/json' } });
    }
    if (parsed.hostname.endsWith('.supabase.co')) {
      return new Response(jpeg, { status: 200, headers: { 'Content-Type': 'image/jpeg' } });
    }
    throw new Error('unexpected outbound host ' + parsed.hostname);
  };
  const sbHost = 'https://fyqpifmrsftsfqibhwhy.supabase.co/storage/v1/object/shop-catalog';
  const authorizedShop = ['top', 'bottom', 'shoes'].map((slot, index) => ({
    id: `auth-${slot}-${index}`,
    name: `${slot} authorized garment`, category: { top: 'tops', bottom: 'bottoms', shoes: 'shoes' }[slot],
    garment_type: slot === 'bottom' ? 'lower_body' : slot === 'shoes' ? 'shoes' : 'upper_body',
    price: 100 + index, currency: 'UZS',
    image_url: `${sbHost}/${slot}.jpg`, source_url: `https://example.com/shop/${slot}`,
    is_active: true, brand: 'Test',
  }));
  const harness = makeIo({ shopRows: authorizedShop });
  const body = JSON.stringify({ request: { ...baseRequest, enableVisual: true } });
  const authorization = `Bearer ${JWT}`;
  const signature = await signStylistTransport({ secret: SECRET, body, authorization });
  const result = await handleStylist6CloudGenerate({
    method: 'POST', headers: { authorization, ...signature }, bodyText: body,
    env: { ...env, NVIDIA_API_KEY: 'nvapi-test-key-0123456789abcdef', STYLIST6_REFERENCE_BUCKET: 'test-bucket' },
    io: harness.io, fetchImpl: stubFetch,
  });
  assert.equal(result.status, 200);
  // The mocked verdict names an unknown candidate, so the run abstains —
  // but the full authorized chain (admit, fetch, dispatch, validate) ran.
  assert.equal(result.body.visualStatus, 'NVIDIA_ABSTAINED');
  assert.ok(result.body.visualReceipt.providerCalls >= 1);
  assert.ok(![...seenHosts].some((host) => host !== 'integrate.api.nvidia.com' && !host.endsWith('.supabase.co')));
});

test('visual status vocabulary never fabricates influence', () => {
  assert.equal(mapCloudVisualStatus(null), 'NVIDIA_NOT_RUN');
  assert.equal(mapCloudVisualStatus({ diagnostics: { visualVerdict: 'NOT_RUN' }, outfits: [] }), 'NVIDIA_NOT_RUN');
  assert.equal(mapCloudVisualStatus({ diagnostics: { visualVerdict: 'ABSTAINED' }, outfits: [] }), 'NVIDIA_ABSTAINED');
  assert.equal(
    mapCloudVisualStatus({ diagnostics: { visualVerdict: 'JUDGED' }, outfits: [{ id: 'a' }] }),
    'NVIDIA_EVALUATED_NO_CHANGE',
  );
  assert.equal(
    mapCloudVisualStatus({ diagnostics: { visualVerdict: 'JUDGED' }, outfits: [{ id: 'a', visualEvidence: ['x'] }] }),
    'NVIDIA_APPLIED',
  );
});

test('wardrobe projector maps slots honestly and drops junk', () => {
  assert.equal(projectWardrobeItem(null), null);
  assert.equal(projectWardrobeItem({ id: 'x', name: 'mystery object', image_url: 'https://example.com/x.jpg' }), null);
  assert.equal(
    projectWardrobeItem({ id: 'x', type: 'shirt', name: 'shirt', image_url: 'http://plain/x.jpg' }), null);
  const shirt = projectWardrobeItem(WARDROBE[0]);
  assert.equal(shirt.macroCategory, 'top');
  assert.equal(shirt.subtype, 'shirt');
  assert.equal(shirt.isShopItem, false);
  const jeans = projectWardrobeItem(WARDROBE[1]);
  assert.equal(jeans.macroCategory, 'bottom');
  const sneakers = projectWardrobeItem(WARDROBE[2]);
  assert.equal(sneakers.macroCategory, 'shoes');
});

test('visual byte fitting passes small JPEGs and rejects the rest', () => {
  assert.equal(fitVisualBytesCloud(null), null);
  assert.equal(fitVisualBytesCloud(Buffer.alloc(200, 0)), null);
  const jpeg = Buffer.concat([Buffer.from([0xff, 0xd8]), Buffer.alloc(300, 7)]);
  assert.ok(Buffer.isBuffer(fitVisualBytesCloud(jpeg)));
  assert.equal(fitVisualBytesCloud(Buffer.concat([Buffer.from([0xff, 0xd8]), Buffer.alloc(10, 1)])), null);
});


test('canonical request fingerprints ignore object order but bind changed contents', () => {
  assert.equal(canonicalRequestFingerprint({style:'Classic',budget:{amount:10,currency:'UZS'}}),
    canonicalRequestFingerprint({budget:{currency:'UZS',amount:10},style:'Classic'}));
  assert.notEqual(canonicalRequestFingerprint({style:'Classic'}), canonicalRequestFingerprint({style:'Casual'}));
});

test('malformed budgets fail before debit', async () => {
  for (const budget of [{amount:-1,currency:'UZS'}, {amount:'100',currency:'UZS'}, {amount:10}, {}, '100']) {
    const result = await signedCall({request:{...baseRequest,budget}});
    assert.equal(result.status,400);
    assert.equal(result.calls.debitQuota,0);
  }
});

test('a debited request with no feasible looks records its terminal outcome', async () => {
  // Every shop row is UZS; a USD budget leaves zero feasible items after
  // admission, so the already-claimed and debited request must still record
  // its terminal outcome for honest 409 replay.
  const request = { request: { ...baseRequest, budget: { amount: 100, currency: 'USD' } } };
  const { status, body, calls, claims } = await signedCall(request);
  assert.equal(status, 502);
  assert.equal(body.code, 'stylist6_no_feasible_looks');
  assert.equal(calls.claimRequest, 1);
  assert.equal(calls.debitQuota, 1);
  assert.equal(calls.finishRequest, 1);
  const [record] = [...claims.values()];
  assert.equal(record.status, 502);
  assert.equal(record.code, 'stylist6_no_feasible_looks');
});


test('changed contents under one logical id conflict without a second debit', async () => {
  const harness = makeIo();
  const call = async (style) => {
    const body = JSON.stringify({request:{...baseRequest,style,generationRequestId:'bound-1'}});
    const authorization = 'Bearer ' + JWT;
    const signature = await signStylistTransport({secret:SECRET,body,authorization});
    return handleStylist6CloudGenerate({method:'POST',headers:{authorization,...signature},bodyText:body,env,io:harness.io});
  };
  assert.equal((await call('old_money')).status,200);
  const changed = await call('casual');
  assert.equal(changed.status,409);
  assert.equal(changed.body.code,'request_content_conflict');
  assert.equal(harness.calls.debitQuota,1);
});


test('server discovery projection without is_active remains usable; explicit inactive rows stay rejected', async (t) => {
  const { createCloudIo } = await import('./cloud_handler.mjs');
  const { adaptCatalog } = await import('./catalog_adapter.mjs');
  const liveProjection = SHOP.map(({ is_active, ...row }) => row);
  liveProjection.push({ ...SHOP[0], id: 'explicit-inactive', is_active: false });
  t.mock.method(globalThis, 'fetch', async (url) => {
    assert.match(String(url), /\/rest\/v1\/rpc\/catalog_discovery_candidates$/);
    return new Response(JSON.stringify(liveProjection), { status: 200 });
  });
  const io = createCloudIo({ url: 'https://example.supabase.co', anonKey: 'anon', serviceKey: 'server', supabaseJs: {} });
  const rows = await io.fetchShopRows();
  const adapted = adaptCatalog(rows);
  assert.equal(adapted.items.length, SHOP.length);
  assert.equal(adapted.skipped.inactive, 1);
  assert.equal(rows.find(row => row.id === 'explicit-inactive').is_active, false);
  // Raw untrusted/standalone adapter input still requires an explicit flag.
  assert.equal(adaptCatalog(liveProjection.filter(row => row.id !== 'explicit-inactive')).items.length, 0);
});

test('server discovery projection without fashion_subtype restores the intelligence descriptor subtype', async (t) => {
  const { createCloudIo } = await import('./cloud_handler.mjs');
  const { adaptCatalog } = await import('./catalog_adapter.mjs');
  // Mirror the real RPC projection: fashion_slot present, subtype column
  // omitted, intelligence descriptor intact.
  const slotOf = { tops: 'top', bottoms: 'bottom', shoes: 'shoes' };
  const liveProjection = SHOP.map((row) => {
    const { is_active, ...rest } = row;
    return {
      ...rest,
      fashion_slot: slotOf[row.category] ?? 'top',
      ...(row.id === 'shoes-1'
        ? { fashion_intelligence: { version: 1, descriptor: { slot: 'shoes', subtype: 'loafer', formality: 'smart' } } }
        : {}),
    };
  });
  t.mock.method(globalThis, 'fetch', async (url) => {
    assert.match(String(url), /\/rest\/v1\/rpc\/catalog_discovery_candidates$/);
    return new Response(JSON.stringify(liveProjection), { status: 200 });
  });
  const io = createCloudIo({ url: 'https://example.supabase.co', anonKey: 'anon', serviceKey: 'server', supabaseJs: {} });
  const rows = await io.fetchShopRows();
  assert.equal(rows.find(row => row.id === 'shoes-1').fashion_subtype, 'loafer');
  assert.equal(rows.find(row => row.id === 'shoes-2').fashion_subtype, undefined);
  const adapted = adaptCatalog(rows);
  const shoes1 = adapted.items.find(item => item.id === 'shoes-1');
  assert.equal(shoes1.macroCategory, 'shoes');
  assert.equal(shoes1.subtype, 'loafer');
  assert.equal(shoes1.projection.kind, 'live_intelligence');
  // Rows without a descriptor stay on the transparent text projection.
  const shoes2 = adapted.items.find(item => item.id === 'shoes-2');
  assert.notEqual(shoes2.projection.kind, 'live_intelligence');
  // The general adapter itself never reads fashion_intelligence: raw RPC
  // rows keep zero live hints — restoration happens only at the server
  // boundary.
  const rawAdapted = adaptCatalog(liveProjection.map((row) => ({ ...row, is_active: true })));
  assert.equal(rawAdapted.items.find(item => item.id === 'shoes-1').projection.kind, 'native_category');
});
