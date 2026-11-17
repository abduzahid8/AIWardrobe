import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import sharp from 'sharp';
import { signStylistTransport } from '../stylist-node/transport/signature.mjs';
import {
  handleStylist6CloudGenerate,
  createCloudIo, hydrateProfiles, catalogStylingDiscoveryHints,
  mapCloudVisualStatus,
  projectWardrobeItem,
  fitVisualBytesCloud,
  STYLIST6_CLOUD_VERSION,
  fashionSlateDecisionBudgetMs,
  prewarmProductPhotos,
  stylist6SafeQualityTrace,
  serverFashionSlateEnabled,
  canonicalRequestFingerprint,
} from './cloud_handler.mjs';
import { HOST_ALLOWLIST, INFERENCE_ALLOWLIST, isInferenceAuthorized } from '../../scripts/stylist6/image_access.mjs';
import { adaptCatalog, sourceHash } from './catalog_adapter.mjs';
import { describeGarment } from '../../src/domain/garmentDescriptor.ts';
import { garmentContextReason } from '../../src/domain/garmentContext.ts';
import { stylingContext } from '../../src/domain/stylingContext.ts';

const SECRET = 'cloud-test-shared-secret-0123456789abcdef';
const JWT = 'good-jwt-token';
const USER = 'user-1';

const shopRow = (id, slot, price = 250000, currency = 'UZS') => ({
  id, name: `${slot} garment ${id}`, category: { top: 'tops', bottom: 'bottoms', shoes: 'shoes', outerwear: 'tops' }[slot],
  garment_type: slot === 'bottom' ? 'lower_body' : slot === 'shoes' ? 'shoes' : 'upper_body',
  price, currency,
  image_url: `https://example.com/images/${id}.jpg`,
  source_url: `https://example.com/shop/${id}`,
  is_active: true, brand: 'Test',
});
const SHOP = [
  shopRow('top-1', 'top', 100000), shopRow('top-2', 'top', 150000),
  shopRow('bottom-1', 'bottom', 200000), shopRow('bottom-2', 'bottom', 120000),
  shopRow('shoes-1', 'shoes', 300000), shopRow('shoes-2', 'shoes', 180000),
  shopRow('outer-1', 'outerwear', 500000),
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
    fetchShopRows: 0, fetchWardrobeRows: 0, fetchVisualProfiles: 0, fetchFeedbackTaste: 0,
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
      async debitQuota(userId, requestId) {
        calls.debitQuota += 1;
        assert.equal(userId, USER);
        assert.ok(claims.has(requestId), 'debit must be tied to a durable request claim');
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
      async fetchShopRows(seed, context) {
        calls.fetchShopRows += 1;
        overrides.onShopSeed?.(seed, context);
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
      // Gate D taste source: newest-first explicit thumbs rows for the JWT
      // user (service-role read in production). Tests inject fixtures or
      // failures through overrides.
      async fetchFeedbackTaste(userId) {
        calls.fetchFeedbackTaste += 1;
        assert.equal(userId, USER);
        if (overrides.feedbackFail) throw new Error('stylist6_feedback_unavailable');
        return overrides.feedbackRows ?? [];
      },
      ...(overrides.fetchBudgetFxSnapshot ? { fetchBudgetFxSnapshot: overrides.fetchBudgetFxSnapshot } : {}),
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

test('authenticated cloud response returns only product-grounded personal style evidence', async () => {
  const request = { ...baseRequest,
    preferences: {favoriteBrands:['Test'],fitPreference:'balanced'},
  };
  const result = await signedCall({request});
  assert.equal(result.status, 200);
  assert.equal(result.body.status, 'complete');
  assert.equal(result.body.generationVersion, STYLIST6_CLOUD_VERSION);
  assert.equal(result.calls.debitQuota, 1);
  assert.equal(result.body.outfits.length,3);
  for (const outfit of result.body.outfits) {
    const personalization = outfit.fashionDiagnostics?.personalization;
    assert.ok(personalization);
    assert.ok(personalization.adjustment>0 && personalization.adjustment<=0.085);
    assert.ok(personalization.matches.some(x=>x.type==='favorite_brand' && x.value==='Test'));
    assert.ok(personalization.matches.every(x=>
      outfit.items.some(item=>item.id===x.productId)));
    assert.equal(outfit.preferences,undefined);
    assert.ok(['signature','refresh','discovery','alternative'].includes(outfit.direction?.id));
    if(outfit.direction.id === 'discovery') {
      assert.ok(outfit.direction.qualified);
      assert.ok(outfit.direction.evidence.structuralSlots.length > 0 ||
        outfit.direction.evidence.paletteSlots.length >= 3);
    }
  }
  assert.equal(result.body.outfits[0].direction.id,'signature');
});

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
          if (name === 'settle_stylist6_request') {
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

test('Stylist 6 quota is charged and failure-settled through claim-bound atomic RPCs only', async () => {
  const { createCloudIo } = await import('./cloud_handler.mjs');
  const used = [];
  const io = createCloudIo({
    url: 'https://example.supabase.co', anonKey: 'anon', serviceKey: 'service',
    supabaseJs: { createClient: () => ({
      rpc: async (name, args) => {
        used.push({ name, args });
        if (name === 'claim_stylist6_request') return { data: [{ duplicate:false, status_code:0, outcome_code:'' }], error:null };
        if (name === 'charge_stylist6_request') return { data:true, error:null };
        if (name === 'settle_stylist6_request') return { data:true, error:null };
        throw Error(`unexpected or obsolete RPC: ${name}`);
      },
    }) },
  });
  const requestId = 'server-bound-transaction-1';
  const claim = await io.claimRequest(USER, requestId, 'a'.repeat(64));
  assert.equal(claim.duplicate, false);
  await io.debitQuota(USER, requestId);
  assert.equal(await io.finishRequest(USER, requestId, 502, 'stylist6_visual_quality_unmet'), true);
  assert.deepEqual(used.map(x=>x.name), [
    'claim_stylist6_request','charge_stylist6_request','settle_stylist6_request',
  ]);
  assert.deepEqual(used[1].args, {p_user:USER,p_request_id:requestId});
  assert.equal(used[2].args.p_status,502);
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
          if (name === 'charge_stylist6_request') { calls.debit += 1; return { data: true, error: null }; }
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

test('language pass-through writes deck copy in the requested language, never a mix', async () => {
  const CYRILLIC = /[А-Яа-яёЁ]/;
  const LATIN = /[A-Za-z]/;
  for (const language of ['ru', 'uz']) {
    const result = await signedCall({ request: { ...baseRequest, language } });
    assert.equal(result.status, 200);
    assert.equal(result.body.outfits.length, 3);
    // Same uniqueness rule in every language: three distinct headlines.
    assert.equal(new Set(result.body.outfits.map(x => (x.concept?.title || '').toLowerCase())).size, 3);
    for (const outfit of result.body.outfits) {
      const title = outfit.concept?.title || '';
      const description = String(outfit.description || '');
      // Monolingual strings only: a card never mixes languages within one string.
      assert.ok(!(CYRILLIC.test(title) && LATIN.test(title)), `mixed title: ${title}`);
      assert.ok(!(CYRILLIC.test(description) && LATIN.test(description)), `mixed description: ${description}`);
      if (language === 'ru') {
        assert.ok(CYRILLIC.test(description), `en description leaked into ru deck: ${description}`);
        assert.ok(!LATIN.test(title) && !LATIN.test(description));
      } else {
        assert.ok(!CYRILLIC.test(title) && !CYRILLIC.test(description));
        assert.ok(LATIN.test(description));
      }
    }
  }
  // An explicit 'en' request keeps the established English copy.
  const english = await signedCall({ request: { ...baseRequest, language: 'en' } });
  assert.equal(english.status, 200);
  for (const outfit of english.body.outfits) {
    assert.ok(!CYRILLIC.test(String(outfit.description || '')));
    assert.match(String(outfit.description), /[a-z]/i);
  }
});

test('an unsupported language value degrades to English instead of failing admission', async () => {
  const result = await signedCall({ request: { ...baseRequest, language: 'de-DE' } });
  assert.equal(result.status, 200);
  assert.match(String(result.body.outfits[0].description), /[a-z]/i);
  assert.ok(!/[А-Яа-яёЁ]/.test(String(result.body.outfits[0].description)));
});

test('every cloud look satisfies the client thumbs-write (stylist_look_feedback) contract', async () => {
  // Mirrors src/services/stylistFeedback.ts canRecordLookFeedback so the
  // screen's thumbs up/down on this exact deck reaches stylist_look_feedback
  // with the server-issued outfit_N look id, request id, concept slug and
  // engine version — plus the per-item ids the Gate-D ranker links against.
  const result = await signedCall({ request: baseRequest });
  assert.equal(result.status, 200);
  assert.ok(result.body.outfits.length >= 1);
  for (const outfit of result.body.outfits) {
    assert.match(outfit.id, /^outfit_[1-3]$/);
    assert.match(outfit.concept?.id || '', /^[a-z][a-z0-9-]{0,63}$/);
    assert.ok(typeof result.body.generationRequestId === 'string' && result.body.generationRequestId.length >= 1);
    assert.ok(typeof outfit.engineVersion === 'string'
      && outfit.engineVersion.length >= 1 && outfit.engineVersion.length <= 80);
    assert.ok(outfit.items.every((item) => typeof item.id === 'string' && item.id.trim().length > 0));
  }
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
    shopRow('top-1', 'top', 100000, 'UZS'), shopRow('top-2', 'top', 50, 'USD'),
    shopRow('bottom-1', 'bottom', 200000, 'UZS'), shopRow('bottom-2', 'bottom', 60, 'USD'),
    shopRow('shoes-1', 'shoes', 300000, 'UZS'), shopRow('shoes-2', 'shoes', 70, 'USD'),
  ];
  const result = await signedCall(
    { request: { ...baseRequest, budget: { amount: 1000000, currency: 'UZS' } } }, {}, { shopRows: variance });
  assert.equal(result.status, 200);
  for (const outfit of result.body.outfits) {
    const uzs = outfit.items.filter((item) => item.currency === 'UZS');
    const subtotal = uzs.reduce((sum, item) => sum + item.price, 0);
    assert.ok(subtotal <= 1000000, `over budget: ${subtotal}`);
    assert.ok(!outfit.items.some((item) => item.isShopItem && item.currency && item.currency !== 'UZS'));
  }
});

test('a UZS budget over a mixed-currency pool still returns a complete UZS-only deck when one is feasible', async () => {
  // Every slot exists in BOTH currencies: the same-currency pre-filter drops
  // the USD offers, but a full UZS inventory remains — the engine must
  // complete, not abstain.
  const mixed = [
    shopRow('m-top-1', 'top', 400000, 'UZS'), shopRow('m-top-2', 'top', 420000, 'UZS'), shopRow('m-top-usd', 'top', 30, 'USD'),
    shopRow('m-bottom-1', 'bottom', 450000, 'UZS'), shopRow('m-bottom-2', 'bottom', 480000, 'UZS'), shopRow('m-bottom-usd', 'bottom', 40, 'USD'),
    shopRow('m-shoes-1', 'shoes', 500000, 'UZS'), shopRow('m-shoes-2', 'shoes', 520000, 'UZS'), shopRow('m-shoes-usd', 'shoes', 60, 'USD'),
    shopRow('m-outer-1', 'outerwear', 700000, 'UZS'), shopRow('m-outer-usd', 'outerwear', 90, 'USD'),
  ];
  const result = await signedCall(
    { request: { ...baseRequest, budget: { amount: 5000000, currency: 'UZS' } } }, {}, { shopRows: mixed });
  assert.equal(result.status, 200);
  assert.equal(result.body.status, 'complete');
  assert.equal(result.body.outfits.length, 3);
  for (const outfit of result.body.outfits) {
    for (const item of outfit.items) {
      if (item.isShopItem !== true) continue;
      assert.equal(item.currency, 'UZS', 'cross-currency shop offer leaked into a UZS deck');
    }
  }
  const shownIds = new Set(result.body.outfits.flatMap((o) => o.items.map((i) => i.id)));
  for (const id of ['m-top-usd', 'm-bottom-usd', 'm-shoes-usd', 'm-outer-usd']) {
    assert.ok(!shownIds.has(id), `USD offer ${id} shown in a UZS deck`);
  }
});

test('merchant offer freshness rides the deck items and stays honest without timestamps', async () => {
  const now = Date.now();
  const iso = (days) => new Date(now - days * 86400000).toISOString();
  const rows = [
    { ...shopRow('fresh-1', 'top', 400000), last_seen_at: iso(2) },
    { ...shopRow('stale-1', 'bottom', 450000), last_seen_at: iso(60) },
    shopRow('unknown-1', 'shoes', 500000), // no last_seen_at: ~4.4k live rows look like this
  ];
  const result = await signedCall({ request: baseRequest }, {}, { shopRows: rows });
  assert.equal(result.status, 200);
  const items = result.body.outfits.flatMap((outfit) => outfit.items);
  const byTier = (tier) => items.filter((item) => item.priceFreshness === tier);
  assert.ok(byTier('fresh').length > 0);
  assert.ok(byTier('stale').length > 0);
  assert.ok(byTier('unknown').length > 0);
  const fresh = byTier('fresh')[0];
  assert.ok(Number.isFinite(fresh.priceAgeDays) && fresh.priceAgeDays >= 2);
  for (const item of items) {
    if (item.priceFreshness === 'unknown') assert.equal(item.priceAgeDays, null);
    if (item.priceFreshness === 'stale') assert.ok(item.priceAgeDays > 45);
  }
  // Rendering contract: the cloud acceptance tolerates the extra fields.
  globalThis.__DEV__ = globalThis.__DEV__ ?? true;
  const { acceptStylist6CloudDeck } = await import('../../src/services/stylist6Dev.ts');
  const acceptance = acceptStylist6CloudDeck(result.body.outfits, 'catalog');
  assert.equal(acceptance.accepted.length, result.body.outfits.length);
  assert.deepEqual(acceptance.rejections, {});
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
  // A real decoder-valid JPEG, not merely a header/EOI marker mock. The
  // cloud now does genuine pixel decoding before a provider dispatch.
  const jpeg = await sharp({
    create: { width: 128, height: 128, channels: 3,
      background: { r: 42, g: 42, b: 42 } },
  }).jpeg({ quality: 82 }).toBuffer();
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
    price: 250000 + index * 50000, currency: 'UZS',
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

test('present but unusable locked footwear returns actionable 422 without fabricating replacement', async () => {
  const wardrobeMismatch = SHOP.map(item => item.id === 'shoes-1'
    ? { ...item, name:'Neon beach flip-flops for swimming', fashion_subtype:'flip_flop',
      garment_type:'shoes', category:'shoes' }
    : item);
  const {status,body,calls} = await signedCall({request:{...baseRequest,
    style:'classic',occasion:'Work',requiredIds:['shoes-1'],
    generationRequestId:'qa-locked-incompatible-shoe-1'}}, {}, {shopRows:wardrobeMismatch});
  assert.equal(status,422);
  assert.equal(body.code,'anchor_outfit_unavailable');
  assert.equal(body.outfits.length,0);
  assert.equal(calls.debitQuota,1);
  assert.equal(calls.finishRequest,1);
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



test('new unanchored requests rotate the shop seed privately while retries remain deterministic', async () => {
  const received = [];
  const request = { ...baseRequest, generationRequestId: 'seed-case-a' };
  const payload = { request };
  const body = JSON.stringify(payload);
  const expected = createHash('sha256').update(USER).update('|').update(body).digest('hex').slice(0, 16);
  const first = await signedCall(payload, {}, { onShopSeed: seed => received.push(seed) });
  assert.equal(first.status, 200);
  assert.deepEqual(received, [expected]);
  const again = await signedCall(payload, {}, { onShopSeed: seed => received.push(seed) });
  assert.equal(again.status, 200); // new isolated io, same bytes => same seed
  assert.deepEqual(received, [expected, expected]);
  const changed = await signedCall({ request: { ...request, generationRequestId: 'seed-case-b' } },
    {}, { onShopSeed: seed => received.push(seed) });
  assert.equal(changed.status, 200);
  assert.match(received[2], /^[a-f0-9]{16}$/);
  assert.notEqual(received[2], expected);
});

test('explicit user anchor keeps the original catalog sampling route', async () => {
  const seeds = [];
  const result = await signedCall({
    request: { ...baseRequest, requiredIds: ['top-1'], generationRequestId: 'locked-anchor' },
  }, {}, { onShopSeed: seed => seeds.push(seed) });
  assert.equal(result.status, 200);
  assert.deepEqual(seeds, [null]);
  assert.ok(result.body.outfits.every(look => look.items.some(item => item.id === 'top-1')));
});

test('brand-balanced shoes expose new makers instead of Geox / two-vendor monopolies', async () => {
  const {balancedShoeCatalogWindow}=await import('./cloud_handler.mjs');
  const brands=['Jack Erwin','BLKBRD','Cobbler Union','Velasca','Beckett Simonon',
    'TLB Mallorca','Meermin','Herring','Septième Largeur','Loake','Zara','Geox',
    'Shoepassion','Grant Stone','Barker','Carlos Santos','Henry Stevens','MANGO','ECCO','Wildsmith'];
  const rows=brands.flatMap((brand,j)=>Array.from({length:20},(_,i)=>({
    id:`shoe-${j}-${i}`,brand, fashion_intelligence:{descriptor:{subtype:i===19?'unknown':'loafer'}},
  })));
  // A discovery index organized by manufacturer used to put 20 items from
  // the same source in front of every other brand.
  const balanced=balancedShoeCatalogWindow(rows,100);
  const counts=new Map();
  for(const row of balanced)counts.set(row.brand,(counts.get(row.brand)||0)+1);
  assert.equal(balanced.length,100);
  assert.equal(counts.size,20);
  assert.ok(Math.max(...counts.values())<=5);
  assert.ok(balanced.every(x=>rows.includes(x)));
  assert.deepEqual(balancedShoeCatalogWindow([...rows,rows[0]],100),balanced,
    'aliases of the same source SKU never increase coverage');
  assert.equal(balancedShoeCatalogWindow([],100).length,0);
});

test('catalog discovery RPC accepts only validated hex seed and stays within 100 per slot', async (t) => {
  const { createCloudIo } = await import('./cloud_handler.mjs');
  const seen = [];
  t.mock.method(globalThis, 'fetch', async (_url, options) => {
    seen.push(JSON.parse(options.body));
    return new Response('[]', { status: 200 });
  });
  const io = createCloudIo({ url: 'https://example.supabase.co', anonKey: 'anon', serviceKey: 'server', supabaseJs: {} });
  const hex = 'a1b2c3d4e5f60708';
  assert.deepEqual(await io.fetchShopRows(hex), []);
  assert.equal(seen.length, 6);
  assert.ok(seen.every(row => row.p_seed.endsWith(':' + hex) && row.p_limit === 100));
  assert.deepEqual(seen.filter(row=>row.p_slot==='shoes').map(row=>row.p_offset),[0,100,200]);
  seen.length = 0;
  await io.fetchShopRows('x'.repeat(256));
  assert.ok(seen.every(row => /^s6cloud:(top|bottom|shoes|outerwear)$/.test(row.p_seed)));
});

test('cloud discovery attaches bounded live import observations to matching SKU metadata', async (t) => {
  const { createCloudIo } = await import('./cloud_handler.mjs');
  const { adaptCatalog } = await import('./catalog_adapter.mjs');
  const observed = '2026-10-26T00:00:00Z';
  const calls = { discovery: 0, observations: 0 };
  const rpcRows = SHOP.map(({ is_active, ...rest }) => rest);
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    const endpoint = new URL(String(url));
    if (endpoint.pathname === '/rest/v1/rpc/catalog_discovery_candidates') {
      calls.discovery++;
      assert.equal(options.method, 'POST');
      return Response.json(rpcRows);
    }
    if (endpoint.pathname === '/rest/v1/shop_catalog') {
      calls.observations++;
      assert.equal(options.method, 'GET');
      const ids = endpoint.searchParams.get('id').replace(/^in\.\(/, '').replace(/\)$/, '').split(',');
      return Response.json(SHOP.filter(item => ids.includes(item.id))
        .map(row => ({ ...row, last_seen_at: observed })));
    }
    assert.fail('unexpected network endpoint ' + endpoint.pathname);
  });
  const io = createCloudIo({ url: 'https://example.supabase.co', anonKey: 'anon',
    serviceKey: 'server', supabaseJs: {} });
  const rows = await io.fetchShopRows();
  assert.equal(calls.discovery, 6);
  assert.equal(calls.observations, 1);
  assert.equal(rows.length, SHOP.length);
  assert.ok(rows.every(row => row.last_seen_at === observed));
  assert.ok(adaptCatalog(rows).items.every(item => item.priceFreshness === 'recent'));
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

// ---------------------------------------------------------------------------
// Cloud-staging visual review: inference allowlist unification, kill-switch,
// verdict cache and honest degradation.
// ---------------------------------------------------------------------------

test('inference authorization covers every audited download host plus first-party storage', () => {
  for (const host of HOST_ALLOWLIST) {
    assert.equal(isInferenceAuthorized(`https://${host}/img.jpg`), true, host);
    assert.ok(INFERENCE_ALLOWLIST.has(host), host);
  }
  assert.equal(isInferenceAuthorized('https://fyqpifmrsftsfqibhwhy.supabase.co/storage/v1/object/x.jpg'), true);
  assert.equal(isInferenceAuthorized('https://unknown-merchant.example/img.jpg'), false);
  assert.equal(isInferenceAuthorized('http://static.zara.net/img.jpg'), false, 'non-https never authorized');
  assert.equal(isInferenceAuthorized('https://static.zara.net:8443/img.jpg'), false, 'ports never authorized');
});

/** Decoder-valid 128x128 JPEG, deterministic per fill with no metadata. */
const smokeJpeg = async (fill) => sharp({
  create: { width: 128, height: 128, channels: 3,
    background: { r: fill, g: fill, b: fill } },
}).jpeg({ quality: 80 }).toBuffer();

/** Merchant-host catalog fixture; every SKU gets distinct image bytes. */
const merchantShop = (salt, host = 'static.zara.net') => {
  const slots = { top: 2, outerwear: 1, bottom: 2, shoes: 2 };
  const rows = [];
  for (const [slot, count] of Object.entries(slots)) {
    for (let index = 0; index < count; index++) {
      const id = `m-${slot}-${index}`;
      const fill = createHash('sha256').update(`${salt}:${id}`).digest()[0] % 200 + 20;
      rows.push({
        id, name: `${slot} merchant garment ${index}`, category: { top: 'tops', bottom: 'bottoms', shoes: 'shoes', outerwear: 'tops' }[slot],
        garment_type: slot === 'bottom' ? 'lower_body' : slot === 'shoes' ? 'shoes' : 'upper_body',
        price: 250000 + index * 50000, currency: 'UZS',
        image_url: `https://${host}/images/${id}.jpg`, source_url: `https://example.com/shop/${id}`,
        is_active: true, brand: 'Test',
      });
    }
  }
  return rows;
};

/**
 * Builds a critic-valid verdict for the prompt the critic actually sent:
 * parses the slot list from the context JSON, then returns three grounded
 * relationships (outfit-only when the run is reference-less, never claiming
 * a reference comparison that did not happen).
 */
function verdictForPrompt(promptText, candidateId) {
  const contextLine = promptText.split('\n').find((line) => line.startsWith('{"style"'));
  const { imageSlots } = JSON.parse(contextLine);
  const referenceless = promptText.includes('No reference image is supplied');
  const slotObservations = Object.fromEntries(imageSlots.map((slot) => [slot, `visible ${slot} garment with clean trim`]));
  const relationships = [
    { kind: 'outfit', slots: [imageSlots[0], imageSlots[1]], observation: 'the upper garments keep one tone line' },
    { kind: 'outfit', slots: [imageSlots[1], imageSlots[2] ?? imageSlots[0]], observation: 'the lower pieces repeat the darker tone' },
    referenceless
      ? { kind: 'outfit', slots: [imageSlots[0], imageSlots.at(-1)], observation: 'the trim echoes the leading garment' }
      : { kind: 'reference', slots: [imageSlots[0]], observation: 'the palette follows the reference' },
  ];
  const row = {
    candidateId, slotObservations, visualCohesion: 0.9, styleFit: 0.9,
    occasionFit: 0.9, formalityFit: 0.9, confidence: 0.95, reject: false, relationships,
  };
  if (!referenceless) row.referenceFidelity = 0.9;
  return row;
}

/** Media stub: merchant/first-party hosts get deterministic JPEG bytes, the
 * NVIDIA endpoint gets a grounded verdict derived from the sent prompt. */
function makeVisualFetchStub(salt, recorder = { prompts: [], hosts: new Set() },
  mediaFormat = 'jpeg', transformVerdict = (row) => row) {
  return async (url, options = {}) => {
    const parsed = new URL(String(url));
    recorder.hosts.add(parsed.hostname);
    if (parsed.hostname === 'integrate.api.nvidia.com') {
      const content = JSON.parse(options.body).messages[0].content;
      const prompt = content[0].text;
      recorder.prompts.push(prompt);
      const id = prompt.match(/candidateId must be exactly "([^"]+)"/)[1];
      return new Response(JSON.stringify({
        choices: [{ finish_reason: 'stop', message: { content: JSON.stringify(
          transformVerdict(verdictForPrompt(prompt, id))) } }],
        usage: { prompt_tokens: 900, completion_tokens: 60, total_tokens: 960 },
      }), { status: 200, headers: { 'Content-Type': 'application/json' } });
    }
    const fill = createHash('sha256').update(`${salt}:${String(url)}`).digest()[0] % 200 + 20;
    const pixel = sharp({
      create:{width:256,height:192,channels:3,background:{r:fill,g:fill,b:fill}},
    });
    const bytes=mediaFormat === 'png' ? await pixel.png().toBuffer()
      : mediaFormat === 'webp' ? await pixel.webp().toBuffer()
      : await smokeJpeg(fill);
    return new Response(bytes, { status: 200, headers: { 'Content-Type':
      mediaFormat === 'webp' ? 'image/webp' : mediaFormat === 'png' ? 'image/png' : 'image/jpeg',
    } });
  };
}

async function cloudVisualCall(request, { salt, env: envExtra = {}, fetchStub } = {}) {
  const harness = makeIo({ shopRows: merchantShop(salt) });
  const body = JSON.stringify({ request });
  const authorization = `Bearer ${JWT}`;
  const signature = await signStylistTransport({ secret: SECRET, body, authorization });
  const result = await handleStylist6CloudGenerate({
    method: 'POST', headers: { authorization, ...signature }, bodyText: body,
    env: { ...env, NVIDIA_API_KEY: 'nvapi-test-key-0123456789abcdef', ...envExtra },
    io: harness.io, fetchImpl: fetchStub,
  });
  return result;
}

test('allowlisted merchant imagery reaches the hosted critic and the deck reports applied evidence', async () => {
  const recorder = { prompts: [], hosts: new Set() };
  const result = await cloudVisualCall(
    { ...baseRequest, enableVisual: true, generationRequestId: 'visual-merchant-1' },
    { salt: 'merchant-activation', env: { STYLIST6_REFERENCE_BUCKET: 'staged-references' }, fetchStub: makeVisualFetchStub('merchant-activation', recorder) },
  );
  assert.equal(result.status, 200);
  assert.equal(result.body.visualStatus, 'NVIDIA_APPLIED');
  const receipt = result.body.visualReceipt;
  assert.ok(['full_three_verified','verified_partial'].includes(receipt.visualApproval));
  assert.equal(receipt.displayedJudged, receipt.displayedLooks,
    'selected looks must have their own verdict');
  for (const look of result.body.outfits) {
    assert.equal(look.fashionDiagnostics.visualVerdict, 'JUDGED');
  }
  assert.ok(receipt.providerCalls >= 1, 'the critic must dispatch at least one provider call');
  assert.ok(receipt.imagesSent >= receipt.providerCalls, 'images sent must be metered');
  assert.ok(receipt.validVerdicts >= 1);
  assert.equal(receipt.retriedCalls, 0);
  assert.equal(receipt.cacheHits, 0);
  assert.equal(receipt.serverDisabled, false);
  assert.equal(receipt.model, 'nvidia/nemotron-3.5-super-vl-120b-a12b');
  assert.ok(receipt.totalTokens >= 1, 'tokens-based cost accounting is present');
  // Transmission gate: only the merchant CDN, NVIDIA and first-party
  // Supabase storage were ever contacted.
  for (const host of recorder.hosts) {
    assert.ok(
      host === 'integrate.api.nvidia.com' || host.endsWith('.supabase.co') || host === 'static.zara.net',
      `unexpected host ${host}`,
    );
  }
  assert.ok(result.body.outfits.some((look) => Array.isArray(look.items) && look.items.length >= 3));
  assert.equal(result.body.diagnostics.visualVerdict, 'JUDGED');
});

test('cloud rejects all low-quality assessed outfits without charging AGAIN or laundering unjudged replacements', async()=>{
  const recorder={prompts:[],hosts:new Set()};
  const degrade = row => ({
    ...row, visualCohesion:0.1, styleFit:0.1,
    occasionFit:0.1, formalityFit:0.1, referenceFidelity:0.1,
    reject:false, confidence:0.95,
  });
  const result=await cloudVisualCall(
    { ...baseRequest, enableVisual:true, generationRequestId:'s6-visual-quality-zero-1' },
    {salt:'quality-zero',fetchStub:makeVisualFetchStub('quality-zero',recorder,'jpeg',degrade)},
  );
  assert.equal(result.status,502);
  assert.equal(result.body.code,'stylist6_visual_quality_unmet');
  assert.equal(result.body.visualStatus,'NVIDIA_EVALUATED_NO_CHANGE');
  assert.deepEqual(result.body.outfits,[]);
  assert.equal(result.body.diagnostics.visualReviewCoverage.approved,0);
  assert.ok(result.body.diagnostics.visualQualityDecisions?.length > 0);
  assert.ok(result.body.diagnostics.visualQualityDecisions.every(x => x.reason === 'below_quality_floor' && x.quality < 0.68));
  assert.ok(recorder.prompts.length>0);
});

test('cloud returns only independent, individually approved partial looks rather than fabricating a third', async()=>{
  const recorder={prompts:[],hosts:new Set()};
  let call=0;
  const degradeLater = row => {
    call++;
    return call<=2 ? row : {
      ...row, visualCohesion:0.1, styleFit:0.1, occasionFit:0.1,
      formalityFit:0.1, referenceFidelity:0.1, confidence:0.95,
    };
  };
  const result=await cloudVisualCall(
    {...baseRequest,enableVisual:true,generationRequestId:'s6-visual-two-approved-1'},
    {salt:'two-approved',fetchStub:makeVisualFetchStub('two-approved',recorder,'jpeg',degradeLater)},
  );
  assert.equal(result.status,200);
  assert.equal(result.body.status,'partial');
  // The two model-approved images may themselves form structurally
  // equivalent looks. Never force both into the deck just to hit a count.
  assert.ok(result.body.outfits.length>=1 && result.body.outfits.length<=2);
  assert.equal(result.body.visualStatus,'NVIDIA_APPLIED');
  assert.equal(result.body.visualReceipt.visualApproval,'verified_partial');
  assert.equal(result.body.visualReceipt.displayedLooks,result.body.outfits.length);
  assert.equal(result.body.visualReceipt.displayedJudged,result.body.outfits.length);
  assert.equal(result.body.diagnostics.visualReviewCoverage.approved,2,
    'both model verdicts are present; displayed pairwise independence is a separate gate');
  assert.ok(result.body.outfits.every(x=>x.fashionDiagnostics.visualVerdict==='JUDGED'));
});

test('valid retail PNG/WebP get in-memory conversion and real critic verdicts',async()=>{
  for(const format of ['png','webp']) {
    const recorder={prompts:[],hosts:new Set()};
    const result=await cloudVisualCall(
      {...baseRequest,enableVisual:true,generationRequestId:'visual-format-'+format},
      {salt:'media-format-'+format,fetchStub:makeVisualFetchStub('media-format-'+format,recorder,format)},
    );
    assert.equal(result.status,200,format);
    assert.ok(['NVIDIA_APPLIED','NVIDIA_EVALUATED_NO_CHANGE'].includes(result.body.visualStatus),
      format+' status '+result.body.visualStatus+' receipt '+JSON.stringify(result.body.visualReceipt));
    assert.ok(result.body.visualReceipt.validVerdicts>0);
    assert.ok(result.body.visualReceipt.providerCalls>0);
    assert.ok(result.body.visualReceipt.imagesSent>=3);
    assert.equal(result.body.visualReceipt.referenceBucketConfigured,false);
    assert.ok(recorder.prompts.some(x=>x.includes('No reference image is supplied')));
    assert.ok(recorder.hosts.has('integrate.api.nvidia.com'));
  }
});

test('staging without a reference bucket judges reference-less looks and never claims a reference', async () => {
  const recorder = { prompts: [], hosts: new Set() };
  // style 'casual' matches no reference metadata in the committed index, so
  // no reference id is attached and STYLIST6_REFERENCE_BUCKET is absent.
  const result = await cloudVisualCall(
    { ...baseRequest, style: 'casual', occasion: 'Work', enableVisual: true, generationRequestId: 'visual-refless-1' },
    { salt: 'refless-activation', fetchStub: makeVisualFetchStub('refless-activation', recorder) },
  );
  assert.equal(result.status, 200);
  assert.ok(['NVIDIA_APPLIED', 'NVIDIA_EVALUATED_NO_CHANGE'].includes(result.body.visualStatus),
    `expected a judged deck, got ${result.body.visualStatus}`);
  assert.ok(recorder.prompts.length >= 1);
  // Every dispatched prompt that lacked a reference is the reference-less
  // contract; no prompt ever scores a reference that was not supplied.
  for (const prompt of recorder.prompts) {
    if (prompt.includes('No reference image is supplied')) {
      assert.ok(!prompt.includes('referenceFidelity'));
      assert.ok(!prompt.includes('The first image is the reference'));
    }
  }
  assert.ok(result.body.visualReceipt.validVerdicts >= 1);
  for (const look of result.body.outfits) {
    if (look.items?.length) {
      assert.equal(look.concept?.referenceId ?? null, null);
    }
  }
});

test('STYLIST6_VISUAL_DISABLED forces an honest NOT_RUN with zero provider calls', async () => {
  const recorder = { prompts: [], hosts: new Set() };
  const result = await cloudVisualCall(
    { ...baseRequest, enableVisual: true, generationRequestId: 'visual-killed-1' },
    { salt: 'kill-switch', env: { STYLIST6_VISUAL_DISABLED: '1', STYLIST6_REFERENCE_BUCKET: 'staged-references' }, fetchStub: makeVisualFetchStub('kill-switch', recorder) },
  );
  assert.equal(result.status, 200);
  assert.equal(result.body.visualStatus, 'NVIDIA_NOT_RUN');
  const receipt = result.body.visualReceipt;
  assert.equal(receipt.requested, true);
  assert.equal(receipt.serverDisabled, true);
  assert.equal(receipt.providerCalls, 0);
  assert.equal(receipt.imagesSent, 0);
  assert.equal(receipt.model, null);
  assert.equal(recorder.hosts.size, 0, 'no outbound traffic at all under the kill-switch');
});

test('a repeated staging request with identical media reuses cached verdicts without another provider call', async () => {
  const recorder = { prompts: [], hosts: new Set() };
  const fetchStub = makeVisualFetchStub('cache-hit-path', recorder);
  const first = await cloudVisualCall(
    { ...baseRequest, enableVisual: true, generationRequestId: 'visual-cache-1' },
    { salt: 'cache-hit-path', env: { STYLIST6_REFERENCE_BUCKET: 'staged-references' }, fetchStub },
  );
  assert.equal(first.status, 200);
  assert.ok(first.body.visualReceipt.providerCalls >= 1);
  const firstIds = first.body.outfits.map((look) => look.items.map((item) => item.id).sort().join('+')).join('|');
  const second = await cloudVisualCall(
    { ...baseRequest, enableVisual: true, generationRequestId: 'visual-cache-2' },
    { salt: 'cache-hit-path', env: { STYLIST6_REFERENCE_BUCKET: 'staged-references' }, fetchStub },
  );
  assert.equal(second.status, 200);
  assert.equal(second.body.visualReceipt.providerCalls, 0, 'identical media must not pay a second provider call');
  assert.ok(second.body.visualReceipt.cacheHits >= 1);
  assert.ok(second.body.visualReceipt.validVerdicts >= 1);
  assert.equal(second.body.diagnostics.visualVerdict, first.body.diagnostics.visualVerdict);
  const secondIds = second.body.outfits.map((look) => look.items.map((item) => item.id).sort().join('+')).join('|');
  assert.equal(secondIds, firstIds, 'the cached run reproduces the same deck');
  const providerCallsAfterSecondRequest = recorder.prompts.length;
  assert.ok(providerCallsAfterSecondRequest >= 1);
});

test('enableVisual without a server key completes the deterministic deck with an honest empty receipt', async () => {
  const recorder = { prompts: [], hosts: new Set() };
  const harness = makeIo({ shopRows: merchantShop('no-key') });
  const body = JSON.stringify({ request: { ...baseRequest, enableVisual: true } });
  const authorization = `Bearer ${JWT}`;
  const signature = await signStylistTransport({ secret: SECRET, body, authorization });
  const result = await handleStylist6CloudGenerate({
    method: 'POST', headers: { authorization, ...signature }, bodyText: body,
    env: { ...env }, io: harness.io, fetchImpl: makeVisualFetchStub('no-key', recorder),
  });
  assert.equal(result.status, 200);
  assert.equal(result.body.visualStatus, 'NVIDIA_NOT_RUN');
  assert.deepEqual(result.body.visualReceipt, {
    model: null, policy: 'explicit_client', requested: true, serverDisabled: false, referenceBucketConfigured: false,
    providerCalls: 0, imagesSent: 0, cacheHits: 0, retriedCalls: 0,
    validVerdicts: 0, abstentions: 0, displayedLooks: result.body.outfits.length, displayedJudged: 0,
    visualApproval: 'unavailable', totalTokens: null, events: [],
  });
  assert.equal(recorder.hosts.size, 0, 'no key means no outbound provider traffic');
  assert.equal(harness.calls.debitQuota, 1);
});

// ---------------------------------------------------------------------------
// Gate D: explicit taste feedback changes ranking deterministically, without
// touching quality floors, hard gates or admission semantics.
// ---------------------------------------------------------------------------

/** Two likes saturate a ±1 tally at any age under the 30-day half-life, so
 * the deck flip below is exact regardless of wall-clock drift. */
const ratedAt = new Date(Date.now() - 86_400_000).toISOString();
const feedbackRow = (rating, itemIds, conceptId, direction, requestId) => ({
  rating, concept_id: conceptId, item_ids: itemIds,
  look_signature: { version: 1, direction, signature: null, features: [] },
  created_at: ratedAt, request_id: requestId,
});
// User A liked the refined-ivy lead (top-1/bottom-1/shoes-1) twice across two
// decks and disliked the top-2 alternative twice; user B is the exact inverse.
const A_LIKED = ['top-1', 'bottom-1', 'shoes-1'];
const A_DISLIKED = ['top-2'];
const likedSet = (rating, requestIdPrefix) => [
  feedbackRow(rating, A_LIKED, 'refined-ivy', 'signature', `${requestIdPrefix}-a`),
  feedbackRow(rating, A_LIKED, 'refined-ivy', 'signature', `${requestIdPrefix}-b`),
  feedbackRow(-rating, A_DISLIKED, 'refined-ivy', 'signature', `${requestIdPrefix}-c`),
  feedbackRow(-rating, A_DISLIKED, 'refined-ivy', 'signature', `${requestIdPrefix}-d`),
];
const deckOf = (result) => result.body.outfits
  .map((look) => look.items.map((item) => item.id).sort().join('+')).join('|');

test('gate D: opposite explicit feedback flips the deck over the same inventory', async () => {
  const userA = await signedCall({ request: baseRequest }, {}, { feedbackRows: likedSet(1, 'like') });
  const userB = await signedCall({ request: baseRequest }, {}, { feedbackRows: likedSet(-1, 'dislike') });
  assert.equal(userA.status, 200);
  assert.equal(userB.status, 200);
  const deckA = deckOf(userA);
  const deckB = deckOf(userB);
  assert.notEqual(deckA, deckB, 'taste feedback must change ranking');
  // A's signature look is the liked lead and never the disliked alternative;
  // B's signature look is exactly the inverse over the SAME shop inventory.
  const signatureA = userA.body.outfits[0].items.map((item) => item.id);
  const signatureB = userB.body.outfits[0].items.map((item) => item.id);
  for (const id of A_LIKED) {
    assert.ok(signatureA.includes(id), `A signature must keep liked ${id}`);
    assert.ok(!signatureB.includes(id), `B signature must drop ${id}`);
  }
  assert.ok(signatureB.includes('top-2'), 'B signature surfaces the item A disliked');
  assert.ok(!signatureA.includes('top-2'), 'A signature drops the disliked alternative');
  // Quality/hard constraints survive the adjustment unchanged.
  for (const result of [userA, userB]) {
    assert.equal(result.body.status, 'complete');
    for (const outfit of result.body.outfits) {
      assert.ok(outfit.items.length >= 3 && outfit.items.length <= 4);
      for (const item of outfit.items) {
        assert.ok(SHOP.some((row) => row.id === item.id), 'server inventory only');
        assert.equal(item.isShopItem, true);
      }
    }
    assert.equal(result.calls.debitQuota, 1);
  }
  // The read receipt is honest, and the adjustment is visible in per-look
  // diagnostics bounded to the feedback cap.
  assert.deepEqual(userA.body.feedbackTaste, {
    status: 'ok', rows: 4, active: true, version: 's6-feedback-taste-v2',
  });
  const diagnostics = userA.body.outfits
    .map((outfit) => outfit.fashionDiagnostics?.personalization?.feedback)
    .filter(Boolean);
  assert.ok(diagnostics.length >= 1, 'feedback adjustment must appear in scoring diagnostics');
  for (const entry of diagnostics) {
    assert.equal(entry.version, 's6-feedback-taste-v2');
    assert.ok(Math.abs(entry.adjustment) <= 0.06);
  }
  assert.ok(diagnostics.some((entry) => entry.adjustment > 0));
});

test('gate D: zero feedback reproduces the exact legacy ranking', async () => {
  const empty = await signedCall({ request: baseRequest }, {}, { feedbackRows: [] });
  // An io without the new method (older harness) is a not_wired read, not a
  // failure — and must rank exactly like an empty feedback set.
  const harness = makeIo();
  delete harness.io.fetchFeedbackTaste;
  const body = JSON.stringify({ request: baseRequest });
  const authorization = `Bearer ${JWT}`;
  const signature = await signStylistTransport({ secret: SECRET, body, authorization });
  const notWired = await handleStylist6CloudGenerate({
    method: 'POST', headers: { authorization, ...signature }, bodyText: body, env, io: harness.io,
  });
  assert.equal(deckOf(empty), deckOf(notWired));
  assert.equal(empty.body.feedbackTaste.status, 'ok');
  assert.equal(empty.body.feedbackTaste.active, false);
  assert.equal(notWired.body.feedbackTaste.status, 'not_wired');
  assert.equal(notWired.body.feedbackTaste.active, false);
  for (const result of [empty, notWired]) {
    for (const outfit of result.body.outfits) {
      assert.equal(outfit.fashionDiagnostics?.personalization?.feedback, undefined);
    }
    assert.equal(result.body.diagnostics.feedbackTasteActive, false);
    assert.equal(result.body.diagnostics.feedbackTasteVersion, 's6-feedback-taste-v2');
  }
});

test('gate D: a feedback read failure degrades honestly and never fails the deck', async () => {
  const result = await signedCall({ request: baseRequest }, {}, { feedbackFail: true });
  assert.equal(result.status, 200);
  assert.equal(result.body.success, true);
  assert.deepEqual(result.body.feedbackTaste, {
    status: 'read_failed', rows: 0, active: false, version: 's6-feedback-taste-v2',
  });
  assert.equal(result.body.outfits.length, 3);
  assert.equal(result.calls.debitQuota, 1);
});

test('gate D: uniform dislikes cannot empty the deck or block any item outright', async () => {
  // Every catalog item disliked twice over: strong negative adjustment only —
  // the engine must still produce a valid, complete deck from the same pool.
  const allDisliked = ['top-1', 'top-2', 'bottom-1', 'bottom-2', 'shoes-1', 'shoes-2', 'outer-1']
    .flatMap((id) => [
      feedbackRow(-1, [id], 'refined-ivy', 'signature', `neg-${id}-1`),
      feedbackRow(-1, [id], 'refined-ivy', 'signature', `neg-${id}-2`),
    ]);
  const result = await signedCall({ request: baseRequest }, {}, { feedbackRows: allDisliked });
  assert.equal(result.status, 200);
  assert.equal(result.body.status, 'complete');
  assert.ok(result.body.outfits.length === 3);
  assert.equal(result.body.feedbackTaste.active, true);
});

test('verified FX enables honest cross-currency budgets while unavailable FX never invents rates', async () => {
  const usdShop = SHOP.map((row) => ({ ...row, price: 60, currency: 'USD' }));
  // Without FX: USD items cannot satisfy an UZS budget (fail closed, zero invented rates).
  const withoutFx = await signedCall({
    request: { ...baseRequest, budget: { amount: 2500000, currency: 'UZS' }, generationRequestId: 'fx-missing-1' },
  }, {}, { shopRows: usdShop });
  assert.equal(withoutFx.status, 502);
  assert.equal(withoutFx.body.code, 'stylist6_no_feasible_looks');
  assert.equal(withoutFx.body.currencyConversion.active, false);
  assert.equal(withoutFx.body.currencyConversion.reason, 'fx_not_configured');

  // With valid CBU FX snapshot: 3 * $60 * 12800 = 2,304,000 UZS <= 2,500,000 UZS passes cleanly.
  const { parseCbuBudgetRates } = await import('../../src/domain/budgetFx.ts');
  const now = Date.now();
  const uzToday = new Date(now + 5 * 3600_000);
  const dd = String(uzToday.getUTCDate()).padStart(2, '0');
  const mm = String(uzToday.getUTCMonth() + 1).padStart(2, '0');
  const yyyy = uzToday.getUTCFullYear();
  const cbuDate = `${dd}.${mm}.${yyyy}`;
  const validSnapshot = parseCbuBudgetRates([
    { Ccy: 'USD', Nominal: '1', Rate: '12800', Date: cbuDate },
    { Ccy: 'EUR', Nominal: '1', Rate: '13800', Date: cbuDate },
    { Ccy: 'GBP', Nominal: '1', Rate: '16200', Date: cbuDate },
  ], now);
  const withFx = await signedCall({
    request: { ...baseRequest, budget: { amount: 2500000, currency: 'UZS' }, generationRequestId: 'fx-valid-1' },
  }, {}, { shopRows: usdShop, fetchBudgetFxSnapshot: async () => validSnapshot });
  assert.equal(withFx.status, 200);
  assert.equal(withFx.body.currencyConversion.active, true);
  assert.equal(withFx.body.currencyConversion.source, 'CBU');
  assert.equal(withFx.body.currencyConversion.asOf, validSnapshot.asOf);
  assert.deepEqual(withFx.body.currencyConversion.snapshot, validSnapshot);
  assert.ok(withFx.body.offerVerification);
  for (const outfit of withFx.body.outfits) {
    assert.equal(outfit.selectionSource, 'deterministic_fallback');
    assert.ok(outfit.offerVerification);
  }

  // Expired/tampered snapshot is rejected and falls back cleanly without inventing rates.
  const expired = await signedCall({
    request: { ...baseRequest, budget: { amount: 2500000, currency: 'UZS' }, generationRequestId: 'fx-expired-1' },
  }, {}, {
    shopRows: usdShop,
    fetchBudgetFxSnapshot: async () => ({ ...validSnapshot, fetchedAt: new Date(now - 2 * 3_600_000).toISOString() }),
  });
  assert.equal(expired.status, 502);
  assert.equal(expired.body.currencyConversion.active, false);
  assert.equal(expired.body.currencyConversion.reason, 'fx_unavailable');
});

test('accepts standard generate-outfits release payloads, ignores client wardrobeItems, and supports secret rotation', async () => {
  const { normalizeStylist6CloudPayload } = await import('./cloud_handler.mjs');
  const releasePayload = {
    prompt: 'refined office look',
    occasion: 'Work',
    stylePreferences: ['old_money'],
    catalogOnly: true,
    anchorItemId: 'shop_top-1',
    excludeItemIds: ['shop_top-2'],
    recentLookSignatures: ['sig-1'],
    preferences: {
      favoriteBrands: ['Test'],
      maxShopBudget: { amount: 2000000, currency: 'UZS' },
    },
    wardrobeItems: [{ id: 'forged-wardrobe-item', price: 0 }],
    generationRequestId: 'release-payload-1',
    language: 'uz',
  };
  const normalized = normalizeStylist6CloudPayload(releasePayload, { STYLIST6_DEFAULT_VISUAL_EVAL: 'false' });
  assert.equal(normalized.style, 'old_money');
  assert.equal(normalized.occasion, 'Work');
  assert.equal(normalized.sourceMode, 'catalog');
  assert.deepEqual(normalized.budget, { amount: 2000000, currency: 'UZS' });
  assert.deepEqual(normalized.requiredIds, ['shop_top-1']);
  assert.deepEqual(normalized.excludeItemIds, ['shop_top-2']);
  assert.deepEqual(normalized.recentSignatures, ['sig-1']);
  assert.equal(normalized.language, 'uz');
  assert.equal(normalized.preferences.maxShopBudget, undefined);

  // Sign with previousSecret while primary secret is rotated
  const previousSecret = 'previous-shared-secret-0123456789abcdef';
  const rotatedEnv = {
    ...env,
    STYLIST6_NODE_SHARED_SECRET: 'new-active-shared-secret-0123456789abcdef',
    STYLIST6_NODE_SHARED_SECRET_PREVIOUS: previousSecret,
  };
  const harness = makeIo();
  const body = JSON.stringify(releasePayload);
  const authorization = `Bearer ${JWT}`;
  const signature = await signStylistTransport({ secret: previousSecret, body, authorization });
  const res = await handleStylist6CloudGenerate({
    method: 'POST',
    headers: { authorization, ...signature },
    bodyText: body,
    env: rotatedEnv,
    io: harness.io,
  });
  assert.equal(res.status, 200);
  assert.equal(res.body.clientInventoryIgnored, true);
  assert.equal(res.body.language, 'uz');
  assert.ok(res.body.outfits.length > 0);
  for (const outfit of res.body.outfits) {
    assert.ok(outfit.items.some((item) => item.id === 'top-1'));
    assert.ok(outfit.items.every((item) => item.id !== 'top-2'));
  }

  // Duplicate replay reports outcomeState: 'terminal_recorded'
  const dup = await handleStylist6CloudGenerate({
    method: 'POST',
    headers: { authorization, ...signature },
    bodyText: body,
    env: rotatedEnv,
    io: harness.io,
  });
  assert.equal(dup.status, 409);
  assert.equal(dup.body.outcomeState, 'terminal_recorded');
  assert.equal(dup.body.priorStatus, 200);

  // Unprefixed catalog SKU anchor in catalogOnly mode stays pure catalog (never loads wardrobe)
  const unprefixedCatalogRes = await signedCall({
    occasion: 'Work',
    stylePreferences: ['old_money'],
    catalogOnly: true,
    anchorItemId: 'top-1',
    generationRequestId: 'release-unprefixed-catalog-anchor-1',
  });
  assert.equal(unprefixedCatalogRes.status, 200);
  assert.equal(unprefixedCatalogRes.body.sourceMode, 'catalog');
  for (const outfit of unprefixedCatalogRes.body.outfits) {
    assert.ok(outfit.items.some((item) => item.id === 'top-1'));
    assert.ok(outfit.items.every((item) => item.isShopItem === true));
  }

  // Locked closet anchor in catalogOnly mode loads only the locked closet anchor + catalog items
  const closetAnchorRes = await signedCall({
    occasion: 'Work',
    stylePreferences: ['old_money'],
    catalogOnly: true,
    anchorItemId: 'w-top-1',
    generationRequestId: 'release-closet-anchor-1',
  });
  assert.equal(closetAnchorRes.status, 200);
  assert.equal(closetAnchorRes.body.sourceMode, 'mixed');
  for (const outfit of closetAnchorRes.body.outfits) {
    assert.ok(outfit.items.some((item) => item.id === 'w-top-1' && item.isShopItem === false));
    assert.ok(outfit.items.filter((item) => item.id !== 'w-top-1').every((item) => item.isShopItem === true));
  }

  // Shop-only occasion defaults to catalog and merges id: product keys into excludeItemIds
  const shopOnlyNormalized = normalizeStylist6CloudPayload({
    stylePreferences: 'business_casual',
    occasion: 'Team Collaboration',
    excludeProductKeys: ['id:top-2', 'name:ignored'],
    previousDeckProductKeys: ['id:shoe-1'],
  }, {});
  assert.equal(shopOnlyNormalized.sourceMode, 'catalog');
  assert.deepEqual(shopOnlyNormalized.excludeItemIds, ['top-2', 'shoe-1']);
});

test('budgeted unanchored cloud discovery forwards budget, currency, and FX rates via p_context', async (t) => {
  const { createCloudIo } = await import('./cloud_handler.mjs');
  const { parseCbuBudgetRates } = await import('../../src/domain/budgetFx.ts');
  const seen = [];
  t.mock.method(globalThis, 'fetch', async (_url, options) => {
    seen.push(JSON.parse(options.body));
    return new Response('[]', { status: 200 });
  });
  const io = createCloudIo({ url: 'https://example.supabase.co', anonKey: 'anon', serviceKey: 'server', supabaseJs: {} });
  const rates = { UZS: 1, USD: 12800, EUR: 13800, GBP: 16200 };
  await io.fetchShopRows(null, { budget: 2500000, currency: 'UZS', rates });
  assert.equal(seen.length, 6);
  assert.ok(seen.every((row) => row.p_context?.budget === 2500000 && row.p_context?.currency === 'UZS' && row.p_context?.rates?.USD === 12800));

  const now = Date.now();
  const futureUz = new Date(now + 5 * 3_600_000);
  const dd = String(futureUz.getUTCDate()).padStart(2, '0');
  const mm = String(futureUz.getUTCMonth() + 1).padStart(2, '0');
  const yyyy = futureUz.getUTCFullYear();
  const cbuDate = `${dd}.${mm}.${yyyy}`;
  const validSnapshot = parseCbuBudgetRates([
    { Ccy: 'USD', Nominal: '1', Rate: '12800', Date: cbuDate },
    { Ccy: 'EUR', Nominal: '1', Rate: '13800', Date: cbuDate },
    { Ccy: 'GBP', Nominal: '1', Rate: '16200', Date: cbuDate },
  ], now);

  const capturedContexts = [];
  await signedCall({
    request: { ...baseRequest, budget: { amount: 2500000, currency: 'UZS' }, generationRequestId: 'budget-ctx-1' },
  }, {}, {
    fetchBudgetFxSnapshot: async () => validSnapshot,
    onShopSeed: (_seed, context) => capturedContexts.push(context),
  });
  assert.deepEqual(capturedContexts[0], {
    budget: 2500000,
    currency: 'UZS',
    rates: validSnapshot.uzsPerUnit,
    ...catalogStylingDiscoveryHints({style:baseRequest.style,
      occasion:baseRequest.occasion,brief:baseRequest.brief,
      weather:baseRequest.weather}),
  });

  // Locked anchors must not be pre-filtered before anchor_not_found validation.
  capturedContexts.length = 0;
  await signedCall({
    request: { ...baseRequest, requiredIds: ['top-1'], budget: { amount: 2500000, currency: 'UZS' }, generationRequestId: 'budget-ctx-anchored' },
  }, {}, {
    fetchBudgetFxSnapshot: async () => validSnapshot,
    onShopSeed: (_seed, context) => capturedContexts.push(context),
  });
  assert.equal(capturedContexts[0], null);
});

test('crash-window recovery: deadline aborts slow provider with 504 stylist6_timeout and stranded claims signal retryWithNewRequestId without re-debit', async () => {
  // 1) Simulated stalled downstream provider inside ranking triggers 504 + finishRequest before 60s kill
  const harness = makeIo({ shopRows: merchantShop('timeout-test') });
  const body = JSON.stringify({ request: { ...baseRequest, enableVisual: true, generationRequestId: 'timeout-recovery-1' } });
  const authorization = `Bearer ${JWT}`;
  const signature = await signStylistTransport({ secret: SECRET, body, authorization });
  const mediaStub = makeVisualFetchStub('timeout-test');
  const slowFetch = async (targetUrl, options = {}) => {
    const u = String(targetUrl);
    if (u.startsWith('https://integrate.api.nvidia.com/')) {
      await new Promise((resolve, reject) => {
        const t = setTimeout(resolve, 1000);
        options.signal?.addEventListener('abort', () => { clearTimeout(t); reject(new Error('aborted')); }, { once: true });
      });
    }
    return mediaStub(targetUrl, options);
  };
  const timeoutRes = await handleStylist6CloudGenerate({
    method: 'POST',
    headers: { authorization, ...signature },
    bodyText: body,
    env: { ...env, NVIDIA_API_KEY: 'nvapi-timeout-test-key' },
    io: harness.io,
    fetchImpl: slowFetch,
    timeoutMs: 120,
  });
  assert.equal(timeoutRes.status, 504);
  assert.equal(timeoutRes.body.code, 'stylist6_timeout');
  assert.equal(timeoutRes.body.retryWithNewRequestId, true);
  assert.equal(harness.calls.debitQuota, 1);
  assert.equal(harness.calls.finishRequest, 1);
  assert.equal(harness.claims.get('timeout-recovery-1').status, 504);
  assert.equal(harness.claims.get('timeout-recovery-1').code, 'stylist6_timeout');

  // Replaying timed-out ID returns 409 duplicate_request with 0 extra debits
  const replayTimeout = await handleStylist6CloudGenerate({
    method: 'POST',
    headers: { authorization, ...signature },
    bodyText: body,
    env,
    io: harness.io,
  });
  assert.equal(replayTimeout.status, 409);
  assert.equal(replayTimeout.body.code, 'duplicate_request');
  assert.equal(replayTimeout.body.priorStatus, 504);
  assert.equal(replayTimeout.body.retryWithNewRequestId, false);
  assert.equal(harness.calls.debitQuota, 1);

  // 2) Stranded status=0 claim (e.g. Lambda SIGKILL mid-flight) blocks same ID and signals retryWithNewRequestId=true
  const strandedHarness = makeIo();
  const strandedReq = { ...baseRequest, generationRequestId: 'sigkill-stranded-1' };
  const strandedBody = JSON.stringify({ request: strandedReq });
  const strandedSig = await signStylistTransport({ secret: SECRET, body: strandedBody, authorization });
  strandedHarness.claims.set('sigkill-stranded-1', {
    status: 0,
    code: '',
    fingerprint: canonicalRequestFingerprint(strandedReq),
  });
  const strandedRes = await handleStylist6CloudGenerate({
    method: 'POST',
    headers: { authorization, ...strandedSig },
    bodyText: strandedBody,
    env,
    io: strandedHarness.io,
  });
  assert.equal(strandedRes.status, 409);
  assert.equal(strandedRes.body.code, 'duplicate_request');
  assert.equal(strandedRes.body.outcomeState, 'in_progress_or_interrupted');
  assert.equal(strandedRes.body.retryWithNewRequestId, true);
  assert.equal(strandedHarness.calls.debitQuota, 0);

  // 3) Visual policy transparency: default_off vs server_default vs server_killswitch vs explicit_client
  const defaultOff = await signedCall({ request: { ...baseRequest, generationRequestId: 'policy-default-off' } });
  assert.equal(defaultOff.body.visualReceipt.policy, 'default_off');

  const explicitBody = JSON.stringify({ request: { ...baseRequest, enableVisual: true, generationRequestId: 'policy-explicit' } });
  const explicitSig = await signStylistTransport({ secret: SECRET, body: explicitBody, authorization });
  const explicitRes = await handleStylist6CloudGenerate({
    method: 'POST',
    headers: { authorization, ...explicitSig },
    bodyText: explicitBody,
    env,
    io: makeIo().io,
  });
  assert.equal(explicitRes.body.visualReceipt.policy, 'explicit_client');

  const srvDefaultBody = JSON.stringify({ prompt: 'work', occasion: 'Work', stylePreferences: ['classic'], generationRequestId: 'policy-srv-default' });
  const srvDefaultSig = await signStylistTransport({ secret: SECRET, body: srvDefaultBody, authorization });
  const srvDefaultRes = await handleStylist6CloudGenerate({
    method: 'POST',
    headers: { authorization, ...srvDefaultSig },
    bodyText: srvDefaultBody,
    env: { ...env, STYLIST6_DEFAULT_VISUAL_EVAL: 'true' },
    io: makeIo().io,
  });
  assert.equal(srvDefaultRes.body.visualReceipt.policy, 'server_default');

  const killRes = await handleStylist6CloudGenerate({
    method: 'POST',
    headers: { authorization, ...explicitSig },
    bodyText: explicitBody,
    env: { ...env, STYLIST6_VISUAL_DISABLED: 'true' },
    io: makeIo().io,
  });
  assert.equal(killRes.body.visualReceipt.policy, 'server_killswitch');
});

test('real workout catalog discovery prioritizes training gear and preserves the four 100-SKU windows', async()=>{
  const { createCloudIo }=await import('./cloud_handler.mjs');
  const originalFetch=globalThis.fetch;
  const searched=[];
  globalThis.fetch=async (_url, options={})=>{
    assert.equal(options.method,'POST','all catalog discovery is server-side RPC');
    const payload=JSON.parse(options.body);
    searched.push(payload);
    const slot=payload.p_slot;
    const search=payload.p_search||'general';
    const count=payload.p_limit;
    return Response.json(Array.from({length:count},(_,n)=>({
      ...shopRow(`${slot}-${search}-${n}`,slot),
      last_seen_at:null,
      fashion_intelligence:{descriptor:{subtype:search}},
    })));
  };
  try{
    const io=createCloudIo({url:'https://example.supabase.co',anonKey:'public',serviceKey:'server',supabaseJs:{}});
    const athletic=await io.fetchShopRows('a'.repeat(16),null,'workout');
    assert.equal(searched.length,11,'six bounded catalog pages plus five targeted search terms');
    assert.deepEqual(searched.filter(x=>x.p_search).map(x=>`${x.p_slot}:${x.p_search}`),[
      'top:training','bottom:jogger','shoes:running','shoes:training','outerwear:training',
    ]);
    assert.ok(searched.every(x=>x.p_context?.dressCode==='athletic'));
    assert.equal(athletic.length,400,'cannot exceed 400 SKU hydration cap');
    assert.ok(athletic.some(x=>x.id==='top-training-0'));
    assert.ok(athletic.some(x=>x.id==='bottom-jogger-0'));
    assert.ok(athletic.some(x=>x.id==='shoes-running-0'));
    assert.ok(athletic.some(x=>x.id==='shoes-training-0'));
    assert.equal(athletic.filter(x=>x.id.startsWith('top-')).length,100);
    assert.equal(athletic.filter(x=>x.id.startsWith('bottom-')).length,100);
    assert.equal(athletic.filter(x=>x.id.startsWith('shoes-')).length,100);
    assert.equal(athletic.filter(x=>x.id.startsWith('outerwear-')).length,100);
    searched.length=0;
    const winter=await io.fetchShopRows('a'.repeat(16),null,'cold');
    assert.equal(searched.length,9,'six standard pages plus three snow-friendly queries');
    assert.deepEqual(searched.filter(x=>x.p_search).map(x=>`${x.p_slot}:${x.p_search}`),[
      'outerwear:puffer','outerwear:coat','shoes:boot',
    ]);
    assert.equal(winter.length,400,'winter discovery must keep exact 400-SKU cap');
    assert.ok(winter.some(x=>x.id==='outerwear-puffer-0'));
    assert.ok(winter.some(x=>x.id==='shoes-boot-0'));
    searched.length=0;
    const everyday=await io.fetchShopRows('a'.repeat(16),null,null);
    assert.equal(searched.length,6);
    assert.ok(searched.every(x=>x.p_search==='' && !x.p_context));
    assert.equal(everyday.length,400);
  }finally{globalThis.fetch=originalFetch;}
});


test('cloud visual-profile REST query actually selects the version needed to hydrate evidence', async () => {
  const queryURLs=[];
  const before=globalThis.fetch;
  try {
    globalThis.fetch=async (url) => {
      const parsed=new URL(String(url));
      queryURLs.push(parsed);
      assert.equal(parsed.pathname,'/rest/v1/catalog_visual_fashion');
      return Response.json([]);
    };
    const io=createCloudIo({url:'https://fake-supabase.example',anonKey:'test-anon',
      serviceKey:'test-service',supabaseJs:{}});
    assert.deepEqual(await io.fetchVisualProfiles(['fashion-item-1']),[]);
  } finally {globalThis.fetch=before;}
  assert.equal(queryURLs.length,1);
  assert.equal(queryURLs[0].searchParams.get('version'),'eq.1');
  assert.deepEqual(queryURLs[0].searchParams.get('select').split(','),
    ['product_id','version','source_hash','image_url','profile'],
    'missing version caused all expensive already-computed image intelligence to be ignored');
});

test('all four 100-SKU visual-profile reads start together, not behind serialized HTTP waits',async()=>{
  const prior=globalThis.fetch, pending=[], urls=[];
  try{
    globalThis.fetch=(url)=>{
      const location=new URL(String(url));
      urls.push(location);
      return new Promise(resolve=>pending.push(resolve));
    };
    const io=createCloudIo({url:'https://fake-supabase.example',
      anonKey:'anon-test',serviceKey:'service-test',supabaseJs:{}});
    const ids=Array.from({length:400},(_,i)=>'safe-product-'+i);
    const read=io.fetchVisualProfiles(ids);
    await Promise.resolve();
    await Promise.resolve();
    assert.equal(urls.length,4,'all four REST reads start before any response');
    for(const location of urls){
      assert.equal(location.pathname,'/rest/v1/catalog_visual_fashion');
      assert.equal(location.searchParams.get('version'),'eq.1');
      assert.deepEqual(location.searchParams.get('select').split(','),
        ['product_id','version','source_hash','image_url','profile']);
      const chosen=location.searchParams.get('product_id')||'';
      assert.match(chosen,/^in\./);
      assert.equal((chosen.match(/safe-product-/g)||[]).length,100);
    }
    pending.forEach((resolve,i)=>resolve(Response.json([
      {product_id:'safe-product-'+(i*100),version:1,
       source_hash:'a'.repeat(32),image_url:'https://example.com/product-'+i+'.jpg',
       profile:{version:1}},
    ])));
    const result=await read;
    assert.deepEqual(result.map(x=>x.product_id),
      ['safe-product-0','safe-product-100','safe-product-200','safe-product-300'],
      'result ordering and source-hash evidence remain stable');
    await assert.rejects(io.fetchVisualProfiles(Array.from({length:401},(_,i)=>'id-'+i)),
      /stylist6_visual_hydration_limit_exceeded/);
  }finally{globalThis.fetch=prior}
});

test('fresh visual image colors exclude otherwise unknown-colored merchant cardigan from personal palette', () => {
  const row={...shopRow('grey-photo-no-color-title','outerwear'),
    name:'100% WOOL REGULAR-FIT CARDIGAN',description:'Wool cardigan. Plain product photo',
    source:'trusted-merchant'};
  const hash=sourceHash(row);
  const profile={version:1,model:'google/siglip2-base-patch16-224',revision:'visual-test-r1',
    imageUrl:row.image_url,sourceHash:hash,imageSha256:'a'.repeat(64),
    attributes:{color:{value:'grey',confidence:0.92,margin:0.65,evidence:'image-model'}}};
  const rows=[row];
  const hydration=hydrateProfiles(rows,[{product_id:row.id,version:1,
    source_hash:hash,image_url:row.image_url,profile}]);
  assert.deepEqual(hydration,{fetched:1,attached:1,drifted:0});
  const mapped=adaptCatalog(rows).items;
  assert.equal(mapped.length,1);
  assert.equal(mapped[0].color,'unknown','merchant title alone does not tell the color');
  const descriptor=describeGarment(mapped[0]);
  assert.equal(descriptor.color,'grey');
  assert.ok(descriptor.evidence.some(x=>x.startsWith('image:color:')));
  const warm=stylingContext({style:'semi_classic',occasion:'Date night',
    sourceMode:'catalog',preferences:{avoidColors:['grey','navy']}});
  assert.equal(garmentContextReason(mapped[0],warm),'excluded_color',
    'photographic evidence must override missing merchant text');
  const other=stylingContext({style:'semi_classic',occasion:'Date night',
    sourceMode:'catalog',preferences:{avoidColors:['beige','brown']}});
  assert.notEqual(garmentContextReason(mapped[0],other),'excluded_color');
  const stale=[{...row}];
  const invalid=hydrateProfiles(stale,[{product_id:row.id,version:1,
    source_hash:'f'.repeat(32),image_url:row.image_url,profile}]);
  assert.deepEqual(invalid,{fetched:1,attached:0,drifted:1},
    'stale photo metadata must not bypass catalog image/revision checks');
});


test('server-owned catalog discovery retrieves occasion-appropriate types before 100-SKU slot caps',()=>{
  const office=catalogStylingDiscoveryHints({style:'classic',occasion:'Work',
    brief:'Professional polished classic office tailoring'});
  assert.equal(office.dressCode,'smart');
  for(const type of ['dress-shirt','blazer','tailored-trouser','oxford','derby'])
    assert.ok(office.preferredSubtypes.includes(type));
  assert.ok(!office.preferredSubtypes.includes('t-shirt'));
  assert.ok(!office.preferredSubtypes.includes('hoodie'));
  const date=catalogStylingDiscoveryHints({style:'semi_classic',occasion:'Date night',
    brief:'Refined semi-classic outfit for a nice dinner date'});
  assert.equal(date.dressCode,'smart');
  for(const type of ['fine-knit','polo','blazer','cardigan','tailored-trouser','loafer'])
    assert.ok(date.preferredSubtypes.includes(type));
  assert.ok(!date.preferredSubtypes.includes('athletic-trouser'));
  const oldMoney=catalogStylingDiscoveryHints({style:'old_money',occasion:'Weekend'});
  for(const type of ['cardigan','overshirt','chino','loafer'])
    assert.ok(oldMoney.preferredSubtypes.includes(type));
  const cold=catalogStylingDiscoveryHints({style:'old_money',occasion:'Weekend',weather:{temp:-2}});
  assert.deepEqual(cold.preferredSubtypes.slice(0,3),['coat','boot','chelsea']);
  const casual=catalogStylingDiscoveryHints({style:'semi_classic',occasion:'Date night',
    brief:'Relaxed casual walk, not a formal date'});
  assert.equal(casual.dressCode,'casual');
  const formal=catalogStylingDiscoveryHints({style:'classic',occasion:'Wedding'});
  assert.equal(formal.dressCode,'formal');
  for(const subtype of ['dress-shirt','blazer','tailored-trouser','oxford','derby','monk'])
    assert.ok(formal.preferredSubtypes.includes(subtype));
  for(const subtype of ['loafer','chelsea','minimal-sneaker','coat'])
    assert.ok(!formal.preferredSubtypes.includes(subtype),
      'formal source relevance prefers dress footwear, not the generic office mix');
  const formalNoWedding=catalogStylingDiscoveryHints({style:'classic',occasion:'Formal'});
  assert.deepEqual(formalNoWedding,formal,
    'plain Formal and Wedding should source the same verified formal roles');
  assert.deepEqual(catalogStylingDiscoveryHints({style:'casual',occasion:'Weekend'}),{});
  assert.deepEqual(catalogStylingDiscoveryHints({style:'classic',occasion:'Workout'}),{});
  assert.ok(!JSON.stringify(date).includes('Refined semi-classic outfit'),
    'free-form user brief must never appear in trusted database SQL context');
});

test('authenticated stylist forwards fixed event/subtype hints to Supabase discovery, ignores client taxonomy injection',async()=>{
  let captured=null;
  await signedCall({request:{...baseRequest,sourceMode:'catalog',
    style:'semi_classic',occasion:'Date night',brief:'Refined dinner date',
    preferredSubtypes:['t-shirt','sneaker','arbitrary-client-SQL-text'],
    preferences:{avoidColors:['navy','grey']},
  }},{},{onShopSeed:(_seed,context)=>{captured=context;}});
  assert.ok(captured && captured.dressCode==='smart');
  assert.ok(captured.preferredSubtypes.includes('tailored-trouser'));
  assert.ok(captured.preferredSubtypes.includes('blazer'));
  assert.ok(!captured.preferredSubtypes.includes('t-shirt'));
  assert.ok(!captured.preferredSubtypes.includes('sneaker'));
  assert.ok(!JSON.stringify(captured).includes('arbitrary-client-SQL-text'));
});

test('live AI slate budget reserves visual photo review while no longer demanding 22.5 seconds',()=>{
  assert.equal(fashionSlateDecisionBudgetMs(NaN),0);
  assert.equal(fashionSlateDecisionBudgetMs(8900),0);
  assert.equal(fashionSlateDecisionBudgetMs(11499),0);
  assert.equal(fashionSlateDecisionBudgetMs(11500),2500);
  assert.equal(fashionSlateDecisionBudgetMs(12000),2600);
  assert.equal(fashionSlateDecisionBudgetMs(18000),2600);
  assert.equal(fashionSlateDecisionBudgetMs(22500),2600);
  assert.equal(fashionSlateDecisionBudgetMs(49000),2600);
});

test('workout SKU-only Fashion Director is independent from freeform creative planner',()=>{
  const enabled={STYLIST6_CREATIVE_DIRECTOR_ENABLED:'1',STYLIST6_AI_SLATE_ENABLED:'1'};
  assert.equal(serverFashionSlateEnabled(enabled,{
    visualAsked:true,visualCriticAvailable:true,
  }),true,'Sport can choose already feasible training outfits before costly photo review');
  for(const disabled of [
    {...enabled,STYLIST6_CREATIVE_DIRECTOR_ENABLED:''},
    {...enabled,STYLIST6_AI_SLATE_ENABLED:''},
  ])assert.equal(serverFashionSlateEnabled(disabled,{
    visualAsked:true,visualCriticAvailable:true,
  }),false,'server-owned switches must fail closed');
  assert.equal(serverFashionSlateEnabled(enabled,{
    visualAsked:false,visualCriticAvailable:true,
  }),false,'client did not request paid vision');
  assert.equal(serverFashionSlateEnabled(enabled,{
    visualAsked:true,visualCriticAvailable:false,
  }),false,'selector cannot pretend the actual-photo critic is configured');
});


test('media prewarm starts all 32 allowlisted SKU photos together, never serial six-image waves',async()=>{
  const urls=Array.from({length:32},(_,i)=>
    'https://static.zara.net/assets/product-'+i+'.jpg');
  let active=0,peak=0,resolved=0;
  const pending=[];
  const fetchPhoto=async()=>{
    active++;peak=Math.max(active,peak);
    await new Promise(resolve=>pending.push(resolve));
    active--;resolved++;
  };
  const operation=prewarmProductPhotos(urls,fetchPhoto,{remainingMs:()=>50000});
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(peak,32,'all 32 safe CDN requests should start without serial barriers');
  pending.splice(0).forEach(resolve=>resolve());
  const result=await operation;
  assert.equal(result.attempted,32);
  assert.equal(result.completed,32);
  assert.equal(resolved,32);
});

test('media prewarm only reaches inference-authorized hosts, is bounded and cannot fake an approval',async()=>{
  const urls=[
    'https://static.zara.net/product.jpg','https://static.zara.net/product.jpg',
    'http://127.0.0.1/secret.jpg','https://evil.example/photo.jpg',
  ];
  const visited=[];
  const result=await prewarmProductPhotos(urls,async({imageUrl})=>{
    visited.push(imageUrl);throw new Error('cdn failure');
  },{remainingMs:()=>6000});
  assert.equal(result.attempted,1,'duplicate/forbidden URLs never enter the prewarm wave');
  assert.equal(result.completed,0,'failed media is never accepted as a successful image');
  assert.deepEqual(visited,['https://static.zara.net/product.jpg']);
  assert.deepEqual(await prewarmProductPhotos(urls,async()=>{},{
    remainingMs:()=>1000}),{attempted:0,completed:0});
});


test('regression: authenticated S6 critic callback REALLY uses concurrent photo prewarm',()=>{
  const source=readFileSync(new URL('./cloud_handler.mjs',import.meta.url),'utf8');
  const handler=source.slice(source.indexOf('visualCritic = async (candidates, critiqueRequest'),
    source.indexOf('// Transmission gate, separate from download eligibility'));
  assert.match(handler,/await prewarmProductPhotos\(urls, resolveProduct,/,
    'prewarm helper MUST be wired into the production S6 visualCritic closure');
  assert.doesNotMatch(handler,/urls\.slice\(i, i \+ 6\)|for\s*\(let i\s*=\s*0;\s*i\s*<\s*urls\.length/,
    'never regress to serial 6-CDN waves that starve NVIDIA of its deadline');
});


test('safe Stage 6 quality trace includes only bounded status counters and reason codes',()=>{
  const out=stylist6SafeQualityTrace({
    personalSecret:'private-identity',request:'disclose this',
    modelFashionSelection:{status:'selected',formalPhotoProbe:true,
      uniqueTripleAvailable:false,chosen:5,customerEmail:'test@example.com'},
    visualReviewCoverage:{mode:'displayed_deck',attempted:8,
      approved:4,displayedEvidence:2,rawPhoto:'super-secret-raw-bytes'},
    visualQualityDecisions:[
      {reason:'approved',title:'private coat'},
      {reason:'style_mismatch',prompt:'private query'},
      {reason:'<malicious user text with contact info>'},
    ],
  },2,48888);
  assert.deepEqual(out,{
    elapsedMs:48888,displayed:2,requestedPhoto:true,
    attempted:8,approved:4,displayedEvidence:2,
    directorSelected:true,directSkuComposition:false,
    skuCompositionStatus:'not_run',skuCompositionProposed:null,
    skuCompositionAdmitted:null,skuCompositionExploration:null,
    photoRevisionStatus:'not_run',photoRevisionAttempted:null,
    photoRevisionApproved:null,
    photoEditorialStatus:'not_run',photoEditorialChosen:null,
    photoEditorialEligibleTriples:null,photoEditorialEligiblePairs:null,
    formalPhotoProbe:true,prePhotoUniqueTriple:false,
    reasons:['approved','style_mismatch','other'],
  });
  assert.doesNotMatch(JSON.stringify(out),/secret|example|private|contact info/);
  assert.equal(stylist6SafeQualityTrace(null,5,-1).displayed,null,
    'trace cannot invent out-of-range deck counts');
});
