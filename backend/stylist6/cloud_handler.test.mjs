import test from 'node:test';
import assert from 'node:assert/strict';
import { signStylistTransport } from '../stylist-node/transport/signature.mjs';
import {
  handleStylist6CloudGenerate,
  mapCloudVisualStatus,
  projectWardrobeItem,
  fitVisualBytesCloud,
  STYLIST6_CLOUD_VERSION,
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
  const calls = { verifyUser: 0, admitQuota: 0, fetchShopRows: 0, fetchWardrobeRows: 0, fetchVisualProfiles: 0 };
  return {
    calls,
    io: {
      async verifyUser(jwt) {
        calls.verifyUser += 1;
        if (jwt !== JWT) throw new Error('stylist6_unauthorized');
        return USER;
      },
      async admitQuota(userId) {
        calls.admitQuota += 1;
        assert.equal(userId, USER);
        if (overrides.quotaDeny) {
          const denied = new Error('denied');
          denied.code = overrides.quotaDeny;
          throw denied;
        }
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

const env = { STYLIST_NODE_SHARED_SECRET: SECRET, SUPABASE_URL: 'https://example.supabase.co' };

async function signedCall(request, { secret = SECRET, jwt = JWT, method = 'POST' } = {}, ioOverride) {
  const { calls, io } = makeIo(ioOverride);
  const body = JSON.stringify(request);
  const authorization = `Bearer ${jwt}`;
  const signature = await signStylistTransport({ secret, body, authorization });
  const headers = { authorization, ...signature };
  const result = await handleStylist6CloudGenerate({ method, headers, bodyText: body, env, io });
  return { ...result, calls };
}

const baseRequest = {
  style: 'old_money', occasion: 'Work', brief: 'work, old money',
  sourceMode: 'catalog', preferences: {}, recentSignatures: [], excludeItemIds: [],
};

test('missing transport secret fails closed without touching auth or quota', async () => {
  const { calls, io } = makeIo();
  const result = await handleStylist6CloudGenerate({
    method: 'POST', headers: {}, bodyText: '{}',
    env: { STYLIST_NODE_SHARED_SECRET: '' }, io,
  });
  assert.equal(result.status, 503);
  assert.equal(result.body.code, 'stylist_compute_unconfigured');
  assert.equal(calls.verifyUser, 0);
  assert.equal(calls.admitQuota, 0);
});

test('method, size and HMAC admission cost no quota', async () => {
  const probe = await makeIo();
  const badSecret = await signedCall({ request: baseRequest }, { secret: 'wrong-secret-000000000000000000000001' });
  assert.equal(badSecret.status, 401);
  assert.equal(badSecret.calls.admitQuota, 0);
  const get = await handleStylist6CloudGenerate({ method: 'GET', headers: {}, bodyText: '', env, io: probe.io });
  assert.equal(get.status, 405);
  const big = await handleStylist6CloudGenerate({
    method: 'POST', headers: {}, bodyText: 'x'.repeat(1024 * 1024 + 1), env, io: probe.io,
  });
  assert.equal(big.status, 413);
  assert.equal(probe.calls.admitQuota, 0);
});

test('invalid JWT fails before inventory and quota', async () => {
  const result = await signedCall({ request: baseRequest }, { jwt: 'bad-token' });
  assert.equal(result.status, 401);
  assert.equal(result.calls.verifyUser, 1);
  assert.equal(result.calls.admitQuota, 0);
  assert.equal(result.calls.fetchShopRows, 0);
});

test('quota denial stops before inventory (exactly-once on success)', async () => {
  const denied = await signedCall({ request: baseRequest }, {}, { quotaDeny: 'feature_limit_reached' });
  assert.equal(denied.status, 403);
  assert.equal(denied.calls.fetchShopRows, 0);
  const limited = await signedCall({ request: baseRequest }, {}, { quotaDeny: 'rate_limited' });
  assert.equal(limited.status, 429);
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
  assert.equal(result.calls.admitQuota, 1);
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
  assert.equal(result.calls.fetchShopRows, 0);
});

test('mixed mode draws from both sources', async () => {
  const result = await signedCall({ request: { ...baseRequest, sourceMode: 'mixed' } });
  assert.equal(result.status, 200);
  const items = result.body.outfits.flatMap((outfit) => outfit.items);
  assert.ok(items.some((item) => item.isShopItem === true));
  assert.ok(items.some((item) => item.isShopItem === false));
});

test('missing locked anchor fails 422, present anchor survives in every outfit', async () => {
  const missing = await signedCall({ request: { ...baseRequest, requiredIds: ['no-such-sku'] } });
  assert.equal(missing.status, 422);
  assert.equal(missing.body.code, 'anchor_not_found');
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
