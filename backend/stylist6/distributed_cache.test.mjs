import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createStylistCache,
  buildCreativeDirectorCacheKey,
  buildVisualVerdictCacheKey,
  buildAppearanceCacheKey,
} from './distributed_cache.mjs';

test('L1 LRU enforces BOTH maxMemoryEntries and maxMemoryBytes so photo buffers can never OOM', async () => {
  // Entry-count bound test
  const entryBounded = createStylistCache({
    maxMemoryEntries: 3,
    maxMemoryBytes: 1024 * 1024,
    namespace: 'photos',
  });
  await entryBounded.set('a', 'v1');
  await entryBounded.set('b', 'v2');
  await entryBounded.set('c', 'v3');
  // Touch 'a' so 'b' becomes least-recently-used
  assert.equal(await entryBounded.get('a'), 'v1');
  await entryBounded.set('d', 'v4');

  assert.equal(await entryBounded.get('b'), undefined, 'LRU entry b should be evicted');
  assert.equal(await entryBounded.get('a'), 'v1');
  assert.equal(await entryBounded.get('c'), 'v3');
  assert.equal(await entryBounded.get('d'), 'v4');
  assert.equal(entryBounded.stats().evictions, 1);

  // Byte-size bound test (e.g., 3 x 1024B buffers inside 2500B cap -> max 2 fit!)
  const byteBounded = createStylistCache({
    maxMemoryEntries: 100,
    maxMemoryBytes: 2500,
    namespace: 'photo-bytes',
  });
  const buf1 = Buffer.alloc(1024, 0xaa);
  const buf2 = Buffer.alloc(1024, 0xbb);
  const buf3 = Buffer.alloc(1024, 0xcc);
  const giantBuf = Buffer.alloc(5000, 0xff); // exceeds entire cap!

  await byteBounded.set('img1', buf1);
  await byteBounded.set('img2', buf2);
  assert.equal(byteBounded.stats().l1Bytes, 2048);

  await byteBounded.set('img3', buf3);
  assert.ok(byteBounded.stats().l1Bytes <= 2500, 'l1Bytes must stay strictly <= maxMemoryBytes');
  assert.equal(await byteBounded.get('img1'), undefined, 'oldest buffer evicted to respect byte cap');
  assert.ok(Buffer.isBuffer(await byteBounded.get('img3')));

  // Single oversized payload is skipped safely without wiping existing healthy entries
  await byteBounded.set('giant', giantBuf);
  assert.equal(await byteBounded.get('giant'), undefined);
  assert.equal(byteBounded.stats().oversizedSkips, 1);
  assert.ok(Buffer.isBuffer(await byteBounded.get('img3')), 'existing entries preserved');
});

test('L1 respects deterministic TTL expiration and del()', async () => {
  let now = 1_700_000_000_000;
  const cache = createStylistCache({
    defaultTtlMs: 5000,
    nowFn: () => now,
  });

  await cache.set('k1', { hello: 'world' }, 5000);
  assert.deepEqual(await cache.get('k1'), { hello: 'world' });

  now += 4999;
  assert.deepEqual(await cache.get('k1'), { hello: 'world' });

  now += 2;
  assert.equal(await cache.get('k1'), undefined, 'expired entry must return undefined');

  await cache.set('k2', 'keep');
  await cache.del('k2');
  assert.equal(await cache.get('k2'), undefined);
});

test('multi-tier L2 Upstash Redis REST + L3 Supabase fallback with backfill and Buffer support', async () => {
  const redisStore = new Map();
  const supabaseStore = new Map();

  const mockFetch = async (url, init) => {
    assert.equal(url, 'https://mock-upstash.example.com');
    assert.equal(init.headers.Authorization, 'Bearer test-token');
    const [cmd, key, val] = JSON.parse(init.body);
    if (cmd === 'GET') {
      return { ok: true, async json() { return { result: redisStore.has(key) ? redisStore.get(key) : null }; } };
    }
    if (cmd === 'SET') {
      redisStore.set(key, val);
      return { ok: true, async json() { return { result: 'OK' }; } };
    }
    if (cmd === 'DEL') {
      redisStore.delete(key);
      return { ok: true, async json() { return { result: 1 }; } };
    }
    throw new Error(`unexpected redis cmd ${cmd}`);
  };

  const mockSupabase = {
    from(table) {
      assert.equal(table, 'stylist6_cache_entries');
      return {
        select() {
          return {
            eq(col, val) {
              assert.equal(col, 'cache_key');
              return {
                async maybeSingle() {
                  return { data: supabaseStore.get(val) || null, error: null };
                },
              };
            },
          };
        },
        async upsert(row) {
          supabaseStore.set(row.cache_key, row);
          return { error: null };
        },
        delete() {
          return {
            async eq(col, val) {
              supabaseStore.delete(val);
              return { error: null };
            },
          };
        },
      };
    },
  };

  // Instance A writes a Buffer and a JSON object through L1 + L2 + L3
  const instanceA = createStylistCache({
    redisUrl: 'https://mock-upstash.example.com',
    redisToken: 'test-token',
    supabaseClient: mockSupabase,
    fetchFn: mockFetch,
    namespace: 's6',
  });

  const sampleJpeg = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0x01, 0x02]);
  await instanceA.set('board-1', sampleJpeg, 60_000);
  await instanceA.set('plan-1', { directions: ['signature', 'refresh', 'discovery'] }, 60_000);

  // Instance B has cold L1 -> hits L2 Upstash and backfills L1!
  const instanceB = createStylistCache({
    redisUrl: 'https://mock-upstash.example.com',
    redisToken: 'test-token',
    supabaseClient: mockSupabase,
    fetchFn: mockFetch,
    namespace: 's6',
  });
  const gotBuf = await instanceB.get('board-1');
  assert.ok(Buffer.isBuffer(gotBuf));
  assert.deepEqual(gotBuf, sampleJpeg);
  assert.equal(instanceB.stats().hits.l2, 1);

  // Second read on Instance B hits L1 directly (0 extra L2 calls)
  const gotBuf2 = await instanceB.get('board-1');
  assert.deepEqual(gotBuf2, sampleJpeg);
  assert.equal(instanceB.stats().hits.l1, 1);
  assert.equal(instanceB.stats().hits.l2, 1);

  // Clear Redis to simulate L2 miss -> Instance C falls through to L3 Supabase!
  redisStore.clear();
  const instanceC = createStylistCache({
    redisUrl: 'https://mock-upstash.example.com',
    redisToken: 'test-token',
    supabaseClient: mockSupabase,
    fetchFn: mockFetch,
    namespace: 's6',
  });
  const gotPlan = await instanceC.get('plan-1');
  assert.deepEqual(gotPlan, { directions: ['signature', 'refresh', 'discovery'] });
  assert.equal(instanceC.stats().hits.l3, 1);
});

test('L2 and L3 failures fail open safely without crashing callers', async () => {
  const failingFetch = async () => { throw new Error('redis_network_down'); };
  const failingSupabase = {
    from() {
      return {
        select() {
          return { eq() { return { async maybeSingle() { return { data: null, error: { message: 'db_timeout' } }; } }; } };
        },
        async upsert() { return { error: { message: 'db_readonly' } }; },
        delete() { return { async eq() { return { error: { message: 'db_readonly' } }; } }; },
      };
    },
  };

  const cache = createStylistCache({
    redisUrl: 'https://down.example.com',
    redisToken: 'tok',
    supabaseClient: failingSupabase,
    fetchFn: failingFetch,
  });

  assert.equal(await cache.get('missing'), undefined);
  assert.equal(cache.stats().errors.l2, 1);
  assert.equal(cache.stats().errors.l3, 1);

  // Write still populates L1 cleanly despite L2/L3 failures!
  await cache.set('resilient', { ok: true });
  assert.deepEqual(await cache.get('resilient'), { ok: true });
  assert.equal(cache.stats().hits.l1, 1);
});

test('getOrSet coalesces concurrent cold stampedes (singleflight) and respects skipCacheIf', async () => {
  const cache = createStylistCache();
  let calls = 0;

  const compute = async () => {
    calls += 1;
    await new Promise((r) => setTimeout(r, 15));
    return { planId: 'dir-42', computedOnce: true };
  };

  const results = await Promise.all(
    Array.from({ length: 25 }, () => cache.getOrSet('hot-key', compute)),
  );

  assert.equal(calls, 1, '25 concurrent cold requests must invoke computeFn exactly ONCE');
  for (const res of results) {
    assert.deepEqual(res, { planId: 'dir-42', computedOnce: true });
  }
  assert.equal(cache.stats().coalescedHits, 24);

  // skipCacheIf prevents caching transient failure results
  let retryCalls = 0;
  const r1 = await cache.getOrSet(
    'transient',
    async () => ({ ok: false, attempt: ++retryCalls }),
    { skipCacheIf: (v) => !v?.ok },
  );
  const r2 = await cache.getOrSet(
    'transient',
    async () => ({ ok: true, attempt: ++retryCalls }),
    { skipCacheIf: (v) => !v?.ok },
  );
  const r3 = await cache.getOrSet(
    'transient',
    async () => ({ ok: true, attempt: ++retryCalls }),
  );
  assert.equal(r1.ok, false);
  assert.equal(r2.ok, true);
  assert.equal(r2.attempt, 2);
  assert.equal(r3.attempt, 2, 'successful result was cached');
});

test('canonical deterministic cache key builders are order-independent, sensitive to changes, and zero-PII', () => {
  // Creative Director key: order-independent DNA objects & temperature bucketing
  const cd1 = buildCreativeDirectorCacheKey({
    presentation: 'menswear',
    style: 'Classic',
    occasion: 'Work',
    temperatureC: 17,
    budget: 500,
    currency: 'USD',
    dnaSignature: { undertone: 'warm', contrast: 'medium' },
  });
  const cd2 = buildCreativeDirectorCacheKey({
    presentation: 'MENSWEAR',
    style: ' classic ',
    occasion: 'work',
    temperatureC: 16, // same 4°C bucket (round(17/4)*4 = 16, round(16/4)*4 = 16)
    budget: 500,
    currency: 'USD',
    dnaSignature: { contrast: 'medium', undertone: 'warm' }, // reversed keys
  });
  const cdWomens = buildCreativeDirectorCacheKey({
    presentation: 'womenswear',
    style: 'classic',
    occasion: 'work',
    temperatureC: 16,
    budget: 500,
    currency: 'USD',
    dnaSignature: { undertone: 'warm', contrast: 'medium' },
  });
  assert.equal(cd1, cd2);
  assert.notEqual(cd1, cdWomens);
  assert.match(cd1, /^cd:[a-f0-9]{40}$/);

  // Visual Verdict key: item order independent, but invalidated when ANY imageSha256 changes!
  const itemA = { id: 'sku-1', imageSha256: '1'.repeat(64) };
  const itemB = { id: 'sku-2', imageSha256: '2'.repeat(64) };
  const itemBChangedPhoto = { id: 'sku-2', imageSha256: '3'.repeat(64) };

  const vv1 = buildVisualVerdictCacheKey({ items: [itemA, itemB], occasion: 'date', modelId: 'kimi-k3.1' });
  const vv2 = buildVisualVerdictCacheKey({ items: [itemB, itemA], occasion: 'Date', modelId: 'kimi-k3.1' });
  const vvChanged = buildVisualVerdictCacheKey({ items: [itemA, itemBChangedPhoto], occasion: 'date', modelId: 'kimi-k3.1' });
  assert.equal(vv1, vv2);
  assert.notEqual(vv1, vvChanged, 'changed product photo hash MUST invalidate verdict key');
  assert.match(vv1, /^vv:[a-f0-9]{40}$/);

  // Appearance key: strictly content-addressed by 64-hex photoSha256
  const validSha = 'a'.repeat(64);
  const ap1 = buildAppearanceCacheKey({ photoSha256: validSha, analyzerVersion: 'v1', modelId: 'gemini-2.5' });
  const ap2 = buildAppearanceCacheKey({ photoSha256: validSha.toUpperCase(), analyzerVersion: 'v1', modelId: 'gemini-2.5' });
  assert.equal(ap1, ap2);
  assert.match(ap1, /^ap:[a-f0-9]{40}$/);
  assert.throws(() => buildAppearanceCacheKey({ photoSha256: 'not-a-sha' }), TypeError);
});
