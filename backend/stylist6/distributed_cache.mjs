/**
 * Stylist 6 — Multi-Tier Distributed Stateless Cache.
 *
 * Three-tier cache architecture designed for 1M+ MAU horizontal scaling with
 * zero additional npm dependencies:
 * - L1: Bounded in-memory LRU enforcing BOTH `maxMemoryEntries` AND
 *       `maxMemoryBytes` so decoded product photo buffers can NEVER OOM a node.
 * - L2: Optional Upstash Redis REST over pure `fetch` (sub-5ms shared cluster
 *       cache across stateless Render / Fly / Cloud Run instances).
 * - L3: Optional Supabase `public.stylist6_cache_entries` table fallback
 *       (supports both Supabase JS client `.from(...)` and direct PostgREST
 *       `{ url, serviceKey, fetchFn }` config).
 * - Singleflight (`getOrSet`) coalesces concurrent cold-cache stampedes for
 *   identical keys into a single provider computation.
 *
 * All L2/L3 errors fail open safely (tracked via `stats().errors`) so an
 * external cache outage can never break outfit generation.
 */

import { createHash } from 'node:crypto';

export const DISTRIBUTED_CACHE_VERSION = 's6-distributed-cache-v1';
export const DEFAULT_MAX_MEMORY_ENTRIES = 1000;
export const DEFAULT_MAX_MEMORY_BYTES = 32 * 1024 * 1024; // 32 MiB hard cap
export const DEFAULT_TTL_MS = 15 * 60 * 1000; // 15 minutes
const DEFAULT_L2_TIMEOUT_MS = 1500;
const MAX_KEY_LENGTH = 256;

const BUFFER_SENTINEL = '__aiw_buf_b64_v1__';

function sha256Hex(text) {
  return createHash('sha256').update(String(text)).digest('hex');
}

function normStr(value, maxLen = 64) {
  if (typeof value !== 'string') return '';
  return value.normalize('NFKC').trim().toLowerCase().replace(/\s+/g, ' ').slice(0, maxLen);
}

function stableNormalize(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === 'boolean' || typeof value === 'string') return value;
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (Array.isArray(value)) return value.map(stableNormalize);
  if (typeof value === 'object') {
    const out = {};
    for (const key of Object.keys(value).sort()) {
      const normalized = stableNormalize(value[key]);
      if (normalized !== undefined) out[key] = normalized;
    }
    return out;
  }
  return null;
}

function stableJson(value) {
  return JSON.stringify(stableNormalize(value));
}

function estimateByteSize(value) {
  if (value === null || value === undefined) return 8;
  if (Buffer.isBuffer(value)) return value.byteLength;
  if (ArrayBuffer.isView(value)) return value.byteLength;
  if (typeof value === 'string') return Buffer.byteLength(value, 'utf8');
  if (typeof value === 'number' || typeof value === 'boolean') return 8;
  try {
    return Buffer.byteLength(JSON.stringify(value), 'utf8');
  } catch {
    return 1024;
  }
}

function encodeWireValue(value) {
  if (Buffer.isBuffer(value)) {
    return { [BUFFER_SENTINEL]: value.toString('base64') };
  }
  if (value instanceof Uint8Array) {
    return { [BUFFER_SENTINEL]: Buffer.from(value.buffer, value.byteOffset, value.byteLength).toString('base64') };
  }
  return value;
}

function decodeWireValue(raw) {
  if (raw && typeof raw === 'object' && !Array.isArray(raw) && typeof raw[BUFFER_SENTINEL] === 'string') {
    return Buffer.from(raw[BUFFER_SENTINEL], 'base64');
  }
  return raw;
}

function buildNamespacedKey(namespace, key) {
  const ns = normStr(namespace || 'stylist6', 48).replace(/[^a-z0-9:_-]/g, '_') || 'stylist6';
  const rawKey = String(key || '').trim();
  const combined = `${ns}:${rawKey}`;
  if (combined.length <= MAX_KEY_LENGTH) return combined;
  return `${ns}:h:${sha256Hex(rawKey)}`;
}

// ---------------------------------------------------------------------------
// Canonical deterministic cache key builders (zero PII, order-independent)
// ---------------------------------------------------------------------------

function bucketTemperature(tempC) {
  const n = Number(tempC);
  if (!Number.isFinite(n)) return 'temp_unknown';
  // 4°C buckets preserve thermal accuracy while maximizing cross-user cache reuse
  const bucket = Math.round(n / 4) * 4;
  return `t${bucket}c`;
}

function bucketBudget(budget, currency) {
  const n = Number(budget);
  if (!Number.isFinite(n) || n <= 0) return 'unbounded';
  const cur = typeof currency === 'string' && /^[A-Z]{3}$/.test(currency.toUpperCase())
    ? currency.toUpperCase()
    : 'USD';
  // Round budget into coarse ~15% logarithmic tiers for clean cache reuse
  const step = n < 100 ? 25 : n < 500 ? 50 : n < 2000 ? 150 : Math.pow(10, Math.floor(Math.log10(n)) - 1);
  const rounded = Math.round(n / step) * step;
  return `${cur}_${rounded}`;
}

/**
 * Deterministic key for Creative Director 3-direction fashion blueprints.
 * Excludes all user PII; hashes normalized style context, presentation,
 * weather/budget buckets, and optional anonymous DNA/taste signatures.
 */
export function buildCreativeDirectorCacheKey({
  version = 'v1',
  modelId = 'default',
  presentation = 'menswear',
  style = 'classic',
  occasion = 'casual',
  weather = null,
  temperatureC = null,
  season = null,
  budget = null,
  currency = null,
  outfitGrammar = 'layered_separates',
  dnaSignature = null,
  tasteSignature = null,
  language = 'en',
  catalogEpoch = null,
} = {}) {
  const tempVal = temperatureC ?? weather?.temperatureC ?? weather?.temp ?? null;
  const condition = normStr(weather?.condition || weather?.summary || '', 32);
  const payload = {
    v: normStr(version, 32),
    m: normStr(modelId, 64),
    p: normStr(presentation, 16) || 'menswear',
    st: normStr(style, 48),
    oc: normStr(occasion, 48),
    tb: bucketTemperature(tempVal),
    wc: condition,
    se: normStr(season, 16),
    bg: bucketBudget(budget, currency),
    gr: normStr(outfitGrammar, 32),
    dna: typeof dnaSignature === 'string' ? dnaSignature.slice(0, 128) : dnaSignature ? sha256Hex(stableJson(dnaSignature)).slice(0, 24) : '',
    ts: typeof tasteSignature === 'string' ? tasteSignature.slice(0, 128) : tasteSignature ? sha256Hex(stableJson(tasteSignature)).slice(0, 24) : '',
    lg: normStr(language, 8) || 'en',
    ce: normStr(catalogEpoch, 32),
  };
  return `cd:${sha256Hex(stableJson(payload)).slice(0, 40)}`;
}

/**
 * Deterministic key for Visual Critic full-outfit verdicts.
 * Bound strictly to exact sorted SKU IDs AND exact image SHA-256 hashes (or
 * URLs) so a changed product photograph ALWAYS invalidates the cached verdict.
 */
export function buildVisualVerdictCacheKey({
  criticVersion = 'v1',
  modelId = 'default',
  items = [],
  skuIds = null,
  imageHashes = null,
  occasion = 'casual',
  style = 'classic',
  presentation = 'menswear',
  temperatureC = null,
  colorSeason = null,
} = {}) {
  const itemTuples = [];
  if (Array.isArray(items) && items.length > 0) {
    for (const item of items) {
      if (!item || typeof item !== 'object') continue;
      const id = String(item.id || item.productId || item.product_id || '').trim().slice(0, 128);
      const imgHash = String(
        item.imageSha256 || item.image_sha256 || item.visual_image_sha256 || item.imageUrl || item.image_url || '',
      ).trim().slice(0, 256);
      itemTuples.push(`${id}|${imgHash}`);
    }
  } else {
    const ids = Array.isArray(skuIds) ? skuIds.map((x) => String(x || '').trim()) : [];
    const hashes = Array.isArray(imageHashes) ? imageHashes.map((x) => String(x || '').trim()) : [];
    const len = Math.max(ids.length, hashes.length);
    for (let i = 0; i < len; i++) {
      itemTuples.push(`${ids[i] || ''}|${hashes[i] || ''}`);
    }
  }
  itemTuples.sort();
  const payload = {
    v: normStr(criticVersion, 32),
    m: normStr(modelId, 64),
    items: itemTuples,
    oc: normStr(occasion, 48),
    st: normStr(style, 48),
    p: normStr(presentation, 16) || 'menswear',
    tb: bucketTemperature(temperatureC),
    cs: normStr(colorSeason, 32),
  };
  return `vv:${sha256Hex(stableJson(payload)).slice(0, 40)}`;
}

/**
 * Strictly content-addressed cache key for opt-in User Appearance Analysis.
 * Never includes userId, email, or raw photo URL — only exact photo SHA-256
 * plus analyzer/model version.
 */
export function buildAppearanceCacheKey({
  photoSha256,
  analyzerVersion = 'v1',
  modelId = 'default',
} = {}) {
  const sha = typeof photoSha256 === 'string' ? photoSha256.trim().toLowerCase() : '';
  if (!/^[a-f0-9]{64}$/.test(sha)) {
    throw new TypeError('buildAppearanceCacheKey requires a valid 64-hex photoSha256');
  }
  const payload = {
    sha,
    v: normStr(analyzerVersion, 32),
    m: normStr(modelId, 64),
  };
  return `ap:${sha256Hex(stableJson(payload)).slice(0, 40)}`;
}

// ---------------------------------------------------------------------------
// L2 Upstash Redis REST adapter (pure fetch, zero extra dependencies)
// ---------------------------------------------------------------------------

function createUpstashAdapter({ redisUrl, redisToken, fetchFn, timeoutMs }) {
  if (!redisUrl || !redisToken) return null;
  const endpoint = String(redisUrl).replace(/\/+$/, '');
  const headers = {
    Authorization: `Bearer ${redisToken}`,
    'Content-Type': 'application/json',
  };
  const runCmd = async (commandArray) => {
    const response = await fetchFn(endpoint, {
      method: 'POST',
      headers,
      body: JSON.stringify(commandArray),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!response.ok) {
      throw new Error(`upstash_http_${response.status}`);
    }
    const body = await response.json();
    if (body && typeof body === 'object' && body.error) {
      throw new Error(`upstash_error_${body.error}`);
    }
    return body?.result ?? null;
  };
  return {
    async get(fullKey) {
      const raw = await runCmd(['GET', fullKey]);
      if (raw === null || raw === undefined) return { hit: false, value: undefined };
      const parsed = typeof raw === 'string' ? JSON.parse(raw) : raw;
      return { hit: true, value: decodeWireValue(parsed) };
    },
    async set(fullKey, value, ttlMs) {
      const serialized = JSON.stringify(encodeWireValue(value));
      const px = Math.max(1000, Math.floor(ttlMs));
      await runCmd(['SET', fullKey, serialized, 'PX', String(px)]);
    },
    async del(fullKey) {
      await runCmd(['DEL', fullKey]);
    },
  };
}

// ---------------------------------------------------------------------------
// L3 Supabase `stylist6_cache_entries` adapter
// ---------------------------------------------------------------------------

function createSupabaseCacheAdapter(supabaseClient, fetchFn, timeoutMs) {
  if (!supabaseClient) return null;

  // Option A: Supabase JS client (`supabase.from('stylist6_cache_entries')`)
  if (typeof supabaseClient.from === 'function') {
    return {
      async get(fullKey, nowMs) {
        const { data, error } = await supabaseClient
          .from('stylist6_cache_entries')
          .select('payload, expires_at')
          .eq('cache_key', fullKey)
          .maybeSingle();
        if (error) throw new Error(`supabase_cache_get_${error.message || error}`);
        if (!data) return { hit: false, value: undefined };
        const expMs = Date.parse(data.expires_at);
        if (!Number.isFinite(expMs) || expMs <= nowMs) {
          return { hit: false, value: undefined };
        }
        return { hit: true, value: decodeWireValue(data.payload), remainingTtlMs: expMs - nowMs };
      },
      async set(fullKey, namespace, value, ttlMs, nowMs) {
        const expiresAt = new Date(nowMs + ttlMs).toISOString();
        const { error } = await supabaseClient
          .from('stylist6_cache_entries')
          .upsert(
            {
              cache_key: fullKey,
              namespace,
              payload: encodeWireValue(value),
              expires_at: expiresAt,
              updated_at: new Date(nowMs).toISOString(),
            },
            { onConflict: 'cache_key' },
          );
        if (error) throw new Error(`supabase_cache_set_${error.message || error}`);
      },
      async del(fullKey) {
        const { error } = await supabaseClient
          .from('stylist6_cache_entries')
          .delete()
          .eq('cache_key', fullKey);
        if (error) throw new Error(`supabase_cache_del_${error.message || error}`);
      },
    };
  }

  // Option B: Lightweight PostgREST config `{ url, serviceKey }`
  if (typeof supabaseClient.url === 'string' && typeof supabaseClient.serviceKey === 'string') {
    const baseUrl = supabaseClient.url.replace(/\/+$/, '');
    const key = supabaseClient.serviceKey;
    const headers = {
      apikey: key,
      Authorization: `Bearer ${key}`,
      'Content-Type': 'application/json',
    };
    const restFetch = supabaseClient.fetchFn || fetchFn;
    return {
      async get(fullKey, nowMs) {
        const params = new URLSearchParams({
          select: 'payload,expires_at',
          cache_key: `eq.${fullKey}`,
          limit: '1',
        });
        const res = await restFetch(`${baseUrl}/rest/v1/stylist6_cache_entries?${params}`, {
          method: 'GET',
          headers,
          signal: AbortSignal.timeout(timeoutMs),
        });
        if (!res.ok) throw new Error(`supabase_rest_get_${res.status}`);
        const rows = await res.json();
        const row = Array.isArray(rows) ? rows[0] : null;
        if (!row) return { hit: false, value: undefined };
        const expMs = Date.parse(row.expires_at);
        if (!Number.isFinite(expMs) || expMs <= nowMs) {
          return { hit: false, value: undefined };
        }
        return { hit: true, value: decodeWireValue(row.payload), remainingTtlMs: expMs - nowMs };
      },
      async set(fullKey, namespace, value, ttlMs, nowMs) {
        const res = await restFetch(
          `${baseUrl}/rest/v1/stylist6_cache_entries?on_conflict=cache_key`,
          {
            method: 'POST',
            headers: { ...headers, Prefer: 'resolution=merge-duplicates,return=minimal' },
            body: JSON.stringify({
              cache_key: fullKey,
              namespace,
              payload: encodeWireValue(value),
              expires_at: new Date(nowMs + ttlMs).toISOString(),
              updated_at: new Date(nowMs).toISOString(),
            }),
            signal: AbortSignal.timeout(timeoutMs),
          },
        );
        if (!res.ok) throw new Error(`supabase_rest_set_${res.status}`);
      },
      async del(fullKey) {
        const params = new URLSearchParams({ cache_key: `eq.${fullKey}` });
        const res = await restFetch(`${baseUrl}/rest/v1/stylist6_cache_entries?${params}`, {
          method: 'DELETE',
          headers,
          signal: AbortSignal.timeout(timeoutMs),
        });
        if (!res.ok) throw new Error(`supabase_rest_del_${res.status}`);
      },
    };
  }

  return null;
}

// ---------------------------------------------------------------------------
// Main multi-tier cache factory
// ---------------------------------------------------------------------------

export function createStylistCache({
  redisUrl = null,
  redisToken = null,
  supabaseClient = null,
  maxMemoryEntries = DEFAULT_MAX_MEMORY_ENTRIES,
  maxMemoryBytes = DEFAULT_MAX_MEMORY_BYTES,
  namespace = 'stylist6',
  defaultTtlMs = DEFAULT_TTL_MS,
  l2TimeoutMs = DEFAULT_L2_TIMEOUT_MS,
  fetchFn = globalThis.fetch,
  nowFn = () => Date.now(),
} = {}) {
  const entryCap = Math.max(1, Math.floor(Number(maxMemoryEntries) || DEFAULT_MAX_MEMORY_ENTRIES));
  const byteCap = Math.max(1024, Math.floor(Number(maxMemoryBytes) || DEFAULT_MAX_MEMORY_BYTES));
  const ttlDefault = Math.max(1000, Math.floor(Number(defaultTtlMs) || DEFAULT_TTL_MS));
  const cleanNs = normStr(namespace, 48).replace(/[^a-z0-9:_-]/g, '_') || 'stylist6';

  const l1 = new Map(); // fullKey -> { value, bytes, expiresAt }
  let l1Bytes = 0;

  const l2 = createUpstashAdapter({ redisUrl, redisToken, fetchFn, timeoutMs: l2TimeoutMs });
  const l3 = createSupabaseCacheAdapter(supabaseClient, fetchFn, l2TimeoutMs);
  const inflight = new Map(); // fullKey -> Promise

  const counters = {
    l1Hits: 0,
    l2Hits: 0,
    l3Hits: 0,
    misses: 0,
    sets: 0,
    deletes: 0,
    evictions: 0,
    oversizedSkips: 0,
    coalescedHits: 0,
    l2Errors: 0,
    l3Errors: 0,
  };

  function l1Delete(fullKey) {
    const existing = l1.get(fullKey);
    if (!existing) return false;
    l1.delete(fullKey);
    l1Bytes = Math.max(0, l1Bytes - existing.bytes);
    return true;
  }

  function l1Set(fullKey, value, ttlMs) {
    const bytes = estimateByteSize(value);
    if (bytes > byteCap) {
      // Never evict the entire L1 cache or OOM for a single oversized payload
      counters.oversizedSkips += 1;
      l1Delete(fullKey);
      return false;
    }
    l1Delete(fullKey);
    const expiresAt = nowFn() + Math.max(1000, ttlMs);
    while (l1.size >= entryCap || l1Bytes + bytes > byteCap) {
      const oldestKey = l1.keys().next().value;
      if (oldestKey === undefined) break;
      l1Delete(oldestKey);
      counters.evictions += 1;
    }
    l1.set(fullKey, { value, bytes, expiresAt });
    l1Bytes += bytes;
    return true;
  }

  function l1Get(fullKey) {
    const entry = l1.get(fullKey);
    if (!entry) return { hit: false, value: undefined };
    if (entry.expiresAt <= nowFn()) {
      l1Delete(fullKey);
      return { hit: false, value: undefined };
    }
    // Refresh LRU recency (O(1) Map re-insertion at tail)
    l1.delete(fullKey);
    l1.set(fullKey, entry);
    return { hit: true, value: entry.value };
  }

  async function get(key) {
    if (!key) return undefined;
    const fullKey = buildNamespacedKey(cleanNs, key);

    // Tier 1: Bounded in-memory LRU
    const t1 = l1Get(fullKey);
    if (t1.hit) {
      counters.l1Hits += 1;
      return t1.value;
    }

    // Tier 2: Upstash Redis REST
    if (l2) {
      try {
        const t2 = await l2.get(fullKey);
        if (t2.hit) {
          counters.l2Hits += 1;
          l1Set(fullKey, t2.value, ttlDefault);
          return t2.value;
        }
      } catch {
        counters.l2Errors += 1;
      }
    }

    // Tier 3: Supabase stylist6_cache_entries fallback
    if (l3) {
      try {
        const t3 = await l3.get(fullKey, nowFn());
        if (t3.hit) {
          counters.l3Hits += 1;
          const ttl = Math.min(ttlDefault, t3.remainingTtlMs || ttlDefault);
          l1Set(fullKey, t3.value, ttl);
          if (l2) {
            l2.set(fullKey, t3.value, ttl).catch(() => { counters.l2Errors += 1; });
          }
          return t3.value;
        }
      } catch {
        counters.l3Errors += 1;
      }
    }

    counters.misses += 1;
    return undefined;
  }

  async function set(key, value, ttlMs = ttlDefault) {
    if (!key || value === undefined) return false;
    const effectiveTtl = Math.max(1000, Math.floor(Number(ttlMs) || ttlDefault));
    const fullKey = buildNamespacedKey(cleanNs, key);
    counters.sets += 1;

    l1Set(fullKey, value, effectiveTtl);

    const writes = [];
    if (l2) {
      writes.push(
        l2.set(fullKey, value, effectiveTtl).catch(() => {
          counters.l2Errors += 1;
        }),
      );
    }
    if (l3) {
      writes.push(
        l3.set(fullKey, cleanNs, value, effectiveTtl, nowFn()).catch(() => {
          counters.l3Errors += 1;
        }),
      );
    }
    if (writes.length > 0) await Promise.all(writes);
    return true;
  }

  async function del(key) {
    if (!key) return false;
    const fullKey = buildNamespacedKey(cleanNs, key);
    counters.deletes += 1;
    l1Delete(fullKey);
    const ops = [];
    if (l2) {
      ops.push(l2.del(fullKey).catch(() => { counters.l2Errors += 1; }));
    }
    if (l3) {
      ops.push(l3.del(fullKey).catch(() => { counters.l3Errors += 1; }));
    }
    if (ops.length > 0) await Promise.all(ops);
    return true;
  }

  /**
   * Cache-aside helper with singleflight stampede coalescing.
   * Concurrent calls for the same cold key share a single `computeFn()` run.
   */
  async function getOrSet(key, computeFn, options = {}) {
    if (typeof computeFn !== 'function') {
      throw new TypeError('getOrSet requires a computeFn function');
    }
    const cached = await get(key);
    if (cached !== undefined) return cached;

    const fullKey = buildNamespacedKey(cleanNs, key);
    const existingFlight = inflight.get(fullKey);
    if (existingFlight) {
      counters.coalescedHits += 1;
      return existingFlight;
    }

    const ttlMs = options.ttlMs ?? ttlDefault;
    const skipCacheIf = typeof options.skipCacheIf === 'function' ? options.skipCacheIf : null;

    const flight = (async () => {
      const computed = await computeFn();
      const shouldSkip = computed === undefined || (skipCacheIf ? skipCacheIf(computed) : false);
      if (!shouldSkip) {
        await set(key, computed, ttlMs);
      }
      return computed;
    })();

    inflight.set(fullKey, flight);
    try {
      return await flight;
    } finally {
      inflight.delete(fullKey);
    }
  }

  function stats() {
    const totalHits = counters.l1Hits + counters.l2Hits + counters.l3Hits;
    const totalReads = totalHits + counters.misses;
    return {
      version: DISTRIBUTED_CACHE_VERSION,
      namespace: cleanNs,
      tiers: {
        l1: true,
        l2Redis: Boolean(l2),
        l3Supabase: Boolean(l3),
      },
      l1Entries: l1.size,
      l1Bytes,
      maxMemoryEntries: entryCap,
      maxMemoryBytes: byteCap,
      hits: {
        l1: counters.l1Hits,
        l2: counters.l2Hits,
        l3: counters.l3Hits,
        total: totalHits,
      },
      misses: counters.misses,
      sets: counters.sets,
      deletes: counters.deletes,
      evictions: counters.evictions,
      oversizedSkips: counters.oversizedSkips,
      coalescedHits: counters.coalescedHits,
      errors: {
        l2: counters.l2Errors,
        l3: counters.l3Errors,
      },
      hitRate: totalReads > 0 ? Math.round((totalHits / totalReads) * 10_000) / 10_000 : 0,
    };
  }

  return { get, set, del, stats, getOrSet };
}
