/**
 * Isolated Stylist 6 cloud staging handler (Node/Vercel).
 *
 * Runs the VERIFIED local Stylist 6 fashion engine server-side:
 * server-fetched catalog inventory, cached visual-profile hydration,
 * recommendFashionDeck ranking, and the hosted NVIDIA visual critic with
 * the same admission rules as the local dev bridge.
 *
 * The Stylist 5 compute route is untouched; this handler answers only the
 * new isolated `stylist6-generate-outfits` route and reports a truthful
 * `stylist-6-*` generation version. It never runs as Stylist 5 and never
 * relabels deterministic ranking as provider judgment.
 *
 * Admission order (fail-closed, at-most-once admission):
 *   transport size/HMAC -> user JWT -> rate bucket -> server inventory ->
 *   anchor validation -> durable claim -> quota debit -> ranking -> response.
 * Claims and debits are separate transactions; planning failures keep debit.
 *
 * Client-supplied inventory is NEVER authoritative: any `catalogRows` in
 * the request body are ignored and reported as ignored. Wardrobe rows
 * always come from `clothing_items` filtered by the verified JWT user;
 * shop rows always come from the server-side discovery RPC.
 */
import { createHash } from 'node:crypto';
import { adaptCatalog, devLookFeasibility, sourceHash } from './catalog_adapter.mjs';
import { hydrateCatalogObservations } from './catalog_observation.mjs';
import { recommendFashionDeck } from './fashion_deck.mjs';
import { feedbackTasteModel, FEEDBACK_TASTE_VERSION, FEEDBACK_TASTE_MAX_ROWS } from './feedback_taste.mjs';
import { referenceFirstBriefs } from './lookbook_briefs.mjs';
import { currentVisual } from '../../src/domain/visualFashion.ts';
import { resolveStylistWeather } from '../../src/domain/outfitComposition.ts';
import { normalizeStylistLanguage } from '../../src/domain/stylingContext.ts';
import {
  createHostedVisualCritic,
  NVIDIA_VISION_MODEL,
  VISUAL_PROVIDER_TIMEOUT_MS,
  VISUAL_TIMEOUT_MS,
} from './visual_critic_hosted.mjs';
import { HOST_ALLOWLIST, isInferenceAuthorized } from '../../scripts/stylist6/image_access.mjs';
import { prepareCloudVisualMedia } from './cloud_visual_media.mjs';
import { verifyStylistTransport, STYLIST_TRANSPORT_MAX_BODY_BYTES } from '../stylist-node/transport/signature.mjs';
import { verifyMerchantOffer, summarizeDeckOffers } from './merchant_offers.mjs';
import { parseCbuBudgetRates, validateBudgetFxSnapshot, referenceAmount } from '../../src/domain/budgetFx.ts';
import referenceIndex from '../../benchmarks/stylist6/references/image_teaste_46.json' with { type: 'json' };

export const STYLIST6_CLOUD_VERSION = 'stylist-6-cloud-staging-1';
const QUOTA_FEATURE = 'aiw-stylist6-staging';
const RATE_LIMIT_PER_MINUTE = 10;
const CATALOG_SLOTS = ['top', 'bottom', 'shoes', 'outerwear'];
const CATALOG_PER_SLOT = 100;
const WARDROBE_PAGE = 500;
const VISUAL_PROFILE_CHUNK = 100;
const PRODUCT_FETCH_TIMEOUT_MS = 8000;
const MAX_PRODUCT_IMAGE_BYTES = 4 * 1024 * 1024;
const MAX_VISUAL_IMAGE_BYTES = 512 * 1024;
const REQUEST_TIMEOUT_MS = 45_000;

const fail = (code, status, extra = {}) => ({
  status,
  body: { success: false, outfits: [], code, error: code, generationVersion: STYLIST6_CLOUD_VERSION, ...extra },
});

const hostOf = (value) => {
  try {
    const url = new URL(String(value));
    if (url.protocol !== 'https:' || url.username || url.password || url.port) return '';
    return url.hostname;
  } catch {
    return '';
  }
};

const seasonOf = (temp) => (temp >= 25 ? 'summer' : temp < 12 ? 'winter' : 'autumn');

const normalizeSourceMode = (mode) => {
  const normalized = String(mode || 'catalog').toLowerCase();
  if (normalized === 'shop' || normalized === 'catalog') return 'catalog';
  if (normalized === 'wardrobe' || normalized === 'mixed') return normalized;
  return null;
};

/** Slot projection for owned wardrobe rows (S5-proven EN+RU keyword mapping). */
const WARDROBE_SLOT_RULES = [
  ['shoes', /кроссовк|кеды|туфл|ботинок|ботинки|лофер|сапог|мокасин|сандал|shoes?|sneakers?|boots?|loafers?|sandals?|heels?|trainers?|derb|footwear|brogue|oxford(?!\s*shirt)/],
  ['outerwear', /жилет|куртк|пальто|пиджак|жакет|кардиган|свитер|худи|толстовк|blazer|overcoat|topcoat|trench|parka|puffer|windbreaker|bomber|coats?|jackets?|cardigans?|sweaters?|hoodies?|vests?|pullovers?|fleece|outerwear/],
  ['bottom', /джинс|брюк|шорт|юбк|леггинс|\b(pants?|trousers?|jeans|chinos?|slacks?|joggers?|shorts?|skirts?|sweatpants?|bottoms?)\b/],
  ['top', /футболк|рубашк|блуз|майк|платье|\b(t-shirts?|tshirts?|tees?|polos?|blouses?|shirts?|tops?|dresses?)\b|upper[_ ]?body/],
];

const WARDROBE_SUBTYPE_RULES = [
  ['shirt', /\b(shirts?|рубашк|блуз)\b/],
  ['blazer', /\b(blazers?|пиджак|жакет)\b/],
  ['fine_knit', /\b(knit|sweaters?|pullovers?|cardigans?|свитер|кардиган|водолазк|гольф)\b/],
  ['polo', /\b(polos?|поло)\b/],
  ['jeans', /\b(jeans?|джинс)\b/],
  ['tailored_trouser', /\b(trousers?|slacks?|chinos?|брюк)\b/],
  ['chino', /\b(chinos?|чинос)\b/],
  ['loafer', /\b(loafers?|moccasins?|лофер|мокасин)\b/],
  ['derby', /\b(derb|brogues?|oxfords?|дерби|броги|оксфорд)\b/],
  ['minimal_sneaker', /\b(sneakers?|trainers?|кроссовк|кеды)\b/],
  ['overshirt', /\b(overshirts?|s?tuleikas?)\b/],
];

export function projectWardrobeItem(row) {
  if (!row || typeof row !== 'object') return null;
  const id = String(row.id || '').trim();
  if (!id) return null;
  const blob = `${row.category || ''} ${row.type || ''} ${row.sub_category || ''} ${row.name || ''}`.toLowerCase();
  const slot = (WARDROBE_SLOT_RULES.find(([, re]) => re.test(blob)) || [])[0] || null;
  if (!slot) return null;
  const imageUrl = String(row.image_url || '');
  if (!imageUrl.startsWith('https://')) return null;
  const subtype = (WARDROBE_SUBTYPE_RULES.find(([, re]) => re.test(blob)) || [])[0] || 'unknown';
  const color = String(
    (Array.isArray(row.color) ? row.color.join(', ') : row.primary_color || row.color) || 'neutral',
  ).slice(0, 64);
  return Object.freeze({
    id,
    name: String(row.name || row.type || row.category || 'Wardrobe item').slice(0, 160),
    brand: String(row.brand || '').slice(0, 80),
    macroCategory: slot,
    subtype,
    color,
    texture: 'unknown',
    imageUrl,
    shopUrl: null,
    price: null,
    currency: null,
    isShopItem: false,
    visualEvidenceVerified: false,
    projection: { kind: 'owned_wardrobe', evidence: [`wardrobe:${slot}/${subtype}`] },
  });
}

/**
 * Fit raw image bytes to the hosted critic contract without a native
 * toolchain: pass through bytes that are already small JPEGs, otherwise
 * abstain honestly. Never invents pixels.
 */
export function fitVisualBytesCloud(bytes) {
  if (!Buffer.isBuffer(bytes) || bytes.length < 128 || bytes.length > MAX_VISUAL_IMAGE_BYTES) return null;
  if (bytes[0] !== 0xff || bytes[1] !== 0xd8) return null;
  return bytes;
}

async function fetchWithTimeout(url, options = {}, timeoutMs = PRODUCT_FETCH_TIMEOUT_MS) {
  // Keep the signal alive while the response body streams; cancelling it
  // immediately after headers allowed slow/hung CDN bodies to consume the
  // entire 18s visual budget despite the advertised 8s merchant timeout.
  return fetch(url, { ...options, signal: options.signal || AbortSignal.timeout(timeoutMs) });
}

/**
 * Server-side IO for the cloud handler. The default implementation talks
 * to Supabase over HTTPS; tests inject fakes. No client-supplied rows are
 * ever read here.
 */
export function createCloudIo({ url, anonKey, serviceKey, supabaseJs }) {
  const authHeaders = (key) => ({ apikey: key, 'Content-Type': 'application/json' });
  const check = async (response, what) => {
    if (!response.ok) throw new Error(what);
    return response.json();
  };
  return {
    async verifyUser(jwt) {
      const { createClient } = supabaseJs;
      const client = createClient(url, anonKey, { auth: { persistSession: false, autoRefreshToken: false } });
      const { data, error } = await client.auth.getUser(jwt);
      if (error || !data?.user?.id) throw new Error('stylist6_unauthorized');
      return data.user.id;
    },
    adminClient() {
      const { createClient } = supabaseJs;
      return createClient(url, serviceKey, { auth: { persistSession: false, autoRefreshToken: false } });
    },
    // Cheap early admission: rate bucket only, never debits quota.
    async checkRate(userId) {
      const admin = this.adminClient();
      const nowMs = Date.now();
      const bucketMs = Math.floor(nowMs / 60_000) * 60_000;
      const bucket = await admin.rpc('increment_rate_bucket', {
        p_user: userId, p_feature: `rl:${QUOTA_FEATURE}:${bucketMs}`,
      });
      if (bucket.error || !Number.isSafeInteger(bucket.data) || bucket.data < 1) {
        throw new Error('stylist6_rate_unavailable');
      }
      if (bucket.data > RATE_LIMIT_PER_MINUTE) {
        const denied = new Error('stylist6_rate_limited');
        denied.code = 'rate_limited';
        throw denied;
      }
    },
    // Paid debit, called exactly once after the request is fully admitted
    // (valid anchors, feasible inventory). Same production outfit-generation
    // bucket: no new entitlement surface, no migration.
    async debitQuota(userId) {
      const admin = this.adminClient();
      const quota = await admin.rpc('consume_edge_quota', { p_user: userId, p_endpoint: 'generate-outfits' });
      if (quota.error || typeof quota.data !== 'boolean') throw new Error('stylist6_rate_unavailable');
      if (!quota.data) {
        const denied = new Error('stylist6_quota_exhausted');
        denied.code = 'feature_limit_reached';
        throw denied;
      }
    },
    /**
     * Durable idempotency claim for one logical generation request via the
     * `claim_stylist6_request` RPC (atomic insert wins; concurrent
     * duplicates serialize on the primary key). There is NO process-local
     * fallback: if durable claim storage is unavailable the request fails
     * closed BEFORE any quota debit. Returns {duplicate, prior?}.
     * Duplicates never re-debit and never re-plan: the caller answers 409
     * from the recorded outcome.
     */
    async claimRequest(userId, requestId, fingerprint) {
      const admin = this.adminClient();
      let claimed;
      try {
        claimed = await admin.rpc('claim_stylist6_request', { p_user: userId, p_request_id: requestId, p_fingerprint: fingerprint });
      } catch {
        throw new Error('stylist6_claim_unavailable');
      }
      if (claimed?.error) throw new Error('stylist6_claim_unavailable');
      // Table-returning RPCs arrive as row arrays; tolerate both shapes.
      const row = Array.isArray(claimed?.data) ? claimed.data[0] : claimed?.data;
      if (!row || typeof row !== 'object' || typeof row.duplicate !== 'boolean') {
        throw new Error('stylist6_claim_unavailable');
      }
      return {
        duplicate: row.duplicate === true,
        conflict: row.outcome_code === 'request_content_conflict',
        prior: row.duplicate === true ? { status: row.status_code, code: row.outcome_code } : null,
        durable: true,
      };
    },
    /**
     * Records the terminal outcome for a claimed request so a redelivery
     * replays it via 409. Returns true when recorded; a false return is
     * surfaced on the response (`idempotencyRecorded`) but never fails an
     * otherwise successful generation — the claim itself already prevents
     * double billing.
     */
    async finishRequest(userId, requestId, status, code) {
      const admin = this.adminClient();
      try {
        const finished = await admin.rpc('finish_stylist6_request', {
          p_user: userId, p_request_id: requestId, p_status: status, p_code: String(code || ''),
        });
        return !finished?.error;
      } catch {
        return false;
      }
    },
    async fetchShopRows(seed = null) {
      const rows = new Map();
      // An unanchored fresh request gets a stable but rotated bounded catalog
      // window. Keep legacy sampling for explicit anchors (no missing locked
      // product due to a random discovery page). No client-supplied raw seed.
      const safeSeed = typeof seed === 'string' && /^[a-f0-9]{16}$/.test(seed) ? seed : null;
      for (const slot of CATALOG_SLOTS) {
        const response = await fetch(`${url}/rest/v1/rpc/catalog_discovery_candidates`, {
          method: 'POST',
          headers: { ...authHeaders(serviceKey), Authorization: `Bearer ${serviceKey}` },
          body: JSON.stringify({ p_slot: slot, p_search: '', p_seed: safeSeed ? `s6cloud:${slot}:${safeSeed}` : `s6cloud:${slot}`, p_limit: CATALOG_PER_SLOT, p_offset: 0 }),
        });
        const data = await check(response, 'stylist6_catalog_unavailable');
        for (const row of Array.isArray(data) ? data : []) {
          if (row && row.id && !rows.has(String(row.id))) {
            // This server RPC selects only active shop_catalog rows, but its
            // JSON projection omits is_active. Restore that contract fact
            // here, never in the general adapter or from client inventory.
            // The projection also omits the intelligence table's subtype
            // column; the projected fashion_intelligence descriptor carries
            // exactly that value (subtype = descriptor.subtype at write
            // time), so restore it for the adapter's live-hint contract.
            // Explicit false stays false; a missing descriptor leaves the
            // row to the adapter's transparent text projection.
            rows.set(String(row.id), {
              ...row,
              is_active: row.is_active ?? true,
              fashion_subtype: row.fashion_subtype
                ?? (row.fashion_intelligence && typeof row.fashion_intelligence === 'object'
                  ? row.fashion_intelligence?.descriptor?.subtype
                  : undefined),
            });
          }
        }
      }
      // catalog_discovery_candidates projects the catalog SKU but omits the
      // last feed-import observation. Reread only selected IDs, bounded and
      // read-only; changed/deactivated rows are dropped. A failed REST read
      // retains UNKNOWN recency (never a fabricated current price).
      return hydrateCatalogObservations({
        rows: [...rows.values()], url, serviceKey,
      });
    },
    async fetchWardrobeRows(userId) {
      const rows = [];
      const pageSize = WARDROBE_PAGE;
      for (let offset = 0; ; offset += pageSize) {
        const params = new URLSearchParams({
          select: 'id,type,category,sub_category,color,primary_color,brand,name,description,image_url',
          user_id: `eq.${userId}`,
          order: 'id.asc',
          offset: String(offset),
          limit: String(pageSize),
        });
        const response = await fetch(`${url}/rest/v1/clothing_items?${params}`, {
          headers: { ...authHeaders(serviceKey), Authorization: `Bearer ${serviceKey}` },
        });
        const data = await check(response, 'stylist6_wardrobe_unavailable');
        if (!Array.isArray(data) || !data.length) break;
        rows.push(...data);
        if (data.length < pageSize) break;
      }
      return rows;
    },
    async fetchVisualProfiles(ids) {
      const profiles = [];
      for (let offset = 0; offset < ids.length; offset += VISUAL_PROFILE_CHUNK) {
        const chunk = ids.slice(offset, offset + VISUAL_PROFILE_CHUNK);
        const params = new URLSearchParams({
          select: 'product_id,source_hash,image_url,profile',
          version: 'eq.1',
          product_id: `in.(${chunk.map((id) => `"${String(id).replace(/"/g, '')}"`).join(',')})`,
        });
        const response = await fetch(`${url}/rest/v1/catalog_visual_fashion?${params}`, {
          headers: { ...authHeaders(serviceKey), Authorization: `Bearer ${serviceKey}` },
        });
        const data = await check(response, 'stylist6_catalog_unavailable');
        if (Array.isArray(data)) profiles.push(...data);
      }
      return profiles;
    },
    // Gate D taste source: this user's own explicit thumbs rows, newest
    // first, bounded to the pure module's row cap. One indexed query on
    // feedback_owner_recent (user_id, created_at desc) — plus deterministic
    // request/look tiebreakers so the same rows always rank the same way.
    // Service-role key: the caller is the verified JWT user (RLS unaffected).
    async fetchFeedbackTaste(userId) {
      const params = new URLSearchParams({
        select: 'rating,concept_id,item_ids,look_signature,created_at',
        user_id: `eq.${userId}`,
        order: 'created_at.desc,request_id.desc,look_id.desc',
        limit: String(FEEDBACK_TASTE_MAX_ROWS),
      });
      const response = await fetch(`${url}/rest/v1/stylist_look_feedback?${params}`, {
        headers: { ...authHeaders(serviceKey), Authorization: `Bearer ${serviceKey}` },
      });
      const data = await check(response, 'stylist6_feedback_unavailable');
      return Array.isArray(data) ? data : [];
    },
    async fetchBudgetFxSnapshot() {
      return fetchCbuBudgetFxSnapshot();
    },
  };
}

const CBU_FX_URL = 'https://cbu.uz/ru/arkhiv-kursov-valyut/json/';
const CBU_FX_CACHE_TTL_MS = 60 * 60 * 1000;
let cbuFxCache = { expiresAt: 0, snapshot: null };

/**
 * Fetches and validates the official Central Bank of Uzbekistan daily FX
 * reference rates. Never invents rates; returns null when unavailable so
 * budgeted requests fail closed or enforce same-currency pricing honestly.
 */
export async function fetchCbuBudgetFxSnapshot(fetchFn = fetch, now = Date.now()) {
  if (cbuFxCache.snapshot && now < cbuFxCache.expiresAt) {
    try {
      return validateBudgetFxSnapshot(cbuFxCache.snapshot, now);
    } catch {
      cbuFxCache = { expiresAt: 0, snapshot: null };
    }
  }
  const response = await fetchFn(CBU_FX_URL, {
    headers: { Accept: 'application/json', 'User-Agent': 'AIWardrobe-Stylist6-Cloud/1.0' },
    signal: AbortSignal.timeout(4000),
  });
  if (!response?.ok) throw new Error(`cbu_fx_http_${response?.status || 'failed'}`);
  const payload = await response.json();
  const snapshot = validateBudgetFxSnapshot(parseCbuBudgetRates(payload, now), now);
  cbuFxCache = { expiresAt: now + CBU_FX_CACHE_TTL_MS, snapshot };
  return snapshot;
}

function hydrateProfiles(rows, profiles) {
  const byId = new Map(profiles.map((entry) => [String(entry.product_id), entry]));
  let attached = 0, drifted = 0;
  for (const row of rows) {
    const entry = byId.get(String(row?.id));
    if (!entry || entry.version !== 1 || !entry.profile) continue;
    if (entry.image_url !== row.image_url || entry.source_hash !== sourceHash(row)) {
      drifted += 1;
      continue;
    }
    if (!currentVisual({ ...row, imageUrl: row.image_url, visualFashion: entry.profile, visualSourceHash: entry.source_hash })) {
      drifted += 1;
      continue;
    }
    row.visual_fashion = entry.profile;
    row.visual_source_hash = entry.source_hash;
    attached += 1;
  }
  return { fetched: profiles.length, attached, drifted };
}

function mapDeckItem(item, rankScore, offerVerification = null) {
  return {
    id: String(item.id),
    type: item.type || item.macroCategory,
    macroCategory: item.macroCategory,
    color: item.color || 'unknown',
    name: item.name,
    brand: item.brand || '',
    description: item.description || item.name,
    imageUrl: item.imageUrl,
    image: item.imageUrl,
    recommendation: `Stylist 6 rank · score ${Number(rankScore).toFixed(2)}`,
    isShopItem: item.isShopItem === true,
    // Truthful owned-piece provenance: the app's quote/buy pipeline skips
    // isWardrobe pieces (no price, no Buy button, zero spend).
    isWardrobe: item.isShopItem !== true,
    price: item.price ?? null,
    currency: item.currency || null,
    shopUrl: item.shopUrl || null,
    // Merchant-offer price freshness (gate F): when the server last saw this
    // listing price, as days and a bounded tier. Null/unknown when the row
    // carries no last_seen_at — never an invented recency claim. Owned
    // wardrobe pieces carry no merchant offer and stay unknown.
    priceAgeDays: Number.isFinite(item.priceAgeDays) ? item.priceAgeDays : null,
    priceFreshness: typeof item.priceFreshness === 'string' && item.isShopItem === true
      ? item.priceFreshness
      : null,
    ...(offerVerification ? { offerVerification } : {}),
  };
}

/** Same truth table as the local bridge: never relabels ranking as NVIDIA. */
export function mapCloudVisualStatus(deck) {
  const verdict = deck?.diagnostics?.visualVerdict;
  if (verdict === 'JUDGED') {
    const applied = (deck?.outfits || []).some((look) => look.visualEvidence);
    return applied ? 'NVIDIA_APPLIED' : 'NVIDIA_EVALUATED_NO_CHANGE';
  }
  if (verdict === 'ABSTAINED') return 'NVIDIA_ABSTAINED';
  return 'NVIDIA_NOT_RUN';
}

const IMAGE_FAILURE_CLASSES = new Set([
  'http_401','http_403','http_404','http_429','http_5xx',
  'invalid_media','oversize','decode_failed','fetch_failed',
  'host_not_allowlisted',
]);

function mediaFailure(code) {
  const error = new Error('product_image_unresolved');
  error.mediaFailure = IMAGE_FAILURE_CLASSES.has(code) ? code : 'fetch_failed';
  return error;
}

/** Memory-bounded streaming reader: hostile Content-Length is not trusted. */
export async function readBoundedProductImage(response, maxBytes = MAX_PRODUCT_IMAGE_BYTES) {
  if (!response || !Number.isSafeInteger(maxBytes) || maxBytes < 128 ||
    maxBytes > MAX_PRODUCT_IMAGE_BYTES) throw mediaFailure('invalid_media');
  const header = response.headers?.get?.('content-length');
  if (header && /^\d+$/.test(header) && Number(header) > maxBytes) {
    void response.body?.cancel?.().catch(() => {});
    throw mediaFailure('oversize');
  }
  if (!response.body?.getReader) throw mediaFailure('invalid_media');
  const reader = response.body.getReader();
  const parts = [];
  let length = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value?.byteLength) continue;
      length += value.byteLength;
      if (length > maxBytes) {
        await reader.cancel().catch(() => {});
        throw mediaFailure('oversize');
      }
      parts.push(Buffer.from(value));
    }
    if (length < 128) throw mediaFailure('invalid_media');
    return Buffer.concat(parts, length);
  } finally {
    reader.releaseLock();
  }
}

/**
 * One request's in-flight and successful photo fetches are shared by URL.
 * Two concurrent looks using the same shoes no longer double-fetch the
 * merchant CDN; a failed response is shared too (no retry storm/429).
 * The cache is created INSIDE the generation handler, never persisted or
 * shared across users, requests or merchant price windows.
 */
export async function resolveProductAsset(imageUrl, cache, fetchFn) {
  const url = String(imageUrl || '');
  if (!HOST_ALLOWLIST.has(hostOf(url))) throw mediaFailure('host_not_allowlisted');
  if (!(cache instanceof Map)) throw new Error('request_image_cache_required');
  if (cache.has(url)) return cache.get(url);
  const task = (async () => {
    const get = fetchFn || fetchWithTimeout;
    let response;
    try {
      response = await get(url, {
        redirect: 'error',
        headers: { 'User-Agent': 'AIWardrobe-Stylist6-Cloud/1.0', Accept: 'image/*,*/*;q=0.8' },
        // Covers both HTTP headers and the image body. Injected test stubs
        // receive the same request shape with an abortable deadline.
        signal: AbortSignal.timeout(PRODUCT_FETCH_TIMEOUT_MS),
      });
    } catch {
      throw mediaFailure('fetch_failed');
    }
    if (!response?.ok) {
      const code = Number(response?.status);
      const failure = [401,403,404,429].includes(code) ? `http_${code}`
        : code >= 500 ? 'http_5xx' : 'fetch_failed';
      void response?.body?.cancel?.().catch(() => {});
      throw mediaFailure(failure);
    }
    let raw;
    try {
      raw = await readBoundedProductImage(response);
    } catch (error) {
      if (IMAGE_FAILURE_CLASSES.has(error?.mediaFailure)) throw error;
      throw mediaFailure('fetch_failed');
    }
    const fitted = await prepareCloudVisualMedia(raw);
    if (!fitted) throw mediaFailure('decode_failed');
    return { bytes: fitted, sha256: createHash('sha256').update(fitted).digest('hex') };
  })();
  cache.set(url, task);
  return task;
}

/**
 * Reference bytes are licensed media and never ship with the bundle. When
 * the owner stages them in a private Supabase Storage bucket
 * (STYLIST6_REFERENCE_BUCKET), reference-guided judging becomes available;
 * otherwise reference admission fails honestly to NOT_RUN.
 */
async function resolveReferenceAsset(env, url, referenceId, fetchFn) {
  const bucket = String(env.STYLIST6_REFERENCE_BUCKET || '').trim();
  if (!bucket || !/^[a-zA-Z0-9_-]{1,64}$/.test(referenceId)) throw new Error('reference_unresolved');
  const entry = (referenceIndex?.images || []).find((item) => item.id === referenceId);
  const filename = entry?.filename ? String(entry.filename).split('/').pop() : '';
  if (!filename || filename.includes('..')) throw new Error('reference_unresolved');
  // First-party staged bytes only: the owner-controlled bucket is the one
  // inference-authorized reference source in cloud.
  const get = fetchFn || fetch;
  const response = await get(
    `${url}/storage/v1/object/${bucket}/${encodeURIComponent(filename)}`,
    { headers: { apikey: env.SUPABASE_SERVICE_ROLE_KEY, Authorization: `Bearer ${env.SUPABASE_SERVICE_ROLE_KEY}` } },
  );
  if (!response.ok) throw new Error('reference_unresolved');
  const raw = Buffer.from(await response.arrayBuffer());
  const fitted = await prepareCloudVisualMedia(raw);
  if (!fitted) throw new Error('reference_unresolved');
  return { bytes: fitted, sha256: createHash('sha256').update(fitted).digest('hex') };
}

// JSON object ordering must not turn an identical logical retry into a conflict.
export function canonicalRequestFingerprint(request) {
  const canonical = (value) => Array.isArray(value) ? value.map(canonical)
    : value && typeof value === 'object'
      ? Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])]))
      : value;
  return createHash('sha256').update(JSON.stringify(canonical(request))).digest('hex');
}

/**
 * Authenticated Stylist 6 cloud generation. `io` performs all Supabase IO
 * (injectable fakes in tests); `env` carries server-only configuration.
 * Returns {status, body} with an edge-shaped payload.
 */
export async function handleStylist6CloudGenerate({ method, headers, bodyText, env, io, fetchImpl }) {
  const secret = env.STYLIST6_NODE_SHARED_SECRET || '';
  if (!secret || secret.length < 32) return fail('stylist_compute_unconfigured', 503);
  if (method !== 'POST') return fail('method_not_allowed', 405);
  const body = String(bodyText || '');
  if (new TextEncoder().encode(body).byteLength > STYLIST_TRANSPORT_MAX_BODY_BYTES) {
    return fail('invalid_generation_request', 413);
  }
  const authorized = await verifyStylistTransport({
    secret, body, authorization: headers.authorization || '',
    timestamp: headers['x-aiw-timestamp'], nonce: headers['x-aiw-nonce'], signature: headers['x-aiw-signature'],
  });
  if (!authorized) return fail('unauthorized', 401);

  let payload;
  try {
    payload = JSON.parse(body);
  } catch {
    return fail('invalid_stylist6_request', 400, { stage: 'stylist6_admission' });
  }
  const request = payload?.request || {};
  const { style, occasion } = request;
  if (typeof style !== 'string' || !style || typeof occasion !== 'string' || !occasion) {
    return fail('invalid_stylist6_request', 400, { stage: 'stylist6_admission' });
  }
  const sourceMode = normalizeSourceMode(request.sourceMode);
  if (!sourceMode) return fail('invalid_stylist6_request', 400, { stage: 'stylist6_admission' });
  // Display language for the deck's editorial copy. Never an admission
  // error: an unsupported value degrades to English per output string.
  const language = normalizeStylistLanguage(request.language);
  if (request.budget != null && (typeof request.budget !== 'object' ||
      typeof request.budget.amount !== 'number' || !Number.isFinite(request.budget.amount) ||
      request.budget.amount <= 0 || !/^[A-Z]{3}$/.test(String(request.budget.currency)))) {
    return fail('invalid_stylist6_request', 400, { stage: 'stylist6_admission' });
  }
  const budget = request.budget ?? undefined;
  if (budget && !/^[A-Z]{3}$/.test(String(budget.currency))) {
    return fail('invalid_stylist6_request', 400, { stage: 'stylist6_admission' });
  }
  const requiredIds = [...new Set((Array.isArray(request.requiredIds) ? request.requiredIds : [])
    .map((id) => String(id || '').trim()).filter(Boolean))].slice(0, 8);
  const excludedIds = [...new Set((Array.isArray(request.excludeItemIds) ? request.excludeItemIds : [])
    .map((id) => String(id || '').trim()).filter(Boolean))].slice(0, 64);
  const clientInventoryIgnored = Array.isArray(payload?.catalogRows) && payload.catalogRows.length > 0;

  // JWT first: no user, no inventory, no quota touch.
  const authorization = headers.authorization || '';
  const jwt = /^Bearer\s+(\S+)$/i.exec(authorization)?.[1] || '';
  let userId;
  try {
    userId = await io.verifyUser(jwt);
  } catch {
    return fail('unauthorized', 401, { stage: 'stylist6_admission' });
  }

  // Cheap early rate limit: bucket only, never debits quota.
  try {
    await io.checkRate(userId);
  } catch (error) {
    if (error?.code === 'rate_limited') return fail('rate_limited', 429, { stage: 'stylist6_admission' });
    return fail('rate_limit_unavailable', 503, { stage: 'stylist6_admission' });
  }

  const started = Date.now();
  const deadline = () => {
    if (Date.now() - started > REQUEST_TIMEOUT_MS) throw new Error('stylist6_timeout');
  };
  // Logical request identity: an explicit client id when supplied,
  // otherwise the body hash (identical bytes = identical request).
  const requestId = (() => {
    const explicit = String(request.generationRequestId || '').trim();
    if (/^[A-Za-z0-9_-]{1,128}$/.test(explicit)) return explicit;
    return `body-${createHash('sha256').update(body).digest('hex').slice(0, 32)}`;
  })();
  const finish = (status, code) => io.finishRequest(userId, requestId, status, code).catch(() => {});
  try {
    // Server-side inventory only. Shop rows come from the discovery RPC;
    // wardrobe rows come from clothing_items filtered by the verified user.
    // Stable over retries of identical request bytes, different on fresh
    // requests (including a new client request id), private user scope; no
    // extra RPCs or cross-user data. Keep anchored inventory stable.
    const discoverySeed = requiredIds.length ? null : createHash('sha256')
      .update(userId).update('|').update(body).digest('hex').slice(0, 16);
    const [shopRows, wardrobeRows] = await Promise.all([
      sourceMode === 'wardrobe' ? [] : io.fetchShopRows(discoverySeed),
      sourceMode === 'catalog' ? [] : io.fetchWardrobeRows(userId),
    ]);
    deadline();
    const visualHydration = await io.fetchVisualProfiles(shopRows.map((row) => row.id))
      .then((profiles) => hydrateProfiles(shopRows, profiles));
    deadline();
    const { items: shopItems } = adaptCatalog(shopRows);
    const wardrobeItems = wardrobeRows.map(projectWardrobeItem).filter(Boolean);
    const pool = sourceMode === 'catalog' ? shopItems
      : sourceMode === 'wardrobe' ? wardrobeItems
        : [...wardrobeItems, ...shopItems];
    if (!pool.length) {
      return fail('stylist6_empty_catalog', 502, { stage: 'catalog_adaptation', serverInventory: true, clientInventoryIgnored });
    }
    // Locked anchors are validated against the server pool BEFORE any
    // quota debit: invalid anchors fail without a generation charge and
    // are never silently dropped.
    const poolIds = new Set(pool.map((item) => String(item.id)));
    const missingAnchors = requiredIds.filter((id) => !poolIds.has(id));
    if (missingAnchors.length) {
      return fail('anchor_not_found', 422, { stage: 'stylist6_admission', missingAnchors, serverInventory: true });
    }
    // Idempotency: durable claim first. A redelivered logical request
    // replays its recorded outcome without another debit and without
    // re-planning. Unavailable claim storage fails closed before debit.
    let claim;
    try {
      claim = await io.claimRequest(userId, requestId, canonicalRequestFingerprint(request));
    } catch {
      return fail('stylist6_claim_unavailable', 503, { stage: 'stylist6_admission' });
    }
    if (claim.conflict) return fail('request_content_conflict', 409, { stage: 'stylist6_admission' });
    if (claim.duplicate) {
      const prior = claim.prior || {};
      return fail('duplicate_request', 409, {
        stage: 'stylist6_admission',
        priorStatus: prior.status ?? null,
        priorCode: prior.code || null,
        replayProtection: 'durable',
      });
    }
    // Paid debit exactly once, after full admission. Planning failures
    // keep the debit, mirroring production semantics.
    try {
      await io.debitQuota(userId);
    } catch (error) {
      if (error?.code === 'feature_limit_reached') {
        await finish(403, 'feature_limit_reached');
        return fail('feature_limit_reached', 403, { stage: 'stylist6_admission' });
      }
      await finish(503, 'rate_limit_unavailable');
      return fail('rate_limit_unavailable', 503, { stage: 'stylist6_admission' });
    }
    // Owned wardrobe pieces are free. When a budget is requested, cross-currency
    // pricing uses only a validated CBU FX snapshot; without valid FX we never
    // invent exchange rates and strictly admit same-currency shop items only.
    let fxSnapshot = null;
    let fxUnavailableReason = null;
    if (budget) {
      if (typeof io.fetchBudgetFxSnapshot === 'function') {
        try {
          fxSnapshot = validateBudgetFxSnapshot(await io.fetchBudgetFxSnapshot());
        } catch {
          fxSnapshot = null;
          fxUnavailableReason = 'fx_unavailable';
        }
      } else {
        fxUnavailableReason = 'fx_not_configured';
      }
    }
    const currencyConversion = !budget
      ? { active: false, reason: 'unbudgeted' }
      : fxSnapshot
        ? {
            active: true,
            source: fxSnapshot.source,
            asOf: fxSnapshot.asOf,
            snapshotId: fxSnapshot.id,
            budgetCurrency: budget.currency,
            snapshot: fxSnapshot,
          }
        : { active: false, reason: fxUnavailableReason || 'same_currency_only', budgetCurrency: budget.currency };
    const priceToBudgetCurrency = fxSnapshot
      ? (item, currency) => {
          const converted = referenceAmount(item.price, item.currency, currency, fxSnapshot);
          return Number.isFinite(converted) ? converted : NaN;
        }
      : (item, currency) => (item.currency === currency ? item.price : NaN);
    const feasible = budget
      ? pool.filter((item) => item.isShopItem !== true || Number.isFinite(priceToBudgetCurrency(item, budget.currency)))
      : pool;
    if (!feasible.length) {
      // Terminal outcome of an already-claimed (and debited) request: record
      // it like the empty-deck path below so a redelivery replays 409 with
      // the real prior code instead of an unrecorded "processing" claim.
      await finish(502, 'stylist6_no_feasible_looks');
      return fail('stylist6_no_feasible_looks', 502, {
        stage: 'stylist6_ranking',
        serverInventory: true,
        currencyConversion,
      });
    }
    // Gate D taste: one bounded, indexed read of this user's explicit thumbs
    // feedback, then a pure, decayed, capped model. This sits on the ranking
    // latency path (a single feedback_owner_recent query); ANY failure here
    // — transport, schema, io shape, missing io method — degrades to an
    // empty set plus an honest diagnostic field, never a failed generation.
    let feedbackRows = [];
    let feedbackTasteStatus = 'ok';
    try {
      if (typeof io.fetchFeedbackTaste === 'function') {
        feedbackRows = (await io.fetchFeedbackTaste(userId)).slice(0, FEEDBACK_TASTE_MAX_ROWS);
      } else {
        feedbackTasteStatus = 'not_wired';
      }
    } catch {
      feedbackRows = [];
      feedbackTasteStatus = 'read_failed';
    }
    const feedbackTaste = feedbackTasteModel(feedbackRows, Date.now());
    const weather = resolveStylistWeather(request.brief || '', request.weather || {});
    const temp = Number(weather?.temp);
    const refs = referenceFirstBriefs(referenceIndex, {
      style, occasion, season: Number.isFinite(temp) ? seasonOf(temp) : undefined,
    });
    // Same metadata-reference match the local bridge uses: the reference is
    // an input common to both engines, only when the shortlist contains the
    // requested style and occasion.
    const norm = (v) => String(v || '').toLowerCase().replaceAll(' ', '_');
    const reference = refs.find(({ referenceId }) => {
      const ref = (referenceIndex?.images || []).find((entry) => entry.id === referenceId);
      return Array.isArray(ref?.styles) && Array.isArray(ref?.occasions)
        && ref.styles.includes(style) && ref.occasions.includes(norm(occasion));
    }) || null;
    const input = {
      brief: request.brief || '',
      weather,
      referenceId: reference?.referenceId,
      recentSignatures: Array.isArray(request.recentSignatures) ? request.recentSignatures.slice(-32) : [],
      excludedIds,
      requiredIds,
      sourceMode,
      budget,
      style,
      occasion,
      optionalLayer: true,
      preferences: request.preferences && typeof request.preferences === 'object' ? request.preferences : {},
      // Language pass-through: fashionStylingContext spreads the request into
      // the styling context, so deck_editorial_copy reads it from there. The
      // fingerprint includes it — copy language is request content.
      language,
    };
    // Hosted NVIDIA review: strictly opt-in, server-side key only, bounded
    // to the critic's own batch limits. Reference-guided judging needs
    // staged reference bytes; without them admission fails honestly and the
    // deck completes deterministically as NOT_RUN. Looks without a matching
    // reference are judged purely on product images (reference-optional).
    // STYLIST6_VISUAL_DISABLED=1 (or true) is the owner kill-switch: it
    // forces an honest NOT_RUN even when the client asks for review.
    let visualCritic, canJudgeOutfit;
    const visualEvents = [];
    const visualAsked = request.enableVisual === true;
    const visualDisabled = ['1', 'true'].includes(String(env.STYLIST6_VISUAL_DISABLED || '').trim().toLowerCase());
    const nvidiaKey = env.NVIDIA_API_KEY || '';
    const photoCache = new Map();
    if (visualAsked && !visualDisabled && nvidiaKey.startsWith('nvapi-')) {
      const resolveProduct = (product) => resolveProductAsset(product.imageUrl || product.image_url, photoCache,
        fetchImpl ? (prodUrl, prodOptions) => fetchImpl(prodUrl, prodOptions) : undefined);
      const rawVisualCritic = createHostedVisualCritic({
        provider: 'nvidia',
        apiKey: nvidiaKey,
        model: NVIDIA_VISION_MODEL,
        // Referenceless visual critique is allowed only when the entire
        // reference bucket was never configured. If an explicitly configured
        // bucket cannot serve the selected reference, the critic abstains.
        referenceMediaConfigured: Boolean(String(env.STYLIST6_REFERENCE_BUCKET || '').trim()),
        timeoutMs: VISUAL_TIMEOUT_MS,
        providerTimeoutMs: VISUAL_PROVIDER_TIMEOUT_MS,
        ...(fetchImpl ? { fetchImpl } : {}),
        resolveReference: (id, signal) => resolveReferenceAsset(env, env.SUPABASE_URL, id,
          fetchImpl ? (refUrl, refOptions) => fetchImpl(refUrl, { ...refOptions, signal }) : undefined),
        resolveProduct,
        onEvent: (event) => {
          visualEvents.push({
            outcome: event.outcome,
            dispatched: event.dispatched === true,
            reason: event.reason || null,
            ...(event.mediaSlot ? { mediaSlot: event.mediaSlot } : {}),
            ...(event.mediaFailure ? { mediaFailure: event.mediaFailure } : {}),
            httpStatus: event.httpStatus ?? null,
            latencyMs: Math.round(event.latencyMs || event.providerMs || 0),
            ...(Number.isSafeInteger(event.images) ? { images: event.images } : {}),
            ...(event.cacheHit === true ? { cacheHit: true } : {}),
            ...(event.retried === true ? { retried: true } : {}),
            ...(event.usage && typeof event.usage === 'object' ? { usage: event.usage } : {}),
          });
        },
      });
      // Bounded concurrent media pre-warm before Wave 1 hits the strict NIM
      // deadline so merchant CDN fetches never wait behind each other inside
      // provider evaluation. Only touches URLs already permitted for inference.
      visualCritic = async (candidates, ...rest) => {
        const urls = [...new Set(
          (Array.isArray(candidates) ? candidates : []).flatMap((look) =>
            (look?.items || [])
              .map((item) => item?.imageUrl || item?.image_url)
              .filter((imageUrl) => isInferenceAuthorized(imageUrl)),
          ),
        )];
        for (let i = 0; i < urls.length; i += 6) {
          await Promise.allSettled(urls.slice(i, i + 6).map((imageUrl) =>
            resolveProduct({ imageUrl }).catch(() => null)));
        }
        return rawVisualCritic(candidates, ...rest);
      };
      // Transmission gate, separate from download eligibility: every product
      // image must carry explicit external-inference authorization, or the
      // look completes as NOT_RUN with full deterministic ranking.
      canJudgeOutfit = async (look) => (look?.items || []).every((item) =>
        isInferenceAuthorized(item.imageUrl || item.image_url));
    }
    deadline();
    const deck = await recommendFashionDeck({
      request: input,
      catalog: feasible,
      referenceBriefs: refs,
      hardGate: async (items) => devLookFeasibility(items),
      priceToBudgetCurrency,
      feedbackTaste,
      // Staging-only visual-first quality pass: review every admissible card
      // in one bounded batch, not just the first selection.
      visualDeckReview: true,
      ...(visualCritic ? { visualCritic, canJudgeOutfit } : {}),
    });
    const visualStatus = mapCloudVisualStatus(deck);
    if ((deck.status !== 'complete' && deck.status !== 'partial') || !deck.outfits.length) {
      // Distinguish a real feasibility shortage from an actual visual
      // rejection. A valid AI verdict that approves zero outfits must not
      // silently reopen the originally feasible (but rejected) catalog deck.
      const qualityFailed = (deck.diagnostics?.visualReviewCoverage?.valid ?? 0) > 0 &&
        (deck.diagnostics?.visualReviewCoverage?.approved ?? 0) === 0;
      const failureCode = qualityFailed ? 'stylist6_visual_quality_unmet' : 'stylist6_no_feasible_looks';
      await finish(502, failureCode);
      return fail(failureCode, 502, {
        stage: 'stylist6_ranking',
        engineVersion: STYLIST6_CLOUD_VERSION,
        visualStatus,
        serverInventory: true,
        clientInventoryIgnored,
        replayProtection: 'durable',
        diagnostics: deck.diagnostics || null,
      });
    }
    const excluded = new Set(excludedIds);
    const previousDeckOverlapBySlot = { top: 0, outerwear: 0, bottom: 0, shoes: 0 };
    for (const look of deck.outfits) {
      for (const item of look.items) {
        if (excluded.has(String(item.id)) && previousDeckOverlapBySlot[item.macroCategory] !== undefined) {
          previousDeckOverlapBySlot[item.macroCategory] += 1;
        }
      }
    }
    let liveOfferMap = null;
    if (typeof io.fetchMerchantOffers === 'function') {
      try {
        const ids = [...new Set(deck.outfits.flatMap((look) => look.items.map((item) => String(item.id))))];
        const fetched = await io.fetchMerchantOffers(ids);
        liveOfferMap = fetched instanceof Map ? fetched : new Map(Object.entries(fetched || {}));
      } catch {
        liveOfferMap = null;
      }
    }
    const mappedOutfits = deck.outfits.map((look, index) => {
      const mappedItems = look.items.map((item) => mapDeckItem(
        item,
        look.fashion?.score,
        verifyMerchantOffer(item, { offerRegistry: liveOfferMap, targetMarket: 'UZ' }),
      ));
      const lookVerified = Array.isArray(look.visualEvidence);
      return {
        // Same positional id convention as the S5 edge (`outfit_${index+1}`):
        // the stylist_look_feedback constraint and the app's feedback guard
        // require it, so a thumbs write can attribute this exact deck.
        id: `outfit_${index + 1}`,
        selectionSource: lookVerified ? 'ai' : 'deterministic_fallback',
        items: mappedItems,
        offerVerification: summarizeDeckOffers([{ items: mappedItems }]),
        description: look.description,
        concept: { id: look.concept?.id, title: look.concept?.title, story: '', score: look.fashion?.score },
        // Actual evidence from the selected SKU deck, never a cosmetic
        // variant label or a promise that the catalog has every direction.
        direction: look.direction ?? null,
        stylingTips: [],
        confidence: look.fashion?.score,
        matchScore: look.fashion?.score,
        engineVersion: STYLIST6_CLOUD_VERSION,
        fashionDiagnostics: {
          score: look.fashion?.score,
          referenceMetadata: look.fashion?.referenceMetadata || null,
          // Verdict of THIS displayed look, not a global batch verdict
          // that may refer to a rejected or non-displayed challenger.
          visualVerdict: lookVerified
            ? 'JUDGED' : deck.diagnostics?.visualVerdict === 'JUDGED'
              ? 'NOT_JUDGED' : deck.diagnostics?.visualVerdict || 'NOT_RUN',
          imageSupportedItems: look.fashion?.imageSupportedItems ?? null,
          // Only catalog-grounded positive matches; never expose the full
          // preference profile or assert body/fit facts without evidence.
          personalization: look.fashion?.personalization || null,
        },
      };
    });
    const offerVerification = summarizeDeckOffers(mappedOutfits);
    const idempotencyRecorded = await finish(200, deck.status);
    return {
      status: 200,
      body: {
        success: true,
        generationVersion: STYLIST6_CLOUD_VERSION,
        source: 'stylist-6-cloud-staging',
        // Logical request identity echoed back so the app can attribute
        // feedback (like/dislike) to this exact deck, as the S5 edge does.
        generationRequestId: requestId,
        compositionProtocol: 'optional-layer-v1',
        deckTarget: 3,
        status: deck.status,
        sourceMode,
        serverInventory: true,
        clientInventoryIgnored,
        currencyConversion,
        offerVerification,
        replayProtection: 'durable',
        idempotencyRecorded,
        previousDeckOverlapBySlot,
        visualHydration,
        visualStatus,
        visualReceipt: {
          // Cost metering is tokens-based only: no per-call pricing table
          // exists in this codebase, so no currency estimate is invented.
          model: visualCritic ? NVIDIA_VISION_MODEL : null,
          requested: visualAsked,
          serverDisabled: visualAsked && visualDisabled,
          // Owner reference storage is optional; without it the critic may
          // still judge actual product images, but never score "reference
          // fidelity". This metadata is non-sensitive and explicitly names
          // the evidence mode of the attempt.
          referenceBucketConfigured: Boolean(String(env.STYLIST6_REFERENCE_BUCKET || '').trim()),
          providerCalls: visualEvents.filter((event) => event.dispatched).length,
          imagesSent: visualEvents.reduce((sum, event) =>
            sum + (event.dispatched && Number.isSafeInteger(event.images) ? event.images : 0), 0),
          cacheHits: visualEvents.filter((event) => event.cacheHit === true).length,
          retriedCalls: visualEvents.filter((event) => event.retried === true).length,
          validVerdicts: visualEvents.filter((event) => event.outcome === 'judged').length,
          abstentions: visualEvents.filter((event) => event.outcome === 'abstain').length,
          displayedLooks: deck.outfits.length,
          displayedJudged: deck.outfits.filter((look) => !!look.visualEvidence).length,
          // Explicitly separate a reviewed deck from a successful request,
          // and full 3-card proof from merely having at least one verdict.
          visualApproval: !visualAsked ? 'not_requested'
            : visualDisabled ? 'disabled'
              : !visualCritic ? 'unavailable'
            : deck.outfits.length === 3 &&
                deck.outfits.every((look) => Array.isArray(look.visualEvidence))
              ? 'full_three_verified'
              : deck.outfits.length > 0 &&
                  deck.outfits.every((look) => Array.isArray(look.visualEvidence))
                ? 'verified_partial'
                : 'unverified_fallback',
          totalTokens: visualEvents.reduce((sum, event) =>
            sum + (Number.isFinite(event.usage?.total_tokens) ? event.usage.total_tokens : 0), 0) || null,
          events: visualEvents,
        },
        diagnostics: deck.diagnostics || null,
        // Gate D read receipt: the bounded feedback read this ranking used
        // (or honestly did not use). Never a taste profile, only a status.
        feedbackTaste: {
          status: feedbackTasteStatus,
          rows: feedbackRows.length,
          active: feedbackTaste.active === true,
          version: FEEDBACK_TASTE_VERSION,
        },
        outfits: mappedOutfits,
      },
    };
  } catch (error) {
    if (error?.message === 'stylist6_timeout') {
      await finish(504, 'stylist6_ranking_failed').catch(() => {});
      return fail('stylist6_ranking_failed', 504, { stage: 'stylist6_ranking' });
    }
    await finish(500, 'stylist6_ranking_failed').catch(() => {});
    return fail('stylist6_ranking_failed', 500, { stage: 'stylist6_ranking' });
  }
}
