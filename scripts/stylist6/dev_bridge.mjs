// Dev-only localhost bridge: runs the REAL Stylist 6 fashion ranker for the
// iOS simulator app. Never deployed, never imported by the app bundle.
//
// Start:  node scripts/stylist6/dev_bridge.mjs [--port 3127]
// Health: GET  http://127.0.0.1:3127/health
// Rank:   POST http://127.0.0.1:3127/generate  { request, catalogRows }
//
// - `request` carries { style, occasion, brief, weather, budget?, preferences?,
//   recentSignatures?, excludeItemIds?, enableVisual? } — the same fields the
//   app already collects for the generate-outfits edge function, plus the
//   dev-only NVIDIA toggle.
// - `catalogRows` are live shop_catalog-shaped rows (id, name, brand, price,
//   currency, image_url, source_url, description, is_active, ...). The bridge
//   hydrates cached visual profiles with the exact SKU + source-hash
//   predicate, adapts rows with the real catalog_adapter and ranks with the
//   real recommendFashionDeck. Cached-profile hydration is a bounded
//   read-only lookup of the already-computed visual cache.
// - Visual review: when `request.enableVisual === true` AND an NVIDIA API key
//   is configured AND every candidate photo is admissible (lookbook reference
//   bytes on local disk, product hosts on the committed allowlist), the real
//   hosted critic (nvidia/nemotron-3.5-super-vl) judges the strongest
//   finalist(s) through the ranker's bounded visual policy (at most one
//   batch of <=2 candidates per deck). Anything else stays NOT_RUN by
//   admission — never faked, never a cached profile relabeled as NVIDIA.
// - Failures are reported as success:false with a Stylist 6 code. The app
//   must surface them, never silently substitute a Stylist 5 deck.
import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import dotenv from 'dotenv';
import { adaptCatalog, DEV_PURCHASE_EVIDENCE, devLookFeasibility, sourceHash } from '../../backend/stylist6/catalog_adapter.mjs';
import { recommendFashionDeck } from '../../backend/stylist6/fashion_deck.mjs';
import { feedbackTasteModel } from '../../backend/stylist6/feedback_taste.mjs';
import { referenceFirstBriefs } from '../../backend/stylist6/lookbook_briefs.mjs';
import { buildReferenceBlueprints } from '../../backend/stylist6/reference_blueprint.mjs';
import { verifyMerchantOffer, summarizeDeckOffers, MERCHANT_OFFER_VERSION } from '../../backend/stylist6/merchant_offers.mjs';
import { projectWardrobeItem } from '../../backend/stylist6/cloud_handler.mjs';
import { HOST_ALLOWLIST } from './image_access.mjs';
import {
  createHostedVisualCritic, NVIDIA_VISION_MODEL,
  VISUAL_CONCURRENCY, VISUAL_PROVIDER_TIMEOUT_MS, VISUAL_TIMEOUT_MS,
} from '../../backend/stylist6/visual_critic_hosted.mjs';
import { currentVisual } from '../../src/domain/visualFashion.ts';
import { resolveStylistWeather } from '../../src/domain/outfitComposition.ts';
import { normalizeStylistLanguage } from '../../src/domain/stylingContext.ts';
import { STYLIST6_DEV_ENGINE_VERSION } from './engine_identity.mjs';
export { STYLIST6_DEV_ENGINE_VERSION } from './engine_identity.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '../..');
dotenv.config({ path: path.join(root, '.env') });

const DEFAULT_PORT = 3127;
const MAX_BODY_BYTES = 12 * 1024 * 1024;
const MAX_PRODUCT_IMAGE_BYTES = 4 * 1024 * 1024;
const MAX_VISUAL_IMAGE_BYTES = 512 * 1024;
const VISUAL_DIMENSION_LIMIT = 2048;
const PRODUCT_FETCH_TIMEOUT_MS = 8000;
const execFileAsync = promisify(execFile);

const indexPath = path.join(root, 'benchmarks/stylist6/references/image_teaste_46.json');
const index = JSON.parse(fs.readFileSync(indexPath, 'utf8'));

const seasonOf = (temp) => (temp >= 25 ? 'summer' : temp < 12 ? 'winter' : 'autumn');

// Same metadata-reference match the offline replay uses: the reference is an
// input common to both engines, only when the shortlist contains the
// requested style and occasion.
function pickReference(refs, { style, occasion }) {
  const norm = (v) => String(v || '').toLowerCase().replaceAll(' ', '_');
  return refs.find(({ referenceId }) => {
    const ref = index.images.find((entry) => entry.id === referenceId);
    return ref.styles.includes(style) && ref.occasions.includes(norm(occasion));
  });
}

const VISUAL_PROFILE_CAP = 500;
const VISUAL_FETCH_CHUNK = 100;

/**
 * Bounded dev-safe hydration of cached visual intelligence. For each raw
 * catalog row, fetches the matching `catalog_visual_fashion` entry (exact
 * SKU + `source_hash` equality + `image_url` equality, version 1 only) and
 * attaches the profile only when the row bytes are unchanged since
 * profiling — verified by recomputing `sourceHash(row)` locally — and the
 * profile passes the shared `currentVisual` gate. Anything drifted,
 * unversioned or unconfident is left absent, never invented.
 * Read-only; failures degrade to unhydrated ranking, never to a failure.
 */
export async function hydrateVisualProfiles(rows) {
  const outcome = { attempted: 0, fetched: 0, attached: 0, drifted: 0, status: 'skipped_no_credentials' };
  const base = process.env.EXPO_PUBLIC_SUPABASE_URL;
  const key = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
  if (!base || !key) return outcome;
  const ids = [...new Set((rows || []).map((row) => String(row?.id || '').trim()).filter(Boolean))]
    .slice(0, VISUAL_PROFILE_CAP);
  if (!ids.length) return { ...outcome, status: 'empty_pool' };
  outcome.attempted = ids.length;
  const cached = [];
  for (let offset = 0; offset < ids.length; offset += VISUAL_FETCH_CHUNK) {
    const chunk = ids.slice(offset, offset + VISUAL_FETCH_CHUNK);
    const url = new URL('/rest/v1/catalog_visual_fashion', base);
    url.searchParams.set('select', 'product_id,source_hash,image_url,profile,version');
    url.searchParams.set('version', 'eq.1');
    url.searchParams.set('product_id', `in.(${chunk.join(',')})`);
    let res;
    try {
      res = await fetch(url, { headers: { apikey: key, Authorization: `Bearer ${key}` } });
    } catch {
      return { ...outcome, status: 'fetch_failed' };
    }
    if (!res.ok) return { ...outcome, status: `fetch_http_${res.status}` };
    cached.push(...(await res.json()));
  }
  outcome.fetched = cached.length;
  const byId = new Map(cached.map((entry) => [entry.product_id, entry]));
  for (const row of rows) {
    const entry = byId.get(String(row?.id));
    if (!entry || entry.version !== 1 || !entry.profile) continue;
    if (entry.image_url !== row.image_url || entry.source_hash !== sourceHash(row)) {
      outcome.drifted += 1;
      continue;
    }
    row.visual_fashion = entry.profile;
    row.visual_source_hash = entry.source_hash;
    // The shared gate reads camelCase item fields (as the adapter projects
    // them); the raw row keeps the snake_case RPC names.
    if (!currentVisual({ ...row, imageUrl: row.image_url, visualFashion: entry.profile, visualSourceHash: entry.source_hash })) {
      delete row.visual_fashion;
      delete row.visual_source_hash;
      outcome.drifted += 1;
      continue;
    }
    outcome.attached += 1;
  }
  return { ...outcome, status: 'ok' };
}

/**
 * Dev-interface visual status vocabulary. The ranker reports the raw
 * verdict (JUDGED/ABSTAINED/NOT_RUN); the dev bridge maps it to what the
 * hosted review actually did to this deck:
 * - NVIDIA_APPLIED: valid model evidence changed or determined the selection.
 * - NVIDIA_EVALUATED_NO_CHANGE: valid verdicts arrived, catalog winner kept.
 * - NVIDIA_ABSTAINED: the evaluation ran but yielded no actionable verdict.
 * - NVIDIA_NOT_RUN: no external visual evaluation occurred.
 * Cached catalog profiles are never reported as NVIDIA evidence.
 */
export function mapDevVisualStatus(deck) {
  const verdict = deck?.diagnostics?.visualVerdict;
  if (verdict === 'JUDGED') {
    const applied = (deck?.outfits || []).some((look) => look.visualEvidence);
    return applied ? 'NVIDIA_APPLIED' : 'NVIDIA_EVALUATED_NO_CHANGE';
  }
  if (verdict === 'ABSTAINED') return 'NVIDIA_ABSTAINED';
  return 'NVIDIA_NOT_RUN';
}

const hostOf = (value) => {
  try {
    const url = new URL(String(value));
    if (url.protocol !== 'https:' || url.username || url.password || url.port) return '';
    return url.hostname;
  } catch {
    return '';
  }
};

// Lookbook reference bytes live next to the repo (owner-provided lookbook
// photos). A reference metadata entry without a matching local photo is
// NOT a usable reference photograph and fails admission honestly.
const referencePhotosDir = path.join(root, 'image-teaste');
export function referenceFileFor(referenceId) {
  const entry = index.images.find((item) => item.id === referenceId);
  const filename = entry?.filename ? path.basename(String(entry.filename)) : '';
  if (!filename) return null;
  const full = path.join(referencePhotosDir, filename);
  if (!full.startsWith(referencePhotosDir + path.sep)) return null;
  try {
    if (!fs.statSync(full).isFile()) return null;
  } catch {
    return null;
  }
  return full;
}

let sharpImpl = null;
async function loadSharp() {
  if (sharpImpl !== null) return sharpImpl;
  try {
    sharpImpl = (await import('sharp')).default;
  } catch {
    sharpImpl = false;
  }
  return sharpImpl;
}

// macOS fallback resizer when sharp is unavailable. Dev-bridge only.
async function sipsFit(bytes) {
  if (process.platform !== 'darwin') throw new Error('image_toolchain_unavailable');
  const dir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 's6vis-'));
  try {
    const src = path.join(dir, 'src.img'), dst = path.join(dir, 'out.jpg');
    await fs.promises.writeFile(src, bytes);
    await execFileAsync('sips', ['-Z', String(VISUAL_DIMENSION_LIMIT), '-s', 'format', 'jpeg',
      '-s', 'formatOptions', '82', src, '--out', dst]);
    return await fs.promises.readFile(dst);
  } finally {
    await fs.promises.rm(dir, { recursive: true, force: true });
  }
}

// Fit raw bytes to the hosted critic's contract (JPEG, <=512KiB,
// <=2048px per side). Returns fitted bytes; never upscales. Throws when no
// local toolchain can do the conversion — the candidate then abstains.
async function prepareVisualBytes(bytes) {
  if (!Buffer.isBuffer(bytes) || bytes.length < 1024) throw new Error('image_bytes_invalid');
  const sharp = await loadSharp();
  if (!sharp) return sipsFit(bytes);
  const meta = await sharp(bytes, { failOn: 'none' }).metadata();
  if (!Number.isFinite(meta.width) || !Number.isFinite(meta.height)) throw new Error('image_bytes_invalid');
  if ((meta.format === 'jpeg' || meta.format === 'jpg') && bytes.length <= MAX_VISUAL_IMAGE_BYTES
      && meta.width <= VISUAL_DIMENSION_LIMIT && meta.height <= VISUAL_DIMENSION_LIMIT) return bytes;
  let size = VISUAL_DIMENSION_LIMIT, quality = 82;
  for (let attempt = 0; attempt < 3; attempt++) {
    const out = await sharp(bytes, { failOn: 'none' }).rotate()
      .resize({ width: size, height: size, fit: 'inside', withoutEnlargement: true })
      .jpeg({ quality }).toBuffer();
    if (out.length <= MAX_VISUAL_IMAGE_BYTES) return out;
    size = attempt === 0 ? 1600 : 1280;
    quality = attempt === 0 ? 75 : 68;
  }
  throw new Error('image_size_invalid');
}

async function resolveReferenceAsset(referenceId) {
  const file = referenceFileFor(referenceId);
  if (!file) throw new Error('reference_unresolved');
  const fitted = await prepareVisualBytes(await fs.promises.readFile(file));
  return { bytes: fitted, sha256: createHash('sha256').update(fitted).digest('hex') };
}

const productPhotoCache = new Map();
async function resolveProductAsset({ imageUrl }, fetchImpl = fetch) {
  const url = String(imageUrl || '');
  if (!HOST_ALLOWLIST.has(hostOf(url))) throw new Error('product_host_not_allowlisted');
  if (fetchImpl === fetch && productPhotoCache.has(url)) return productPhotoCache.get(url);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), PRODUCT_FETCH_TIMEOUT_MS);
  try {
    const response = await fetchImpl(url, {
      signal: controller.signal, redirect: 'follow',
      headers: { 'User-Agent': 'AIWardrobe-Stylist6-Dev/1.0', Accept: 'image/*,*/*;q=0.8' },
    });
    if (!response.ok) throw new Error('product_image_unresolved');
    const raw = Buffer.from(await response.arrayBuffer());
    if (raw.length > MAX_PRODUCT_IMAGE_BYTES) throw new Error('image_size_invalid');
    const fitted = await prepareVisualBytes(raw);
    const asset = { bytes: fitted, sha256: createHash('sha256').update(fitted).digest('hex') };
    if (fetchImpl === fetch) productPhotoCache.set(url, asset);
    return asset;
  } finally {
    clearTimeout(timer);
  }
}

function normalizeBridgeSourceMode(mode) {
  if (mode == null) return 'catalog';
  const value = String(mode).trim().toLowerCase();
  if (!value || value === 'shop' || value === 'catalog') return 'catalog';
  if (value === 'hybrid' || value === 'mixed') return 'mixed';
  if (value === 'wardrobe') return 'wardrobe';
  return null;
}

function adaptWardrobeRow(row) {
  if (!row || typeof row !== 'object') return null;
  const imageUrl = row.image_url || row.imageUrl || (typeof row.image === 'string' ? row.image : '');
  const normalized = {
    ...row,
    image_url: imageUrl,
    primary_color: row.primary_color || row.color,
    sub_category: row.sub_category || row.subCategory || row.type,
    type: row.type || row.macroCategory,
    category: row.category || row.macroCategory,
  };
  const projected = projectWardrobeItem(normalized);
  if (!projected) return null;
  const explicitSlot = ['top', 'outerwear', 'bottom', 'shoes'].includes(String(row.macroCategory || '').toLowerCase())
    ? String(row.macroCategory).toLowerCase()
    : projected.macroCategory;
  return explicitSlot === projected.macroCategory
    ? projected
    : Object.freeze({ ...projected, macroCategory: explicitSlot });
}

function mapDeckItem(item, rankScore, offerVerification = null) {
  return {
    id: String(item.id),
    type: item.type || item.macroCategory,
    macroCategory: item.macroCategory,
    subtype: item.subtype || item.fashion_subtype || (typeof item.type === 'string' && item.type !== item.macroCategory ? item.type : '') || '',
    color: item.color || 'unknown',
    name: item.name,
    brand: item.brand || '',
    description: item.description || item.name,
    imageUrl: item.imageUrl,
    image: item.imageUrl,
    recommendation: `Stylist 6 rank · score ${Number(rankScore).toFixed(2)}`,
    isShopItem: item.isShopItem === true,
    isWardrobe: item.isShopItem !== true,
    price: item.price ?? null,
    currency: item.currency || null,
    shopUrl: item.shopUrl || null,
    // Gate F price freshness, identical to the cloud contract: when the
    // adapter last saw this listing price, honestly unknown when the catalog
    // row carries no last_seen_at.
    priceAgeDays: Number.isFinite(item.priceAgeDays) ? item.priceAgeDays : null,
    priceFreshness: typeof item.priceFreshness === 'string' && item.isShopItem === true
      ? item.priceFreshness
      : null,
    ...(offerVerification ? { offerVerification } : {}),
  };
}

export async function generateStylist6Deck(payload, options = {}) {
  const { request = {}, catalogRows = [] } = payload || {};
  // Dev-only feedback fixture (gate D local testing): the bridge has no
  // database, so an optional `feedbackTaste` request array of
  // stylist_look_feedback-shaped rows feeds the same pure model the cloud
  // handler builds from real reads. Never persisted, never used outside
  // this dev process; production reads feedback only in the cloud handler.
  const feedbackTaste = feedbackTasteModel(
    Array.isArray(payload?.feedbackTaste) ? payload.feedbackTaste.slice(0, 60) : [],
    Date.now(),
  );
  const { style, occasion } = request;
  if (typeof style !== 'string' || !style || typeof occasion !== 'string' || !occasion) {
    return { status: 400, body: { success: false, code: 'invalid_stylist6_request', stage: 'stylist6_admission' } };
  }
  const sourceMode = normalizeBridgeSourceMode(request.sourceMode);
  if (!sourceMode) {
    return { status: 400, body: { success: false, code: 'invalid_stylist6_request', stage: 'stylist6_admission' } };
  }
  const language = normalizeStylistLanguage(request.language ?? request.locale);
  const requiredIds = [...new Set(
    (Array.isArray(request.requiredIds) ? request.requiredIds : [])
      .map((id) => String(id || '').trim())
      .filter(Boolean),
  )].slice(0, 4);
  const rawWardrobeRows = Array.isArray(payload?.wardrobeItems)
    ? payload.wardrobeItems
    : Array.isArray(request?.wardrobeItems)
      ? request.wardrobeItems
      : [];
  const rawCatalogRows = Array.isArray(catalogRows) ? catalogRows : [];
  if (sourceMode === 'catalog' && !rawCatalogRows.length) {
    return { status: 400, body: { success: false, code: 'stylist6_empty_catalog', stage: 'catalog_adaptation' } };
  }
  // Live discovery rows come from the catalog_discovery_candidates RPC, which
  // filters is_active=true server-side and omits the key. The shared adapter
  // treats a missing key as inactive, so default it here — explicit false
  // still filters out. No production code is touched by this normalization.
  let visualHydration = { attempted: 0, fetched: 0, attached: 0, drifted: 0, status: 'skipped_wardrobe_mode' };
  let shopItems = [];
  if (sourceMode !== 'wardrobe' && rawCatalogRows.length) {
    const liveRows = rawCatalogRows.map((row) =>
      row && typeof row === 'object' && !('is_active' in row) ? { ...row, is_active: true } : row);
    visualHydration = options.skipHydration === true
      ? { attempted: 0, fetched: 0, attached: 0, drifted: 0, status: 'skipped_by_option' }
      : await hydrateVisualProfiles(liveRows);
    shopItems = adaptCatalog(liveRows).items;
  }
  const wardrobeItems = sourceMode !== 'catalog'
    ? rawWardrobeRows.map(adaptWardrobeRow).filter(Boolean)
    : [];
  const items = sourceMode === 'catalog'
    ? shopItems
    : sourceMode === 'wardrobe'
      ? wardrobeItems
      : [...wardrobeItems, ...shopItems];
  if (!items.length) {
    return { status: 502, body: { success: false, code: 'stylist6_empty_catalog', stage: 'catalog_adaptation' } };
  }
  const itemIds = new Set(items.map((item) => String(item.id)));
  const missingAnchors = requiredIds.filter((id) => !itemIds.has(id));
  if (missingAnchors.length) {
    return {
      status: 422,
      body: {
        success: false,
        code: 'anchor_not_found',
        stage: 'stylist6_admission',
        missingAnchors,
      },
    };
  }
  const budget = request.budget?.amount && request.budget?.currency ? request.budget : undefined;
  if (budget && !/^[A-Z]{3}$/.test(String(budget.currency))) {
    return { status: 400, body: { success: false, code: 'invalid_stylist6_request', stage: 'stylist6_admission' } };
  }
  // Same-currency pool, mirroring the offline replay: cross-currency looks
  // cannot be priced honestly without an FX snapshot the ranker owns. Owned
  // wardrobe pieces are free and remain eligible under any currency cap.
  const pool = budget
    ? items.filter((item) => item.isShopItem !== true || item.currency === budget.currency)
    : items;
  if (!pool.length) {
    return { status: 502, body: { success: false, code: 'stylist6_no_feasible_looks', stage: 'stylist6_ranking' } };
  }
  const weather = resolveStylistWeather(request.brief || '', request.weather || {});
  const temp = Number(weather?.temp);
  const refs = referenceFirstBriefs(index, {
    style, occasion, season: Number.isFinite(temp) ? seasonOf(temp) : undefined,
  });
  const reference = pickReference(refs, { style, occasion });
  const excludedIds = Array.isArray(request.excludeItemIds)
    ? [...new Set(request.excludeItemIds.map((id) => String(id || '').trim()).filter(Boolean))]
    : [];
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
    language,
  };
  // Hosted NVIDIA review is strictly opt-in per request (`enableVisual`),
  // strictly local-backend (the key never leaves this process), and strictly
  // bounded. When no local lookbook photo matches `reference.referenceId`,
  // textual editorial blueprints inform the critique without claiming
  // photographic reference fidelity (matching cloud_handler.mjs).
  let visualCritic, canJudgeOutfit;
  const visualEvents = [];
  const visualAsked = request.enableVisual === true;
  const nvidiaKey = options.nvidiaApiKey ?? (process.env.NVIDIA_API_KEY || '');
  const hasLocalReference = Boolean(reference && referenceFileFor(reference.referenceId));
  if (visualAsked && nvidiaKey.startsWith('nvapi-') && nvidiaKey.length >= 20) {
    visualCritic = createHostedVisualCritic({
      provider: 'nvidia',
      apiKey: nvidiaKey,
      model: NVIDIA_VISION_MODEL,
      referenceMediaConfigured: hasLocalReference,
      referenceBlueprints: buildReferenceBlueprints(index),
      timeoutMs: VISUAL_TIMEOUT_MS,
      providerTimeoutMs: VISUAL_PROVIDER_TIMEOUT_MS,
      ...(options.fetchImpl ? { fetchImpl: options.fetchImpl } : {}),
      resolveReference: (id) => resolveReferenceAsset(id),
      resolveProduct: (product) => resolveProductAsset(product, options.fetchImpl || fetch),
      onEvent: (event) => {
        visualEvents.push({
          outcome: event.outcome,
          dispatched: event.dispatched === true,
          reason: event.reason || null,
          ...(event.mediaSlot ? { mediaSlot: event.mediaSlot } : {}),
          ...(event.mediaFailure ? { mediaFailure: event.mediaFailure } : {}),
          httpStatus: event.httpStatus ?? null,
          latencyMs: Math.round(event.latencyMs || event.providerMs || 0),
          ...(event.usage && typeof event.usage === 'object' ? { usage: event.usage } : {}),
        });
      },
    });
    canJudgeOutfit = async (look) => {
      if (hasLocalReference && reference && !referenceFileFor(reference.referenceId)) return false;
      return (look?.items || []).every((item) =>
        HOST_ALLOWLIST.has(hostOf(item.imageUrl || item.image_url)));
    };
  }
  const deck = await recommendFashionDeck({
    request: input,
    catalog: pool,
    referenceBriefs: refs,
    // Real feasibility checks (known slot, reachable photo, verified
    // merchant offer, price/currency, identity) — never an open gate.
    hardGate: async (lookItems) => devLookFeasibility(lookItems),
    priceToBudgetCurrency: (item, currency) => (item.currency === currency ? item.price : NaN),
    feedbackTaste,
    visualDeckReview: true,
    ...(visualCritic ? { visualCritic, canJudgeOutfit } : {}),
  });
  const visualStatus = mapDevVisualStatus(deck);
  const providerCalls = visualEvents.filter((event) => event.dispatched).length;
  const displayedLooks = deck.outfits.length;
  const displayedJudged = deck.outfits.filter((look) => Array.isArray(look.visualEvidence)).length;
  const visualApproval = !visualAsked ? 'not_requested'
    : !visualCritic ? 'unavailable'
      : displayedLooks === 3 && displayedJudged === 3 ? 'full_three_verified'
        : displayedLooks > 0 && displayedJudged === displayedLooks ? 'verified_partial'
          : 'unverified_fallback';
  const visualReceipt = {
    model: visualAsked ? NVIDIA_VISION_MODEL : null,
    requested: visualAsked,
    referenceMediaConfigured: hasLocalReference,
    providerCalls,
    validVerdicts: visualEvents.filter((event) => event.outcome === 'judged').length,
    abstentions: visualEvents.filter((event) => event.outcome === 'abstain').length,
    displayedLooks,
    displayedJudged,
    visualApproval,
    totalTokens: visualEvents.reduce((sum, event) =>
      sum + (Number.isFinite(event.usage?.total_tokens) ? event.usage.total_tokens : 0), 0) || null,
    events: visualEvents,
  };
  process.stdout.write(`[stylist6-dev-bridge] visual requested=${visualAsked} key=${nvidiaKey.startsWith('nvapi-') ? 'present' : 'missing'} ` +
    `reference=${reference ? reference.referenceId : 'none'} localMedia=${hasLocalReference} status=${visualStatus} providerCalls=${providerCalls}\n`);
  if ((deck.status !== 'complete' && deck.status !== 'partial') || !deck.outfits.length) {
    return {
      status: 502,
      body: {
        success: false, code: 'stylist6_no_feasible_looks', stage: 'stylist6_ranking',
        engineVersion: STYLIST6_DEV_ENGINE_VERSION, visualHydration, diagnostics: deck.diagnostics || null,
      },
    };
  }
  // Dev observability: one compact line per served deck so repeat-tap
  // diversity and SKU/price identity can be audited from bridge output.
  process.stdout.write(`[stylist6-dev-bridge] deck style=${request.style} occasion=${request.occasion} lang=${language} mode=${sourceMode} ` +
    deck.outfits.map((look) => look.items.map((i) => `${i.id}@${i.price ?? 0}${i.currency || 'OWNED'}`).join('+')).join(' | ') + '\n');
  const excluded = new Set(Array.isArray(request.excludeItemIds) ? request.excludeItemIds.map(String) : []);
  const previousDeckOverlapBySlot = {};
  for (const slot of ['top', 'outerwear', 'bottom', 'shoes']) previousDeckOverlapBySlot[slot] = 0;
  for (const look of deck.outfits) {
    for (const item of look.items) {
      if (excluded.has(String(item.id)) && previousDeckOverlapBySlot[item.macroCategory] !== undefined) {
        previousDeckOverlapBySlot[item.macroCategory] += 1;
      }
    }
  }
  const mappedOutfits = deck.outfits.map((look, index) => {
    const mappedItems = look.items.map((item) => mapDeckItem(
      item,
      look.fashion?.score,
      verifyMerchantOffer(item, { targetMarket: 'UZ' }),
    ));
    const lookVerified = Array.isArray(look.visualEvidence);
    return {
      // Same positional id convention as the cloud handler and the S5 edge,
      // so locally-served decks carry feedback-attributable look ids too.
      id: `outfit_${index + 1}`,
      selectionSource: lookVerified ? 'ai' : 'deterministic_fallback',
      items: mappedItems,
      offerVerification: summarizeDeckOffers([{ items: mappedItems }]),
      description: look.description,
      concept: { id: look.concept?.id, title: look.concept?.title, story: '', score: look.fashion?.score },
      direction: look.direction ?? null,
      stylingTips: [],
      confidence: look.fashion?.score,
      matchScore: look.fashion?.score,
      engineVersion: STYLIST6_DEV_ENGINE_VERSION,
      ...(look.fashionQuality || look.fashion?.fashionQuality ? {
        fashionQuality: look.fashionQuality || look.fashion?.fashionQuality,
      } : {}),
      fashionDiagnostics: {
        score: look.fashion?.score,
        ...(look.fashionQuality || look.fashion?.fashionQuality ? {
          fashionQuality: look.fashionQuality || look.fashion?.fashionQuality,
        } : {}),
        referenceMetadata: look.fashion?.referenceMetadata || null,
        referenceEvaluationMode: lookVerified ? look.visualReferenceMode || 'unknown' : 'not_judged',
        visualVerdict: lookVerified
          ? 'JUDGED' : deck.diagnostics?.visualVerdict === 'JUDGED'
            ? 'NOT_JUDGED' : deck.diagnostics?.visualVerdict || 'NOT_RUN',
        imageSupportedItems: look.fashion?.imageSupportedItems ?? null,
        personalization: look.fashion?.personalization || null,
      },
    };
  });
  const offerVerification = summarizeDeckOffers(mappedOutfits);
  return {
    status: 200,
    body: {
      success: true,
      generationVersion: STYLIST6_DEV_ENGINE_VERSION,
      source: 'stylist-6-dev-local',
      compositionProtocol: 'optional-layer-v1',
      deckTarget: 3,
      status: deck.status,
      sourceMode,
      language,
      feedbackTasteActive: feedbackTaste.active === true,
      previousDeckOverlapBySlot,
      visualHydration,
      visualStatus,
      visualReceipt,
      offerVerification,
      purchaseEvidence: DEV_PURCHASE_EVIDENCE,
      diagnostics: deck.diagnostics || null,
      meta: {
        language,
        sourceMode,
        offerTruth: {
          version: MERCHANT_OFFER_VERSION,
          ...offerVerification,
        },
        editorialRegister: deck.diagnostics?.editorialRegister || null,
      },
      outfits: mappedOutfits,
    },
  };
}

const handleGenerate = (payload) => generateStylist6Deck(payload);

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', (chunk) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(Object.assign(new Error('request_too_large'), { status: 413 }));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

export function createDevBridgeServer() {
  return http.createServer(async (req, res) => {
    res.setHeader('Content-Type', 'application/json');
    try {
      if (req.method === 'GET' && req.url === '/health') {
        res.writeHead(200);
        res.end(JSON.stringify({ ok: true, engine: STYLIST6_DEV_ENGINE_VERSION, references: index.images.length }));
        return;
      }
      if (req.method === 'POST' && req.url === '/generate') {
        const raw = await readBody(req);
        let payload;
        try {
          payload = JSON.parse(raw);
        } catch {
          res.writeHead(400);
          res.end(JSON.stringify({ success: false, code: 'invalid_stylist6_request', stage: 'stylist6_admission' }));
          return;
        }
        const { status, body } = await handleGenerate(payload);
        process.stdout.write(`[stylist6-dev-bridge] POST /generate -> ${status} success=${body?.success} code=${body?.code || '-'} outfits=${body?.outfits?.length ?? 0}\n`);
        res.writeHead(status);
        res.end(JSON.stringify(body));
        return;
      }
      res.writeHead(404);
      res.end(JSON.stringify({ success: false, code: 'not_found', stage: 'stylist6_dev_bridge' }));
    } catch (error) {
      const status = error?.status === 413 ? 413 : 500;
      res.writeHead(status);
      res.end(JSON.stringify({
        success: false,
        code: status === 413 ? 'invalid_generation_request' : 'stylist6_ranking_failed',
        stage: 'stylist6_ranking',
      }));
    }
  });
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const argvPort = Number(process.argv.slice(2).find((a) => a.startsWith('--port='))?.slice('--port='.length));
  const port = Number.isFinite(argvPort) && argvPort > 0 ? argvPort
    : Number(process.env.STYLIST6_DEV_PORT) > 0 ? Number(process.env.STYLIST6_DEV_PORT) : DEFAULT_PORT;
  const server = createDevBridgeServer();
  server.listen(port, '127.0.0.1', () => {
    process.stdout.write(`[stylist6-dev-bridge] ${STYLIST6_DEV_ENGINE_VERSION} on http://127.0.0.1:${port} (${index.images.length} references)\n`);
  });
}
