// Dev-only localhost bridge: runs the FULL World-Class Stylist 6 V3 fashion
// engine for the iOS Simulator app. Never deployed, never imported by the app
// bundle.
//
// Start:      node scripts/stylist6/dev_bridge.mjs [--port 3127]
// Health:     GET  http://127.0.0.1:3127/health
// Generate:   POST http://127.0.0.1:3127/generate  { request, catalogRows?, wardrobeItems? }
//             (also routes action:'swap_slot' and action:'analyze_appearance')
// Swap slot:  POST http://127.0.0.1:3127/swap-slot { outfit, targetSlot, reason?, context?, excludedIds? }
// Appearance: POST http://127.0.0.1:3127/analyze-appearance { photoUrl, consentGranted }
//
// Full V3 Pipeline (local Simulator dev bridge):
// - Hydrates & caches full active Supabase `shop_catalog` +
//   `catalog_garment_intelligence` + `catalog_visual_fashion` (10-min TTL)
//   and live CBU `budget_exchange_rates` (15-min TTL) in live server mode
//   when `catalogRows` is omitted/small, while strictly honoring passed
//   `catalogRows` fixtures in deterministic offline unit tests.
// - Runs PersonalStyleDna, filterHygienicCatalog, Creative Director,
//   buildPerDirectionCandidatePools, Visual-First Showroom Composition
//   (`allowLocalShowroomVisualComposition: true` when visual keys exist),
//   1024x1024 Studio Board 6-Axis Visual Critique, Parallel Surgical 1-Slot
//   Visual Refinement, Cross-Lab Portfolio Jury, and grounded Stylist's Notes.
// - Enforces exact 4 shoppable items per outfit (`top`, `outerwear`, `bottom`,
//   `shoes`, `optionalLayer: false`, `compositionProtocol: 'four-slot-v1'`)
//   on catalog mode.
// - Failures are reported honestly as success:false with a Stylist 6 code;
//   never silently substitutes Stylist 5.
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
import { filterHygienicCatalog } from '../../backend/stylist6/catalog_hygiene.mjs';
import { recommendFashionDeck } from '../../backend/stylist6/fashion_deck.mjs';
import { designCreativeBriefsWithFallback, CREATIVE_DIRECTOR_VERSION } from '../../backend/stylist6/creative_director_ai.mjs';
import { composeCatalogSkuLooks } from '../../backend/stylist6/sku_composer_ai.mjs';
import { selectFashionSlate } from '../../backend/stylist6/fashion_director_slate.mjs';
import { selectPhotoEditorialDeck } from '../../backend/stylist6/photo_editor_final.mjs';
import { feedbackTasteModel, FEEDBACK_TASTE_VERSION } from '../../backend/stylist6/feedback_taste.mjs';
import { buildUnifiedTasteProfile } from '../../backend/stylist6/taste_learning.mjs';
import { referenceFirstBriefs, explicitLookbookPhotoRequested } from '../../backend/stylist6/lookbook_briefs.mjs';
import { buildReferenceBlueprints } from '../../backend/stylist6/reference_blueprint.mjs';
import { verifyMerchantOffer, summarizeDeckOffers, MERCHANT_OFFER_VERSION } from '../../backend/stylist6/merchant_offers.mjs';
import {
  projectWardrobeItem,
  enforceCatalogFourItemDeck,
  catalogStylingDiscoveryHints,
  prioritizeCatalogForBudgetAndClimate,
  balancedGarmentCatalogWindow,
  balancedShoeCatalogWindow,
  fetchCbuBudgetFxSnapshot,
  loadOptionalVisualAtelier,
} from '../../backend/stylist6/cloud_handler.mjs';
import { createQualityFlywheel, buildQualityLookupMap } from '../../backend/stylist6/catalog_quality_flywheel.mjs';
import { createStylistCache } from '../../backend/stylist6/distributed_cache.mjs';
import { swapSingleSlot } from '../../backend/stylist6/surgical_refiner.mjs';
import { analyzeUserAppearance } from '../../backend/stylist6/user_appearance_analyzer.mjs';
import { buildStylistNotes } from '../../backend/stylist6/deck_editorial_copy.mjs';
import { createProviderRouter, getSharedProviderRouter } from '../../backend/stylist6/ai_provider_router.mjs';
import { renderEditorialLookBoard, createPhotoByteCache } from '../../backend/stylist6/editorial_board_renderer.mjs';
import { runAutonomousVisualStylist, AUTONOMOUS_STYLIST_VERSION } from '../../backend/stylist6/autonomous_stylist.mjs';
import { HOST_ALLOWLIST } from './image_access.mjs';
import {
  createHostedVisualCritic,
  NVIDIA_VISION_MODEL,
  OPENROUTER_VISION_MODEL,
  MAX_VISUAL_CANDIDATES,
  VISUAL_CONCURRENCY,
  VISUAL_PROVIDER_TIMEOUT_MS,
  VISUAL_TIMEOUT_MS,
} from '../../backend/stylist6/visual_critic_hosted.mjs';
import { currentVisual } from '../../src/domain/visualFashion.ts';
import { resolveStylistWeather } from '../../src/domain/outfitComposition.ts';
import { isWorkoutOccasion } from '../../src/domain/occasionRules.ts';
import { normalizeStylistLanguage } from '../../src/domain/stylingContext.ts';
import { normalizePersonalStyleDna, buildStyleDnaPromptBlock } from '../../src/domain/personalStyleDna.ts';
import { validateBudgetFxSnapshot, referenceAmount } from '../../src/domain/budgetFx.ts';
import { STYLIST6_DEV_ENGINE_VERSION } from './engine_identity.mjs';
export { STYLIST6_DEV_ENGINE_VERSION } from './engine_identity.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '../..');

// Load local worktree .env first, then non-overriding fallbacks from the main
// AIWardrobe checkout so NVIDIA_API_KEY, OPENROUTER_API_KEY, and Supabase
// credentials resolve cleanly without ever printing raw values.
for (const envPath of [
  path.join(root, '.env'),
  '/Users/v.zahid/AIWardrobe/.env',
  '/Users/v.zahid/AIWardrobe/backend/api/.env',
]) {
  try {
    if (fs.existsSync(envPath)) {
      dotenv.config({ path: envPath, override: false, quiet: true });
    }
  } catch {
    // Non-fatal: missing or unreadable optional fallback env file
  }
}

const DEFAULT_PORT = 3127;
const MAX_BODY_BYTES = 12 * 1024 * 1024;
const MAX_PRODUCT_IMAGE_BYTES = 4 * 1024 * 1024;
const MAX_VISUAL_IMAGE_BYTES = 512 * 1024;
const VISUAL_DIMENSION_LIMIT = 2048;
const PRODUCT_FETCH_TIMEOUT_MS = 8000;
const BRIDGE_GENERATION_BUDGET_MS = 75_000;
const LIVE_CATALOG_TTL_MS = 10 * 60 * 1000;
const LIVE_FX_TTL_MS = 15 * 60 * 1000;
const MIN_FULL_CATALOG_ROWS = 1000;
const BRIDGE_WINDOW_PER_SLOT = 160;
const CANONICAL_SLOTS = ['top', 'outerwear', 'bottom', 'shoes'];
const VALID_SWAP_SLOTS = new Set(['top', 'outerwear', 'bottom', 'shoes', 'accessory', 'dress']);
const execFileAsync = promisify(execFile);

const indexPath = path.join(root, 'benchmarks/stylist6/references/image_teaste_46.json');
const index = JSON.parse(fs.readFileSync(indexPath, 'utf8'));

const devQualityFlywheel = createQualityFlywheel();
const devAppearanceCache = createStylistCache({
  namespace: 's6:dev:appearance',
  maxMemoryEntries: 128,
  maxMemoryBytes: 8 * 1024 * 1024,
});
const devDirectorCache = createStylistCache({
  namespace: 's6:dev:director',
  maxMemoryEntries: 64,
  maxMemoryBytes: 4 * 1024 * 1024,
});
const sharedPhotoByteCache = createPhotoByteCache({
  maxEntries: 256,
  maxBytes: 48 * 1024 * 1024,
});

let liveCatalogCache = { rows: null, fetchedAt: 0, visualAttached: 0, intelAttached: 0 };
let liveCatalogInFlight = null;
let liveFxCache = { snapshot: null, fetchedAt: 0 };
let liveFxInFlight = null;

export function clearDevBridgeCaches() {
  liveCatalogCache = { rows: null, fetchedAt: 0, visualAttached: 0, intelAttached: 0 };
  liveCatalogInFlight = null;
  liveFxCache = { snapshot: null, fetchedAt: 0 };
  liveFxInFlight = null;
  productPhotoCache.clear();
}

const seasonOf = (temp) => (temp >= 25 ? 'summer' : temp < 12 ? 'winter' : 'autumn');

function resolveSupabaseReadConfig(options = {}) {
  const url = String(
    options.supabaseUrl ?? process.env.EXPO_PUBLIC_SUPABASE_URL ?? process.env.SUPABASE_URL ?? '',
  ).trim().replace(/\/+$/, '');
  const anonKey = String(
    options.supabaseAnonKey ?? process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY ?? process.env.SUPABASE_ANON_KEY ?? '',
  ).trim();
  const serviceKey = String(
    options.supabaseServiceKey ?? process.env.SUPABASE_SERVICE_ROLE_KEY ?? '',
  ).trim();
  const readKey = anonKey || serviceKey;
  return { url, anonKey, serviceKey, readKey };
}

function resolveAiKeys(options = {}) {
  const explicitNvidia = options.nvidiaApiKey !== undefined;
  const explicitOpenrouter = options.openrouterApiKey !== undefined;
  const nvidiaKey = String(explicitNvidia ? options.nvidiaApiKey : (process.env.NVIDIA_API_KEY || '')).trim();
  // When a test injects `options.nvidiaApiKey` without `options.openrouterApiKey`,
  // do NOT silently pick up a developer's real `.env` OpenRouter key.
  const openrouterKey = String(
    explicitOpenrouter
      ? options.openrouterApiKey
      : (explicitNvidia ? '' : (process.env.OPENROUTER_API_KEY || '')),
  ).trim();
  const hasNvidiaKey = nvidiaKey.startsWith('nvapi-') && nvidiaKey.length >= 20;
  const hasOpenrouterKey = openrouterKey.length >= 16;
  return { nvidiaKey, openrouterKey, hasNvidiaKey, hasOpenrouterKey };
}

// Same metadata-reference match the offline replay uses: the reference is an
// input common to both engines, only when the shortlist contains the
// requested style and occasion.
function pickReference(refs, { style, occasion }) {
  const norm = (v) => String(v || '').toLowerCase().replaceAll(' ', '_');
  const normStyle = norm(style);
  const normOccasion = norm(occasion);
  return refs.find(({ referenceId }) => {
    const ref = index.images.find((entry) => entry.id === referenceId);
    return ref.styles.includes(normStyle) && ref.occasions.includes(normOccasion);
  });
}

const VISUAL_PROFILE_CAP = 500;
const VISUAL_FETCH_CHUNK = 100;

/**
 * Paginate a public read-only Supabase REST table cleanly (Range 1000/page).
 */
async function restPaginate({ baseUrl, key, table, select, extra = {}, order = 'id.asc', fetchImpl = fetch }) {
  const rows = [];
  const headers = { apikey: key, Authorization: `Bearer ${key}` };
  for (let offset = 0; offset < 40_000; offset += 1000) {
    const url = new URL(`/rest/v1/${table}`, baseUrl);
    url.searchParams.set('select', select);
    url.searchParams.set('order', order);
    for (const [k, v] of Object.entries(extra)) url.searchParams.set(k, v);
    const res = await fetchImpl(url, {
      headers: { ...headers, Range: `${offset}-${offset + 999}` },
      signal: AbortSignal.timeout(20_000),
    });
    if (!res?.ok) throw new Error(`${table}_http_${res?.status || 'failed'}`);
    const page = await res.json();
    if (!Array.isArray(page) || page.length === 0) break;
    rows.push(...page);
    if (page.length < 1000) break;
  }
  return rows;
}

/**
 * Hydrate and cache (10-min TTL) the full active Supabase `shop_catalog`
 * joined with `catalog_garment_intelligence` (version 1) and source-hash-verified
 * `catalog_visual_fashion` (version 1) so every live iOS Simulator generation
 * searches the full hygienic 8,554-SKU catalog.
 */
export async function hydrateFullLiveCatalog(options = {}) {
  const now = options.now ?? Date.now();
  const forceRefresh = options.forceRefresh === true;
  if (!forceRefresh && Array.isArray(liveCatalogCache.rows) && (now - liveCatalogCache.fetchedAt) < LIVE_CATALOG_TTL_MS) {
    return {
      rows: liveCatalogCache.rows,
      status: 'cached',
      totalRows: liveCatalogCache.rows.length,
      intelAttached: liveCatalogCache.intelAttached,
      visualAttached: liveCatalogCache.visualAttached,
    };
  }
  const { url, readKey } = resolveSupabaseReadConfig(options);
  if (!url || !readKey) {
    return { rows: null, status: 'skipped_no_credentials', totalRows: 0, intelAttached: 0, visualAttached: 0 };
  }
  const fetchImpl = options.fetchImpl || fetch;
  if (!forceRefresh && liveCatalogInFlight) {
    return liveCatalogInFlight;
  }
  const task = (async () => {
    try {
      const [catalog, intelligence, visual] = await Promise.all([
        restPaginate({
          baseUrl: url,
          key: readKey,
          table: 'shop_catalog',
          select: 'id,name,brand,description,category,garment_type,image_url,source_url,price,currency,is_active,last_seen_at',
          extra: { is_active: 'eq.true' },
          order: 'id.asc',
          fetchImpl,
        }),
        restPaginate({
          baseUrl: url,
          key: readKey,
          table: 'catalog_garment_intelligence',
          select: 'product_id,source_hash,slot,subtype',
          extra: { version: 'eq.1' },
          order: 'product_id.asc',
          fetchImpl,
        }).catch(() => []),
        restPaginate({
          baseUrl: url,
          key: readKey,
          table: 'catalog_visual_fashion',
          select: 'product_id,source_hash,image_url,profile,version',
          extra: { version: 'eq.1' },
          order: 'product_id.asc',
          fetchImpl,
        }).catch(() => []),
      ]);
      const intelBySku = new Map(intelligence.map((r) => [String(r.product_id), r]));
      const visualBySku = new Map(visual.map((r) => [String(r.product_id), r]));
      let intelAttached = 0;
      let visualAttached = 0;
      const hydrated = catalog.map((raw) => {
        const id = String(raw?.id || '');
        const intel = intelBySku.get(id);
        const vis = visualBySku.get(id);
        const row = { ...raw, is_active: raw?.is_active ?? true };
        if (intel && typeof intel.slot === 'string' && intel.slot) {
          row.fashion_slot = intel.slot;
          if (typeof intel.subtype === 'string' && intel.subtype) {
            row.fashion_subtype = intel.subtype;
          }
          intelAttached += 1;
        }
        if (vis && vis.version === 1 && vis.profile && vis.image_url === row.image_url && vis.source_hash === sourceHash(row)) {
          if (currentVisual({ ...row, imageUrl: row.image_url, visualFashion: vis.profile, visualSourceHash: vis.source_hash })) {
            row.visual_fashion = vis.profile;
            row.visual_source_hash = vis.source_hash;
            visualAttached += 1;
          }
        }
        return row;
      });
      liveCatalogCache = { rows: hydrated, fetchedAt: now, visualAttached, intelAttached };
      return {
        rows: hydrated,
        status: 'ok',
        totalRows: hydrated.length,
        intelAttached,
        visualAttached,
      };
    } catch (error) {
      return {
        rows: null,
        status: `fetch_failed:${error?.message || 'unknown'}`,
        totalRows: 0,
        intelAttached: 0,
        visualAttached: 0,
      };
    } finally {
      liveCatalogInFlight = null;
    }
  })();
  if (!forceRefresh) liveCatalogInFlight = task;
  return task;
}

/**
 * Hydrate and cache (15-min TTL) live CBU FX rates — first from Supabase
 * `budget_exchange_rates` when service credentials are available, then falling
 * back directly to official Central Bank of Uzbekistan JSON rates. Never invents
 * exchange rates when unavailable.
 */
export async function fetchLiveCbuFxSnapshot(options = {}) {
  const now = options.now ?? Date.now();
  if (options.fxSnapshot) {
    try {
      return { snapshot: validateBudgetFxSnapshot(options.fxSnapshot, now), source: 'injected' };
    } catch {
      return { snapshot: null, source: 'invalid_injected' };
    }
  }
  const forceRefresh = options.forceRefresh === true;
  if (!forceRefresh && liveFxCache.snapshot && (now - liveFxCache.fetchedAt) < LIVE_FX_TTL_MS) {
    try {
      return { snapshot: validateBudgetFxSnapshot(liveFxCache.snapshot, now), source: 'cached' };
    } catch {
      liveFxCache = { snapshot: null, fetchedAt: 0 };
    }
  }
  if (!forceRefresh && liveFxInFlight) return liveFxInFlight;
  const fetchImpl = options.fetchImpl || fetch;
  const task = (async () => {
    try {
      const { url, serviceKey } = resolveSupabaseReadConfig(options);
      if (url && serviceKey) {
        try {
          const tableUrl = new URL('/rest/v1/budget_exchange_rates', url);
          tableUrl.searchParams.set('select', 'snapshot');
          tableUrl.searchParams.set('order', 'fetched_at.desc');
          tableUrl.searchParams.set('limit', '1');
          const res = await fetchImpl(tableUrl, {
            headers: { apikey: serviceKey, Authorization: `Bearer ${serviceKey}` },
            signal: AbortSignal.timeout(4000),
          });
          if (res?.ok) {
            const rows = await res.json();
            const rawSnap = Array.isArray(rows) ? rows[0]?.snapshot : null;
            if (rawSnap) {
              const validated = validateBudgetFxSnapshot(rawSnap, now);
              liveFxCache = { snapshot: validated, fetchedAt: now };
              return { snapshot: validated, source: 'supabase_budget_exchange_rates' };
            }
          }
        } catch {
          // Fall through cleanly to direct official CBU reference rates
        }
      }
      const snapshot = await fetchCbuBudgetFxSnapshot(fetchImpl, now);
      if (snapshot) {
        liveFxCache = { snapshot, fetchedAt: now };
        return { snapshot, source: 'cbu_live' };
      }
      return { snapshot: null, source: 'unavailable' };
    } catch {
      return { snapshot: null, source: 'unavailable' };
    } finally {
      liveFxInFlight = null;
    }
  })();
  if (!forceRefresh) liveFxInFlight = task;
  return task;
}

/**
 * Bounded dev-safe hydration of cached visual intelligence for small explicit
 * `catalogRows` pools (legacy / fixture hydration path).
 */
export async function hydrateVisualProfiles(rows) {
  const outcome = { attempted: 0, fetched: 0, attached: 0, drifted: 0, status: 'skipped_no_credentials' };
  const { url: base, readKey: key } = resolveSupabaseReadConfig();
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
 * Select a balanced, style/occasion/weather/budget-prioritized window of up to
 * ~640 items (160/slot) from the full 8,554-SKU catalog so `recommendFashionDeck`
 * gets maximum relevant diversity in <200ms CPU while keeping all `requiredIds`.
 * Small pools (<= 650 items, such as unit test fixtures) pass through untouched.
 */
export function selectBridgeCatalogWindow(items, {
  style,
  occasion,
  brief = '',
  weather = null,
  budget = null,
  fxSnapshot = null,
  priceToBudgetCurrency = null,
  requiredIds = [],
  excludedIds = [],
  perSlot = BRIDGE_WINDOW_PER_SLOT,
} = {}) {
  if (!Array.isArray(items) || items.length <= perSlot * CANONICAL_SLOTS.length) {
    return Array.isArray(items) ? items : [];
  }
  const pinnedSet = new Set(Array.from(requiredIds || [], String));
  const excludedSet = new Set(Array.from(excludedIds || [], String));
  const hints = catalogStylingDiscoveryHints({ style, occasion, brief, weather });
  const preferredSubtypes = new Set(
    (Array.isArray(hints?.preferredSubtypes) ? hints.preferredSubtypes : [])
      .map((s) => String(s || '').trim().toLowerCase().replace(/_/g, '-')),
  );
  const workout = isWorkoutOccasion(occasion);
  const cold = Number.isFinite(weather?.temp) && weather.temp < 5;

  // Apply budget-headroom & hot-weather breathability sorting first across ALL 8,554 SKUs!
  const prioritized = prioritizeCatalogForBudgetAndClimate(items, {
    budget,
    priceToBudgetCurrency,
    weather,
    fxSnapshot,
    requiredIds,
  });

  const pinned = [];
  const bySlot = Object.fromEntries(CANONICAL_SLOTS.map((s) => [s, []]));
  for (const item of prioritized) {
    if (!item?.id) continue;
    const id = String(item.id);
    if (pinnedSet.has(id)) {
      pinned.push(item);
      continue;
    }
    if (excludedSet.has(id)) continue;
    if (bySlot[item.macroCategory]) bySlot[item.macroCategory].push(item);
  }

  const slotScore = (item) => {
    const sub = String(item.subtype || item.fashion_subtype || '').trim().toLowerCase().replace(/_/g, '-');
    const name = String(item.name || '').toLowerCase();
    let score = 0;
    if (preferredSubtypes.has(sub)) score += 3;
    if (currentVisual(item)) score += 1.5;
    if (workout && /\b(?:training|running|gym|athletic|jogger|activewear|sport)\b/.test(name)) score += 4;
    if (cold && item.macroCategory === 'outerwear' && /\b(?:coat|overcoat|puffer|parka|wool)\b/.test(name)) score += 3;
    if (cold && item.macroCategory === 'shoes' && /\b(?:boot|chelsea)\b/.test(name)) score += 3;
    return score;
  };

  const windowedBySlot = {};
  for (const slot of CANONICAL_SLOTS) {
    const list = bySlot[slot];
    // Preserve budget/climate priority tier while lifting style-relevant + visually profiled items
    const sorted = [...list].sort((a, b) => slotScore(b) - slotScore(a));
    // Interleave balanced brand/subtype diversity (up to 100/slot) + top prioritized tail up to `perSlot`
    const capBalanced = Math.min(100, perSlot);
    const diverse = slot === 'shoes'
      ? balancedShoeCatalogWindow(sorted, capBalanced)
      : balancedGarmentCatalogWindow(sorted, capBalanced);
    const chosen = [...diverse];
    const seenIds = new Set(chosen.map((it) => String(it.id)));
    for (const item of list) {
      if (chosen.length >= perSlot) break;
      if (!seenIds.has(String(item.id))) {
        seenIds.add(String(item.id));
        chosen.push(item);
      }
    }
    windowedBySlot[slot] = chosen;
  }

  // Round-robin across the 4 slots so every slot stays evenly represented
  const out = [...pinned];
  const seen = new Set(pinned.map((it) => String(it.id)));
  for (let i = 0; i < perSlot; i++) {
    for (const slot of CANONICAL_SLOTS) {
      const it = windowedBySlot[slot][i];
      if (it && !seen.has(String(it.id))) {
        seen.add(String(it.id));
        out.push(it);
      }
    }
  }
  return out;
}

/**
 * Dev-interface visual status vocabulary.
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
const referencePhotosDir = [
  process.env.STYLIST6_REFERENCE_PHOTOS_DIR,
  path.join(root, 'image-teaste'),
  '/Users/v.zahid/AIWardrobe/image-teaste',
].find((candidate) => {
  if (!candidate) return false;
  try {
    return fs.statSync(candidate).isDirectory();
  } catch {
    return false;
  }
}) || path.join(root, 'image-teaste');

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

// Fit raw bytes to the hosted critic's contract (JPEG, <=512KiB, <=2048px).
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
async function resolveProductAsset(product, fetchImpl = fetch, photoByteCache = null, signal = null) {
  const url = String(product?.imageUrl || product?.image_url || '');
  if (!HOST_ALLOWLIST.has(hostOf(url))) throw new Error('product_host_not_allowlisted');
  if (fetchImpl === fetch && productPhotoCache.has(url)) {
    const cached = productPhotoCache.get(url);
    if (photoByteCache && cached?.bytes) photoByteCache.set(url, cached.bytes);
    return cached;
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 5000);
  const combinedSignal = signal ? AbortSignal.any([signal, controller.signal]) : controller.signal;
  try {
    const response = await fetchImpl(url, {
      signal: combinedSignal, redirect: 'follow',
      headers: { 'User-Agent': 'AIWardrobe-Stylist6-Dev/1.0', Accept: 'image/*,*/*;q=0.8' },
    });
    if (!response.ok) throw new Error('product_image_unresolved');
    const raw = Buffer.from(await response.arrayBuffer());
    if (raw.length > MAX_PRODUCT_IMAGE_BYTES) throw new Error('image_size_invalid');
    const fitted = await prepareVisualBytes(raw);
    const asset = { bytes: fitted, sha256: createHash('sha256').update(fitted).digest('hex') };
    if (fetchImpl === fetch) productPhotoCache.set(url, asset);
    if (photoByteCache) photoByteCache.set(url, fitted);
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
  const vfColor = item?.visualFashion?.attributes?.color;
  const resolvedColor = (item?.color && item.color !== 'unknown')
    ? item.color
    : (vfColor && Number.isFinite(vfColor.confidence) && vfColor.confidence >= 0.8 &&
        (!Number.isFinite(vfColor.margin) || vfColor.margin >= 0.1) &&
        typeof vfColor.value === 'string' && vfColor.value && vfColor.value !== 'unknown'
        ? vfColor.value.toLowerCase()
        : 'unknown');
  return {
    id: String(item.id),
    type: item.type || item.macroCategory,
    macroCategory: item.macroCategory,
    subtype: item.subtype || item.fashion_subtype || (typeof item.type === 'string' && item.type !== item.macroCategory ? item.type : '') || '',
    color: resolvedColor,
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
    priceAgeDays: Number.isFinite(item.priceAgeDays) ? item.priceAgeDays : null,
    priceFreshness: typeof item.priceFreshness === 'string' && item.isShopItem === true
      ? item.priceFreshness
      : null,
    ...(offerVerification ? { offerVerification } : {}),
  };
}

function buildBridgePersonalStyleDna(request, style, rawWardrobeRows = []) {
  const rawPrefs = request.preferences && typeof request.preferences === 'object' ? request.preferences : {};
  const mergedPrefs = {
    ...rawPrefs,
    ...(Array.isArray(request.dislikedPatterns) && !rawPrefs.dislikedPatterns
      ? { dislikedPatterns: request.dislikedPatterns }
      : {}),
  };
  const safeAppearance = request.appearanceProfile && request.appearanceProfile.consentGranted === true
    ? { ...request.appearanceProfile, provenance: 'opt_in_photo_vlm' }
    : null;
  const hasInput = Boolean(
    request.presentation ||
    request.bodyProfile ||
    safeAppearance ||
    request.ownedWardrobeSummary ||
    (Array.isArray(rawWardrobeRows) && rawWardrobeRows.length > 0) ||
    Object.keys(mergedPrefs).length > 0,
  );
  if (!hasInput) return { personalStyleDna: null, styleDnaPromptBlock: '', safeAppearance: null };
  const personalStyleDna = normalizePersonalStyleDna({
    preferences: mergedPrefs,
    style,
    presentation: request.presentation || request.bodyProfile?.gender || null,
    body: request.bodyProfile || null,
    appearance: safeAppearance,
    wardrobeItems: Array.isArray(rawWardrobeRows) && rawWardrobeRows.length > 0 ? rawWardrobeRows : null,
    ...(request.ownedWardrobeSummary && (!Array.isArray(rawWardrobeRows) || rawWardrobeRows.length === 0)
      ? {
          ownedWardrobeDna: {
            totalItems: Number(request.ownedWardrobeSummary.totalItems) || 0,
            dominantPalette: Array.isArray(request.ownedWardrobeSummary.dominantColors)
              ? request.ownedWardrobeSummary.dominantColors
              : [],
            dominantCategories: Array.isArray(request.ownedWardrobeSummary.dominantCategories)
              ? request.ownedWardrobeSummary.dominantCategories
              : [],
            topBrands: Array.isArray(request.ownedWardrobeSummary.topBrands)
              ? request.ownedWardrobeSummary.topBrands
              : [],
          },
        }
      : {}),
  });
  return {
    personalStyleDna,
    styleDnaPromptBlock: personalStyleDna ? buildStyleDnaPromptBlock(personalStyleDna) : '',
    safeAppearance,
  };
}

async function resolveBridgeCatalogRows({ rawCatalogRows, sourceMode, options }) {
  const allowLive = options.liveServer === true || options.allowLiveCatalogHydration === true;
  if (sourceMode === 'wardrobe') {
    return {
      rows: [],
      catalogSource: 'wardrobe_only',
      visualHydration: { attempted: 0, fetched: 0, attached: 0, drifted: 0, status: 'skipped_wardrobe_mode' },
    };
  }
  const clientRows = Array.isArray(rawCatalogRows) ? rawCatalogRows : [];
  if (allowLive && options.skipHydration !== true && clientRows.length < MIN_FULL_CATALOG_ROWS) {
    const live = await hydrateFullLiveCatalog(options);
    if (Array.isArray(live.rows) && live.rows.length > 0) {
      // Merge any extra client-supplied rows (such as pinned anchors) without losing full catalog
      const byId = new Map(live.rows.map((r) => [String(r.id), r]));
      for (const row of clientRows) {
        if (row && typeof row === 'object' && row.id && !byId.has(String(row.id))) {
          byId.set(String(row.id), 'is_active' in row ? row : { ...row, is_active: true });
        }
      }
      return {
        rows: [...byId.values()],
        catalogSource: `supabase_live_${live.status}`,
        visualHydration: {
          attempted: live.totalRows,
          fetched: live.visualAttached,
          attached: live.visualAttached,
          drifted: 0,
          status: live.status === 'cached' ? 'ok_cached' : 'ok',
        },
      };
    }
  }
  if (!clientRows.length) {
    return {
      rows: [],
      catalogSource: 'empty',
      visualHydration: { attempted: 0, fetched: 0, attached: 0, drifted: 0, status: 'empty_pool' },
    };
  }
  const liveRows = clientRows.map((row) =>
    row && typeof row === 'object' && !('is_active' in row) ? { ...row, is_active: true } : row,
  );
  const visualHydration = options.skipHydration === true
    ? { attempted: 0, fetched: 0, attached: 0, drifted: 0, status: 'skipped_by_option' }
    : await hydrateVisualProfiles(liveRows);
  return {
    rows: liveRows,
    catalogSource: 'client_fixture',
    visualHydration,
  };
}

function createBridgeAppearanceProvider(options = {}) {
  const { nvidiaKey, openrouterKey, hasNvidiaKey, hasOpenrouterKey } = resolveAiKeys(options);
  if (!hasNvidiaKey && !hasOpenrouterKey) return null;
  const router = createProviderRouter({
    nvidiaApiKey: hasNvidiaKey ? nvidiaKey : '',
    openrouterApiKey: hasOpenrouterKey ? openrouterKey : '',
    ...(options.fetchImpl ? { fetchImpl: options.fetchImpl } : {}),
  });
  return async ({ mimeType, imageBase64, prompt, responseFormat }) => {
    const res = await router.callVisionModel({
      preferCrossLabFastVision: hasOpenrouterKey,
      allowFailover: true,
      parseJson: true,
      timeoutMs: 18_000,
      primaryTimeoutMs: 10_000,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: prompt },
            { type: 'image_url', image_url: { url: `data:${mimeType};base64,${imageBase64}` } },
          ],
        },
      ],
      response_format: responseFormat,
      max_tokens: 500,
      temperature: 0.1,
    });
    if (!res?.ok) throw new Error(res?.error || 'provider_call_failed');
    if (res.parsedJson && typeof res.parsedJson === 'object') return res.parsedJson;
    if (typeof res.content === 'string') return JSON.parse(res.content);
    throw new Error('appearance_invalid_json');
  };
}

/**
 * Local dev-bridge opt-in Appearance Analysis (`action: 'analyze_appearance'`
 * and `POST /analyze-appearance`). Strictly enforces fail-closed consent.
 */
export async function handleDevBridgeAnalyzeAppearance(payload, options = {}) {
  const started = Date.now();
  const source = payload?.request && typeof payload.request === 'object'
    ? { ...payload, ...payload.request }
    : (payload || {});
  const consent = source.consentGranted === true || source.consent === true;
  if (!consent) {
    return {
      status: 200,
      body: {
        success: true,
        action: 'analyze_appearance',
        status: 'consent_required',
        appearance: null,
        generationVersion: STYLIST6_DEV_ENGINE_VERSION,
      },
    };
  }
  const providerFn = typeof options.providerFn === 'function'
    ? options.providerFn
    : createBridgeAppearanceProvider(options);
  const cache = options.appearanceCache || devAppearanceCache;
  const result = await analyzeUserAppearance({
    photoUrl: source.photoUrl,
    photoBytes: source.photoBytes,
    consent: true,
    userId: typeof source.userId === 'string' && source.userId.trim() ? source.userId.trim().slice(0, 64) : 'dev-simulator',
    providerFn,
    cache,
    ...(options.fetchImpl ? { fetchFn: options.fetchImpl } : {}),
    ...(typeof options.dnsLookupFn === 'function' ? { dnsLookupFn: options.dnsLookupFn } : {}),
    ...(Array.isArray(source.allowedHosts) ? { allowedHosts: source.allowedHosts } : {}),
  });
  process.stdout.write(
    `[stylist6-dev-bridge] analyze_appearance status=${result.status} cached=${result.cached === true} latencyMs=${Date.now() - started}\n`,
  );
  if (result.status === 'blocked_url' || result.status === 'invalid_photo') {
    return {
      status: 400,
      body: {
        success: false,
        code: 'invalid_stylist6_request',
        stage: 'stylist6_appearance',
        reason: result.reason || result.status,
        generationVersion: STYLIST6_DEV_ENGINE_VERSION,
      },
    };
  }
  if (result.status === 'ok' || result.status === 'low_confidence') {
    return {
      status: 200,
      body: {
        success: true,
        action: 'analyze_appearance',
        status: result.status === 'ok' ? 'analyzed' : 'low_confidence',
        appearance: result.appearance
          ? {
              ...result.appearance,
              consentGranted: true,
              ...(result.photoSha256 ? { photoSha256: result.photoSha256 } : {}),
            }
          : null,
        cached: result.cached === true,
        photoSha256: result.photoSha256 || null,
        generationVersion: STYLIST6_DEV_ENGINE_VERSION,
      },
    };
  }
  return {
    status: 200,
    body: {
      success: true,
      action: 'analyze_appearance',
      status: 'unavailable',
      appearance: null,
      reason: result.reason || result.status,
      generationVersion: STYLIST6_DEV_ENGINE_VERSION,
    },
  };
}

/**
 * Local dev-bridge 1-Tap Surgical Slot Swap (`action: 'swap_slot'` and
 * `POST /swap-slot`): locks 3 items, searches full hygienic catalog with live
 * CBU FX, supports Visual Swap Boards (`selectVisualSlotReplacement`) + hard
 * gate & budget verification + updated Stylist's Notes + honest offer truth.
 */
export async function handleDevBridgeSwapSlot(payload, options = {}) {
  const started = Date.now();
  const source = payload?.request && typeof payload.request === 'object'
    ? { ...payload, ...payload.request }
    : (payload || {});
  const outfit = source.outfit && typeof source.outfit === 'object' && !Array.isArray(source.outfit)
    ? source.outfit
    : null;
  const targetSlot = typeof source.targetSlot === 'string' ? source.targetSlot.trim().toLowerCase() : '';
  if (!outfit || !Array.isArray(outfit.items) || outfit.items.length < 3 || outfit.items.length > 4 || !VALID_SWAP_SLOTS.has(targetSlot)) {
    return {
      status: 400,
      body: {
        success: false,
        code: 'invalid_stylist6_request',
        stage: 'stylist6_swap',
        generationVersion: STYLIST6_DEV_ENGINE_VERSION,
      },
    };
  }
  const rawContext = source.context && typeof source.context === 'object' && !Array.isArray(source.context)
    ? source.context
    : source;
  const style = typeof rawContext.style === 'string' && rawContext.style.trim() ? rawContext.style.trim() : 'classic';
  const occasion = typeof rawContext.occasion === 'string' && rawContext.occasion.trim() ? rawContext.occasion.trim() : 'Casual';
  const sourceMode = normalizeBridgeSourceMode(rawContext.sourceMode ?? source.sourceMode) || 'catalog';
  const language = normalizeStylistLanguage(rawContext.language ?? rawContext.locale ?? source.language);
  const budget = rawContext.budget?.amount && rawContext.budget?.currency &&
    Number.isFinite(Number(rawContext.budget.amount)) && Number(rawContext.budget.amount) > 0 &&
    /^[A-Z]{3}$/.test(String(rawContext.budget.currency))
    ? { amount: Number(rawContext.budget.amount), currency: String(rawContext.budget.currency) }
    : undefined;
  const weather = resolveStylistWeather(rawContext.brief || '', rawContext.weather || {});

  const shouldHydrateFx = options.fxSnapshot !== undefined ||
    ((options.liveServer === true || options.allowLiveFx === true) && options.skipHydration !== true);
  const fxRes = shouldHydrateFx ? await fetchLiveCbuFxSnapshot(options) : { snapshot: null, source: 'skipped' };
  const fxSnapshot = fxRes.snapshot;
  const priceToBudgetCurrency = fxSnapshot
    ? (item, currency = budget?.currency) => {
        if (typeof currency !== 'string' || !currency.trim()) return NaN;
        const converted = referenceAmount(item.price, item.currency, currency, fxSnapshot);
        return Number.isFinite(converted) ? converted : NaN;
      }
    : (item, currency = budget?.currency) => (currency && item.currency === currency ? item.price : NaN);

  const rawCatalogRows = Array.isArray(payload?.catalogRows)
    ? payload.catalogRows
    : Array.isArray(source.catalogRows)
      ? source.catalogRows
      : [];
  const rawWardrobeRows = Array.isArray(payload?.wardrobeItems)
    ? payload.wardrobeItems
    : Array.isArray(source.wardrobeItems)
      ? source.wardrobeItems
      : [];
  const resolvedCat = await resolveBridgeCatalogRows({ rawCatalogRows, sourceMode, options });
  const rawShopItems = resolvedCat.rows.length ? adaptCatalog(resolvedCat.rows).items : [];
  const qualityMap = options.qualityMap
    ? buildQualityLookupMap(options.qualityMap)
    : devQualityFlywheel.getLookupMap();
  const hygiene = filterHygienicCatalog(
    rawShopItems,
    {
      occasion,
      brief: rawContext.brief,
      presentation: rawContext.presentation,
      bodyProfile: rawContext.bodyProfile,
    },
    qualityMap,
  );
  const shopItems = hygiene.items;
  const wardrobeItems = sourceMode !== 'catalog'
    ? rawWardrobeRows.map(adaptWardrobeRow).filter(Boolean)
    : [];
  const pool = sourceMode === 'catalog'
    ? shopItems
    : sourceMode === 'wardrobe'
      ? wardrobeItems
      : [...wardrobeItems, ...shopItems];
  const feasible = budget
    ? pool.filter((item) => item.isShopItem !== true || Number.isFinite(priceToBudgetCurrency(item, budget.currency)))
    : pool;
  if (!feasible.length) {
    return {
      status: 502,
      body: {
        success: false,
        code: 'stylist6_no_feasible_looks',
        stage: 'stylist6_swap',
        generationVersion: STYLIST6_DEV_ENGINE_VERSION,
      },
    };
  }

  // Rehydrate locked outfit items from pool when available so hardGate/ranker see full metadata,
  // while preserving original client item fields verbatim for any non-catalog/wardrobe piece.
  const poolById = new Map(pool.map((item) => [String(item.id), item]));
  const hydratedLook = {
    ...outfit,
    items: outfit.items.map((item) => poolById.get(String(item?.id)) || item),
  };
  const { personalStyleDna, styleDnaPromptBlock } = buildBridgePersonalStyleDna(rawContext, style, rawWardrobeRows);
  const excludedIds = [
    ...(Array.isArray(source.excludedIds) ? source.excludedIds : []),
    ...(Array.isArray(rawContext.excludeItemIds) ? rawContext.excludeItemIds : []),
  ].map((id) => String(id || '').trim()).filter(Boolean);

  const { nvidiaKey, openrouterKey, hasNvidiaKey, hasOpenrouterKey } = resolveAiKeys(options);
  const visualEnabled = rawContext.enableVisual !== false && source.enableVisual !== false && (hasNvidiaKey || hasOpenrouterKey);
  const allowLocalShowroom = options.allowLocalShowroomVisualComposition ?? true;
  const photoByteCache = options.photoByteCache || sharedPhotoByteCache;
  const fetchImpl = options.fetchImpl || fetch;
  const resolveProduct = (product, sig) => resolveProductAsset(product, fetchImpl, photoByteCache, sig);
  const providerRouter = (visualEnabled || typeof options.visualReplacementSelector === 'function') && (hasNvidiaKey || hasOpenrouterKey)
    ? createProviderRouter({
        nvidiaApiKey: hasNvidiaKey ? nvidiaKey : '',
        openrouterApiKey: hasOpenrouterKey ? openrouterKey : '',
        ...(options.fetchImpl ? { fetchImpl: options.fetchImpl } : {}),
      })
    : null;
  const authorizePhoto = (url) => HOST_ALLOWLIST.has(hostOf(url));
  const debugBoardDir = String(options.debugBoardDir || process.env.STYLIST6_DEBUG_BOARD_DIR || '').trim();
  const debugSlug = String(
    source.debugScenarioTag || rawContext.debugScenarioTag || `${style}_${occasion}`
  ).toLowerCase().replace(/[^a-z0-9_-]+/g, '_').replace(/^_+|_+$/g, '');
  let visualReplacementSelector = typeof options.visualReplacementSelector === 'function'
    ? options.visualReplacementSelector
    : null;
  if (!visualReplacementSelector && allowLocalShowroom && visualEnabled) {
    const atelierMod = await loadOptionalVisualAtelier();
    if (typeof atelierMod?.selectVisualSlotReplacement === 'function') {
      const baseSelector = atelierMod.selectVisualSlotReplacement;
      visualReplacementSelector = async (selArgs) => {
        const res = await baseSelector(selArgs);
        if (debugBoardDir && Buffer.isBuffer(res?.swapBoard?.jpegBuffer)) {
          try {
            fs.mkdirSync(debugBoardDir, { recursive: true });
            fs.writeFileSync(path.join(debugBoardDir, `swap_board_${debugSlug}.jpg`), res.swapBoard.jpegBuffer);
          } catch { /* non-fatal debug dump */ }
        }
        return res;
      };
    }
  }
  const swapVisualCritic = typeof options.visualCritic === 'function'
    ? options.visualCritic
    : null;
  const boardRenderer = swapVisualCritic
    ? (items, opts = {}) => renderEditorialLookBoard(items, {
        resolvePhoto: resolveProduct,
        authorizePhoto,
        photoCache: photoByteCache,
        anonymizeLabels: true,
        ...opts,
      })
    : null;

  const swapResult = await swapSingleSlot({
    look: hydratedLook,
    targetSlot,
    reason: typeof source.reason === 'string' ? source.reason : '',
    catalogPool: feasible,
    context: {
      ...rawContext,
      style,
      occasion,
      weather,
      sourceMode,
      budget,
    },
    personalStyleDna,
    ...(styleDnaPromptBlock ? { styleDnaPromptBlock } : {}),
    hardGateValidator: async (candidateItems) => {
      if (sourceMode === 'catalog' && (!Array.isArray(candidateItems) || candidateItems.length !== 4)) {
        return { ok: false, reason: 'catalog_four_item_contract' };
      }
      const feasibility = devLookFeasibility(candidateItems);
      if (feasibility === false || (feasibility && typeof feasibility === 'object' && feasibility.ok === false)) {
        return { ok: false, reason: feasibility?.reason || 'infeasible_look' };
      }
      if (budget) {
        let total = 0;
        for (const it of candidateItems) {
          if (it?.isShopItem !== false) {
            const conv = priceToBudgetCurrency(it, budget.currency);
            if (!Number.isFinite(conv)) return { ok: false, reason: 'currency_unconvertible' };
            total += conv;
          }
        }
        if (total > budget.amount) return { ok: false, reason: 'over_budget' };
      }
      return { ok: true };
    },
    visualCritic: swapVisualCritic,
    ...(boardRenderer ? { boardRenderer } : {}),
    ...(allowLocalShowroom && visualReplacementSelector ? { visualReplacementSelector } : {}),
    ...(allowLocalShowroom && providerRouter ? { providerRouter } : {}),
    ...(allowLocalShowroom && (visualReplacementSelector || providerRouter) ? {
      photoCache: photoByteCache,
      resolvePhoto: resolveProduct,
      authorizePhoto,
    } : {}),
    excludedItemIds: excludedIds,
    qualityMap,
    priceToBudgetCurrency,
    deadlineMs: Math.min(20_000, Math.max(1000, Number(options.timeoutMs) || 18_000)),
  });

  if (!swapResult.swapped || !swapResult.look || !swapResult.newItem) {
    process.stdout.write(
      `[stylist6-dev-bridge] swap_slot status=${swapResult.status} slot=${targetSlot} latencyMs=${Date.now() - started}\n`,
    );
    return {
      status: 502,
      body: {
        success: false,
        code: 'stylist6_no_feasible_looks',
        stage: 'stylist6_swap',
        swapStatus: swapResult.status,
        generationVersion: STYLIST6_DEV_ENGINE_VERSION,
      },
    };
  }

  const rankScore = Number.isFinite(Number(outfit.confidence)) ? Number(outfit.confidence) : 0.85;
  const mappedItems = swapResult.look.items.map((item) => mapDeckItem(
    item,
    rankScore,
    verifyMerchantOffer(item, { targetMarket: 'UZ' }),
  ));
  const mappedReplacement = mappedItems.find((it) => String(it.macroCategory || '').toLowerCase() === targetSlot)
    || mapDeckItem(swapResult.newItem, rankScore, verifyMerchantOffer(swapResult.newItem, { targetMarket: 'UZ' }));
  const notes = buildStylistNotes(
    { ...swapResult.look, items: mappedItems },
    { style, occasion, language, personalStyleDna, preferences: rawContext.preferences || {} },
    outfit.direction || { id: 'signature' },
    { personalStyleDna, visualVerdict: swapResult.critique },
  );
  const updatedOutfit = {
    ...outfit,
    id: String(outfit.id || 'outfit_1'),
    items: mappedItems,
    offerVerification: summarizeDeckOffers([{ items: mappedItems }]),
    description: notes.stylistNote || outfit.description || '',
    stylistNote: notes.stylistNote,
    whyItFlattersYou: notes.whyItFlattersYou,
    stylingTips: notes.stylingTips,
    colorStory: notes.colorStory,
    silhouetteFormula: notes.silhouetteFormula,
    engineVersion: STYLIST6_DEV_ENGINE_VERSION,
  };
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
      : { active: false, reason: 'same_currency_only', budgetCurrency: budget.currency };

  process.stdout.write(
    `[stylist6-dev-bridge] swap_slot slot=${targetSlot} prev=${swapResult.previousItem?.id || '-'} next=${mappedReplacement.id} ` +
    `visualVerified=${swapResult.visualVerified === true} latencyMs=${Date.now() - started}\n`,
  );

  return {
    status: 200,
    body: {
      success: true,
      action: 'swap_slot',
      status: 'swapped',
      generationVersion: STYLIST6_DEV_ENGINE_VERSION,
      source: 'stylist-6-dev-local',
      sourceMode,
      language,
      currencyConversion,
      swappedSlot: targetSlot,
      previousItemId: swapResult.previousItem?.id ? String(swapResult.previousItem.id) : null,
      replacementItem: mappedReplacement,
      visualVerified: swapResult.visualVerified === true,
      ...(swapResult.visualSwapUsed ? {
        visualSwapUsed: true,
        visuallySelected: swapResult.visuallySelected === true,
        visualSwapToken: swapResult.visualSwapToken || null,
        visualSwapRationale: swapResult.visualSwapRationale || null,
      } : {}),
      outfit: updatedOutfit,
      outfits: [updatedOutfit],
    },
  };
}

export async function generateStylist6Deck(payload, options = {}) {
  const rawAction = typeof payload?.action === 'string'
    ? payload.action.trim().toLowerCase()
    : (typeof payload?.request?.action === 'string' ? payload.request.action.trim().toLowerCase() : '');
  if (rawAction === 'swap_slot') {
    return handleDevBridgeSwapSlot(payload, options);
  }
  if (rawAction === 'analyze_appearance') {
    return handleDevBridgeAnalyzeAppearance(payload, options);
  }

  const started = Date.now();
  const budgetMs = Number.isFinite(Number(options.timeoutMs)) && Number(options.timeoutMs) > 0
    ? Number(options.timeoutMs)
    : BRIDGE_GENERATION_BUDGET_MS;
  const remainingMs = () => budgetMs - (Date.now() - started);

  const { request = {}, catalogRows = [] } = payload || {};
  const feedbackRows = Array.isArray(payload?.feedbackTaste)
    ? payload.feedbackTaste.slice(0, 60)
    : (Array.isArray(request?.feedbackTaste) ? request.feedbackTaste.slice(0, 60) : []);
  const feedbackTaste = feedbackTasteModel(feedbackRows, Date.now());

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

  const resolvedCat = await resolveBridgeCatalogRows({ rawCatalogRows, sourceMode, options });
  const visualHydration = resolvedCat.visualHydration;
  if (sourceMode === 'catalog' && !resolvedCat.rows.length) {
    return { status: 400, body: { success: false, code: 'stylist6_empty_catalog', stage: 'catalog_adaptation' } };
  }

  const budget = request.budget?.amount && request.budget?.currency ? request.budget : undefined;
  if (budget && !/^[A-Z]{3}$/.test(String(budget.currency))) {
    return { status: 400, body: { success: false, code: 'invalid_stylist6_request', stage: 'stylist6_admission' } };
  }
  const shouldHydrateFx = options.fxSnapshot !== undefined ||
    ((options.liveServer === true || options.allowLiveFx === true) && options.skipHydration !== true);
  const fxRes = shouldHydrateFx ? await fetchLiveCbuFxSnapshot(options) : { snapshot: null, source: 'skipped' };
  const fxSnapshot = fxRes.snapshot;
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
      : { active: false, reason: 'same_currency_only', budgetCurrency: budget.currency };
  const priceToBudgetCurrency = fxSnapshot
    ? (item, currency = budget?.currency) => {
        if (typeof currency !== 'string' || !currency.trim()) return NaN;
        const converted = referenceAmount(item.price, item.currency, currency, fxSnapshot);
        return Number.isFinite(converted) ? converted : NaN;
      }
    : (item, currency = budget?.currency) => (currency && item.currency === currency ? item.price : NaN);

  const rawShopItems = resolvedCat.rows.length ? adaptCatalog(resolvedCat.rows).items : [];
  const qualityMap = options.qualityMap
    ? buildQualityLookupMap(options.qualityMap)
    : devQualityFlywheel.getLookupMap();
  const catalogHygiene = filterHygienicCatalog(
    rawShopItems,
    {
      occasion,
      brief: request.brief,
      presentation: request.presentation,
      bodyProfile: request.bodyProfile,
      requiredIds,
    },
    qualityMap,
  );
  const shopItems = catalogHygiene.items;
  const wardrobeItems = sourceMode !== 'catalog'
    ? rawWardrobeRows.map(adaptWardrobeRow).filter(Boolean)
    : [];
  const rawItems = sourceMode === 'catalog'
    ? rawShopItems
    : sourceMode === 'wardrobe'
      ? wardrobeItems
      : [...wardrobeItems, ...rawShopItems];
  const items = sourceMode === 'catalog'
    ? shopItems
    : sourceMode === 'wardrobe'
      ? wardrobeItems
      : [...wardrobeItems, ...shopItems];
  if (!rawItems.length || !items.length) {
    return { status: 502, body: { success: false, code: 'stylist6_empty_catalog', stage: 'catalog_adaptation' } };
  }
  const itemIds = new Set(rawItems.map((item) => String(item.id)));
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

  const budgetFeasible = budget
    ? items.filter((item) => item.isShopItem !== true || Number.isFinite(priceToBudgetCurrency(item, budget.currency)))
    : items;
  if (!budgetFeasible.length) {
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

  const pool = sourceMode === 'wardrobe'
    ? budgetFeasible
    : selectBridgeCatalogWindow(budgetFeasible, {
        style,
        occasion,
        brief: request.brief || '',
        weather,
        budget,
        fxSnapshot,
        priceToBudgetCurrency,
        requiredIds,
        excludedIds,
      });

  const { personalStyleDna, styleDnaPromptBlock, safeAppearance } = buildBridgePersonalStyleDna(
    request,
    style,
    rawWardrobeRows,
  );
  const catalogByIdMap = new Map(pool.map((item) => [String(item.id), item]));
  const tasteProfile = buildUnifiedTasteProfile({
    explicitFeedback: feedbackRows,
    savedOutfits: Array.isArray(request.savedOutfits) ? request.savedOutfits : [],
    wearLogs: Array.isArray(request.wearLogs) ? request.wearLogs : [],
    brandClicks: Array.isArray(request.brandClicks) ? request.brandClicks : [],
    catalogById: catalogByIdMap,
    nowMs: Date.now(),
  });

  // Catalog mode strictly enforces exact 4 items per outfit (top, outerwear,
  // bottom, shoes); wardrobe & mixed modes retain flexible layering.
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
    optionalLayer: sourceMode !== 'catalog',
    preferences: request.preferences && typeof request.preferences === 'object' ? request.preferences : {},
    language,
    ...(request.presentation ? { presentation: request.presentation } : {}),
    ...(request.bodyProfile ? { bodyProfile: request.bodyProfile } : {}),
    ...(safeAppearance ? { appearanceProfile: safeAppearance } : {}),
  };

  const visualAsked = request.enableVisual === true;
  const { nvidiaKey, openrouterKey, hasNvidiaKey, hasOpenrouterKey } = resolveAiKeys(options);
  const hasVisionKeys = hasNvidiaKey || hasOpenrouterKey;
  const fastVisualMode = request.fastVisualMode !== false && (hasOpenrouterKey || request.fastVisualMode === true);
  const allowLocalShowroomVisualComposition = options.allowLocalShowroomVisualComposition ?? true;
  const fetchImpl = options.fetchImpl || fetch;
  const photoByteCache = options.photoByteCache || (options.fetchImpl
    ? createPhotoByteCache({ maxEntries: 64, maxBytes: 16 * 1024 * 1024 })
    : sharedPhotoByteCache);

  const hasLocalReference = Boolean(reference &&
    explicitLookbookPhotoRequested(request, reference) &&
    referenceFileFor(reference.referenceId));

  let creativePlan = { status: 'disabled', briefs: [], providerCalls: 0 };
  let visualCritic, canJudgeOutfit, studioBoardRenderer, prewarmCandidatePhotos;
  let selectDirectorFinalists = null;
  let selectCatalogCompositions = null;
  let selectCatalogRescueCompositions = null;
  let selectPhotoApprovedPortfolio = null;
  let visualAtelierCompose = typeof options.visualAtelierCompose === 'function' ? options.visualAtelierCompose : null;
  let visualReplacementSelector = typeof options.visualReplacementSelector === 'function' ? options.visualReplacementSelector : null;
  let providerRouter = null;
  let visualCriticActiveModel = hasOpenrouterKey ? OPENROUTER_VISION_MODEL : NVIDIA_VISION_MODEL;
  const visualEvents = [];
  let autonomousApplied = false;
  let computedDeck = null;

  // ── PRIMARY PATH (Phase 11 — Parity with Cloud Handler): Unified Autonomous Visual Stylist ──
  const autonomousKillSwitchOff = /^(0|false)$/i.test(String(process.env.STYLIST6_AUTONOMOUS_VISUAL_ENABLED || '').trim())
    || options.disableAutonomousStylist === true;
  const legacyAtelierSeamOverride = typeof options.visualAtelierCompose === 'function'
    && typeof options.autonomousStylist !== 'function';
  const canRunAutonomousVisual = Boolean(
    !autonomousKillSwitchOff &&
    !legacyAtelierSeamOverride &&
    visualAsked &&
    hasVisionKeys &&
    sourceMode === 'catalog',
  );

  if (canRunAutonomousVisual) {
    const resolveProduct = (product, sig) => resolveProductAsset(product, fetchImpl, photoByteCache, sig);
    const authorizePhoto = (url) => HOST_ALLOWLIST.has(hostOf(url));
    providerRouter = getSharedProviderRouter({
      nvidiaApiKey: hasNvidiaKey ? nvidiaKey : '',
      openrouterApiKey: hasOpenrouterKey ? openrouterKey : '',
      ...(typeof options.fetchImpl === 'function' ? { fetchImpl: options.fetchImpl } : {}),
    });
    const debugBoardDir = String(options.debugBoardDir || process.env.STYLIST6_DEBUG_BOARD_DIR || '').trim();
    const debugSlug = String(
      request.debugScenarioTag || payload?.debugScenarioTag || `${style}_${occasion}`
    ).toLowerCase().replace(/[^a-z0-9_-]+/g, '_').replace(/^_+|_+$/g, '');
    const autonomousRunner = typeof options.autonomousStylist === 'function'
      ? options.autonomousStylist
      : runAutonomousVisualStylist;
    let autonomousRes = null;
    try {
      autonomousRes = await autonomousRunner({
        request: input,
        catalog: budgetFeasible,
        personalStyleDna,
        styleDnaPromptBlock,
        tasteProfile,
        priceToBudgetCurrency,
        hardGate: async (lookItems) => devLookFeasibility(lookItems),
        providerRouter,
        nvidiaApiKey: hasNvidiaKey ? nvidiaKey : '',
        openrouterApiKey: hasOpenrouterKey ? openrouterKey : '',
        photoCache: photoByteCache,
        resolvePhoto: resolveProduct,
        authorizePhoto,
        ...(options.fetchImpl ? { fetchImpl: options.fetchImpl } : {}),
        timeoutMs: Math.min(62_000, Math.max(12_000, remainingMs() - 4_000)),
        hygieneAlreadyApplied: true,
        qualityMap,
        photoProvenance: 'host_allowlisted_staging',
        merchantRightsVerified: false,
        captureDebugBoards: Boolean(debugBoardDir),
      });
    } catch {
      autonomousRes = { status: 'fallback', reason: 'autonomous_error' };
    }
    if (debugBoardDir && autonomousRes?.debugBoards) {
      try {
        fs.mkdirSync(debugBoardDir, { recursive: true });
        for (const [key, buf] of Object.entries(autonomousRes.debugBoards)) {
          if (Buffer.isBuffer(buf)) {
            fs.writeFileSync(path.join(debugBoardDir, `${key}_${debugSlug}.jpg`), buf);
          }
        }
      } catch { /* non-fatal debug dump */ }
    }
    if (autonomousRes?.status === 'applied' && Array.isArray(autonomousRes?.deck?.outfits) && autonomousRes.deck.outfits.length > 0) {
      autonomousApplied = true;
      computedDeck = autonomousRes.deck;
      if (Array.isArray(autonomousRes.visualEvents)) {
        visualEvents.push(...autonomousRes.visualEvents);
      }
      visualCriticActiveModel = autonomousRes.diagnostics?.editorModel
        || autonomousRes.diagnostics?.composerModel
        || visualCriticActiveModel;
      creativePlan = { status: 'autonomous_visual', briefs: [], providerCalls: 0 };
    }
  }

  if (!autonomousApplied && visualAsked && hasVisionKeys && !isWorkoutOccasion(occasion) && options.disableCreativeDirector !== true) {
    creativePlan = await designCreativeBriefsWithFallback({
      catalog: pool,
      request: input,
      feedbackTaste,
      tasteProfile,
      personalStyleDna,
      styleDnaPromptBlock,
      cacheStore: devDirectorCache,
      allowIndependentFallback: true,
      parallelIndependentFallback: true,
      backupTimeoutMs: 6500,
      apiKey: hasNvidiaKey ? nvidiaKey : '',
      ...(hasOpenrouterKey ? { openrouterApiKey: openrouterKey } : {}),
      ...(fastVisualMode ? { fastMode: true } : {}),
      ...(options.fetchImpl ? { fetchImpl: options.fetchImpl } : {}),
      timeoutMs: Math.min(9500, Math.max(1000, remainingMs() - 7000)),
      highTaste: true,
    });
  }

  if (!autonomousApplied && visualAsked && hasVisionKeys) {
    const resolveProduct = (product, sig) => resolveProductAsset(product, fetchImpl, photoByteCache, sig);
    const authorizePhoto = (url) => HOST_ALLOWLIST.has(hostOf(url));
    providerRouter = createProviderRouter({
      nvidiaApiKey: hasNvidiaKey ? nvidiaKey : '',
      openrouterApiKey: hasOpenrouterKey ? openrouterKey : '',
      ...(options.fetchImpl ? { fetchImpl: options.fetchImpl } : {}),
    });

    if (!visualAtelierCompose || !visualReplacementSelector) {
      const atelierMod = await loadOptionalVisualAtelier();
      const debugBoardDir = String(options.debugBoardDir || process.env.STYLIST6_DEBUG_BOARD_DIR || '').trim();
      const debugSlug = String(
        request.debugScenarioTag || payload?.debugScenarioTag || `${style}_${occasion}`
      ).toLowerCase().replace(/[^a-z0-9_-]+/g, '_').replace(/^_+|_+$/g, '');
      if (!visualAtelierCompose && typeof atelierMod?.composeOutfitsVisually === 'function') {
        const baseComposer = atelierMod.composeOutfitsVisually;
        visualAtelierCompose = async (compArgs) => {
          const res = await baseComposer(compArgs);
          if (debugBoardDir && Buffer.isBuffer(res?.showroomBoard?.jpegBuffer)) {
            try {
              fs.mkdirSync(debugBoardDir, { recursive: true });
              fs.writeFileSync(path.join(debugBoardDir, `showroom_board_${debugSlug}.jpg`), res.showroomBoard.jpegBuffer);
            } catch { /* non-fatal debug dump */ }
          }
          return res;
        };
      }
      if (!visualReplacementSelector && typeof atelierMod?.selectVisualSlotReplacement === 'function') {
        const baseSelector = atelierMod.selectVisualSlotReplacement;
        visualReplacementSelector = async (selArgs) => {
          const res = await baseSelector(selArgs);
          if (debugBoardDir && Buffer.isBuffer(res?.swapBoard?.jpegBuffer)) {
            try {
              fs.mkdirSync(debugBoardDir, { recursive: true });
              fs.writeFileSync(path.join(debugBoardDir, `swap_board_${debugSlug}.jpg`), res.swapBoard.jpegBuffer);
            } catch { /* non-fatal debug dump */ }
          }
          return res;
        };
      }
    }

    studioBoardRenderer = (lookItems, opts = {}) => renderEditorialLookBoard(lookItems, {
      resolvePhoto: resolveProduct,
      authorizePhoto,
      photoCache: photoByteCache,
      anonymizeLabels: true,
      ...opts,
    });
    prewarmCandidatePhotos = async (candidateLooks) => {
      const urls = [...new Set(
        (Array.isArray(candidateLooks) ? candidateLooks : []).flatMap((look) =>
          (look?.items || look?.products || [])
            .map((item) => item?.imageUrl || item?.image_url)
            .filter(authorizePhoto),
        ),
      )].slice(0, MAX_VISUAL_CANDIDATES * 4);
      await Promise.allSettled(urls.map((imageUrl) => resolveProduct({ imageUrl })));
    };

    const rawVisualCritic = createHostedVisualCritic({
      provider: hasNvidiaKey ? 'nvidia' : 'openrouter',
      apiKey: hasNvidiaKey ? nvidiaKey : openrouterKey,
      openrouterApiKey: hasOpenrouterKey ? openrouterKey : '',
      model: hasNvidiaKey ? NVIDIA_VISION_MODEL : OPENROUTER_VISION_MODEL,
      enableFailover: hasOpenrouterKey || fastVisualMode,
      preferCrossLabFastVision: fastVisualMode && hasOpenrouterKey,
      fastVisualMode,
      boardRenderer: studioBoardRenderer,
      referenceMediaConfigured: hasLocalReference,
      referenceBlueprints: buildReferenceBlueprints(index),
      timeoutMs: Math.min(VISUAL_TIMEOUT_MS, Math.max(2000, remainingMs() - 1000)),
      providerTimeoutMs: (fastVisualMode && hasOpenrouterKey)
        ? Math.min(VISUAL_PROVIDER_TIMEOUT_MS, 14_000)
        : VISUAL_PROVIDER_TIMEOUT_MS,
      concurrency: Math.min(MAX_VISUAL_CANDIDATES, Math.max(VISUAL_CONCURRENCY, 6)),
      ...(options.fetchImpl ? { fetchImpl: options.fetchImpl } : {}),
      resolveReference: (id) => resolveReferenceAsset(id),
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
          ...(event.evaluatedEditorialBoard === true ? { evaluatedEditorialBoard: true } : {}),
          ...(event.routedModel ? { routedModel: event.routedModel } : {}),
          ...(event.routedProvider ? { routedProvider: event.routedProvider } : {}),
          ...(event.usage && typeof event.usage === 'object' ? { usage: event.usage } : {}),
        });
      },
    });
    visualCriticActiveModel = rawVisualCritic.activeModel || visualCriticActiveModel;
    visualCritic = async (candidates, critiqueRequest, criticOptions = {}) => rawVisualCritic(
      candidates,
      {
        ...critiqueRequest,
        ...(Number.isFinite(weather?.temp) ? { weather } : {}),
        ...(styleDnaPromptBlock ? { personalStyleDnaPromptBlock: styleDnaPromptBlock } : {}),
      },
      criticOptions,
    );
    canJudgeOutfit = async (look) => {
      if (hasLocalReference && reference && !referenceFileFor(reference.referenceId)) return false;
      return (look?.items || []).every((item) => authorizePhoto(item.imageUrl || item.image_url));
    };

    if (sourceMode === 'catalog' && !requiredIds.length) {
      selectDirectorFinalists = async ({ options: slateOptions, context: slateContext }) => {
        const left = remainingMs();
        if (left < 12_000 || !hasNvidiaKey) return { status: 'not_configured', tokens: [], providerCalls: 0 };
        return selectFashionSlate({
          options: slateOptions,
          context: slateContext,
          apiKey: nvidiaKey,
          ...(options.fetchImpl ? { fetchImpl: options.fetchImpl } : {}),
          timeoutMs: Math.min(2600, Math.max(600, left - 9000)),
        });
      };
      selectCatalogCompositions = async ({
        pool: compPool,
        perDirectionPools,
        personalStyleDna: dnaOpt,
        styleDnaPromptBlock: dnaBlockOpt,
        context: compContext,
        directions,
        layerRequired,
        allowLayer,
        outfitCount = 6,
        feedbackTaste: fbOpt,
        parallel = false,
      }) => {
        const left = remainingMs();
        if (left < 20_000) return { status: 'deadline', looks: [], providerCalls: 0 };
        return composeCatalogSkuLooks({
          pool: compPool,
          perDirectionPools,
          personalStyleDna: dnaOpt || personalStyleDna,
          styleDnaPromptBlock: dnaBlockOpt || styleDnaPromptBlock,
          context: compContext,
          directions,
          layerRequired,
          allowLayer,
          outfitCount,
          feedbackTaste: fbOpt,
          parallel,
          initialStudioDuel: parallel && hasNvidiaKey,
          apiKey: hasNvidiaKey ? nvidiaKey : '',
          ...(hasOpenrouterKey ? { openrouterApiKey: openrouterKey } : {}),
          ...(fastVisualMode ? { fastMode: true } : {}),
          ...(options.fetchImpl ? { fetchImpl: options.fetchImpl } : {}),
          timeoutMs: parallel ? Math.min(16_000, Math.max(2000, left - 18_000)) : Math.min(4000, Math.max(600, left - 18_000)),
        });
      };
      selectCatalogRescueCompositions = async ({
        pool: compPool,
        perDirectionPools,
        personalStyleDna: dnaOpt,
        styleDnaPromptBlock: dnaBlockOpt,
        context: compContext,
        directions,
        critiqueFeedback,
        layerRequired,
        allowLayer,
        feedbackTaste: fbOpt,
      }) => {
        const left = remainingMs();
        if (left < 14_000) return { status: 'deadline', looks: [], providerCalls: 0 };
        return composeCatalogSkuLooks({
          pool: compPool,
          perDirectionPools,
          personalStyleDna: dnaOpt || personalStyleDna,
          styleDnaPromptBlock: dnaBlockOpt || styleDnaPromptBlock,
          context: compContext,
          directions,
          critiqueFeedback,
          layerRequired,
          allowLayer,
          feedbackTaste: fbOpt,
          crossLabRevision: hasNvidiaKey,
          fastPhotoRevision: true,
          apiKey: hasNvidiaKey ? nvidiaKey : '',
          ...(hasOpenrouterKey ? { openrouterApiKey: openrouterKey } : {}),
          ...(fastVisualMode ? { fastMode: true } : {}),
          ...(options.fetchImpl ? { fetchImpl: options.fetchImpl } : {}),
          timeoutMs: Math.min(8000, Math.max(600, left - 12_000)),
        });
      };
      selectPhotoApprovedPortfolio = async ({ looks, critiques, portfolios, context: edContext, creativePlans }) => {
        const left = remainingMs();
        if (left < 4000) return { status: 'deadline', tokens: [], providerCalls: 0 };
        return selectPhotoEditorialDeck({
          looks,
          critiques,
          portfolios,
          context: edContext,
          creativePlans,
          apiKey: hasNvidiaKey ? nvidiaKey : '',
          openrouterApiKey: hasOpenrouterKey ? openrouterKey : '',
          enableCrossLabFailover: true,
          fastVisualMode,
          preserveDistinctVerifiedTriple: true,
          photoCache: photoByteCache,
          resolvePhoto: resolveProduct,
          authorizePhoto,
          independentVisionJury: hasNvidiaKey,
          crossLabTieBreak: hasNvidiaKey,
          ...(options.fetchImpl ? { fetchImpl: options.fetchImpl } : {}),
          timeoutMs: fastVisualMode ? Math.min(6000, Math.max(1200, left - 2000)) : Math.min(9600, Math.max(1000, left - 2500)),
        });
      };
    }
  }

  const visualAtelierConfigured = Boolean(
    visualAtelierCompose || (visualAsked && hasVisionKeys && sourceMode === 'catalog' && !requiredIds.length && allowLocalShowroomVisualComposition === false),
  );
  const visualAtelierOptions = (visualAsked && hasVisionKeys && allowLocalShowroomVisualComposition)
    ? {
        providerRouter,
        photoCache: photoByteCache,
        resolvePhoto: (product, sig) => resolveProductAsset(product, fetchImpl, photoByteCache, sig),
        authorizePhoto: (url) => HOST_ALLOWLIST.has(hostOf(url)),
        ...(options.fetchImpl ? { fetchImpl: options.fetchImpl } : {}),
        timeoutMs: Math.min(26_000, Math.max(3500, remainingMs() - 18_000)),
        maxPerSlot: 8,
      }
    : null;

  if (!autonomousApplied) {
    computedDeck = await recommendFashionDeck({
      request: input,
      catalog: pool,
      referenceBriefs: refs,
      aiCreativeBriefs: creativePlan.briefs,
      hardGate: async (lookItems) => devLookFeasibility(lookItems),
      priceToBudgetCurrency,
      ...(fxSnapshot ? { fxSnapshot } : {}),
      valueFxRates: sourceMode === 'catalog' && !budget ? fxSnapshot?.uzsPerUnit : null,
      feedbackTaste,
      personalStyleDna,
      styleDnaPromptBlock,
      tasteProfile,
      qualityFlywheel: devQualityFlywheel,
      qualityMap,
      fastVisualMode,
      ...(studioBoardRenderer ? { boardRenderer: studioBoardRenderer } : {}),
      photoByteCache,
      ...(prewarmCandidatePhotos ? { prewarmPhotos: prewarmCandidatePhotos } : {}),
      visualDeckReview: true,
      visualCandidateLimit: 8,
      allowUnreviewedVisualFill: false,
      photoRescueRemainingMs: visualCritic ? remainingMs : null,
      ...(visualAtelierCompose ? { visualAtelierCompose } : {}),
      visualAtelierAllowed: Boolean(allowLocalShowroomVisualComposition && visualAsked && hasVisionKeys),
      visualAtelierConfigured,
      ...(visualAtelierOptions ? { visualAtelierOptions } : {}),
      ...(visualReplacementSelector ? { visualReplacementSelector } : {}),
      ...(providerRouter ? { providerRouter } : {}),
      ...(selectDirectorFinalists ? { selectDirectorFinalists } : {}),
      ...(selectCatalogCompositions ? { selectCatalogCompositions } : {}),
      ...(selectCatalogRescueCompositions ? { selectCatalogRescueCompositions } : {}),
      ...(selectPhotoApprovedPortfolio ? { selectPhotoApprovedPortfolio } : {}),
      ...(visualCritic ? { visualCritic, canJudgeOutfit } : {}),
    });
  }

  // Enforce exact 4 authentic shoppable items per look in catalog mode.
  const deck = enforceCatalogFourItemDeck(computedDeck, sourceMode);
  const visualStatus = mapDevVisualStatus(deck);
  const providerCalls = visualEvents.filter((event) => event.dispatched).length;
  const displayedLooks = deck.outfits.length;
  const displayedJudged = deck.outfits.filter((look) => Array.isArray(look.visualEvidence)).length;
  const visualApproval = !visualAsked ? 'not_requested'
    : !(visualCritic || autonomousApplied) ? 'unavailable'
      : displayedLooks === 3 && displayedJudged === 3 ? 'full_three_verified'
        : displayedLooks > 0 && displayedJudged === displayedLooks ? 'verified_partial'
          : 'unverified_fallback';
  const visualReceipt = {
    model: visualAsked ? visualCriticActiveModel : null,
    ...(fastVisualMode ? { fastVisualMode: true } : {}),
    ...(autonomousApplied ? { autonomousVersion: AUTONOMOUS_STYLIST_VERSION } : {}),
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
  const visualAtelierStatus = deck?.diagnostics?.visualAtelierStatus || 'not_configured';
  const surgicalSwaps = Number(deck?.diagnostics?.modelFashionSelection?.surgicalRefinement?.approved || 0);
  const latencyMs = Date.now() - started;

  if ((deck.status !== 'complete' && deck.status !== 'partial') || !deck.outfits.length) {
    const diagnostic = deck.diagnostics || {};
    const rejectionCounts = Object.entries((diagnostic.visualQualityDecisions || []).reduce(
      (counts, item) => ({
        ...counts,
        [String(item?.reason || 'unknown')]: (counts[String(item?.reason || 'unknown')] || 0) + 1,
      }),
      {},
    ));
    process.stdout.write(
      `[stylist6-dev-bridge] run scenario=${style}/${occasion} mode=${sourceMode} lang=${language} ` +
      `latencyMs=${latencyMs} looks=0 visualStatus=${visualStatus} visualAtelier=${visualAtelierStatus} ` +
      `surgicalSwaps=${surgicalSwaps} status=${deck.status} rejections=${JSON.stringify(Object.fromEntries(rejectionCounts))}\n`,
    );
    const qualityFailed = (diagnostic.visualReviewCoverage?.valid ?? 0) > 0 &&
      (diagnostic.visualReviewCoverage?.approved ?? 0) === 0;
    return {
      status: 502,
      body: {
        success: false,
        code: qualityFailed ? 'stylist6_visual_quality_unmet' : 'stylist6_no_feasible_looks',
        stage: 'stylist6_ranking',
        engineVersion: STYLIST6_DEV_ENGINE_VERSION,
        visualHydration,
        visualStatus,
        visualReceipt,
        diagnostics: deck.diagnostics || null,
      },
    };
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

  const mappedOutfits = deck.outfits.map((look, lookIndex) => {
    const mappedItems = look.items.map((item) => mapDeckItem(
      item,
      look.fashion?.score,
      verifyMerchantOffer(item, { targetMarket: 'UZ' }),
    ));
    const lookVerified = Array.isArray(look.visualEvidence);
    const fallbackNotes = (!look.stylistNote || !look.whyItFlattersYou || !look.colorStory || !look.silhouetteFormula)
      ? buildStylistNotes(
          { ...look, items: mappedItems },
          { style, occasion, language, personalStyleDna, preferences: input.preferences },
          look.direction || { id: 'signature' },
          { personalStyleDna },
        )
      : null;
    const stylistNote = look.stylistNote || fallbackNotes?.stylistNote || look.description || '';
    const whyItFlattersYou = look.whyItFlattersYou || fallbackNotes?.whyItFlattersYou || '';
    const stylingTips = Array.isArray(look.stylingTips) && look.stylingTips.length > 0
      ? look.stylingTips
      : (Array.isArray(fallbackNotes?.stylingTips) ? fallbackNotes.stylingTips : []);
    const colorStory = look.colorStory || fallbackNotes?.colorStory || '';
    const silhouetteFormula = look.silhouetteFormula || fallbackNotes?.silhouetteFormula || '';

    return {
      id: `outfit_${lookIndex + 1}`,
      selectionSource: lookVerified ? 'ai' : 'deterministic_fallback',
      items: mappedItems,
      offerVerification: summarizeDeckOffers([{ items: mappedItems }]),
      description: look.description || stylistNote,
      stylistNote,
      whyItFlattersYou,
      stylingTips,
      colorStory,
      silhouetteFormula,
      concept: { id: look.concept?.id, title: look.concept?.title, story: '', score: look.fashion?.score },
      direction: look.direction ?? null,
      confidence: look.fashion?.score,
      matchScore: look.fashion?.score,
      engineVersion: STYLIST6_DEV_ENGINE_VERSION,
      ...(look.surgicalRefinement ? { surgicalRefinement: look.surgicalRefinement } : {}),
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

  const skuSummary = mappedOutfits
    .map((look) => look.items.map((i) => String(i.id)).join('+'))
    .join(' | ');
  // Clean structured per-generation QA trace (zero secrets, zero user PII).
  process.stdout.write(
    `[stylist6-dev-bridge] run scenario=${style}/${occasion} mode=${sourceMode} lang=${language} ` +
    `catalogSource=${resolvedCat.catalogSource}(${pool.length}/${items.length}) latencyMs=${latencyMs} ` +
    `looks=${mappedOutfits.length} visualStatus=${visualStatus} visualAtelier=${visualAtelierStatus} ` +
    `surgicalSwaps=${surgicalSwaps} providerCalls=${providerCalls} skus=[${skuSummary}]\n`,
  );

  const offerVerification = summarizeDeckOffers(mappedOutfits);
  return {
    status: 200,
    body: {
      success: true,
      generationVersion: STYLIST6_DEV_ENGINE_VERSION,
      source: 'stylist-6-dev-local',
      compositionProtocol: sourceMode === 'catalog' ? 'four-slot-v1' : 'optional-layer-v1',
      deckTarget: 3,
      status: deck.status,
      sourceMode,
      language,
      currencyConversion,
      feedbackTasteActive: feedbackTaste.active === true,
      previousDeckOverlapBySlot,
      visualHydration,
      visualStatus,
      visualAtelierStatus,
      visualReceipt,
      offerVerification,
      purchaseEvidence: DEV_PURCHASE_EVIDENCE,
      creativeDirector: {
        version: CREATIVE_DIRECTOR_VERSION,
        status: creativePlan.status,
        providerCalls: creativePlan.providerCalls,
        briefsAccepted: creativePlan.briefs.length,
      },
      feedbackTaste: {
        rows: feedbackRows.length,
        active: feedbackTaste.active === true,
        version: FEEDBACK_TASTE_VERSION,
      },
      diagnostics: deck.diagnostics || null,
      meta: {
        language,
        sourceMode,
        catalogSource: resolvedCat.catalogSource,
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

export function createDevBridgeServer(serverOptions = {}) {
  const defaultOpts = { liveServer: true, ...serverOptions };
  return http.createServer(async (req, res) => {
    res.setHeader('Content-Type', 'application/json');
    const route = String(req.url || '').split('?')[0];
    try {
      if (req.method === 'GET' && route === '/health') {
        const { hasNvidiaKey, hasOpenrouterKey } = resolveAiKeys(defaultOpts);
        const { url, readKey } = resolveSupabaseReadConfig(defaultOpts);
        const atelierMod = await loadOptionalVisualAtelier();
        const visualAtelierReady = Boolean(
          typeof atelierMod?.composeOutfitsVisually === 'function' &&
          (hasNvidiaKey || hasOpenrouterKey),
        );
        res.writeHead(200);
        res.end(JSON.stringify({
          ok: true,
          engine: STYLIST6_DEV_ENGINE_VERSION,
          engineVersion: STYLIST6_DEV_ENGINE_VERSION,
          pipeline: 'stylist6-phase11-autonomous-visual',
          autonomousStylist: hasNvidiaKey || hasOpenrouterKey,
          autonomousStylistVersion: AUTONOMOUS_STYLIST_VERSION,
          visualAtelier: visualAtelierReady,
          references: index.images.length,
          endpoints: ['/health', '/generate', '/swap-slot', '/analyze-appearance'],
          capabilities: {
            autonomousStylist: hasNvidiaKey || hasOpenrouterKey,
            visualAtelier: visualAtelierReady,
            liveCatalogConfigured: Boolean(url && readKey),
            cbuFxConfigured: true,
            nvidiaConfigured: hasNvidiaKey,
            openrouterConfigured: hasOpenrouterKey,
          },
          nvidiaConfigured: hasNvidiaKey,
          openrouterConfigured: hasOpenrouterKey,
          supabaseConfigured: Boolean(url && readKey),
          cachedCatalogSkus: Array.isArray(liveCatalogCache.rows) ? liveCatalogCache.rows.length : 0,
        }));
        return;
      }
      if (req.method === 'POST' && (route === '/generate' || route === '/swap-slot' || route === '/analyze-appearance')) {
        const raw = await readBody(req);
        let payload;
        try {
          payload = JSON.parse(raw);
        } catch {
          res.writeHead(400);
          res.end(JSON.stringify({ success: false, code: 'invalid_stylist6_request', stage: 'stylist6_admission' }));
          return;
        }
        let result;
        if (route === '/swap-slot') {
          result = await handleDevBridgeSwapSlot(payload, defaultOpts);
        } else if (route === '/analyze-appearance') {
          result = await handleDevBridgeAnalyzeAppearance(payload, defaultOpts);
        } else {
          result = await generateStylist6Deck(payload, defaultOpts);
        }
        res.writeHead(result.status);
        res.end(JSON.stringify(result.body));
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
  const { hasNvidiaKey, hasOpenrouterKey } = resolveAiKeys();
  const { url, readKey } = resolveSupabaseReadConfig();
  const server = createDevBridgeServer({ liveServer: true });
  server.listen(port, '127.0.0.1', () => {
    process.stdout.write(
      `[stylist6-dev-bridge] ${STYLIST6_DEV_ENGINE_VERSION} (v3-visual-atelier) on http://127.0.0.1:${port} ` +
      `(refs=${index.images.length} supabase=${Boolean(url && readKey)} nvidia=${hasNvidiaKey} openrouter=${hasOpenrouterKey})\n`,
    );
  });
}
