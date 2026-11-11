/**
 * Build the frozen UZS catalog input for the Stylist 39-scenario fashion replay.
 *
 * Read-only public RLS GETs only (the same reader pattern as
 * `scripts/validation/stylist-quality-catalog.mjs`). No writes, no service
 * credentials, no provider calls, no image requests. The output is a private
 * evaluation artifact and is git-ignored (`.private/`).
 *
 * Join rules follow the production migrations:
 *   - slot/subtype come from `catalog_garment_intelligence` (version 1);
 *   - `catalog_visual_fashion` (version 1) is attached only when its
 *     `source_hash` equals
 *     md5(jsonb_build_array(name,description,brand,category,garment_type,image_url)::text)
 *     AND its `image_url` equals the current catalog image URL. A stale or
 *     re-shot profile is never carried forward.
 *
 * Sampling (deterministic, documented, reproducible):
 *   1. keep active, positive-price, HTTPS image+shop, native-UZS rows whose
 *      intelligence slot is one of the four outfit slots;
 *   2. one row per base product (cheapest variant wins);
 *   3. per slot: cheapest row per (subtype, cached-or-inferred color) group,
 *      then fill to the quota ordered by md5(id) with current-visual rows
 *      first.
 *
 * Usage:
 *   node scripts/stylist6/frozen_catalog.mjs .private/catalog-uz.json [--per-slot 96]
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const args = process.argv.slice(2);
const output = args.find((a) => !a.startsWith('--'));
const perSlotIndex = args.indexOf('--per-slot');
const PER_SLOT = perSlotIndex > 0 ? Number(args[perSlotIndex + 1]) : 96;
if (!output || !Number.isInteger(PER_SLOT) || PER_SLOT < 1) {
  throw new Error('usage: node frozen_catalog.mjs <output.json> [--per-slot 96]');
}

const dotenv = await import('dotenv');
const env = dotenv.parse(fs.readFileSync(path.resolve(import.meta.dirname, '../../.env')));
const BASE = env.EXPO_PUBLIC_SUPABASE_URL;
const KEY = env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
if (!BASE || !KEY) throw new Error('missing public Supabase reader configuration');
const headers = { apikey: KEY, Authorization: `Bearer ${KEY}` };

const SLOTS = ['top', 'outerwear', 'bottom', 'shoes'];
const isHttps = (value) => {
  try {
    const url = new URL(String(value));
    return url.protocol === 'https:' && !!url.hostname;
  } catch {
    return false;
  }
};

/** PostgreSQL jsonb::text renders arrays with ", " separators. */
function sourceHash(row) {
  const parts = [row.name, row.description, row.brand, row.category, row.garment_type, row.image_url]
    .map((value) => JSON.stringify(value ?? null));
  return crypto.createHash('md5').update(`[${parts.join(', ')}]`).digest('hex');
}

async function rest(table, select, extra = {}, order = 'id.asc') {
  const rows = [];
  for (let offset = 0; offset < 40000; offset += 1000) {
    const url = new URL(`/rest/v1/${table}`, BASE);
    url.searchParams.set('select', select);
    url.searchParams.set('order', order);
    for (const [key, value] of Object.entries(extra)) url.searchParams.set(key, value);
    const response = await fetch(url, {
      headers: { ...headers, Range: `${offset}-${offset + 999}` },
      signal: AbortSignal.timeout(30000),
    });
    if (!response.ok) throw new Error(`${table}_read_${response.status}`);
    const page = await response.json();
    rows.push(...page);
    if (page.length < 1000) break;
  }
  return rows;
}

const baseProductId = (id) => String(id).replace(/-v\d+$/i, '');
const colorOf = (row, profile) => {
  const observed = profile?.attributes?.color;
  if (observed && observed.confidence >= 0.8 && typeof observed.value === 'string') return observed.value;
  return null;
};

const [catalog, intelligence, visual] = await Promise.all([
  rest('shop_catalog', 'id,name,brand,description,category,garment_type,image_url,source_url,price,currency,is_active', { is_active: 'eq.true' }),
  rest('catalog_garment_intelligence', 'product_id,source_hash,slot,subtype', { version: 'eq.1' }, 'product_id.asc'),
  rest('catalog_visual_fashion', 'product_id,source_hash,image_url,profile', { version: 'eq.1' }, 'product_id.asc'),
]);

const intelBySku = new Map(intelligence.map((row) => [row.product_id, row]));
const visualBySku = new Map(visual.map((row) => [row.product_id, row]));

const eligible = [];
const skipped = {};
const bump = (reason) => { skipped[reason] = (skipped[reason] || 0) + 1; };
for (const row of catalog) {
  if (row.currency !== 'UZS') { bump('not_uzs'); continue; }
  if (!(Number(row.price) > 0)) { bump('unpriced'); continue; }
  if (!isHttps(row.image_url) || !isHttps(row.source_url)) { bump('non_https_media'); continue; }
  const intel = intelBySku.get(row.id);
  if (!intel || !SLOTS.includes(intel.slot)) { bump('no_outfit_slot'); continue; }
  const hash = sourceHash(row);
  const candidate = visualBySku.get(row.id);
  const current = candidate && candidate.source_hash === hash && candidate.image_url === row.image_url
    ? candidate : null;
  eligible.push({
    ...row,
    fashion_slot: intel.slot,
    fashion_subtype: intel.subtype,
    ...(current ? { visual_fashion: current.profile, visual_source_hash: current.source_hash } : {}),
  });
}

// One row per base product; cheapest variant wins (deterministic tie-break by id).
const cheapest = new Map();
for (const row of eligible) {
  const base = baseProductId(row.id);
  const held = cheapest.get(base);
  if (!held || row.price < held.price || (row.price === held.price && row.id < held.id)) cheapest.set(base, row);
}

const byIdHash = (a, b) => (a.hash < b.hash ? -1 : a.hash > b.hash ? 1 : a.id.localeCompare(b.id));
const selected = [];
const perSlot = {};
for (const slot of SLOTS) {
  const pool = [...cheapest.values()].filter((row) => row.fashion_slot === slot);
  for (const row of pool) row.hash = crypto.createHash('md5').update(row.id).digest('hex');
  const groups = new Map();
  for (const row of pool) {
    const key = `${row.fashion_subtype || 'unknown'}|${colorOf(row, row.visual_fashion) || 'unknown'}`;
    const held = groups.get(key);
    if (!held || row.price < held.price || (row.price === held.price && row.id < held.id)) groups.set(key, row);
  }
  const chosen = new Set();
  for (const row of [...groups.values()].sort(byIdHash)) {
    if (chosen.size >= PER_SLOT) break;
    chosen.add(row);
  }
  const rest = pool
    .filter((row) => !chosen.has(row))
    .sort((a, b) => Number(!!b.visual_fashion) - Number(!!a.visual_fashion) || byIdHash(a, b));
  for (const row of rest) {
    if (chosen.size >= PER_SLOT) break;
    chosen.add(row);
  }
  perSlot[slot] = { pool: pool.length, chosen: chosen.size, withVisual: [...chosen].filter((r) => r.visual_fashion).length };
  selected.push(...chosen);
}

selected.sort((a, b) => a.id.localeCompare(b.id));
for (const row of selected) delete row.hash;
const bytes = JSON.stringify(selected) + '\n';
fs.mkdirSync(path.dirname(path.resolve(output)), { recursive: true, mode: 0o700 });
fs.writeFileSync(output, bytes, { mode: 0o600 });
console.log(JSON.stringify({
  output,
  rows: selected.length,
  perSlot,
  withVisualProfiles: selected.filter((row) => row.visual_fashion).length,
  distinctBaseProducts: new Set(selected.map((row) => baseProductId(row.id))).size,
  snapshotSha256: crypto.createHash('sha256').update(bytes).digest('hex'),
  sourceRows: catalog.length,
  eligibleRows: eligible.length,
  skipped,
  providerCalls: 0,
  imageRequests: 0,
}, null, 1));
