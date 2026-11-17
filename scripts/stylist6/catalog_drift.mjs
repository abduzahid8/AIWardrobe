/**
 * Offline-vs-live catalog drift report for the Stylist 6 pilot.
 *
 * Compares the frozen pilot snapshot (committed `data/shop_catalog_*.json`,
 * 40,989 rows) against a recent read-only export of the live production
 * catalog (expected ~7,020 active rows). Never touches production; both
 * inputs are files. Never merges the two universes silently.
 *
 * Live export (run by the OWNER with server-side credentials; NOT run here):
 *
 *   -- read-only; replace :out with a private path, keep user tables out
 *   copy (
 *     select s.id, s.name, s.brand, s.description, s.category, s.garment_type,
 *       s.image_url, s.price, s.currency, s.source, s.source_url,
 *       s.external_id, s.is_active,
 *       i.slot as fashion_slot,
 *       i.subtype as fashion_subtype,
 *       md5(jsonb_build_array(s.name,s.description,s.brand,s.category,
 *         s.garment_type,s.image_url)::text) as source_hash
 *     from public.shop_catalog s
 *     left join public.catalog_garment_intelligence i
 *       on i.product_id = s.id and i.version = 1
 *   ) to ':out' with csv header;
 *   -- then convert to JSON array of objects with the same keys.
 *
 * Usage:
 *   node scripts/stylist6/catalog_drift.mjs \
 *     --live /private/live-catalog-20261110.json \
 *     --report /private/drift-report.json
 *
 * Exit 0 with a report even under total divergence. SKUs that vanished,
 * deactivated, repriced or lost HTTPS media are listed, not averaged away.
 */

import fs from 'node:fs';
import path from 'node:path';

export const DRIFT_VERSION = 'catalog-drift-v1';

const args = Object.fromEntries(
  process.argv.slice(2).flatMap((a, i, all) => {
    if (!a.startsWith('--')) return [];
    const eq = a.indexOf('=');
    if (eq > 0) return [[a.slice(2, eq), a.slice(eq + 1)]];
    const next = all[i + 1];
    return [[a.slice(2), next && !next.startsWith('--') ? next : 'true']];
  }),
);

const baseProductId = (id) => String(id).replace(/-v\d+$/i, '');
const isHttps = (v) => {
  try {
    const u = new URL(String(v));
    return u.protocol === 'https:' && !!u.hostname;
  } catch {
    return false;
  }
};

export function snapshotRows(root, dirName = 'data', prefix = 'shop_catalog_') {
  const dir = path.join(root, dirName);
  const rows = [];
  for (const file of fs.readdirSync(dir)) {
    if (!file.startsWith(prefix) || !file.endsWith('.json')) continue;
    rows.push(...JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8')));
  }
  return rows;
}

export function driftReport(offlineRows, liveRows) {
  const off = new Map((offlineRows || []).map((r) => [String(r.id), r]));
  const live = new Map((liveRows || []).map((r) => [String(r.id), r]));
  const onlyOffline = [];
  const onlyLive = [];
  const changed = [];
  const stable = [];
  for (const [id, o] of off) {
    const l = live.get(id);
    if (!l) {
      onlyOffline.push(id);
      continue;
    }
    const diffs = [];
    if (Number(o.price) !== Number(l.price)) diffs.push(`price:${o.price}->${l.price}`);
    if (o.currency !== l.currency) diffs.push(`currency:${o.currency}->${l.currency}`);
    if (o.image_url !== l.image_url) diffs.push('image_url');
    if (o.source_url !== l.source_url) diffs.push('source_url');
    if (Boolean(o.is_active) !== Boolean(l.is_active)) diffs.push(`active:${o.is_active}->${l.is_active}`);
    if (!l.is_active) diffs.push('live_inactive');
    if (!isHttps(l.image_url) || !isHttps(l.source_url)) diffs.push('live_non_https_media');
    if (!(Number(l.price) > 0)) diffs.push('live_unpriced');
    if (diffs.length) changed.push({ id, diffs });
    else stable.push(id);
  }
  for (const [id] of live) {
    if (!off.has(id)) onlyLive.push(id);
  }
  const baseIds = (rows) => new Set(rows.map((r) => baseProductId(r.id ?? r)));
  const offBase = baseIds(offlineRows);
  const liveBase = baseIds(liveRows);
  const staleVariants = [...offBase].filter((b) => !liveBase.has(b)).length;
  const activeLive = liveRows.filter((r) => r.is_active).length;
  return {
    version: DRIFT_VERSION,
    offlineRows: offlineRows.length,
    liveRows: liveRows.length,
    liveActive: activeLive,
    inBoth: stable.length + changed.length,
    stable: stable.length,
    changed: changed.length,
    onlyOffline: onlyOffline.length,
    onlyLive: onlyLive.length,
    distinctBaseOffline: offBase.size,
    distinctBaseLive: liveBase.size,
    staleVariantBases: staleVariants,
    changedSample: changed.slice(0, 50),
    onlyOfflineSample: onlyOffline.slice(0, 50),
    onlyLiveSample: onlyLive.slice(0, 50),
  };
}

async function main() {
  const root = path.resolve(import.meta.dirname, '../..');
  const offline = snapshotRows(root);
  if (!args.live) {
    // Offline-only characterization (no live access in this environment).
    const active = offline.filter((r) => r.is_active);
    const httpsPriced = active.filter((r) => Number(r.price) > 0 && isHttps(r.image_url) && isHttps(r.source_url));
    const byCurrency = {};
    for (const r of httpsPriced) byCurrency[r.currency] = (byCurrency[r.currency] || 0) + 1;
    const base = new Set(offline.map((r) => baseProductId(r.id)));
    console.log(JSON.stringify({
      version: DRIFT_VERSION, mode: 'offline_characterization_no_live_access',
      offlineRows: offline.length, offlineActive: active.length,
      offlineShoppable: httpsPriced.length, byCurrency,
      distinctBaseProducts: base.size,
      variantDuplicates: offline.length - base.size,
      liveExport: 'MISSING — run the documented read-only COPY with server-side credentials',
    }, null, 1));
    return;
  }
  const live = JSON.parse(fs.readFileSync(args.live, 'utf8'));
  const report = driftReport(offline, live);
  if (args.report) fs.writeFileSync(args.report, JSON.stringify(report, null, 1) + '\n', { mode: 0o600 });
  console.log(JSON.stringify(report, null, 1));
}

const isMain = process.argv[1] && path.resolve(process.argv[1]).endsWith('catalog_drift.mjs');
if (isMain) await main();
