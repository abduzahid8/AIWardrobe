#!/usr/bin/env node
/**
 * scripts/upload-shop-images-to-supabase.js
 *
 * Uploads the locally vendored white-bg product shots (assets/shop/) to the
 * Supabase `shop-catalog` storage bucket and upserts matching rows into the
 * `shop_catalog` table — so the 1.4GB local bundle can be deleted from the
 * app without losing the images.
 *
 * Source of truth: assets/shop/manifest.json (written by
 * scripts/vendor-shop-images.py). Each entry:
 *   { id, file, brand, name, price, currency, garmentType,
 *     description, sourceUrl, image }
 *
 * Storage layout:  vendored/{zara|massimo}/{file}
 * Table source:    'vendored-white-bg'
 *
 * Usage:
 *   node scripts/upload-shop-images-to-supabase.js --dry-run [--limit 5] [--brand zara]
 *   node scripts/upload-shop-images-to-supabase.js --limit 5   # real upload, 5 files
 *   node scripts/upload-shop-images-to-supabase.js             # full upload (~4k files)
 *   node scripts/upload-shop-images-to-supabase.js --force     # re-upload even if present
 *
 * Env (via .env or shell):
 *   SUPABASE_URL (or EXPO_PUBLIC_SUPABASE_URL)
 *   SUPABASE_SERVICE_KEY  (service_role — required for storage upload + table write)
 *
 * Safe to re-run: existing storage objects are skipped (unless --force) and
 * table writes use upsert on id.
 */

'use strict';

try {
  require('dotenv').config();
} catch (_) {
  /* dotenv optional */
}

const fs = require('fs');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');

const ROOT = path.join(__dirname, '..');
const MANIFEST = path.join(ROOT, 'assets', 'shop', 'manifest.json');
const BUCKET = 'shop-catalog';
const SOURCE = 'vendored-white-bg';

const SUPABASE_URL =
  process.env.SUPABASE_URL || process.env.EXPO_PUBLIC_SUPABASE_URL;

const argv = process.argv.slice(2);
const DRY_RUN = argv.includes('--dry-run');
const FORCE = argv.includes('--force');
const limitIdx = argv.indexOf('--limit');
const LIMIT = limitIdx !== -1 ? parseInt(argv[limitIdx + 1], 10) : Infinity;
const brandIdx = argv.indexOf('--brand');
const BRAND_FILTER = brandIdx !== -1 ? argv[brandIdx + 1].toLowerCase() : null;

// service_role bypasses RLS for storage + table writes. Anon key is enough
// for --dry-run (public reads only).
const SUPABASE_KEY =
  process.env.SUPABASE_SERVICE_KEY ||
  (DRY_RUN ? process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY : null);

if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error('Missing SUPABASE_URL / SUPABASE_SERVICE_KEY. Set them in .env first.');
  process.exit(1);
}

function garmentToCategory(garmentType) {
  switch ((garmentType || '').toLowerCase()) {
    case 'upper_body':
      return 'tops';
    case 'lower_body':
      return 'bottoms';
    case 'shoes':
      return 'shoes';
    case 'dresses':
      return 'dresses';
    case 'outfit':
      return 'outfits';
    case 'accessory':
      return 'accessories';
    default:
      return 'tops';
  }
}

function brandDir(brand) {
  const b = (brand || '').toLowerCase();
  if (b.includes('zara')) return 'zara';
  return 'massimo';
}

async function listExistingStoragePaths(supabase) {
  // Collect already-uploaded objects under vendored/ so re-runs skip them.
  const existing = new Set();
  for (const dir of ['vendored/zara', 'vendored/massimo']) {
    let offset = 0;
    const pageSize = 1000;
    for (;;) {
      const { data, error } = await supabase.storage
        .from(BUCKET)
        .list(dir, { limit: pageSize, offset });
      if (error) {
        console.error(`storage.list failed for ${dir}:`, error.message);
        break;
      }
      if (!data || data.length === 0) break;
      for (const obj of data) {
        if (obj && obj.name) existing.add(`${dir}/${obj.name}`);
      }
      if (data.length < pageSize) break;
      offset += pageSize;
    }
  }
  return existing;
}

async function main() {
  if (!fs.existsSync(MANIFEST)) {
    console.error(`Manifest not found: ${MANIFEST}`);
    process.exit(1);
  }
  const manifest = JSON.parse(fs.readFileSync(MANIFEST, 'utf-8'));
  console.log(`Manifest entries: ${manifest.length}`);

  let entries = manifest.filter((e) => e && e.id && (e.file || e.image));
  if (BRAND_FILTER) {
    entries = entries.filter((e) => brandDir(e.brand) === BRAND_FILTER);
    console.log(`Brand filter '${BRAND_FILTER}': ${entries.length} entries`);
  }
  if (Number.isFinite(LIMIT)) entries = entries.slice(0, LIMIT);

  const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

  // Verify bucket is reachable before touching 1.4GB of files.
  const { error: bucketError } = await supabase.storage.from(BUCKET).list('', { limit: 1 });
  if (bucketError) {
    console.error(`Cannot access bucket '${BUCKET}':`, bucketError.message);
    process.exit(1);
  }

  const existing = DRY_RUN ? new Set() : await listExistingStoragePaths(supabase);
  console.log(`Already in storage: ${existing.size} objects`);

  const CONCURRENCY = parseInt(process.env.UPLOAD_CONCURRENCY || '10', 10) || 10;
  const rows = new Array(entries.length).fill(null);
  let uploaded = 0;
  let skipped = 0;
  let missing = 0;
  let done = 0;

  async function processOne(entry, i) {
    const dir = brandDir(entry.brand);
    const file = entry.file || path.basename(entry.image || '');
    const localPath = path.join(ROOT, 'assets', 'shop', dir, file);
    const storagePath = `vendored/${dir}/${file}`;

    if (!fs.existsSync(localPath)) {
      missing += 1;
      if (missing <= 10) console.warn(`  skip (missing file): ${localPath}`);
      return;
    }

    let publicUrl = null;
    if (!FORCE && existing.has(storagePath)) {
      skipped += 1;
      publicUrl = supabase.storage.from(BUCKET).getPublicUrl(storagePath).data.publicUrl;
    } else if (DRY_RUN) {
      publicUrl = supabase.storage.from(BUCKET).getPublicUrl(storagePath).data.publicUrl;
    } else {
      const buffer = fs.readFileSync(localPath);
      const { error } = await supabase.storage
        .from(BUCKET)
        .upload(storagePath, buffer, { contentType: 'image/jpeg', upsert: true });
      if (error) {
        console.error(`  upload failed ${storagePath}: ${error.message}`);
        return;
      }
      uploaded += 1;
      publicUrl = supabase.storage.from(BUCKET).getPublicUrl(storagePath).data.publicUrl;
    }

    rows[i] = {
      id: entry.id,
      brand: entry.brand || '',
      name: entry.name || entry.id,
      price: typeof entry.price === 'number' ? entry.price : 0,
      currency: entry.currency || 'EUR',
      category: garmentToCategory(entry.garmentType),
      garment_type: entry.garmentType || 'upper_body',
      description: entry.description || '',
      image_url: publicUrl,
      source_url: entry.sourceUrl || '',
      source: SOURCE,
      is_active: true,
      sort_order: i,
      last_seen_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
  }

  // Worker pool: CONCURRENCY parallel uploads.
  async function worker(pool) {
    for (;;) {
      const i = pool.next++;
      if (i >= entries.length) return;
      await processOne(entries[i], i);
      done += 1;
      if (done % 200 === 0) {
        console.log(`  … ${done}/${entries.length} (uploaded: ${uploaded}, skipped: ${skipped})`);
      }
    }
  }
  const pool = { next: 0 };
  await Promise.all(
    Array.from({ length: Math.min(CONCURRENCY, entries.length) }, () => worker(pool)),
  );

  const validRows = rows.filter(Boolean);
  console.log(`\nFiles — uploaded: ${uploaded}, already present: ${skipped}, missing: ${missing}`);
  console.log(`Rows to upsert: ${validRows.length}`);

  if (DRY_RUN) {
    console.log('[dry-run] No writes performed. First row preview:');
    console.log(JSON.stringify(validRows[0], null, 2));
    return;
  }

  const BATCH = 100;
  let total = 0;
  for (let i = 0; i < validRows.length; i += BATCH) {
    const batch = validRows.slice(i, i + BATCH);
    const { error } = await supabase.from('shop_catalog').upsert(batch, { onConflict: 'id' });
    if (error) {
      console.error(`\nDB upsert failed at batch ${i / BATCH + 1}:`, error.message);
      process.exit(1);
    }
    total += batch.length;
    console.log(`  upserted ${total}/${validRows.length}`);
  }

  console.log(`\nDone — ${uploaded} files uploaded, ${total} shop_catalog rows upserted (source='${SOURCE}').`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
