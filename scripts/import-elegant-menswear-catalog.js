#!/usr/bin/env node
/**
 * Uploads the locally reviewed menswear photos to the shop-catalog bucket and
 * upserts the corresponding rows into public.shop_catalog.
 *
 * Preview: node scripts/import-elegant-menswear-catalog.js --dry-run
 * Apply:   node scripts/import-elegant-menswear-catalog.js --apply
 *
 * A service-role key is required for uploads and table writes. The dry run
 * uses the public anon key for read-only duplicate discovery.
 */
'use strict';

try { require('dotenv').config(); } catch (_) { /* env may be supplied by shell */ }

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { createClient } = require('@supabase/supabase-js');

const ROOT = path.join(__dirname, '..');
const CATALOG = path.join(ROOT, 'data', 'elegant_menswear_catalog_2026.json');
const BUCKET = 'shop-catalog';
const SOURCE = 'elegant-menswear-2026';
const MAX_IMAGE_BYTES = 5 * 1024 * 1024;
const APPLY = process.argv.includes('--apply');
const DRY_RUN = process.argv.includes('--dry-run') || !APPLY;
const BATCH_SIZE = 50;

const url = process.env.SUPABASE_URL || process.env.EXPO_PUBLIC_SUPABASE_URL;
const anonKey = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
const serviceKey = process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY;
const key = APPLY ? serviceKey : (serviceKey || anonKey);

if (!url || !key) throw new Error('Missing Supabase URL/key.');
if (APPLY && !serviceKey) {
  throw new Error('SUPABASE_SERVICE_KEY is missing. A service-role key is required for storage and shop_catalog writes.');
}

function categoryFor(item) {
  const n = String(item.name || item.item_name || '').toLowerCase();
  const current = String(item.category || '').toLowerCase();
  if (current === 'shoes' || item.garment_type === 'footwear' || /\b(oxford shoes|derby shoes|loafers|boots)\b/.test(n)) {
    return { category: 'shoes', garment_type: 'shoes' };
  }
  if (current === 'bottoms' || item.garment_type === 'lower_body' || /\b(trousers|trouser|chinos|chino|pants)\b/.test(n)) {
    return { category: 'bottoms', garment_type: 'lower_body' };
  }
  if (current === 'outerwear' || /\b(overcoat|coat)\b/.test(n)) {
    return { category: 'outerwear', garment_type: 'upper_body' };
  }
  if (/\bsuit\b/.test(n) && !/\b(jacket|trouser|trousers|pants)\b/.test(n)) {
    return { category: 'outfit', garment_type: 'outfit' };
  }
  return { category: 'tops', garment_type: 'upper_body' };
}

function validateCatalog(items) {
  if (!Array.isArray(items) || items.length !== 70) {
    throw new Error(`Expected 70 reviewed products; got ${Array.isArray(items) ? items.length : 'non-array'}.`);
  }
  const ids = new Set();
  const urls = new Set();
  for (const item of items) {
    if (!item.id || !item.brand || !item.name || !item.product_url || !item.image_file) {
      throw new Error(`Required catalog field missing for ${item.brand || '(unknown brand)'}.`);
    }
    if (!Number.isFinite(Number(item.price)) || !item.currency) {
      throw new Error(`Invalid price/currency for ${item.brand} / ${item.name}.`);
    }
    if (ids.has(item.id)) throw new Error(`Duplicate ID: ${item.id}`);
    if (urls.has(item.product_url)) throw new Error(`Duplicate product URL: ${item.product_url}`);
    ids.add(item.id); urls.add(item.product_url);
    const imagePath = path.resolve(ROOT, item.local_image_path || item.image_file);
    if (!imagePath.startsWith(`${ROOT}${path.sep}`) || !fs.existsSync(imagePath)) {
      throw new Error(`Image file missing or outside project: ${item.local_image_path || item.image_file}`);
    }
    const stat = fs.statSync(imagePath);
    if (stat.size > MAX_IMAGE_BYTES) throw new Error(`Image exceeds 5 MB bucket limit: ${item.brand}`);
    if (item.image_format !== 'JPEG') throw new Error(`Expected JPEG image: ${item.brand}`);
    if (item.image_review !== 'Visually checked; no mannequin is visible.') {
      throw new Error(`Image review flag missing for ${item.brand}.`);
    }
    const sha = crypto.createHash('sha256').update(fs.readFileSync(imagePath)).digest('hex');
    if (sha !== item.image_sha256) throw new Error(`Image checksum mismatch: ${item.brand}`);
  }
  if (new Set(items.map(x => x.brand)).size !== 50) throw new Error('Expected 50 distinct menswear brands.');
}

async function findExistingBySourceUrl(supabase, items) {
  const urls = [...new Set(items.map(x => x.source_url || x.product_url))];
  const existing = new Map();
  for (let i = 0; i < urls.length; i += 20) {
    const { data, error } = await supabase.from('shop_catalog')
      .select('id,brand,name,source_url,source,is_active')
      .in('source_url', urls.slice(i, i + 20));
    if (error) throw error;
    for (const row of data || []) existing.set(row.source_url, row);
  }
  return existing;
}

async function main() {
  if (!fs.existsSync(CATALOG)) throw new Error(`Catalog missing: ${CATALOG}`);
  const items = JSON.parse(fs.readFileSync(CATALOG, 'utf8'));
  validateCatalog(items);
  const supabase = createClient(url, key, { auth: { persistSession: false, autoRefreshToken: false } });

  const existing = await findExistingBySourceUrl(supabase, items);
  const inserts = items.filter(x => !existing.has(x.source_url || x.product_url)).length;
  const refreshes = items.length - inserts;
  console.log(`Validated ${items.length} products across 50 brands.`);
  console.log(`Exact product-page matches already in shop_catalog: ${refreshes}; new rows: ${inserts}.`);
  console.log(`Images to upload/update in Storage bucket '${BUCKET}': ${items.length}.`);
  if (DRY_RUN) {
    console.log('[dry-run] No files or database rows were changed.');
    return;
  }

  const rows = [];
  let uploaded = 0;
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    const matched = existing.get(item.source_url || item.product_url);
    const dbId = matched ? matched.id : item.id;
    const rel = item.local_image_path || item.image_file;
    const localPath = path.resolve(ROOT, rel);
    const storagePath = `elegant-menswear/${item.id}.jpg`;
    const file = fs.readFileSync(localPath);
    const { error: uploadError } = await supabase.storage.from(BUCKET).upload(storagePath, file, {
      contentType: 'image/jpeg', upsert: true,
    });
    if (uploadError) throw new Error(`Storage upload failed (${item.brand}): ${uploadError.message}`);
    uploaded++;
    const { data: publicUrl } = supabase.storage.from(BUCKET).getPublicUrl(storagePath);
    const category = categoryFor(item);
    rows.push({
      id: dbId,
      brand: item.brand,
      name: item.name || item.item_name,
      price: Number(item.price),
      currency: item.currency,
      category: category.category,
      garment_type: category.garment_type,
      description: item.description || '',
      image_url: publicUrl.publicUrl,
      source_url: item.source_url || item.product_url,
      source: matched?.source || SOURCE,
      is_active: true,
      sort_order: Number.isInteger(item.sort_order) ? item.sort_order : i,
      updated_at: new Date().toISOString(),
    });
  }

  let upserted = 0;
  for (let i = 0; i < rows.length; i += BATCH_SIZE) {
    const batch = rows.slice(i, i + BATCH_SIZE);
    const { error } = await supabase.from('shop_catalog').upsert(batch, { onConflict: 'id' });
    if (error) throw new Error(`shop_catalog upsert failed after ${upserted} rows: ${error.message}`);
    upserted += batch.length;
    console.log(`Upserted ${upserted}/${rows.length} shop_catalog rows.`);
  }
  console.log(`Done: ${uploaded} photos uploaded, ${upserted} rows upserted (${inserts} new, ${refreshes} matching existing products refreshed).`);
}

main().catch(error => {
  console.error('Elegant menswear Supabase import failed:', error.message || error);
  process.exit(1);
});
