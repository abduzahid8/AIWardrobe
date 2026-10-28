#!/usr/bin/env node
/**
 * Idempotently imports the curated GANT men's feed produced by the
 * website-downloader into Supabase shop_catalog.
 *
 * Usage:
 *   node scripts/import-gant-catalog.js             # preview
 *   node scripts/import-gant-catalog.js --apply      # upsert into Supabase
 */
'use strict';
try { require('dotenv').config(); } catch (_) { /* env may be set externally */ }
const fs = require('node:fs');
const path = require('node:path');
const { createClient } = require('@supabase/supabase-js');

const itemsPath = path.join(__dirname, '..', 'data', 'shop_catalog_uz_gant.json');
const items = JSON.parse(fs.readFileSync(itemsPath, 'utf8'));
const apply = process.argv.includes('--apply');
const url = process.env.SUPABASE_URL || process.env.EXPO_PUBLIC_SUPABASE_URL;
const key = process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY;

for (const item of items) {
  if (!item.id || item.brand !== 'GANT' || !item.image_url || !item.source_url) {
    throw new Error(`Invalid catalog row: ${item.id || '(missing id)'}`);
  }
  if (!['tops', 'bottoms', 'shoes'].includes(item.category)) {
    throw new Error(`Unexpected category for ${item.id}: ${item.category}`);
  }
  if (!item.image_url.includes('-flat-')) {
    throw new Error(`Expected flat product photo for ${item.id}`);
  }
}

console.log(`Validated ${items.length} GANT men's items from ${itemsPath}`);
console.log('Categories:', items.reduce((counts, item) => {
  counts[item.category] = (counts[item.category] || 0) + 1;
  return counts;
}, {}));
if (!apply) {
  console.log('Preview only. Add --apply to upsert into shop_catalog.');
  process.exit(0);
}
if (!url || !key) throw new Error('Set SUPABASE_URL and SUPABASE_SERVICE_KEY before --apply.');
const supabase = createClient(url, key, { auth: { persistSession: false } });
(async () => {
  for (let i = 0; i < items.length; i += 50) {
    const batch = items.slice(i, i + 50);
    const { error } = await supabase.from('shop_catalog').upsert(batch, { onConflict: 'id' });
    if (error) throw error;
    console.log(`Upserted ${Math.min(i + batch.length, items.length)}/${items.length}`);
  }
})().catch((error) => {
  console.error('GANT catalog import failed:', error.message || error);
  process.exit(1);
});
