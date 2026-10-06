#!/usr/bin/env node
/**
 * scripts/import-uz-catalog.js
 * Imports shop_catalog_uz_*.json from website-downloader into shop_catalog.
 * Usage: SUPABASE_SERVICE_KEY=eyJ... node scripts/import-uz-catalog.js
 */
require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');

const URL = process.env.SUPABASE_URL || process.env.EXPO_PUBLIC_SUPABASE_URL;
const KEY = process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY;
if (!URL || !KEY) {
  console.error('❌ Need SUPABASE_SERVICE_KEY env var.');
  process.exit(1);
}
const SRC_DIR = '/Users/v.zahid/startups/website-downloader';
const FILES = [
  'shop_catalog_uz_boss.json',
  'shop_catalog_uz_ck.json',
  'shop_catalog_uz_mens.json',
  'shop_catalog_uz_tommy.json',
  'shop_catalog_uz_zara.json',
];

const sb = createClient(URL, KEY);
const now = new Date().toISOString();

function sanitize(raw, fallbackSort) {
  const price = typeof raw.price === 'number' ? raw.price : parseFloat(raw.price) || 0;
  return {
    id: String(raw.id || '').slice(0, 120),
    brand: String(raw.brand || '').slice(0, 120),
    name: String(raw.name || raw.etln_name || 'Untitled').slice(0, 500),
    price,
    currency: raw.currency || 'UZS',
    category: raw.category || 'other',
    garment_type: raw.garment_type || 'upper_body',
    description: String(raw.description || '').slice(0, 2000),
    image_url: String(raw.image_url || ''),
    source: String(raw.source || 'uz-import').slice(0, 120),
    source_url: String(raw.source_url || raw.official_url || '').slice(0, 1000),
    external_id: null,
    is_active: raw.is_active !== false,
    sort_order: typeof raw.sort_order === 'number' ? raw.sort_order : fallbackSort,
    last_seen_at: now,
    updated_at: now,
  };
}

async function main() {
  let items = [];
  for (const f of FILES) {
    const p = path.join(SRC_DIR, f);
    if (!fs.existsSync(p)) {
      console.log(`⚠ missing ${f}, skip`);
      continue;
    }
    const arr = JSON.parse(fs.readFileSync(p, 'utf-8'));
    console.log(`📦 ${f}: ${arr.length}`);
    arr.forEach((r, i) => {
      if (!r.id || !r.image_url) return;
      items.push(sanitize(r, items.length));
    });
  }
  // de-dupe by id (last wins)
  const map = new Map(items.map((it) => [it.id, it]));
  items = [...map.values()];
  console.log(`\n🚀 Upserting ${items.length} unique items → ${URL}\n`);

  const BATCH = 100;
  let done = 0;
  for (let i = 0; i < items.length; i += BATCH) {
    const batch = items.slice(i, i + BATCH);
    const { error } = await sb.from('shop_catalog').upsert(batch, { onConflict: 'id' });
    if (error) {
      console.error(`❌ batch ${i / BATCH + 1} failed:`, error.message);
      console.error(JSON.stringify(batch[0]).slice(0, 500));
      process.exit(1);
    }
    done += batch.length;
    console.log(`  ✓ ${done}/${items.length}`);
  }
  const { count } = await sb.from('shop_catalog').select('*', { count: 'exact', head: true });
  console.log(`\n✅ Done. shop_catalog count now: ${count}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
