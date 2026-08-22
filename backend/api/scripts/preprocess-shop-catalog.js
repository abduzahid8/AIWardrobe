/**
 * Batch-preprocess every shop_catalog item's primary image.
 *
 * Reads `shop_catalog` from Supabase, calls the deterministic renderer's
 * preprocess step on each `image_url`, and writes the cleaned cutout to
 * the local on-disk cache (api/cache/garments/<sha1>.png).
 *
 * Run from the api/ folder:
 *   node scripts/preprocess-shop-catalog.js
 *
 * Re-runs are safe: items already cached are skipped instantly.
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import { preprocessGarmentAndCache } from '../services/tryonRenderer.js';

const SUPABASE_URL = process.env.SUPABASE_URL || process.env.EXPO_PUBLIC_SUPABASE_URL;
const SERVICE_ROLE_KEY =
  process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY;

if (!SUPABASE_URL || !SERVICE_ROLE_KEY) {
  console.error('Missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_SERVICE_KEY) in api/.env');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SERVICE_ROLE_KEY);

async function main() {
  const { data, error } = await supabase
    .from('shop_catalog')
    .select('id, name, image_url, garment_type')
    .not('image_url', 'is', null);

  if (error) throw error;

  console.log(`Preprocessing ${data.length} catalog items…`);
  let ok = 0;
  let failed = 0;
  for (let i = 0; i < data.length; i++) {
    const item = data[i];
    const start = Date.now();
    try {
      await preprocessGarmentAndCache(item.image_url);
      ok++;
      console.log(
        `[${i + 1}/${data.length}] ✓ ${item.garment_type ?? '?'} — ${item.name?.slice(0, 40) ?? item.id} (${Date.now() - start}ms)`,
      );
    } catch (err) {
      failed++;
      console.warn(`[${i + 1}/${data.length}] ✗ ${item.id}: ${err?.message ?? err}`);
    }
  }
  console.log(`Done. ok=${ok} failed=${failed} total=${data.length}`);
}

main().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});
