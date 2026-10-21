#!/usr/bin/env node
/**
 * scripts/clear-shop-catalog.js
 *
 * Deletes ALL rows from Supabase `shop_catalog`.
 * Requires SUPABASE_SERVICE_KEY (service_role) — anon key is blocked by RLS.
 *
 * Usage:
 *   node scripts/clear-shop-catalog.js [--dry-run]
 *
 * Env (from .env):
 *   EXPO_PUBLIC_SUPABASE_URL + EXPO_PUBLIC_SUPABASE_ANON_KEY
 *   or SUPABASE_URL + SUPABASE_SERVICE_KEY (preferred, bypasses RLS)
 */
require('dotenv').config();
const { createClient } = require('@supabase/supabase-js');

const URL =
  process.env.SUPABASE_URL || process.env.EXPO_PUBLIC_SUPABASE_URL;
const SERVICE_KEY =
  process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY;
const ANON_KEY = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
const KEY = SERVICE_KEY || ANON_KEY;

if (!URL || !KEY) {
  console.error('❌ Set SUPABASE_URL + key in .env first.');
  process.exit(1);
}
if (!SERVICE_KEY) {
  console.error(
    '❌ SUPABASE_SERVICE_KEY is empty — anon key is blocked by RLS (deletes silently affect 0 rows).\n' +
      '   Option A (fastest): run this SQL in Supabase Dashboard → SQL Editor:\n' +
      '     DELETE FROM public.shop_catalog;\n' +
      '   Option B: put your service_role key into .env as SUPABASE_SERVICE_KEY=eyJ... then re-run this script.\n' +
      '   Get it from: Supabase Dashboard → Project Settings → API → service_role (secret).'
  );
  process.exit(1);
}

const DRY_RUN = process.argv.includes('--dry-run');
const BATCH = 200;

const supabase = createClient(URL, KEY);

async function main() {
  const { count } = await supabase
    .from('shop_catalog')
    .select('*', { count: 'exact', head: true });
  console.log(`📦 shop_catalog rows before: ${count}`);
  console.log(`🔑 Using ${SERVICE_KEY ? 'SERVICE_KEY (bypasses RLS)' : 'ANON_KEY'}`);

  if (count === 0) {
    console.log('✅ Already empty.');
    return;
  }
  if (DRY_RUN) {
    console.log('🔍 Dry run — nothing deleted. Re-run without --dry-run to wipe.');
    return;
  }

  let deleted = 0;
  let staleRounds = 0;
  for (;;) {
    // Fetch a batch of ids
    const { data: rows, error: fetchErr } = await supabase
      .from('shop_catalog')
      .select('id')
      .limit(BATCH);

    if (fetchErr) {
      console.error('❌ Fetch failed:', fetchErr.message);
      process.exit(1);
    }
    if (!rows || rows.length === 0) break;

    const ids = rows.map((r) => r.id);
    const { data: delData, error: delErr } = await supabase
      .from('shop_catalog')
      .delete()
      .in('id', ids)
      .select('id');

    if (delErr) {
      console.error('❌ Delete failed:', delErr.message);
      process.exit(1);
    }
    const affected = delData ? delData.length : ids.length;
    if (affected === 0) {
      staleRounds += 1;
      console.error(
        `❌ Delete affected 0 rows (RLS blocked). Stopped after ${deleted} deleted. Use SERVICE_KEY or dashboard SQL.`
      );
      process.exit(1);
    }
    staleRounds = 0;
    deleted += affected;
    console.log(`  ✓ Deleted ${deleted} rows...`);
    if (ids.length < BATCH) break;
  }

  const { count: after } = await supabase
    .from('shop_catalog')
    .select('*', { count: 'exact', head: true });
  console.log(`\n✅ Done — deleted ~${deleted} rows. Remaining: ${after}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
