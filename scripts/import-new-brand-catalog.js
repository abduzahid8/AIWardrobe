#!/usr/bin/env node
/** Idempotently imports a locally curated men's brand feed into shop_catalog. */
'use strict';
try { require('dotenv').config(); } catch (_) { /* env may be set externally */ }
const fs = require('node:fs');
const path = require('node:path');
const { createClient } = require('@supabase/supabase-js');

const files = {
  thursday: 'shop_catalog_thursday.json',
  allsaints: 'shop_catalog_allsaints.json',
  ralph_lauren: 'shop_catalog_ralph_lauren.json',
  reiss: 'shop_catalog_reiss.json',
  aurelien: 'shop_catalog_aurelien.json',
  loro_piana: 'shop_catalog_loro_piana.json',
  emporio_armani: 'shop_catalog_emporio_armani.json',
  giorgio_armani: 'shop_catalog_giorgio_armani.json',
  armani_exchange: 'shop_catalog_armani_exchange.json',
  ea7: 'shop_catalog_ea7.json',
  mango: 'shop_catalog_mango.json',
  uniqlo: 'shop_catalog_uniqlo.json',
  paul_shark: 'shop_catalog_paul_shark.json',
  harmont_blaine: 'shop_catalog_harmont_blaine.json',
  geox: 'shop_catalog_geox.json',
  lacoste: 'shop_catalog_lacoste.json',
  everlane: 'shop_catalog_everlane.json',
  gap: 'shop_catalog_gap.json',
  banana_republic: 'shop_catalog_banana_republic.json',
  champion: 'shop_catalog_champion.json',
  reebok: 'shop_catalog_reebok.json',
  barbour: 'shop_catalog_barbour.json',
  fjallraven: 'shop_catalog_fjallraven.json',
  jcrew: 'shop_catalog_jcrew.json',
  orvis: 'shop_catalog_orvis.json',
  selected_homme: 'shop_catalog_selected_homme.json',
  fred_perry: 'shop_catalog_fred_perry.json',
  only_sons: 'shop_catalog_only_sons.json',
  jack_jones: 'shop_catalog_jack_jones.json',
  matinique: 'shop_catalog_matinique.json',
  johnston_murphy: 'shop_catalog_johnston_murphy.json',
  charles_tyrwhitt: 'shop_catalog_charles_tyrwhitt.json',
  lands_end: 'shop_catalog_lands_end.json',
  c_and_a: 'shop_catalog_c_and_a.json',
  hm: 'shop_catalog_hm.json',
  ms: 'shop_catalog_ms.json',
  old_navy: 'shop_catalog_old_navy.json',
  gildan: 'shop_catalog_gildan.json',
  fruit_of_the_loom: 'shop_catalog_fruit_of_the_loom.json',
  patagonia: 'shop_catalog_patagonia.json',
  brooks_brothers: 'shop_catalog_brooks_brothers.json',
  jockey: 'shop_catalog_jockey.json',
  vans: 'shop_catalog_vans.json',
  hanes: 'shop_catalog_hanes.json',
  converse: 'shop_catalog_converse.json',
  american_eagle: 'shop_catalog_american_eagle.json',
  columbia: 'shop_catalog_columbia.json',
  puma: 'shop_catalog_puma.json',
  cos: 'shop_catalog_cos.json',
  arket: 'shop_catalog_arket.json',
  mango_additional: 'shop_catalog_mango_additional.json',
  additional_500: 'shop_catalog_additional_500.json',
  elegant_additional: 'shop_catalog_elegant_additional.json',
  canali: 'shop_catalog_canali.json',
  selected_midrange_20261028: 'shop_catalog_selected_midrange_20261028.json',
  jack_midrange_20261028: 'shop_catalog_jack_midrange_20261028.json',
};
const brandKey = process.argv[2];
if (!files[brandKey]) throw new Error('Choose one of: ' + Object.keys(files).join(', ') + '.');
const items = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'data', files[brandKey]), 'utf8'));
const apply = process.argv.includes('--apply');
const url = process.env.SUPABASE_URL || process.env.EXPO_PUBLIC_SUPABASE_URL;
const key = process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY;

for (const item of items) {
  if (!item.id || !item.image_url || !item.source_url || !['EUR', 'USD', 'GBP', 'UZS'].includes(item.currency)) {
    throw new Error(`Invalid row: ${item.id || '(missing id)'}`);
  }
}
console.log(`Validated ${items.length} ${brandKey} items.`);
if (!apply) {
  console.log('Preview only. Add --apply to upsert into shop_catalog.');
  process.exit(0);
}
if (!url || !key) throw new Error('Set SUPABASE_URL and SUPABASE_SERVICE_KEY before --apply.');
const supabase = createClient(url, key, { auth: { persistSession: false } });
(async () => {
  for (let i = 0; i < items.length; i += 50) {
    const { error } = await supabase.from('shop_catalog').upsert(items.slice(i, i + 50), { onConflict: 'id' });
    if (error) throw error;
    console.log(`Upserted ${Math.min(i + 50, items.length)}/${items.length}`);
  }
})().catch((error) => {
  console.error('Brand catalog import failed:', error.message || error);
  process.exit(1);
});
