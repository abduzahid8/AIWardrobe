#!/usr/bin/env node
'use strict';

// Build a reproducible cross-budget selection from existing project feeds.
// Default: write local artifacts. --check: validate without writing.
// --apply: also import into Supabase, preserving existing product IDs.
const fs = require('node:fs');
const path = require('node:path');
const { createClient } = require('@supabase/supabase-js');
require('dotenv').config({ quiet: true });
const root = path.resolve(__dirname, '..');
// Every supplemental item must originate in a feed explicitly marked as a
// product-only/no-model gallery. Do not loosen this check merely to increase
// brand count: the catalog's image rule is more important than coverage.
const PRODUCT_ONLY_SOURCE = /(?:no[-\s]?model|no[-\s]?mannequin|product[-\s]?only|plain[-\s]?background|white[-\s]?(?:bg|background))/i;
const TIER_BY_FEED = {
  affordable: [
    'a_p_c', 'american_eagle', 'asos_design', 'bershka', 'bonobos', 'burton', 'c_a', 'c_p_company', 'carhartt_wip', 'barbour', 'celio', 'cos_affordable', 'defacto', 'fatface', 'gap', 'hm_premium', 'jack_jones_premium', 'jack_wills', 'kiabi', 'lc_waikiki', 'lefties', 'mango', 'marks_spencer', 'new_yorker', 'next', 'old_navy', 'only_sons', 'oviesse', 'pepe_jeans', 'primark_elegant', 'pull_bear', 'reserved', 'river_island', 's_oliver', 'springfield', 'stradivarius_man', 'superdry', 'terranova', 'todd_snyder', 'topman', 'uniqlo', 'uniqlo_u',
    'zara_downloader', 'zara_new_2026',
  ],
  midrange: [
    'allsaints', 'arket', 'banana_republic', 'charles_tyrwhitt', 'cos', 'ecco_20261030',
    'everlane', 'gant_downloader', 'geox_shoes_20261030', 'jack_jones', 'jcrew',
    'john_smedley', 'm_j_bale', 'moss', 'nn07', 'theory', 'tiger_of_sweden', 'vince', 'johnston_murphy', 'selected_homme', 'uz_gant',
  ],
  premium: [
    'alden', 'allen_edmonds', 'armani_exchange', 'aurelien', 'aurelien_global', 'berluti_shoes', 'berwick_1707', 'duke_shoes', 'boss', 'brunello', 'calvin_klein', 'canali', 'carlos_santos', 'carmina', 'cobbler_union', 'crockett_jones', 'ea7', 'emporio_armani', 'emporio_global',
    'eidos', 'giorgio_armani', 'hackett', 'harmont_blaine', 'herno', 'herring_shoes', 'incotex', 'lacoste', 'lemaire', 'loake', 'loake2', 'loro_piana', 'luca_faloni',
    'alfred_sargent', 'andre_shoes', 'barker', 'bexley', 'bow_tie_shoes', 'church_english', 'jl_co_shoes', 'churchs_shoes', 'clarks_originals', 'corthay', 'ecco_shoes2', 'enzo_bonafe', 'finsbury', 'loding', 'gaziano_girling', 'grenson_shoes', 'john_doe_shoes', 'magnanni', 'malfini_shoes', 'nettle_shoes', 'john_lobb_2', 'john_white', 'johnston_murphy_shoes', 'massimo_dutti', 'meermin', 'morjas_shoes', 'oliver_cabell', 'paraboot', 'paul_evans', 'rm_williams', 'saxone', 'scarosso', 'septieme_largeur', 'stefano_bemer', 'trickers', 'trickers2', 'vass', 'velasca', 'paul_shark', 'private_white_vc', 'ralph_lauren', 'reiss', 'orlebar_brown', 'ring_jacket', 'rubato', 'pini_parma', 'santoni', 'sunspel', 'tommy_hilfiger', 'oscar_jacobson', 'the_armoury', 'valstar', 'frescobol_carioca',
  ],
};
const elegant = /\b(chinos?|trousers?|pants?|oxford|blazer|merino|cashmere|polo|linen|dress shirt|suit|loafer|derby|overcoat|poplin|cardigan|overshirt|knitwear?|wool|shirt|jacket|coat|belt|tie|shoe|boots?|sneakers?|brogues?|monk)\b/i;
const excluded = /\b(shorts?|joggers?|swim\w*|underwear|jeans|t-shirt|baggy|bermuda)\b/i;
const visiblePersonEvidence = /(?:\b(?:photo|image)\s+(?:shows?|features?)\b[^.]*\b(?:model|mannequin)\b|\bgarment\s+worn\s+by\s+(?:a\s+)?(?:male\s+)?model\b|\bmodel\s+(?:styled|wearing|wears)\b)/i;
const rows = new Map();
const tiers = new Map();

function add(item, tier, file) {
  const sourceUrl = item.source_url || item.product_url;
  const footwear = item.category === 'shoes' || item.garment_type === 'footwear';
  const suit = /\bsuit\b/i.test(item.name) && !/jacket|trouser|pant/i.test(item.name);
  const row = {
    id: item.id, brand: item.brand, name: item.name,
    price: Number(item.price), currency: item.currency,
    category: footwear ? 'shoes' : suit ? 'outfit' : ['suits', 'knitwear'].includes(item.category) ? 'tops' : item.category,
    garment_type: footwear ? 'shoes' : suit ? 'outfit' : item.garment_type === 'full_body' ? 'upper_body' : item.garment_type,
    description: item.description || '', image_url: item.image_url,
    source_url: sourceUrl, source: item.source || 'elegant-menswear-expansion',
    is_active: true, sort_order: rows.size,
  };
  if (!row.id || !row.brand || !row.name || !Number.isFinite(row.price) || row.price <= 0 ||
      !['USD', 'EUR', 'GBP', 'UZS'].includes(row.currency) ||
      !['upper_body', 'lower_body', 'shoes', 'outfit', 'accessory'].includes(row.garment_type)) {
    throw new Error(`Invalid product in ${file}: ${item.id}`);
  }
  for (const url of [sourceUrl, row.image_url]) {
    if (new URL(url).protocol !== 'https:') throw new Error(`Expected HTTPS: ${item.id}`);
  }
  if (rows.has(sourceUrl)) return;
  rows.set(sourceUrl, row);
  tiers.set(sourceUrl, { tier, source_file: file });
}

const baseFile = 'elegant_menswear_catalog_2026.json';
for (const item of require(path.join(root, 'data', baseFile))) {
  if (item.image_review !== 'Visually checked; no mannequin is visible.') {
    throw new Error(`Base catalog image has not passed visual product-only review: ${item.id}`);
  }
  if (visiblePersonEvidence.test(`${item.name || ''} ${item.description || ''}`)) {
    continue;
  }
  add(item, 'premium_and_luxury', baseFile);
}
for (const [tier, names] of Object.entries(TIER_BY_FEED)) {
  for (const name of names) {
    const file = `shop_catalog_${name}.json`;
    const items = require(path.join(root, 'data', file));
    // Use only store feeds whose own provenance says the gallery has no model
    // or mannequin.  Keep category variety, then fill remaining slots.
    const candidates = items.filter(x => x.is_active !== false && (elegant.test(x.name) || x.category === 'shoes') &&
      !excluded.test(x.name) && !visiblePersonEvidence.test(`${x.name || ''} ${x.description || ''}`) &&
      PRODUCT_ONLY_SOURCE.test(`${x.source || ''} ${x.description || ''}`));
    const selected = [];
    const categories = new Set();
    for (const item of candidates) {
      if (!categories.has(item.category)) { selected.push(item); categories.add(item.category); }
    }
    for (const item of candidates) if (!selected.includes(item)) selected.push(item);
    for (const item of selected.slice(0, 150)) add(item, tier, file);
  }
}
const items = [...rows.values()];
if (new Set(items.map(x => x.id)).size !== items.length) throw new Error('Duplicate product IDs');
const brands = [...new Set(items.map(x => x.brand))].sort();
console.log(`Validated ${items.length} products across ${brands.length} brands.`);
const manifest = {
  description: 'Curated global elegant menswear selection across accessible project feeds; every supplemental entry is restricted to source galleries labelled product-only/no-model. This is not a claim to be an exhaustive directory of every worldwide store.',
  pricing_note: 'Prices and availability come from existing project feeds and require periodic refresh. Tiers describe brand positioning, not currency-converted product prices.',
  brands,
  products: items.map(x => ({ id: x.id, brand: x.brand, source_url: x.source_url, ...tiers.get(x.source_url) })),
};

async function main() {
  if (!process.argv.includes('--check')) {
    fs.writeFileSync(path.join(root, 'data/shop_catalog_elegant_international.json'), JSON.stringify(items, null, 2) + '\n');
    fs.writeFileSync(path.join(root, 'data/elegant_international_manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  }
  if (!process.argv.includes('--apply')) return;
  const url = process.env.SUPABASE_URL || process.env.EXPO_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) throw new Error('Live import requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY. Local catalog is ready.');
  const client = createClient(url, key, { auth: { persistSession: false } });
  const existing = new Map();
  for (let i = 0; i < items.length; i += 20) {
    const { data, error } = await client.from('shop_catalog').select('id,source_url').in('source_url', items.slice(i, i + 20).map(x => x.source_url));
    if (error) throw error;
    for (const row of data) {
      if (existing.has(row.source_url)) throw new Error(`Resolve duplicate live product before import: ${row.source_url}`);
      existing.set(row.source_url, row.id);
    }
  }
  const writes = items.map(x => ({ ...x, id: existing.get(x.source_url) || x.id, updated_at: new Date().toISOString() }));
  if (new Set(writes.map(x => x.id)).size !== writes.length) throw new Error('Conflicting live product IDs');
  for (let i = 0; i < writes.length; i += 50) {
    const { error } = await client.from('shop_catalog').upsert(writes.slice(i, i + 50), { onConflict: 'id' });
    if (error) throw error;
    console.log(`Imported ${Math.min(i + 50, writes.length)}/${writes.length}`);
  }
}
main().catch(error => { console.error(error.message); process.exitCode = 1; });
