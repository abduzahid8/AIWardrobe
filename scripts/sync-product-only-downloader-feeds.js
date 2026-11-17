#!/usr/bin/env node
'use strict';

/**
 * Bring audited product-only feeds from the companion website-downloader
 * workspace into this project. The downloader validates a white/plain
 * background and no detected person before producing these source files.
 *
 * Override the default location with WEBSITE_DOWNLOADER_DIR when needed.
 */
const fs = require('node:fs');
const path = require('node:path');
const { normalizeCatalogImageUrl } = require('./lib/catalogImageUrl');

const ROOT = path.resolve(__dirname, '..');
const DOWNLOADER = process.env.WEBSITE_DOWNLOADER_DIR || '/Users/v.zahid/startups/website-downloader';
const PRODUCT_ONLY = /(?:no[-\s]?model|no[-\s]?mannequin|product[-\s]?only|plain[-\s]?background|white[-\s]?(?:bg|background))/i;
const VISIBLE_PERSON_EVIDENCE = /(?:\b(?:photo|image)\s+(?:shows?|features?)\b[^.]*\b(?:model|mannequin)\b|\bgarment\s+worn\s+by\s+(?:a\s+)?(?:male\s+)?model\b|\bmodel\s+(?:styled|wearing|wears)\b)/i;
const FEEDS = [
  ['boss', 'shop_catalog_uz_boss.json'],
  ['calvin_klein', 'shop_catalog_uz_ck.json'],
  ['gant_downloader', 'shop_catalog_uz_gant.json'],
  ['massimo_dutti', 'shop_catalog_uz_mens.json'],
  ['tommy_hilfiger', 'shop_catalog_uz_tommy.json'],
  ['zara_downloader', 'shop_catalog_uz_zara.json'],
];

function validateItem(item, input) {
  // Repair the known Zara `.jpg&w=…` construction before validation. The
  // downloader emits usable CDN images with a missing query separator; the
  // old validator rejected all of them. A continuation that cannot be
  // repaired is still refused below.
  item.image_url = normalizeCatalogImageUrl(item.image_url);
  for (const field of ['id', 'brand', 'name', 'price', 'currency', 'image_url', 'source_url']) {
    if (item[field] === undefined || item[field] === null || item[field] === '') {
      throw new Error(`${input}: missing ${field} for ${item.id || '(unknown id)'}`);
    }
  }
  if (!Number.isFinite(Number(item.price)) || Number(item.price) <= 0) {
    throw new Error(`${input}: invalid price for ${item.id}`);
  }
  if (/\.(?:jpe?g|png|webp|avif|gif)&/i.test(item.image_url.split('?')[0])) {
    throw new Error(`${input}: malformed image URL for ${item.id}`);
  }
  for (const url of [item.image_url, item.source_url]) {
    if (new URL(url).protocol !== 'https:') throw new Error(`${input}: expected HTTPS URL for ${item.id}`);
  }
  if (!PRODUCT_ONLY.test(`${item.source || ''} ${item.description || ''}`)) {
    throw new Error(`${input}: product-only provenance missing for ${item.id}`);
  }
}

let total = 0;
for (const [name, inputName] of FEEDS) {
  const input = path.join(DOWNLOADER, inputName);
  if (!fs.existsSync(input)) throw new Error(`Downloader feed missing: ${input}`);
  const items = JSON.parse(fs.readFileSync(input, 'utf8'));
  if (!Array.isArray(items) || items.length === 0) throw new Error(`${input}: expected a non-empty array`);
  const ids = new Set();
  const safeItems = [];
  let rejected = 0;
  for (const item of items) {
    validateItem(item, inputName);
    if (ids.has(item.id)) throw new Error(`${inputName}: duplicate id ${item.id}`);
    ids.add(item.id);
    if (VISIBLE_PERSON_EVIDENCE.test(`${item.name || ''} ${item.description || ''}`)) {
      rejected++;
      continue;
    }
    safeItems.push(item);
  }
  const output = path.join(ROOT, 'data', `shop_catalog_${name}.json`);
  if (!process.argv.includes('--check')) fs.writeFileSync(output, `${JSON.stringify(safeItems, null, 2)}\n`);
  total += safeItems.length;
  console.log(`${process.argv.includes('--check') ? 'Validated' : 'Synced'} ${safeItems.length} product-only ${name} items${rejected ? ` (${rejected} explicit model-photo rows excluded)` : ''}.`);
}
console.log(`${process.argv.includes('--check') ? 'Validated' : 'Synced'} ${total} products from ${FEEDS.length} downloader feeds.`);
