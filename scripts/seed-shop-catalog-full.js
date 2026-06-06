#!/usr/bin/env node
/**
 * scripts/seed-shop-catalog-full.js
 *
 * Reads shopCatalogItems.ts and upserts all items into Supabase shop_catalog.
 *
 * Usage:
 *   SUPABASE_URL=https://xxx.supabase.co \
 *   SUPABASE_SERVICE_KEY=eyJ... \
 *   node scripts/seed-shop-catalog-full.js
 */

const { createClient } = require('@supabase/supabase-js');
const path = require('path');
const fs = require('fs');

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY;

if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  console.error('❌ Set SUPABASE_URL and SUPABASE_SERVICE_KEY env vars first.');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

/**
 * Parse item objects from a TS file by extracting content between
 * top-level `{` and `}`, then extracting simple properties via regex.
 *
 * Handles:
 *   - Single-line and multi-line comments (removed before parsing)
 *   - Runtime expressions (e.g. outfitItems: [...SHOP_CATALOG_ITEMS.find(...)])
 *   - Trailing commas in objects
 */
function parseItems(filePath) {
  let content = fs.readFileSync(filePath, 'utf-8');

  // Remove multi-line comments first (safe since /* */ can't appear inside strings/URLs)
  content = content.replace(/\/\*[\s\S]*?\*\//g, '');
  // Remove line comments that start at the beginning of a line (with optional whitespace)
  // This is safe — comments like `// ── Shirts ──` start on their own line
  // We DO NOT strip `//` inline (e.g. `https://`) by only matching line-start comments
  content = content.replace(/^\s*\/\/[^\n]*$/gm, '');

  // Find the main array export — match up to '= [' so we find the right bracket
  const arrayMatch = content.match(/export\s+const\s+\w+\s*:\s*ShopCatalogItem\[\]\s*=\s*\[/);
  if (!arrayMatch) throw new Error(`Could not find array export in ${filePath}`);

  // The match ends at the '[' we want (it's part of the matched text)
  const startIdx = arrayMatch.index + arrayMatch[0].length - 1;
  if (startIdx === -1) throw new Error('Could not find opening bracket');

  // Walk to find matching closing bracket of the array
  let depth = 0;
  let endIdx = startIdx;
  for (let i = startIdx; i < content.length; i++) {
    const ch = content[i];
    if (ch === '[') depth++;
    else if (ch === ']') {
      depth--;
      if (depth === 0) { endIdx = i; break; }
    }
  }

  const arrayContent = content.slice(startIdx + 1, endIdx);

  // Extract each top-level object: { ... },
  const items = [];
  let i = 0;
  while (i < arrayContent.length) {
    // Skip to next opening brace
    const braceStart = arrayContent.indexOf('{', i);
    if (braceStart === -1) break;

    // Find matching closing brace
    depth = 0;
    let braceEnd = braceStart;
    for (let j = braceStart; j < arrayContent.length; j++) {
      if (arrayContent[j] === '{') depth++;
      else if (arrayContent[j] === '}') {
        depth--;
        if (depth === 0) { braceEnd = j; break; }
      }
    }
    if (depth !== 0) break; // malformed

    const objStr = arrayContent.slice(braceStart, braceEnd + 1);
    i = braceEnd + 1;

    // Extract simple properties with regex
    const getStr = (key) => {
      const re = new RegExp(`${key}\\s*:\\s*'([^']*)'`);
      const m = objStr.match(re);
      return m ? m[1] : null;
    };
    const getNum = (key) => {
      const re = new RegExp(`${key}\\s*:\\s*([\\d.]+)`);
      const m = objStr.match(re);
      return m ? parseFloat(m[1]) : null;
    };

    const id = getStr('id');
    if (!id) continue;

    const garment_type = getStr('garmentType') || '';
    const image_url = getStr('imageUrl') || '';
    if (!image_url || !garment_type) {
      if (!image_url) console.warn(`  ⚠ Skipping ${id} — no image_url`);
      continue;
    }

    const gt = garment_type;
    items.push({
      id,
      brand: getStr('brand') || '',
      name: getStr('name') || '',
      price: getNum('price') || 0,
      currency: getStr('currency') || 'USD',
      garment_type: gt,
      description: getStr('description') || '',
      image_url,
      category: gt === 'upper_body' ? 'tops'
        : gt === 'lower_body' ? 'bottoms'
        : gt === 'shoes' ? 'shoes'
        : gt === 'accessory' ? 'accessories'
        : gt === 'outfit' ? 'outfits'
        : 'other',
    });
  }

  return items;
}

async function main() {
  console.log(`\n🚀 Seeding full shop_catalog → ${SUPABASE_URL}\n`);

  const ROOT = path.join(__dirname, '..');
  const catalogPath = path.join(ROOT, 'data', 'shopCatalogItems.ts');

  const items = parseItems(catalogPath);
  console.log(`📦 Parsed ${items.length} items from catalog\n`);

  if (items.length === 0) {
    console.log('⚠ No items found. Aborting.');
    process.exit(1);
  }

  // Count by garment type
  const counts = {};
  for (const item of items) {
    counts[item.garment_type] = (counts[item.garment_type] || 0) + 1;
  }
  for (const [k, v] of Object.entries(counts)) {
    console.log(`  ${k}: ${v}`);
  }
  console.log('');

  // Upsert in batches of 50
  const BATCH_SIZE = 50;
  let total = 0;
  for (let i = 0; i < items.length; i += BATCH_SIZE) {
    const batch = items.slice(i, i + BATCH_SIZE);
    const { error } = await supabase
      .from('shop_catalog')
      .upsert(batch, { onConflict: 'id' });

    if (error) {
      console.error(`\n❌ Batch ${i / BATCH_SIZE + 1} failed:`, error.message);
      console.error('First item in failed batch:', JSON.stringify(batch[0], null, 2));
      process.exit(1);
    }
    total += batch.length;
    console.log(`  ✓ Upserted ${total}/${items.length} items`);
  }

  console.log(`\n✅ Successfully seeded ${total} items to shop_catalog.\n`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
