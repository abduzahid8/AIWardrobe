import fs from 'fs';
import path from 'path';

const DATA_DIR = path.join(process.cwd(), 'data');

const CATEGORY_TO_GARMENT = {
  'Tops': 'upper_body',
  'T-Shirts': 'upper_body',
  'Sweatshirts & Hoodies': 'upper_body',
  'Shirts & Polos': 'upper_body',
  'Polos': 'upper_body',
  'Sweaters': 'upper_body',
  'Bottoms': 'lower_body',
  'Jeans': 'lower_body',
  'Shorts': 'lower_body',
  'Outerwear': 'upper_body',
  'Accessories': 'accessory',
  'Innerwear': null, // skip
};

const EXCLUDE_KEYWORDS = [
  'boxer brief', 'boxer', 'trunks', 'briefs', 'underwear',
  'socks', 'innerwear', 'gift bag',
];

function shouldInclude(product) {
  const name = (product.name || '').toLowerCase();
  for (const kw of EXCLUDE_KEYWORDS) {
    if (name.includes(kw)) return false;
  }
  return true;
}

function generateId(product) {
  const prefix = product.productId;
  const nameSlug = product.name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .substring(0, 40);
  return `uniqlo-m-${prefix}-${nameSlug}`;
}

function parsePrice(priceStr) {
  if (!priceStr) return 29.90;
  const num = parseFloat(priceStr.replace(/[^0-9.]/g, ''));
  return isNaN(num) ? 29.90 : num;
}

function generateCatalogEntry(product) {
  const garmentType = CATEGORY_TO_GARMENT[product.category];
  if (!garmentType) return null;
  if (!shouldInclude(product)) return null;

  const price = parsePrice(product.price);
  const id = generateId(product);

  return {
    id,
    brand: 'UNIQLO',
    name: product.name,
    price,
    currency: 'USD',
    garmentType,
    description: `UNIQLO ${product.category} - ${product.name}. ${product.category === 'Accessories' ? 'Accessory' : 'Men\'s apparel'} from UNIQLO.`,
    imageUrl: product.imageUrl,
  };
}

function generateTypeScriptCode(entries) {
  const lines = [];
  for (const e of entries) {
    lines.push('    {');
    lines.push(`        id: '${e.id}',`);
    lines.push(`        brand: '${e.brand}',`);
    lines.push(`        name: '${e.name.replace(/'/g, "\\'")}',`);
    lines.push(`        price: ${e.price},`);
    lines.push(`        currency: '${e.currency}',`);
    lines.push(`        garmentType: '${e.garmentType}',`);
    lines.push(`        description: '${e.description.replace(/'/g, "\\'")}',`);
    lines.push(`        imageUrl: '${e.imageUrl}',`);
    lines.push('    },');
  }
  return lines.join('\n');
}

function main() {
  const inputPath = path.join(DATA_DIR, 'uniqlo-products.json');
  const raw = JSON.parse(fs.readFileSync(inputPath, 'utf-8'));

  console.log(`Loaded ${raw.length} raw products`);

  // Filter and map
  const entries = raw
    .map(generateCatalogEntry)
    .filter(Boolean);

  console.log(`After filtering: ${entries.length} entries`);

  // Generate TypeScript snippet
  const tsCode = generateTypeScriptCode(entries);

  // Save just the entries as JSON for the inspo file
  const outputPath = path.join(DATA_DIR, 'uniqlo-catalog-entries.json');
  fs.writeFileSync(outputPath, JSON.stringify(entries, null, 2), 'utf-8');
  console.log(`Saved catalog entries to ${outputPath}`);

  // Print the TypeScript code
  console.log('\n=== TypeScript Catalog Entries ===');
  console.log(tsCode);

  // Count by garment type
  const counts = {};
  for (const e of entries) {
    counts[e.garmentType] = (counts[e.garmentType] || 0) + 1;
  }
  console.log('\n=== Counts by garmentType ===');
  for (const [k, v] of Object.entries(counts)) {
    console.log(`  ${k}: ${v}`);
  }
  console.log(`  TOTAL: ${entries.length}`);
}

main();
