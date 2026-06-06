/**
 * Generate catalog entries from scraped MD products.
 * Filters for summer-appropriate men's items.
 */
import fs from 'fs';

const scraped = JSON.parse(fs.readFileSync('/tmp/md-scraped-products.json', 'utf-8'));

// Read existing catalog to find the last ID number
const existingCatalog = fs.readFileSync('./data/shopCatalogItems.ts', 'utf-8');

// Find the last MD item ID
const mdIds = [...existingCatalog.matchAll(/id:\s*"(classic-m-([a-z]+)-(\d+))"/g)].map(m => ({
  full: m[1], category: m[2], num: parseInt(m[3])
}));
const maxNums = {};
for (const { category, num } of mdIds) {
  maxNums[category] = Math.max(maxNums[category] || 0, num);
}
console.log('Current max IDs:', maxNums);

// Men's summer-appropriate items
// womanExclude list for cleanup
const womenIds = ['05083773', '05104704', '05049948', '05013674', '05340660', '05039937', '06239633', '06429979', '06738151'];

const keep = scraped.filter(r => {
  // Remove women's items by checking first 8 digits of product ID
  const prefix = r.productId.slice(0, 8);
  if (womenIds.includes(prefix)) return false;
  
  // Keep only items with verified CDN URLs
  if (!r.verified) return false;
  
  return true;
});

console.log(`\nAfter women's filter: ${keep.length}`);

// Categorize each product
function categorize(item) {
  const title = item.title.toLowerCase();
  const name = item.title.split('·')[0].trim();
  
  // Determine garment type
  let garmentType = 'upper_body';
  let category = 'shirt';
  
  if (title.includes('trouser') || title.includes('pant') || title.includes('jean')) {
    garmentType = 'lower_body';
    category = title.includes('jean') ? 'jean' : 'trouser';
  } else if (title.includes('short') || title.includes('bermuda')) {
    garmentType = 'lower_body';
    category = 'short';
  } else if (title.includes('shoe') || title.includes('derby') || title.includes('loafer') || title.includes('trainer') || title.includes('espadrille')) {
    garmentType = 'shoes';
    category = 'shoe';
  } else if (title.includes('polo')) {
    garmentType = 'upper_body';
    category = 'polo';
  } else if (title.includes('t-shirt') || title.includes('t shirt')) {
    garmentType = 'upper_body';
    category = 'tshirt';
  } else if (title.includes('blazer') || title.includes('jacket') || title.includes('coat') || title.includes('parka')) {
    garmentType = 'upper_body';
    category = 'jacket';
  } else if (title.includes('belt')) {
    garmentType = 'accessory';
    category = 'belt';
  } else if (title.includes('bag') || title.includes('backpack')) {
    garmentType = 'accessory';
    category = 'bag';
  } else if (title.includes('tie')) {
    garmentType = 'accessory';
    category = 'accessory';
  } else if (title.includes('cardigan') || title.includes('sweater') || title.includes('knit')) {
    garmentType = 'upper_body';
    category = 'knit';
  }
  
  return { garmentType, category };
}

// Generate entries
const entries = [];
const ids = { ...maxNums };

for (const item of keep) {
  const name = item.title.split('·')[0].trim();
  const { garmentType, category } = categorize(item);
  
  // Increment ID
  ids[category] = (ids[category] || 100) + 1;
  const itemId = `classic-m-${category}-${ids[category]}`;
  
  // Clean price
  let price = parseInt(item.price.replace(/[^0-9]/g, '')) || 0;
  if (price > 1000) price = Math.round(price / 100); // Convert pence/cents to pounds
  if (price > 500) price = Math.round(price / 100); // Handle different formats
  if (price > 300) price = 99; // Fallback
  
  entries.push({
    id: itemId,
    name,
    price,
    garmentType,
    imageUrl: item.cdnUrl,
    source: 'manual',
  });
}

// Generate TS output
let output = `// Generated MD catalog entries\n\n`;

for (const e of entries) {
  output += `  {\n`;
  output += `    id: '${e.id}',\n`;
  output += `    name: '${e.name.replace(/'/g, "\\'")}',\n`;
  output += `    price: ${e.price},\n`;
  output += `    garmentType: '${e.garmentType}',\n`;
  output += `    imageUrl: '${e.imageUrl}',\n`;
  output += `    source: 'manual',\n`;
  output += `  },\n`;
}

console.log(output);

// Also save as JSON for easy processing
const jsonEntries = entries.map(e => ({
  ...e,
  scrapedTitle: scraped.find(s => s.productId && s.productId === e.imageUrl.match(/(\d+)-o1/)?.[1])?.title || '',
}));

fs.writeFileSync('/tmp/new-md-entries.json', JSON.stringify(jsonEntries, null, 2));
console.log(`\n\nGenerated ${entries.length} entries`);
console.log('Saved to /tmp/new-md-entries.json');
