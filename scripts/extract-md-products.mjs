/**
 * Extract ALL product data from the saved MD HTML
 */
import fs from 'fs';

const html = fs.readFileSync('/tmp/md-Shirts.html', 'utf-8');

// 1. Extract all product image URLs (static.massimodutti.net assets with -o1 pattern)
const imgRegex = /static\.massimodutti\.net\/assets\/public\/([a-f0-9]+\/[a-f0-9]+\/[a-f0-9]+\/[a-f0-9]+)\/(\d+-o1)\/\2\.jpg/g;

const products = [];
const seen = new Set();
let match;

while ((match = imgRegex.exec(html)) !== null) {
  const hash = match[1];
  const productId = match[2].replace('-o1', '');
  const imageUrl = `https://static.massimodutti.net/assets/public/${hash}/${match[2]}/${match[2]}.jpg`;
  
  if (!seen.has(productId)) {
    seen.add(productId);
    
    // Try to find the product name/price near this URL in the HTML
    // Search around the match position (within 5000 chars)
    const start = Math.max(0, match.index - 5000);
    const end = Math.min(html.length, match.index + 2000);
    const context = html.slice(start, end);
    
    // Extract name-like patterns
    const nameMatch = context.match(/<h3[^>]*>([^<]+)<\/h3>/);
    const name = nameMatch ? nameMatch[1].trim() : '';
    
    // Extract price-like patterns
    const priceMatch = context.match(/(?:£|\$|€)\s*(\d+[.,]\d{2})/);
    const price = priceMatch ? priceMatch[0] : '';
    
    // Extract alt text from nearby img tags
    const altMatch = context.match(/alt="([^"]*)"/);
    const alt = altMatch ? altMatch[1] : '';
    
    products.push({ productId, imageUrl, name, price, alt: alt.slice(0, 100) });
  }
}

console.log(`Total unique products found: ${products.length}`);

// Show first 10
products.slice(0, 10).forEach((p, i) => {
  console.log(`\n${i + 1}. ID: ${p.productId}`);
  console.log(`   Name: ${p.name}`);
  console.log(`   Price: ${p.price}`);
  console.log(`   Alt: ${p.alt}`);
  console.log(`   Image: ${p.imageUrl.slice(0, 100)}...`);
});

// Check product categories by looking at the alt text
const menProducts = products.filter(p => p.alt.toLowerCase().includes('men') || !p.alt.toLowerCase().includes('women'));
const womenProducts = products.filter(p => p.alt.toLowerCase().includes('women'));
console.log(`\nMen (or unisex): ${menProducts.length}`);
console.log(`Women: ${womenProducts.length}`);

// Output as JSON
fs.writeFileSync('/tmp/md-extracted-products.json', JSON.stringify(products, null, 2));
console.log('\nSaved to /tmp/md-extracted-products.json');
