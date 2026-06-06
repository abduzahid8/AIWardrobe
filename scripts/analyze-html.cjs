const fs = require('fs');
const html = fs.readFileSync('/tmp/md-Shirts.html', 'utf-8');
console.log('HTML size:', html.length);

// Look for product IDs (Inditex product codes are 11 digits)
const productIds = html.match(/[0-9]{11}/g);
if (productIds) {
  const unique = [...new Set(productIds)];
  console.log('11-digit numbers found:', unique.length);
  console.log('Samples:', unique.slice(0, 20));
}

// Look for price patterns  
const prices = html.match(/["']\$\d+[.,]\d+["']/g);
if (prices) console.log('Prices found:', prices.slice(0, 10));

// Search for product div blocks
const productBlocksCount = (html.match(/product/g) || []).length;
console.log('"product" occurrences:', productBlocksCount);

// Check for ng-template or Angular component instances
const ngTemplates = html.match(/<!--[^>]*product[^>]*-->/gi);
console.log('Product comments:', ngTemplates ? ngTemplates.length : 0);

// Look for Angular server-side rendered state
const serverState = html.match(/<script[^>]*server-state[^>]*>[\s\S]*?<\/script>/g);
console.log('Server state scripts:', serverState ? serverState.length : 0);
if (serverState) {
  serverState.forEach((s, i) => {
    console.log(`  Script ${i}: ${s.slice(0, 200)}`);
  });
}

// Look for JSON in script tags
const jsonScripts = html.match(/<script[^>]*type="application\/json"[^>]*>[\s\S]*?<\/script>/g);
console.log('\nJSON scripts:', jsonScripts ? jsonScripts.length : 0);
if (jsonScripts) {
  jsonScripts.forEach((s, i) => {
    console.log(`  Script ${i}: ${s.slice(0, 300)}`);
  });
}

// Look for productName in the HTML
const pn = html.match(/"productName"/g);
console.log('\n"productName" occurrences:', pn ? pn.length : 0);

// Look for any image URLs with product IDs
const imgUrls = html.match(/https?:\/\/[^"']*(?:massimodutti|xmedia)[^"']*\.(?:jpg|png|webp)/g);
if (imgUrls) {
  console.log('\nProduct image URLs:', imgUrls.length);
  imgUrls.slice(0, 5).forEach(u => console.log('  ' + u));
}

// Get all static.massimodutti.net asset URLs
const assetUrls = html.match(/static\.massimodutti\.net\/assets\/public\/[^"']+/g);
if (assetUrls) {
  const unique = [...new Set(assetUrls)];
  console.log('\nCDN asset URLs:', unique.length);
  unique.slice(0, 3).forEach(u => console.log('  ' + u));
}
