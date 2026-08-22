/**
 * Scrape Massimo Dutti products from Lyst using puppeteer-extra
 * Lyst doesn't have Akamai protection.
 */
import puppeteer from 'puppeteer-extra';
import StealthPlugin from 'puppeteer-extra-plugin-stealth';
import fs from 'fs';

puppeteer.use(StealthPlugin());

const browser = await puppeteer.launch({
  headless: true,
  args: ['--no-sandbox'],
});

const page = await browser.newPage();
await page.setUserAgent('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36');
await page.setViewport({ width: 1440, height: 900 });
page.setDefaultTimeout(60000);

// Collect API responses for product data
const apiData = [];
page.on('response', async response => {
  const url = response.url();
  const ct = response.headers()['content-type'] || '';
  if (ct.includes('json') && response.status() === 200) {
    try {
      const text = await response.text();
      if (text.length > 500 && text.length < 500000 && (url.includes('lyst') || url.includes('product'))) {
        apiData.push({ url, body: text.slice(0, 30000) });
      }
    } catch {}
  }
});

const categories = [
  { name: 'linen-shirts', url: '/shop/mens-massimo-dutti-shirts/?colour=White,Blue,Beige,Natural,Grey&material=Linen' },
  { name: 'linen-trousers', url: '/shop/mens-massimo-dutti-trousers/?material=Linen' },
  { name: 'polos', url: '/shop/mens-massimo-dutti-polo-shirts/' },
];

const allProducts = [];

for (const cat of categories) {
  const url = `https://www.lyst.co.uk${cat.url}`;
  console.log(`\n=== ${cat.name} ===`);
  console.log(`URL: ${url}`);
  
  try {
    await page.goto(url, { waitUntil: 'networkidle0', timeout: 60000 });
    await new Promise(r => setTimeout(r, 5000));
    
    const title = await page.title();
    console.log(`Title: ${title}`);
    
    // Try to extract product data from the page
    const products = await page.evaluate(() => {
      const results = [];
      
      // Look for product tiles in the rendered DOM
      const tiles = document.querySelectorAll('[class*="product"], [data-testid*="product"], article, [class*="tile"]');
      
      tiles.forEach(tile => {
        const img = tile.querySelector('img');
        const nameEl = tile.querySelector('[class*="name"], h3, [class*="title"], [class*="brand"]');
        const priceEl = tile.querySelector('[class*="price"], [class*="amount"], [class*="cost"]');
        const link = tile.querySelector('a[href]');
        const brandEl = tile.querySelector('[class*="brand"], [class*="designer"]');
        
        const imgSrc = img ? (img.getAttribute('src') || img.getAttribute('data-src') || '') : '';
        const name = nameEl ? nameEl.textContent.trim() : '';
        const price = priceEl ? priceEl.textContent.trim() : '';
        const brand = brandEl ? brandEl.textContent.trim() : '';
        const href = link ? link.getAttribute('href') : '';
        
        if (name) {
          results.push({ name: name.slice(0, 100), price: price.slice(0, 30), brand: brand.slice(0, 30), imgSrc: imgSrc.slice(0, 200), href });
        }
      });
      
      // Also check for data in script tags (JSON-LD, etc.)
      const scripts = document.querySelectorAll('script[type="application/ld+json"]');
      scripts.forEach(s => {
        try {
          const data = JSON.parse(s.textContent);
          if (data['@type'] === 'Product' || data['@graph']) {
            results.push({ type: 'jsonld', data: JSON.stringify(data).slice(0, 1000) });
          }
        } catch {}
      });
      
      return results;
    });
    
    console.log(`Found ${products.length} product references`);
    
    // Filter for actual product tiles (not nav elements)
    const realProducts = products.filter(p => p.imgSrc || (p.name && p.price));
    console.log(`Real products: ${realProducts.length}`);
    
    // Show samples
    realProducts.slice(0, 5).forEach((p, i) => {
      console.log(`  ${i+1}. ${p.name} | ${p.price} | ${p.imgSrc.slice(0, 80)}`);
    });
    
    allProducts.push({ category: cat.name, products: realProducts });
    
    // Save screenshot
    await page.screenshot({ path: `/tmp/lyst-${cat.name}.png` });
    
  } catch (err) {
    console.log(`Error: ${err.message.slice(0, 100)}`);
  }
}

console.log(`\n\n=== TOTAL ===`);
let total = 0;
for (const cat of allProducts) {
  console.log(`${cat.category}: ${cat.products.length}`);
  total += cat.products.length;
}
console.log(`Total: ${total}`);

// Save
fs.writeFileSync('/tmp/lyst-products.json', JSON.stringify(allProducts, null, 2));
console.log('\nSaved to /tmp/lyst-products.json');

// Also check API responses for product endpoints
console.log(`\nAPI responses: ${apiData.length}`);
const seen = new Set();
for (const r of apiData) {
  if (!seen.has(r.url)) {
    seen.add(r.url);
    if (r.url.includes('/api/') || r.url.includes('search')) {
      console.log(`\n${r.url}`);
      console.log(r.body.slice(0, 500));
    }
  }
}

await browser.close();
