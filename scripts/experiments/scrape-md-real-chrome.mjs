/**
 * Scrape Massimo Dutti using a real Chrome instance with remote debugging.
 * This bypasses Playwright's automation detection since the browser is a real user session.
 */
import { chromium } from 'playwright';
import { execSync, spawn } from 'child_process';
import fs from 'fs';

// Kill any previous Chrome remote debugging instances
try { execSync('pkill -f "remote-debugging-port=9222"'); } catch {}

// Launch real Chrome with remote debugging
const chromePath = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const userDataDir = '/tmp/chrome-md-scrape';

const args = [
  `--remote-debugging-port=9222`,
  `--user-data-dir=${userDataDir}`,
  '--no-first-run',
  '--no-default-browser-check',
  '--disable-fre',
];

const chromeProcess = spawn(chromePath, args, {
  stdio: 'ignore',
  detached: true,
});

// Wait for Chrome to start
await new Promise(r => setTimeout(r, 3000));

// Connect Playwright to the real Chrome
const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
const defaultContext = browser.contexts()[0];
const page = await defaultContext.newPage();
page.setDefaultTimeout(60000);

const categories = [
  { name: 'Shirts', url: 'https://www.massimodutti.com/gb/men/shirts-n1447' },
  { name: 'T-shirts & Polos', url: 'https://www.massimodutti.com/gb/men/t-shirts-n1447' },
  { name: 'Trousers', url: 'https://www.massimodutti.com/gb/men/trousers-n1447' },
  { name: 'Jeans', url: 'https://www.massimodutti.com/gb/men/jeans-n1447' },
  { name: 'Blazers', url: 'https://www.massimodutti.com/gb/men/blazers-n1447' },
  { name: 'Knitwear', url: 'https://www.massimodutti.com/gb/men/knitwear-n1447' },
  { name: 'Shoes', url: 'https://www.massimodutti.com/gb/men/shoes-n1447' },
];

// Capture API responses
const apiData = [];
page.on('response', async response => {
  const url = response.url();
  if (url.includes('massimodutti')) {
    const ct = response.headers()['content-type'] || '';
    if (ct.includes('json') || url.includes('/api/') || url.includes('itxrest') || url.includes('xmedia')) {
      try {
        const text = await response.text();
        if (text.length > 200 && text.length < 500000) {
          apiData.push({ url, status: response.status(), body: text.slice(0, 20000) });
        }
      } catch {}
    }
  }
  // Also capture product image page loads
  if (url.includes('massimodutti.net') && url.includes('meta.json')) {
    try {
      const text = await response.text();
      apiData.push({ url, status: response.status(), body: text });
    } catch {}
  }
});

const allProducts = [];

for (const cat of categories) {
  console.log(`\n=== ${cat.name} ===`);
  console.log(`Loading: ${cat.url}`);
  
  try {
    await page.goto(cat.url, { waitUntil: 'domcontentloaded', timeout: 60000 });
    
    // Wait for Akamai to resolve and products to load
    let challengeDone = false;
    let attempts = 0;
    while (attempts < 30) {
      const title = await page.title();
      if (title && !title.includes('Access Denied') && title !== '\u00a0' && title.length > 2) {
        challengeDone = true;
        break;
      }
      await page.waitForTimeout(1000);
      attempts++;
    }
    
    if (!challengeDone) {
      console.log('  Failed to bypass Akamai challenge');
      continue;
    }
    
    console.log(`  Page title: ${await page.title()}`);
    
    // Wait for products to render
    await page.waitForTimeout(5000);
    
    // Extract product data from the DOM
    const products = await page.evaluate(() => {
      const results = [];
      
      // Find all product tiles
      const tiles = document.querySelectorAll('[class*="product-grid"] [class*="product"], article, [data-testid*="product"], [class*="product-tile"]');
      console.log('Tiles found:', tiles.length);
      
      tiles.forEach(tile => {
        // Try to extract name, price, image, link
        const nameEl = tile.querySelector('[class*="name"], [class*="title"], [class*="product-name"], h3, h2, a[class*="product"]');
        const priceEl = tile.querySelector('[class*="price"], [class*="Price"], span[class*="price"]');
        const imgEl = tile.querySelector('img[src*="massimodutti"], img[src*="xmedia"]');
        const linkEl = tile.querySelector('a[href*="product"]');
        
        const name = nameEl ? (nameEl.textContent || nameEl.getAttribute('title') || '').trim() : '';
        const priceText = priceEl ? (priceEl.textContent || '').trim() : '';
        const imgSrc = imgEl ? (imgEl.getAttribute('src') || imgEl.getAttribute('data-src') || '') : '';
        const link = linkEl ? linkEl.getAttribute('href') : '';
        
        if (name || imgSrc) {
          results.push({ name, priceText, imgSrc, link });
        }
      });
      
      // Also look for JSON-LD structured data
      const jsonldScripts = document.querySelectorAll('script[type="application/ld+json"]');
      jsonldScripts.forEach(script => {
        try {
          const data = JSON.parse(script.textContent);
          if (data['@type'] === 'Product') {
            results.push({ type: 'jsonld', data });
          } else if (data['@graph']) {
            data['@graph'].forEach(item => {
              if (item['@type'] === 'Product') {
                results.push({ type: 'jsonld', data: item });
              }
            });
          }
        } catch {}
      });
      
      // Try to find product data in script tags or __NEXT_DATA__
      const scripts = document.querySelectorAll('script[type="application/json"], script#__NEXT_DATA__');
      scripts.forEach(s => {
        try {
          const data = JSON.parse(s.textContent);
          results.push({ type: 'state', data: JSON.stringify(data).slice(0, 3000) });
        } catch {}
      });
      
      return results;
    });
    
    console.log(`  Products extracted: ${products.length}`);
    
    // Show a few samples
    for (const p of products.slice(0, 3)) {
      if (p.type === 'jsonld') {
        console.log(`  JSON-LD: ${p.data.name} - ${p.data.offers?.price || 'N/A'}`);
      } else if (p.name) {
        console.log(`  Tile: ${p.name} - ${p.priceText}`);
      }
    }
    
    allProducts.push(...products);
    
    // Take screenshot
    await page.screenshot({ path: `/tmp/md-${cat.name.toLowerCase().replace(/[&\s]+/g, '-')}.png`, fullPage: false });
    
  } catch (err) {
    console.log(`  Error: ${err.message}`);
  }
}

console.log(`\n\nTotal products found: ${allProducts.length}`);

// Save results
const output = {
  products: allProducts.slice(0, 200),
  apiResponses: apiData,
};
fs.writeFileSync('/tmp/md-scraped-data.json', JSON.stringify(output, null, 2));
console.log('Data saved to /tmp/md-scraped-data.json');

if (apiData.length > 0) {
  console.log(`\nAPI responses captured: ${apiData.length}`);
  const urls = [...new Set(apiData.map(r => r.url))];
  for (const u of urls) {
    console.log(`  ${apiData.find(r => r.url === u).status} ${u.slice(0, 150)}`);
  }
}

// Close Chrome after a moment
await new Promise(r => setTimeout(r, 2000));
await browser.close();
chromeProcess.kill();
try { execSync('pkill -f "remote-debugging-port=9222"'); } catch {}
