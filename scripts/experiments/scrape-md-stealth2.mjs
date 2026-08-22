/**
 * Scrape MD using puppeteer-extra stealth - fixed version
 */
import puppeteer from 'puppeteer-extra';
import StealthPlugin from 'puppeteer-extra-plugin-stealth';
import fs from 'fs';

puppeteer.use(StealthPlugin());

const browser = await puppeteer.launch({
  headless: true,
  args: ['--no-sandbox', '--disable-setuid-sandbox'],
});

const page = await browser.newPage();
await page.setUserAgent('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36');
await page.setViewport({ width: 1280, height: 900 });
page.setDefaultTimeout(120000);

// Capture all XHR/fetch responses
const apiData = [];
page.on('response', async response => {
  const url = response.url();
  const ct = response.headers()['content-type'] || '';
  if (ct.includes('json') || url.includes('itxrest') || url.includes('/api/')) {
    try {
      const text = await response.text();
      if (text.length > 100 && text.length < 500000) {
        apiData.push({ url, status: response.status(), body: text.slice(0, 20000) });
      }
    } catch {}
  }
});

const categoryUrls = [
  { name: 'Shirts', url: 'https://www.massimodutti.com/gb/men/shirts-n1447' },
  { name: 'Tshirts', url: 'https://www.massimodutti.com/gb/men/t-shirts-n1447' },
  { name: 'Trousers', url: 'https://www.massimodutti.com/gb/men/trousers-n1447' },
  { name: 'Jeans', url: 'https://www.massimodutti.com/gb/men/jeans-n1447' },
  { name: 'Blazers', url: 'https://www.massimodutti.com/gb/men/blazers-n1447' },
  { name: 'Knitwear', url: 'https://www.massimodutti.com/gb/men/knitwear-n1447' },
  { name: 'Shoes', url: 'https://www.massimodutti.com/gb/men/shoes-n1447' },
];

const allResults = [];

for (const cat of categoryUrls) {
  console.log(`\n=== ${cat.name} ===`);
  
  try {
    await page.goto(cat.url, { waitUntil: 'domcontentloaded', timeout: 90000 });
    
    // Wait for Akamai challenge to resolve (check title)
    let resolved = false;
    for (let i = 0; i < 30; i++) {
      const title = await page.title();
      if (title && !title.includes('Access Denied') && title.length > 3 && title !== '\u00a0') {
        resolved = true;
        break;
      }
      await new Promise(r => setTimeout(r, 1000));
    }
    
    if (!resolved) {
      console.log('  Still blocked after 30s');
      continue;
    }
    
    console.log(`  Title: "${await page.title()}"`);
    
    // Give products time to render
    await new Promise(r => setTimeout(r, 5000));
    
    // Extract product data
    const products = await page.evaluate(() => {
      const results = [];
      
      // Find all product grid items
      // MD uses a specific product grid structure
      const gridItems = document.querySelectorAll('[class*="product-grid"] [class*="product"], [class*="grid__item"], li[class*="product"], article');
      
      gridItems.forEach(item => {
        const img = item.querySelector('img');
        const link = item.querySelector('a');
        const nameEl = item.querySelector('[class*="name"], [class*="title"], h3');
        const priceEl = item.querySelector('[class*="price"], [class*="Price"], [class*="sale"]');
        
        const imgSrc = img ? (img.getAttribute('src') || img.getAttribute('data-src') || img.getAttribute('data-original') || '') : '';
        const href = link ? link.getAttribute('href') || '' : '';
        const name = nameEl ? (nameEl.textContent || '').trim() : '';
        const price = priceEl ? (priceEl.textContent || '').trim() : '';
        
        if (imgSrc && imgSrc.includes('massimodutti') && (name || href)) {
          results.push({
            imgSrc: imgSrc.startsWith('http') ? imgSrc : `https://www.massimodutti.com${imgSrc}`,
            href: href.startsWith('http') ? href : `https://www.massimodutti.com${href}`,
            name, price,
          });
        }
      });
      
      // Also look for the product data in the page source (SSR)
      const scripts = document.querySelectorAll('script');
      for (const s of scripts) {
        const t = s.textContent || '';
        // Look for JSON product data
        const matches = t.match(/"productId":\s*"(\d+)"/g);
        if (matches && matches.length > 5) {
          results.push({ type: 'hasProductData', count: matches.length, preview: t.slice(0, 1000) });
        }
        // Look for __NEXT_DATA__ or redux state
        if (t.includes('__NEXT_DATA__') || t.includes('window.__INITIAL_STATE__') || t.includes('window.__DATA__')) {
          results.push({ type: 'appState', preview: t.slice(0, 2000) });
        }
      }
      
      return results;
    });
    
    console.log(`  Found ${products.length} product references`);
    allResults.push({ category: cat.name, products });
    
    // Show samples
    const withImages = products.filter(p => p.imgSrc);
    console.log(`  With CDN images: ${withImages.length}`);
    for (const p of withImages.slice(0, 3)) {
      console.log(`    ${p.name} | ${p.price} | ${p.imgSrc.slice(0, 80)}`);
    }
    
    // Save HTML for analysis
    const html = await page.content();
    fs.writeFileSync(`/tmp/md-${cat.name}.html`, html);
    console.log(`  HTML saved (${html.length} bytes)`);
    
  } catch (err) {
    console.log(`  Error: ${err.message.slice(0, 150)}`);
  }
}

console.log(`\n\n=== SUMMARY ===`);
for (const r of allResults) {
  console.log(`  ${r.category}: ${r.products.length} items`);
}

// Save results
fs.writeFileSync('/tmp/md-all-data.json', JSON.stringify(allResults, null, 2));
console.log('\nResults saved to /tmp/md-all-data.json');

// Analyze API responses for product endpoints
console.log(`\nAPI responses: ${apiData.length}`);
const apiUrls = [...new Set(apiData.map(r => r.url))];
for (const url of apiUrls) {
  const entry = apiData.find(r => r.url === url);
  console.log(`  ${entry.status} ${url.slice(0, 200)}`);
  
  // Check if it has product data
  if (entry.status === 200 && entry.body && (entry.body.includes('product') || entry.body.includes('Product'))) {
    const preview = entry.body.slice(0, 500);
    console.log(`    Preview: ${preview}`);
  }
}

await browser.close();
