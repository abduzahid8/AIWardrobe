/**
 * Scrape MD - accept cookies, scroll, wait for lazy-loaded products
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

// Block unnecessary resources to speed up
await page.setRequestInterception(true);
page.on('request', req => {
  const type = req.resourceType();
  if (type === 'image' || type === 'stylesheet' || type === 'font' || type === 'media') {
    req.continue();
  } else {
    req.continue();
  }
});

// Collect ALL itxrest API responses
const productData = [];
page.on('response', async response => {
  const url = response.url();
  if (url.includes('itxrest/') || url.includes('/api/catalog') || url.includes('product')) {
    const ct = response.headers()['content-type'] || '';
    if (ct.includes('json')) {
      try {
        const text = await response.text();
        if (text.length > 100 && text.length < 1000000) {
          productData.push({ url, status: response.status(), body: text });
        }
      } catch {}
    }
  }
});

const categories = [
  { name: 'shirts', id: 'shirts-n1447' },
  { name: 't-shirts', id: 't-shirts-n1447' },
  { name: 'trousers', id: 'trousers-n1447' },
  { name: 'blazers', id: 'blazers-n1447' },
  { name: 'knitwear', id: 'knitwear-n1447' },
];

let allItems = [];

for (const cat of categories) {
  const url = `https://www.massimodutti.com/gb/men/${cat.id}`;
  console.log(`\n=== ${cat.name} ===`);
  console.log(`Loading: ${url}`);
  
  const catData = [];
  
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
  
  // Wait for Akamai challenge
  for (let i = 0; i < 30; i++) {
    const title = await page.title();
    if (title && title.length > 3 && title !== '\u00a0' && !title.includes('Access Denied')) {
      break;
    }
    await new Promise(r => setTimeout(r, 1000));
  }
  
  console.log('Title:', await page.title());
  
  // Accept cookies if banner present
  try {
    const cookieBtn = await page.$('button:has-text("Accept All"), button:has-text("Accept all"), [aria-label*="cookie"], [class*="cookie"] button');
    if (cookieBtn) {
      await cookieBtn.click();
      console.log('Cookie banner accepted');
      await new Promise(r => setTimeout(r, 2000));
    }
  } catch {}
  
  // Wait for product grid to load
  await new Promise(r => setTimeout(r, 8000));
  
  // Scroll down to trigger lazy loading
  await page.evaluate(() => window.scrollTo(0, 500));
  await new Promise(r => setTimeout(r, 2000));
  await page.evaluate(() => window.scrollTo(0, 1000));
  await new Promise(r => setTimeout(r, 2000));
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await new Promise(r => setTimeout(r, 3000));
  
  // Check if there's a "View more" or "Load more" button
  try {
    const loadMoreBtn = await page.$('button:has-text("View more"), button:has-text("Load more"), [class*="load-more"], [class*="show-more"]');
    if (loadMoreBtn) {
      await loadMoreBtn.click();
      console.log('Clicked "View more"');
      await new Promise(r => setTimeout(r, 3000));
    }
  } catch {}
  
  // Save full HTML
  const html = await page.content();
  fs.writeFileSync(`/tmp/md-${cat.name}-full.html`, html);
  console.log('HTML saved');
  
  // Extract product image URLs
  const imgRegex = /static\.massimodutti\.net\/assets\/public\/([a-f0-9]+\/[a-f0-9]+\/[a-f0-9]+\/[a-f0-9]+)\/(\d+-o1)\/\2\.jpg/g;
  const seen = new Set();
  let m;
  
  while ((m = imgRegex.exec(html)) !== null) {
    const pid = m[2].replace('-o1', '');
    if (!seen.has(pid)) {
      seen.add(pid);
      catData.push({
        productId: pid,
        imageUrl: `https://static.massimodutti.net/assets/public/${m[1]}/${m[2]}/${m[2]}.jpg`,
      });
    }
  }
  
  console.log(`Found ${catData.length} products`);
  
  // Also extract from the DOM
  const domProducts = await page.evaluate(() => {
    const results = [];
    
    // MD uses Angular and renders products server-side with specific selectors
    // Look for product grid items
    const grid = document.querySelector('[class*="product-grid"], [class*="grid"]');
    if (grid) {
      const items = grid.querySelectorAll('[class*="item"], li, article, div[class*="product"]');
      items.forEach(item => {
        const img = item.querySelector('img');
        const nameEl = item.querySelector('[class*="name"], h3, [class*="title"]');
        const priceEl = item.querySelector('[class*="price"], [class*="sale"]');
        const link = item.querySelector('a');
        
        const imgSrc = img ? (img.getAttribute('src') || img.getAttribute('data-src') || '') : '';
        const name = nameEl ? nameEl.textContent.trim() : '';
        const price = priceEl ? priceEl.textContent.trim() : '';
        const href = link ? link.getAttribute('href') : '';
        
        if (imgSrc || name) {
          results.push({ imgSrc: imgSrc.slice(0, 200), name: name.slice(0, 100), price: price.slice(0, 50), href });
        }
      });
    }
    
    // Also look for any product tile with specific MD classes
    const tiles = document.querySelectorAll('[class*="product-tile"], [class*="productTile"], [data-testid*="product"]');
    tiles.forEach(tile => {
      const img = tile.querySelector('img');
      const nameEl = tile.querySelector('[class*="name"], h3, [class*="title"]');
      const priceEl = tile.querySelector('[class*="price"]');
      
      results.push({
        imgSrc: img ? (img.getAttribute('src') || img.getAttribute('data-src') || '') : '',
        name: nameEl ? nameEl.textContent.trim() : '',
        price: priceEl ? priceEl.textContent.trim() : '',
        source: 'tile'
      });
    });
    
    return results;
  });
  
  console.log(`DOM products: ${domProducts.length}`);
  if (domProducts.length > 0) {
    domProducts.slice(0, 3).forEach(p => {
      console.log(`  ${p.name} | ${p.price} | ${p.imgSrc.slice(0, 80)}`);
    });
  }
  
  allItems.push({ category: cat.name, imageProducts: catData, domProducts });
  
  // Take screenshot
  await page.screenshot({ path: `/tmp/md-ss-${cat.name}.png` });
}

console.log(`\n\n=== SUMMARY ===`);
for (const cat of allItems) {
  console.log(`${cat.category}: ${cat.imageProducts.length} image URLs, ${cat.domProducts.length} DOM products`);
}

// Save combined results
fs.writeFileSync('/tmp/md-all-products.json', JSON.stringify(allItems, null, 2));

// Also check if the product data is in the API responses
console.log(`\n=== API RESPONSES ===`);
const seenUrls = new Set();
for (const r of productData) {
  if (!seenUrls.has(r.url)) {
    seenUrls.add(r.url);
    if (r.status === 200 && r.body.length > 500) {
      const hasProductData = r.body.includes('productId') || r.body.includes('"name"') || r.body.includes('products');
      if (hasProductData) {
        console.log(`\n${r.url}`);
        console.log(r.body.slice(0, 500));
      }
    }
  }
}

await browser.close();
