/**
 * Scrape Massimo Dutti using puppeteer-extra with stealth plugin
 * to bypass Akamai bot detection.
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
page.setDefaultTimeout(60000);

// Capture API responses
const apiData = [];
page.on('response', async response => {
  const url = response.url();
  if (url.includes('massimodutti') && !url.includes('.css') && !url.includes('.js') && !url.includes('.png') && !url.includes('.jpg')) {
    const ct = response.headers()['content-type'] || '';
    if (ct.includes('json')) {
      try {
        const text = await response.text();
        if (text.length > 200 && text.length < 500000) {
          apiData.push({ url, status: response.status(), body: text.slice(0, 20000) });
        }
      } catch {}
    }
  }
});

const categories = [
  { name: 'Shirts', url: 'https://www.massimodutti.com/gb/men/shirts-n1447' },
  { name: 'T-shirts & Polos', url: 'https://www.massimodutti.com/gb/men/t-shirts-n1447?page=1' },
  { name: 'Trousers', url: 'https://www.massimodutti.com/gb/men/trousers-n1447' },
  { name: 'Jeans', url: 'https://www.massimodutti.com/gb/men/jeans-n1447' },
  { name: 'Blazers', url: 'https://www.massimodutti.com/gb/men/blazers-n1447' },
  { name: 'Knitwear', url: 'https://www.massimodutti.com/gb/men/knitwear-n1447' },
  { name: 'Shoes', url: 'https://www.massimodutti.com/gb/men/shoes-n1447' },
];

const allProducts = [];

for (const cat of categories) {
  console.log(`\n=== ${cat.name} ===`);
  
  try {
    await page.goto(cat.url, { waitUntil: 'networkidle0', timeout: 60000 });
    
    // Check if we got past the challenge
    const title = await page.title();
    console.log(`  Title: "${title}"`);
    
    if (title.includes('Access Denied') || title === '\u00a0' || !title) {
      console.log('  Blocked by Akamai challenge');
      continue;
    }
    
    // Wait for products to render
    await page.waitForTimeout(3000);
    
    // Save HTML for debugging
    const html = await page.content();
    fs.writeFileSync(`/tmp/md-${cat.name.toLowerCase()}.html`, html.slice(0, 100000));
    
    // Extract product data from the DOM
    const products = await page.evaluate(() => {
      const results = [];
      
      // Try to find any product-like elements
      const allElements = document.querySelectorAll('*');
      const productElements = [];
      
      // Look for elements with common product class patterns
      for (const el of allElements) {
        const cls = el.className || '';
        if (typeof cls === 'string' && (
          cls.includes('product') || cls.includes('Product') || 
          cls.includes('tile') || cls.includes('grid-item') ||
          cls.includes('item-card') || cls.includes('product-card')
        )) {
          productElements.push(el);
        }
      }
      
      console.log('Product elements:', productElements.length);
      
      productElements.forEach(tile => {
        const text = tile.textContent?.trim() || '';
        const html = tile.innerHTML?.slice(0, 500) || '';
        
        // Find images
        const imgs = tile.querySelectorAll('img');
        const imgSrc = imgs[0] ? (imgs[0].getAttribute('src') || imgs[0].getAttribute('data-src') || '') : '';
        
        // Find links
        const links = tile.querySelectorAll('a');
        const href = links[0] ? links[0].getAttribute('href') || '' : '';
        
        if ((imgSrc && text) || (href && text)) {
          results.push({ text: text.slice(0, 200), imgSrc: imgSrc.slice(0, 200), href: href.slice(0, 200) });
        }
      });
      
      // Also look for JSON-LD
      const jsonldScripts = document.querySelectorAll('script[type="application/ld+json"]');
      jsonldScripts.forEach(script => {
        try {
          const data = JSON.parse(script.textContent);
          if (data['@type'] === 'Product') {
            results.push({ type: 'jsonld', name: data.name, price: data.offers?.price, image: data.image });
          } else if (data['@graph']) {
            data['@graph'].filter(i => i['@type'] === 'Product').forEach(p => {
              results.push({ type: 'jsonld', name: p.name, price: p.offers?.price, image: p.image });
            });
          }
        } catch {}
      });
      
      // Check for __NEXT_DATA__ or Apollo state
      const dataScripts = document.querySelectorAll('script[type="application/json"], script#__NEXT_DATA__');
      dataScripts.forEach(s => {
        results.push({ type: 'app-state', length: (s.textContent || '').length, preview: (s.textContent || '').slice(0, 500) });
      });
      
      return results;
    });
    
    console.log(`  Products: ${products.length}`);
    allProducts.push(...products);
    
    if (products.length > 0) {
      for (const p of products.slice(0, 3)) {
        if (p.type === 'jsonld') console.log(`  JSON-LD: ${p.name} - $${p.price}`);
        else if (p.name) console.log(`  ${p.name} - ${p.price}`);
        else console.log(`  ${JSON.stringify(p).slice(0, 150)}`);
      }
    }
    
    await page.screenshot({ path: `/tmp/md-puppeteer-${cat.name.toLowerCase().replace(/[&\s]+/g, '-')}.png` });
    
  } catch (err) {
    console.log(`  Error: ${err.message}`);
  }
}

console.log(`\n\nTotal items: ${allProducts.length}`);

// Save
const output = { products: allProducts, apis: apiData };
fs.writeFileSync('/tmp/md-puppeteer-data.json', JSON.stringify(output, null, 2));
console.log('Saved to /tmp/md-puppeteer-data.json');

if (apiData.length > 0) {
  console.log(`\nAPI responses (${apiData.length}):`);
  const seen = new Set();
  for (const r of apiData) {
    if (!seen.has(r.url)) {
      seen.add(r.url);
      console.log(`  ${r.status} ${r.url.slice(0, 150)}`);
    }
  }
}

await browser.close();
