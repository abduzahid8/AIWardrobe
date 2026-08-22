/**
 * Scrape MD - wait for product API calls explicitly
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

// Intercept API responses CAREFULLY
const productApis = [];
page.on('response', async response => {
  const url = response.url();
  // Only capture the itxrest API calls (product data)
  if (url.includes('itxrest/2/catalog') || url.includes('itxrest/1/catalog')) {
    const ct = response.headers()['content-type'] || '';
    if (ct.includes('json')) {
      try {
        const text = await response.text();
        if (text.length > 500 && text.length < 1000000) {
          productApis.push({ url, status: response.status(), body: text });
        }
      } catch {}
    }
  }
});

// Load a specific category page
const url = 'https://www.massimodutti.com/gb/men/shirts-n1447';
console.log(`Loading: ${url}`);

await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });

// Wait for Akamai challenge to resolve
for (let i = 0; i < 30; i++) {
  const title = await page.title();
  if (title && title.length > 3 && title !== '\u00a0' && !title.includes('Access Denied')) {
    console.log(`Page loaded: "${title}"`);
    break;
  }
  await new Promise(r => setTimeout(r, 1000));
}

// Now wait specifically for product API responses
console.log('Waiting for product API data...');
await new Promise(r => setTimeout(r, 15000));

console.log(`\nProduct API responses captured: ${productApis.length}`);
for (const api of productApis) {
  console.log(`\n${api.status} ${api.url}`);
  // Parse and summarize
  try {
    const data = JSON.parse(api.body);
    if (Array.isArray(data)) {
      console.log(`  Array with ${data.length} items`);
      if (data.length > 0) {
        console.log(`  First item keys: ${Object.keys(data[0]).slice(0, 10).join(', ')}`);
        console.log(`  Sample: ${JSON.stringify(data[0]).slice(0, 500)}`);
      }
    } else if (data.content && Array.isArray(data.content)) {
      console.log(`  Content array: ${data.content.length} items`);
      if (data.content.length > 0) {
        console.log(`  Sample: ${JSON.stringify(data.content[0]).slice(0, 500)}`);
      }
    }
    // Check for any product-related keys
    const objKeys = Object.keys(data).filter(k => k.toLowerCase().includes('product') || k.toLowerCase().includes('item') || k === 'elements');
    if (objKeys.length > 0) {
      console.log(`  Product keys: ${objKeys.join(', ')}`);
    }
    // Show top-level keys
    console.log(`  Top keys: ${Object.keys(data).slice(0, 15).join(', ')}`);
    
    // Save full data for processing
    fs.writeFileSync('/tmp/md-product-data.json', JSON.stringify(data, null, 2));
    console.log('  Full data saved to /tmp/md-product-data.json');
    
  } catch (e) {
    console.log(`  (not JSON or parse error: ${e.message})`);
    console.log(`  Body preview: ${api.body.slice(0, 300)}`);
  }
}

if (productApis.length === 0) {
  console.log('\nNo product APIs found. Saving page content for analysis...');
  const html = await page.content();
  // Search for API URLs in the HTML
  const apiMatches = html.match(/itxrest\/[12]\/catalog\/store\/\d+[^"'"'"]+/g);
  if (apiMatches) {
    console.log('API URLs found in page:', [...new Set(apiMatches)].slice(0, 10));
  }
  
  // Search for the actual API call patterns in JS
  const scripts = await page.$$('script');
  for (const script of scripts) {
    const text = await script.evaluate(el => el.textContent);
    if (text && text.includes('itxrest') && text.includes('catalog')) {
      const lines = text.split('\n').filter(l => l.includes('itxrest'));
      console.log('Found API references in script:', lines.slice(0, 5));
    }
  }
}

await browser.close();
