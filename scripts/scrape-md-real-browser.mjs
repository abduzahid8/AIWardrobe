/**
 * Scrape Massimo Dutti products using real browser (non-headless)
 * to bypass Akamai JavaScript challenge.
 * 
 * Opens a visible browser window briefly to access MD and extract product data.
 */
import { chromium } from 'playwright';
import fs from 'fs';

const browser = await chromium.launch({
  headless: false,
  args: ['--disable-blink-features=AutomationControlled'],
});

const context = await browser.newContext({
  userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
  locale: 'en-GB',
  viewport: { width: 1280, height: 900 },
});

const page = await context.newPage();
page.setDefaultTimeout(60000);

// Capture API responses that contain product data
const apiData = [];
page.on('response', async response => {
  const url = response.url();
  if (url.includes('massimodutti') && response.status() === 200) {
    const ct = response.headers()['content-type'] || '';
    if (ct.includes('json') || url.includes('/api/') || url.includes('.json')) {
      try {
        const text = await response.text();
        if (text.length > 500 && text.length < 500000) {
          apiData.push({ url, body: text.slice(0, 10000) });
        }
      } catch {}
    }
  }
});

const categories = [
  { name: 'Shirts', url: 'https://www.massimodutti.com/gb/men/shirts-n1447' },
  { name: 'T-shirts & Polos', url: 'https://www.massimodutti.com/gb/men/t-shirts-n1447' },
  { name: 'Trousers', url: 'https://www.massimodutti.com/gb/men/trousers-n1447' },
  { name: 'Jeans', url: 'https://www.massimodutti.com/gb/men/jeans-n1447' },
  { name: 'Blazers', url: 'https://www.massimodutti.com/gb/men/blazers-n1447' },
  { name: 'Knitwear', url: 'https://www.massimodutti.com/gb/men/knitwear-n1447' },
  { name: 'Shoes', url: 'https://www.massimodutti.com/gb/men/shoes-n1447' },
  { name: 'Accessories', url: 'https://www.massimodutti.com/gb/men/accessories-n1447' },
];

const allProducts = [];

for (const cat of categories) {
  console.log(`\n=== ${cat.name} ===`);
  console.log(`Navigating to: ${cat.url}`);
  
  try {
    await page.goto(cat.url, { waitUntil: 'networkidle', timeout: 60000 });
    
    // Wait for Akamai challenge to complete and page to render
    // We'll check if we're past the challenge by looking for product content
    let pastChallenge = false;
    for (let i = 0; i < 20; i++) {
      const title = await page.title();
      if (!title.includes('Access Denied') && title !== '\u00a0' && title.length > 0) {
        pastChallenge = true;
        break;
      }
      console.log(`  Waiting for Akamai challenge (${i + 1}s)...`);
      await page.waitForTimeout(1000);
    }
    
    if (!pastChallenge) {
      console.log('  Could not bypass Akamai challenge');
      continue;
    }
    
    const title = await page.title();
    console.log(`  Title: ${title}`);
    
    // Take screenshot to see the state
    await page.screenshot({ path: `/tmp/md-${cat.name.toLowerCase().replace(/[ &]+/g, '-')}.png` });
    
    // Wait for product grid to load
    await page.waitForTimeout(3000);
    
    // Extract product data from the page
    const products = await page.evaluate(() => {
      const items = [];
      
      // Try to find product data in JSON-LD
      const jsonld = document.querySelectorAll('script[type="application/ld+json"]');
      for (const script of jsonld) {
        try {
          const data = JSON.parse(script.textContent);
          if (data['@type'] === 'Product' || data['@graph']) {
            items.push(data);
          }
        } catch {}
      }
      
      // Try to find product tiles in the DOM
      // Look for common MD product tile patterns
      const tiles = document.querySelectorAll('[class*="product"], [data-product], article, li[class*="grid"]');
      
      // Extract structured product data from the page's state (e.g., window.__INITIAL_STATE__)
      for (const script of document.querySelectorAll('script')) {
        const text = script.textContent || '';
        if (text.includes('__INITIAL_STATE__') || text.includes('window.__DATA__')) {
          items.push({ type: 'state', data: text.slice(0, 50000) });
        }
      }
      
      return items;
    });
    
    console.log(`  Products found: ${products.length}`);
    allProducts.push(...products);
    
    // If we found a state variable, try to extract products from it
    for (const p of products) {
      if (p.type === 'state') {
        const stateMatch = p.data.match(/window\.__INITIAL_STATE__\s*=\s*(\{.+?\});/s);
        if (stateMatch) {
          try {
            const state = JSON.parse(stateMatch[1]);
            console.log(`  State parsed, keys: ${Object.keys(state).join(', ')}`);
          } catch {}
        }
        
        // Also try to find product arrays in the state
        const jsonMatch = p.data.match(/products\s*:\s*(\[.+?\])\s*,\s*\w+\s*:/s);
        if (jsonMatch) {
          try {
            const prods = JSON.parse(jsonMatch[1]);
            console.log(`  Products in state: ${prods.length}`);
          } catch {}
        }
      }
    }
    
  } catch (err) {
    console.log(`  Error: ${err.message}`);
  }
}

console.log(`\n\nTotal product records captured: ${allProducts.length}`);

if (apiData.length > 0) {
  console.log(`\nAPI responses captured: ${apiData.length}`);
  for (const r of apiData.slice(0, 5)) {
    console.log(`\n${r.url}`);
    console.log(r.body.slice(0, 500));
  }
  fs.writeFileSync('/tmp/md-api-data.json', JSON.stringify(apiData, null, 2));
  console.log('\nAPI data saved to /tmp/md-api-data.json');
}

await browser.close();
