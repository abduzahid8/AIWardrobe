/**
 * Scrape Massimo Dutti products from Lyst using Playwright
 */
import { chromium } from 'playwright';

const browser = await chromium.launch({ headless: true });

const categories = [
  { name: 'linen-shirts', url: 'https://www.lyst.com/shop/mens-massimo-dutti-shirts/?colour=White,Blue,Beige,Natural,Grey&material=Linen' },
  { name: 'linen-trousers', url: 'https://www.lyst.com/shop/mens-massimo-dutti-trousers/?material=Linen' },
  { name: 'polos', url: 'https://www.lyst.com/shop/mens-massimo-dutti-polo-shirts/' },
];

const allProducts = [];

for (const cat of categories) {
  console.log(`\n=== ${cat.name} ===`);
  console.log(`URL: ${cat.url}`);
  
  const context = await browser.newContext({
    userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    viewport: { width: 1440, height: 900 },
  });
  const page = await context.newPage();
  
  // Capture API responses
  const apiResponses = [];
  page.on('response', async response => {
    const ct = response.headers()['content-type'] || '';
    const url = response.url();
    if ((ct.includes('json') || url.includes('/api/')) && response.status() === 200) {
      try {
        const text = await response.text();
        if (text.length > 200 && text.length < 500000) {
          apiResponses.push({ url: url.slice(0, 200), body: text.slice(0, 3000) });
        }
      } catch {}
    }
  });
  
  try {
    await page.goto(cat.url, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForTimeout(3000);
    
    const title = await page.title();
    console.log(`Title: ${title}`);
    
    // Get page content
    const bodyText = await page.evaluate(() => document.body.innerText.slice(0, 2000));
    console.log(`Body (first 2000 chars):\n${bodyText}`);
    
    // Try to extract products from rendered DOM
    const products = await page.evaluate(() => {
      const results = [];
      
      // Try common product selectors
      const selectors = [
        'a[class*="product"]',
        'div[class*="product"]',
        '[data-testid*="product"]',
        'article',
        'li[class*="product"]',
        'div[class*="Grid"] > div',
        '[class*="tile"]',
      ];
      
      for (const sel of selectors) {
        const els = document.querySelectorAll(sel);
        if (els.length > 0) {
          results.push({ selector: sel, count: els.length });
          
          els.forEach(el => {
            const img = el.querySelector('img');
            const links = el.querySelectorAll('a');
            const text = el.textContent.trim().slice(0, 200);
            const imgSrc = img ? (img.getAttribute('src') || img.getAttribute('data-src') || '') : '';
            
            if (text && (imgSrc || links.length > 0)) {
              results.push({
                text: text.slice(0, 100),
                imgSrc: imgSrc.slice(0, 150),
                href: links[0] ? links[0].getAttribute('href')?.slice(0, 150) : '',
              });
            }
          });
          break; // Use first matching selector
        }
      }
      
      // Check all img tags
      results.push({ selector: 'all_imgs', count: document.querySelectorAll('img').length });
      
      // Check for JSON-LD
      const scripts = document.querySelectorAll('script[type="application/ld+json"]');
      scripts.forEach(s => {
        try {
          const data = JSON.parse(s.textContent);
          results.push({ jsonld: JSON.stringify(data).slice(0, 1000) });
        } catch {}
      });
      
      // Check window.__INITIAL_STATE__ or similar
      const html = document.documentElement.innerHTML;
      const matches = html.match(/window\.__[A-Z_]+__\s*=\s*({[^;]+})/g);
      if (matches) {
        results.push({ initialState: matches.slice(0, 2).map(m => m.slice(0, 200)) });
      }
      
      return results;
    });
    
    console.log(`\nExtracted data:`);
    products.forEach((p, i) => {
      if (i < 30) console.log(`  ${JSON.stringify(p)}`);
    });
    
    allProducts.push({ category: cat.name, apiResponses, products });
    
  } catch (err) {
    console.log(`Error: ${err.message}`);
  }
  
  await context.close();
}

console.log(`\n\n=== API RESPONSES ===`);
for (const cat of allProducts) {
  console.log(`\n${cat.category}: ${cat.apiResponses.length} API responses`);
  cat.apiResponses.slice(0, 5).forEach(r => {
    console.log(`  ${r.url}`);
    console.log(`  ${r.body.slice(0, 300)}`);
  });
}

await browser.close();
