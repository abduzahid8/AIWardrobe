/**
 * Test scraping MD GB site with Playwright
 */
import { chromium } from 'playwright';

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({
  userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
});

const page = await context.newPage();
page.setDefaultTimeout(30000);

const url = 'https://www.massimodutti.com/gb/men/shirts-n1447';
console.log(`Navigating to: ${url}`);

try {
  await page.goto(url, { waitUntil: 'networkidle', timeout: 30000 });
  const title = await page.title();
  console.log(`Title: ${title}`);
  
  // Wait for content to load
  await page.waitForTimeout(3000);
  
  // Check page content
  const bodyText = await page.textContent('body');
  console.log(`Body length: ${bodyText.length}`);
  console.log(`First 500 chars:\n${bodyText.slice(0, 500)}`);
  
  // Look for product data in script tags
  const scripts = await page.$$eval('script', els => els.map(e => ({
    type: e.type,
    src: e.src,
    innerLength: (e.innerHTML || '').length,
    id: e.id,
  })));
  
  const dataScripts = scripts.filter(s => s.innerLength > 1000 && !s.src);
  console.log(`\nScripts with >1k inline content: ${dataScripts.length}`);
  for (const s of dataScripts.slice(0, 5)) {
    console.log(`  type=${s.type}, id=${s.id}, ${s.innerLength} chars`);
  }
  
  // Look for product tiles
  const links = await page.$$('a[href*="product"]');
  console.log(`\nProduct links: ${links.length}`);
  
  // Also check for JSON-LD structured data
  const jsonld = await page.$$('script[type="application/ld+json"]');
  console.log(`JSON-LD scripts: ${jsonld.length}`);
  
  // Take screenshot
  await page.screenshot({ path: '/tmp/md-gb-test.png', fullPage: false });
  console.log('\nScreenshot saved to /tmp/md-gb-test.png');
  
} catch (err) {
  console.log(`FAILED: ${err.message}`);
}

await browser.close();
