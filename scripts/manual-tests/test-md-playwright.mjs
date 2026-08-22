/**
 * Quick test: can Playwright access Massimo Dutti?
 */
import { chromium } from 'playwright';

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
page.setDefaultTimeout(30000);

// Try men's landing page
const urls = [
  'https://www.massimodutti.com/us/en/man/landing.html',
  'https://www.massimodutti.com/us/en/man/shirts-catalog',
  'https://www.massimodutti.com/us/en/man/trousers-catalog',
];

for (const url of urls) {
  try {
    console.log(`\nTrying: ${url}`);
    await page.goto(url, { waitUntil: 'networkidle', timeout: 30000 });
    const title = await page.title();
    console.log(`  Title: ${title}`);
    const bodyText = await page.textContent('body');
    console.log(`  Body length: ${bodyText.length}`);
    
    // Check for product tiles
    const tiles = await page.$$('[class*="product"], [class*="tile"], [class*="item"], a[href*="product"]');
    console.log(`  Product-like elements: ${tiles.length}`);
    
    // Take screenshot
    await page.screenshot({ path: `/tmp/md-test-${urls.indexOf(url)}.png`, fullPage: false });
    console.log('  Screenshot saved');
  } catch (err) {
    console.log(`  FAILED: ${err.message.slice(0, 100)}`);
  }
}

await browser.close();
