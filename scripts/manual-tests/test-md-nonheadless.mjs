/**
 * Try MD with non-headless Playwright to capture API calls
 */
import { chromium } from 'playwright';

const browser = await chromium.launch({
  headless: false, // visible browser
  args: ['--disable-blink-features=AutomationControlled'],
});

const context = await browser.newContext({
  userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
  locale: 'en-GB',
});

const page = await context.newPage();

// Monitor all API/XHR requests
page.on('request', request => {
  const url = request.url();
  if (url.includes('massimodutti') && !url.includes('static.') && !url.includes('.css') && !url.includes('.js')) {
    console.log('REQ:', request.method(), url.slice(0, 200));
  }
});

page.on('response', response => {
  const url = response.url();
  if (url.includes('massimodutti') && !url.includes('static.') && !url.includes('.css') && !url.includes('.js')) {
    console.log('RES:', response.status(), url.slice(0, 200));
    if (response.status() === 200 && url.includes('/api/')) {
      response.text().then(t => {
        console.log('  BODY:', t.slice(0, 300));
      }).catch(() => {});
    }
  }
});

try {
  await page.goto('https://www.massimodutti.com/gb/men/shirts-n1447', {
    waitUntil: 'domcontentloaded',
    timeout: 30000,
  });
  
  // Wait a bit for the interstitial challenge
  await page.waitForTimeout(10000);
  
  const title = await page.title();
  console.log('\nTitle:', title);
  const bodyText = await page.textContent('body');
  console.log('Body length:', bodyText.length);
  console.log('Body preview:', bodyText.slice(0, 300));
  
} catch (err) {
  console.log('ERROR:', err.message);
}

await browser.close();
