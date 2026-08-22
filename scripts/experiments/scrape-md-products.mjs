/**
 * Scrape Massimo Dutti product pages using puppeteer-extra stealth.
 * Creates a fresh browser context per product to avoid Akamai rate limiting.
 * Extracts CDN image URLs, product names, and prices.
 */
import puppeteer from 'puppeteer-extra';
import StealthPlugin from 'puppeteer-extra-plugin-stealth';
import fs from 'fs';

puppeteer.use(StealthPlugin());

// Parse the sitemap to get men's product URLs
const sitemapXml = fs.readFileSync('/tmp/md-gb-sitemap.xml', 'utf-8');
const allUrls = [...sitemapXml.matchAll(/<loc>([^<]+)<\/loc>/g)].map(m => m[1]);

// Filter for men's products
const womenKeywords = ['dress', 'skirt', 'blouse', 'bra', 'bikini', 'thong', 
  'handbag', 'earring', 'necklace', 'jumpsuit', 'bodysuit',
  'women', 'woman', 'female', 'lingerie', 'gown', 'stiletto', 
  'pumps', 'flat-sandal', 'high-heel', 'wedge', 'romper',
  'camisole', 'corset', 'babydoll', 'nightdress'];

const menKeywords = ['shirt', 'trouser', 'jean', 'polo', 'blazer', 'suit', 'jacket', 
  'coat', 'shoe', 'trainer', 'loafer', 'sneaker', 'belt', 'wallet', 'sweater',
  'jumper', 'cardigan', 't-shirt', 'vest', 'short', 'swim', 'trunk',
  'boot', 'derby', 'espadrille', 'sandal', 'backpack', 'bag', 'briefcase',
  'hat', 'cap', 'beanie', 'sunglass', 'tie', 'pant', 'men'];

const menUrls = allUrls.filter(url => {
  const slug = url.toLowerCase();
  for (const kw of womenKeywords) {
    if (slug.includes(kw)) return false;
  }
  for (const kw of menKeywords) {
    if (slug.includes(kw)) return true;
  }
  return false;
});

console.log(`Total: ${allUrls.length}, Men: ${menUrls.length}`);

// Save all men's URLs for reference
fs.writeFileSync('/tmp/md-men-urls.json', JSON.stringify(menUrls, null, 2));

// Pick a manageable set focusing on linen, polo, shorts we don't already have
const wantedPatterns = [
  'linen-trouser', 'linen-pants', 'linen-shirt', 'linen-polo',
  'linen-short', 'bermuda', 'linen-blazer', 'linen-suit',
  'cotton-polo', 'polo-shirt', 'knit-polo',
];

const wantedUrls = menUrls.filter(url => {
  const slug = url.toLowerCase();
  // Exclude URLs we already have (checking by product ID prefix)
  const id = slug.match(/l(\d+)/)?.[1];
  return wantedPatterns.some(kw => slug.includes(kw));
});

// Remove duplicates (same product in EN and ES)
const seen = new Set();
const uniqueWanted = [];
for (const url of wantedUrls) {
  const id = url.match(/l(\d+)/)?.[1];
  if (!seen.has(id)) {
    seen.add(id);
    uniqueWanted.push(url);
  }
}

// Read existing catalog to avoid duplicating products
const catalogContent = fs.readFileSync('./data/shopCatalogItems.ts', 'utf-8');
const existingIds = new Set();
const idMatches = catalogContent.matchAll(/https:\/\/static\.massimodutti\.net\/assets\/public\/[0-9a-f\/]+\/(\d+)-o1\/\1-o1\.jpg/g);
for (const m of idMatches) {
  existingIds.add(m[1]);
}
console.log(`Existing MD products in catalog: ${existingIds.size}`);

// Filter out already-known products (match by first 8 digits of product ID)
const newWanted = uniqueWanted.filter(url => {
  const slugId = url.match(/l(\d+)/)?.[1];
  // Check if any existing product ID starts with this slug ID
  for (const existingId of existingIds) {
    if (existingId.startsWith(slugId)) return false;
  }
  return true;
});

console.log(`\nNew products to scrape: ${newWanted.length} (skipping ${uniqueWanted.length - newWanted.length} existing)`);
newWanted.forEach((url, i) => {
  const slug = url.split('/').pop();
  const id = url.match(/l(\d+)/)?.[1];
  console.log(`  ${i+1}. ${slug} (ID: l${id})`);
});

// Process products
const results = [];
const maxProducts = 20; // Limit for this run

for (let i = 0; i < Math.min(maxProducts, newWanted.length); i++) {
  const url = newWanted[i];
  const slug = url.split('/').pop();
  
  console.log(`\n[${i+1}/${Math.min(maxProducts, uniqueWanted.length)}] ${slug}`);
  
  // Create a fresh browser instance for each product
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });
  
  try {
    const page = await browser.newPage();
    await page.setUserAgent('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36');
    await page.setViewport({ 
      width: 1280 + Math.floor(Math.random() * 200), 
      height: 900 + Math.floor(Math.random() * 100) 
    });
    page.setDefaultTimeout(60000);
    
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
    
    // Wait a bit for images to load
    await new Promise(r => setTimeout(r, 3000));
    
    const title = await page.title();
    console.log(`  Title: "${title}"`);
    
    if (!title || title.includes('Access Denied') || title.includes('Just a moment')) {
      console.log('  ⚠ Blocked by Akamai');
      await browser.close();
      continue;
    }
    
    // Extract product data from page
    const data = await page.evaluate(() => {
      const result = {};
      
      // CDN images
      const cdnUrls = [];
      document.querySelectorAll('img').forEach(img => {
        const src = img.getAttribute('src') || img.getAttribute('data-src') || '';
        if (src.includes('static.massimodutti.net')) {
          cdnUrls.push(src);
        }
      });
      
      // Meta tags
      const ogImage = document.querySelector('meta[property="og:image"]');
      if (ogImage) cdnUrls.push(ogImage.getAttribute('content'));
      
      result.cdnUrls = [...new Set(cdnUrls)];
      
      // Price
      const priceEl = document.querySelector('[class*="price"], [class*="Price"], [data-price]');
      result.price = priceEl ? priceEl.textContent.trim() : '';
      
      // Description
      const descEl = document.querySelector('meta[name="description"]');
      result.description = descEl ? descEl.getAttribute('content') : '';
      
      // JSON-LD
      const jsonld = document.querySelector('script[type="application/ld+json"]');
      if (jsonld) {
        try { result.jsonld = JSON.parse(jsonld.textContent); } catch {}
      }
      
      return result;
    });
    
    // Extract CDN hash and product ID from the first good CDN URL
    let cdnHash = '';
    let productId = '';
    
    for (const cdnUrl of data.cdnUrls) {
      const match = cdnUrl.match(/static\.massimodutti\.net\/assets\/public\/([0-9a-f]{4})\/([0-9a-f]{4})\/([0-9a-f]{12})\/([0-9a-f]{12})\/(\d+)-o1\/\5-o1/);
      if (match) {
        productId = match[5];
        cdnHash = match[1] + match[2] + match[3] + match[4];
        break;
      }
    }
    
    if (cdnHash && productId) {
      const mainUrl = `https://static.massimodutti.net/assets/public/${cdnHash.slice(0,4)}/${cdnHash.slice(4,8)}/${cdnHash.slice(8,20)}/${cdnHash.slice(20,32)}/${productId}-o1/${productId}-o1.jpg`;
      console.log(`  ✅ Product ID: ${productId}`);
      console.log(`  ✅ CDN URL: ${mainUrl}`);
      console.log(`  Price: ${data.price}`);
      
      results.push({
        sitemapUrl: url,
        slug,
        productId,
        cdnHash,
        cdnUrl: mainUrl,
        title,
        price: data.price || 'N/A',
        description: data.description || '',
        allCdnUrls: data.cdnUrls,
      });
    } else {
      console.log(`  ⚠ Could not extract CDN data`);
      console.log(`  CDN URLs found: ${data.cdnUrls.length}`);
      data.cdnUrls.slice(0, 5).forEach(u => console.log(`    ${u}`));
    }
    
    await browser.close();
    
    // Add random delay between scrapes (3-8 seconds)
    const delay = 3000 + Math.random() * 5000;
    console.log(`  Waiting ${Math.round(delay/1000)}s...`);
    await new Promise(r => setTimeout(r, delay));
    
  } catch (err) {
    console.log(`  ❌ Error: ${err.message.slice(0, 120)}`);
    await browser.close();
  }
}

// Save results
console.log(`\n\n=== RESULTS ===`);
console.log(`Successful: ${results.length}/${Math.min(maxProducts, newWanted.length)}`);

// Verify CDN URLs
let verified = 0;
for (const r of results) {
  try {
    const resp = await fetch(r.cdnUrl, {
      headers: { 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36' }
    });
    r.verified = resp.status === 200;
    if (r.verified) verified++;
    console.log(`  ${r.productId}: ${resp.status} ${r.verified ? '✅' : '❌'}`);
  } catch {
    r.verified = false;
    console.log(`  ${r.productId}: fetch error ❌`);
  }
}

console.log(`\nVerified OK: ${verified}/${results.length}`);

fs.writeFileSync('/tmp/md-scraped-products.json', JSON.stringify(results, null, 2));
console.log(`\nSaved to /tmp/md-scraped-products.json`);
