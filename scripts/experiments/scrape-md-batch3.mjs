/**
 * Batch 3: Scrape MD men's products with better gender filtering.
 * Uses single browser with incognito contexts for speed.
 */
import puppeteer from 'puppeteer-extra';
import StealthPlugin from 'puppeteer-extra-plugin-stealth';
import fs from 'fs';

puppeteer.use(StealthPlugin());

const sitemapXml = fs.readFileSync('/tmp/md-gb-sitemap.xml', 'utf-8');
const allUrls = [...sitemapXml.matchAll(/<loc>([^<]+)<\/loc>/g)].map(m => m[1]);

// Exclusion keywords that strongly indicate women's products
const womenExclude = [
  'dress', 'skirt', 'blouse', 'bra', 'bikini', 'thong',
  'high-waist', 'palazzo', 'pintuck', 'peplum', 'ruche', 'ruffle',
  'flounce', 'mermaid', 'asymmetric', 'lace', 'sheer', 'mesh',
  'corset', 'babydoll', 'camisole', 'bralet',
  'stiletto', 'pumps', 'wedge', 'kitten-heel',
  'tote-bag', 'clutch', 'crossbody',
  'necklace', 'earring', 'bracelet', 'choker',
  'bodysuit', 'jumpsuit', 'playsuit', 'romper',
  'nightdress', 'nightie', 'nightgown', 'robe',
  'hipster', 'makeup', 'cosmetic', 'perfume', 'eau-de',
  'hair', 'wig', 'extensions',
  'silk-scarf', 'headband', 'hair-claw',
  'watch', 'ring', 'cufflink',
  'bride', 'wedding', 'bridal',
];

// Men's inclusion keywords - must match at least one
const menInclude = [
  'shirt', 'trouser', 'jean', 'polo', 'blazer', 'jacket', 
  'coat', 'shoe', 'loafer', 'trainer', 'sneaker', 'belt', 
  'wallet', 'sweater', 'jumper', 'cardigan', 't-shirt', 
  'vest', 'short', 'bermuda', 'swim', 'trunk',
  'boot', 'derby', 'espadrille', 'sandal', 'backpack', 'bag',
  'hat', 'cap', 'beanie', 'sunglass', 'tie', 'pant', 
  'knit', 'pique', 'suit', 'linen', 'cotton', 'merino',
  'wool', 'cable-knit', 'crew-neck', 'v-neck',
  'overshirt', 'parka', 'bomber', 'denim', 'chino',
  'belt-bag', 'bumbag', 'backpack',
];

const menUrls = allUrls.filter(url => {
  const slug = url.toLowerCase();
  // First check exclusion keywords
  for (const kw of womenExclude) {
    if (slug.includes(kw)) return false;
  }
  // Then check inclusion keywords
  for (const kw of menInclude) {
    if (slug.includes(kw)) return true;
  }
  return false;
});

// Deduplicate by product ID
const seen = new Set();
const uniqueMen = [];
for (const url of menUrls) {
  const id = url.match(/l(\d+)/)?.[1];
  if (!seen.has(id)) {
    seen.add(id);
    uniqueMen.push(url);
  }
}

console.log(`Total URLs: ${allUrls.length}`);
console.log(`Filtered men: ${uniqueMen.length}`);

// Exclude already-cataloged products
const catalogContent = fs.readFileSync('./data/shopCatalogItems.ts', 'utf-8');
const existingIds = new Set();
const idMatches = catalogContent.matchAll(/https:\/\/static\.massimodutti\.net\/assets\/public\/[0-9a-f\/]+\/(\d+)-o1\/\1-o1\.jpg/g);
for (const m of idMatches) existingIds.add(m[1]);

const prevResults = JSON.parse(fs.readFileSync('/tmp/md-scraped-products.json', 'utf-8'));
const alreadyScraped = new Set(prevResults.map(r => r.productId));

const newUrls = uniqueMen.filter(url => {
  const slugId = url.match(/l(\d+)/)?.[1];
  for (const eid of existingIds) if (eid.startsWith(slugId)) return false;
  for (const sid of alreadyScraped) if (sid.startsWith(slugId)) return false;
  return true;
});

console.log(`New to scrape: ${newUrls.length}`);
newUrls.slice(0, 20).forEach((u, i) => console.log(`  ${i+1}. ${u.split('/').pop()}`));

// Use single browser for efficiency
const browser = await puppeteer.launch({
  headless: true,
  args: ['--no-sandbox', '--disable-setuid-sandbox'],
});

const results = [...prevResults];
const maxProducts = Math.min(30, newUrls.length);

for (let i = 0; i < maxProducts; i++) {
  const url = newUrls[i];
  const slug = url.split('/').pop();
  
  console.log(`\n[${i+1}/${maxProducts}] ${slug}`);
  
  // New incognito context for each product
    const context = await browser.createBrowserContext();
  
  try {
    const page = await context.newPage();
    await page.setUserAgent('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36');
    await page.setViewport({ 
      width: 1280 + Math.floor(Math.random() * 200), 
      height: 900 + Math.floor(Math.random() * 100) 
    });
    page.setDefaultTimeout(45000);
    
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 45000 });
    await new Promise(r => setTimeout(r, 2500));
    
    const title = await page.title();
    console.log(`  Title: "${title}"`);
    
    if (!title || title.includes('Access Denied') || title.includes('Just a moment')) {
      console.log('  ⚠ Blocked');
      await context.close();
      continue;
    }
    
    // Check if this is men's or women's by looking at the category in the title
    const catMatch = title.match(/·\s*(\w[\w\s&]+?)\s*\|/);
    const category = catMatch ? catMatch[1].trim() : '';
    
    // Women's clothing categories that slipped through
    const womenCats = ['Dresses', 'Skirts', 'Blouses', 'Lingerie', 'Nightwear'];
    if (womenCats.some(c => category.includes(c))) {
      console.log(`  ⚠ Women's category: ${category}`);
      await context.close();
      continue;
    }
    
    const data = await page.evaluate(() => {
      const result = {};
      const cdnUrls = [];
      document.querySelectorAll('img').forEach(img => {
        const src = img.getAttribute('src') || img.getAttribute('data-src') || '';
        if (src.includes('static.massimodutti.net')) cdnUrls.push(src);
      });
      const ogImage = document.querySelector('meta[property="og:image"]');
      if (ogImage) cdnUrls.push(ogImage.getAttribute('content'));
      result.cdnUrls = [...new Set(cdnUrls)];
      const priceEl = document.querySelector('[class*="price"], [class*="Price"], [data-price]');
      result.price = priceEl ? priceEl.textContent.trim() : '';
      const descEl = document.querySelector('meta[name="description"]');
      result.description = descEl ? descEl.getAttribute('content') : '';
      return result;
    });
    
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
      console.log(`  ✅ ${productId} (${category || 'unknown'})`);
      
      let verified = false;
      try {
        const resp = await fetch(mainUrl, {
          headers: { 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36' }
        });
        verified = resp.status === 200;
        console.log(`  ${resp.status} ${verified ? '✅' : '❌'}`);
      } catch {}
      
      results.push({
        sitemapUrl: url, slug, productId, cdnHash,
        cdnUrl: mainUrl, title, price: data.price || '',
        description: data.description || '',
        category,
        allCdnUrls: data.cdnUrls, verified,
      });
    } else {
      console.log(`  ⚠ No CDN data. URLs: ${data.cdnUrls.length}`);
      data.cdnUrls.slice(0, 3).forEach(u => console.log(`    ${u}`));
    }
    
    await context.close();
    
    const delay = 2000 + Math.random() * 3000;
    await new Promise(r => setTimeout(r, delay));
    
  } catch (err) {
    console.log(`  ❌ ${err.message.slice(0, 100)}`);
    try { await context.close(); } catch {}
  }
  
  fs.writeFileSync('/tmp/md-scraped-products.json', JSON.stringify(results, null, 2));
}

await browser.close();

console.log(`\n\n=== DONE ===`);
console.log(`Total scraped: ${results.length}`);
const verified = results.filter(r => r.verified).length;
console.log(`Verified: ${verified}/${results.length}`);
fs.writeFileSync('/tmp/md-scraped-products.json', JSON.stringify(results, null, 2));
