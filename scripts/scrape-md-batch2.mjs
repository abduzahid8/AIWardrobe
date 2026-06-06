/**
 * Batch 2: Scrape remaining MD products from the GB sitemap.
 */
import puppeteer from 'puppeteer-extra';
import StealthPlugin from 'puppeteer-extra-plugin-stealth';
import fs from 'fs';

puppeteer.use(StealthPlugin());

const sitemapXml = fs.readFileSync('/tmp/md-gb-sitemap.xml', 'utf-8');
const allUrls = [...sitemapXml.matchAll(/<loc>([^<]+)<\/loc>/g)].map(m => m[1]);

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

const seen = new Set();
const uniqueMen = [];
for (const url of menUrls) {
  const id = url.match(/l(\d+)/)?.[1];
  if (!seen.has(id)) {
    seen.add(id);
    uniqueMen.push(url);
  }
}

// Read existing catalog to find already-known products
const catalogContent = fs.readFileSync('./data/shopCatalogItems.ts', 'utf-8');
const existingIds = new Set();
const idMatches = catalogContent.matchAll(/https:\/\/static\.massimodutti\.net\/assets\/public\/[0-9a-f\/]+\/(\d+)-o1\/\1-o1\.jpg/g);
for (const m of idMatches) {
  existingIds.add(m[1]);
}

// Read previously scraped products
const prevResults = JSON.parse(fs.readFileSync('/tmp/md-scraped-products.json', 'utf-8'));
const alreadyScraped = new Set(prevResults.map(r => r.productId));

console.log(`Total men's products in sitemap: ${uniqueMen.length}`);
console.log(`Existing in catalog: ${existingIds.size}`);
console.log(`Already scraped: ${alreadyScraped.size}`);

// Filter: exclude already-known and already-scraped
const newUrls = uniqueMen.filter(url => {
  const slugId = url.match(/l(\d+)/)?.[1];
  for (const existingId of existingIds) {
    if (existingId.startsWith(slugId)) return false;
  }
  for (const scrapedId of alreadyScraped) {
    if (scrapedId.startsWith(slugId)) return false;
  }
  return true;
});

console.log(`New products to scrape: ${newUrls.length}`);

// Focus on summer-appropriate items: linen, cotton, polo, bermuda, t-shirt, loafer, espadrille
const summerPatterns = ['linen', 'cotton', 'polo', 'bermuda', 'short', 't-shirt', 
  'loafer', 'espadrille', 'trainer', 'sneaker', 'sandal',
  'trouser', 'pant', 'jean', 'blazer', 'shirt', 'belt', 'bag'];

const summerUrls = newUrls.filter(url => {
  const slug = url.toLowerCase();
  return summerPatterns.some(p => slug.includes(p));
});

console.log(`Summer-appropriate: ${summerUrls.length}`);
summerUrls.slice(0, 10).forEach(u => console.log(`  ${u.split('/').pop()}`));

// Process products
const results = [...prevResults];
const maxProducts = Math.min(25, summerUrls.length);

for (let i = 0; i < maxProducts; i++) {
  const url = summerUrls[i];
  const slug = url.split('/').pop();
  
  console.log(`\n[${i+1}/${maxProducts}] ${slug}`);
  
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
    await new Promise(r => setTimeout(r, 3000));
    
    const title = await page.title();
    console.log(`  Title: "${title}"`);
    
    if (!title || title.includes('Access Denied') || title.includes('Just a moment')) {
      console.log('  ⚠ Blocked');
      await browser.close();
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
      console.log(`  ✅ ${productId}`);
      
      // Verify
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
        allCdnUrls: data.cdnUrls, verified,
      });
    } else {
      console.log(`  ⚠ No CDN data. URLs: ${data.cdnUrls.length}`);
      data.cdnUrls.slice(0, 3).forEach(u => console.log(`    ${u}`));
    }
    
    await browser.close();
    
    const delay = 3000 + Math.random() * 5000;
    console.log(`  Wait ${Math.round(delay/1000)}s...`);
    await new Promise(r => setTimeout(r, delay));
    
  } catch (err) {
    console.log(`  ❌ ${err.message.slice(0, 100)}`);
    await browser.close();
  }
  
  // Save progress after each product
  fs.writeFileSync('/tmp/md-scraped-products.json', JSON.stringify(results, null, 2));
}

console.log(`\n\n=== DONE ===`);
console.log(`Total: ${results.length}/${prevResults.length + maxProducts}`);
fs.writeFileSync('/tmp/md-scraped-products.json', JSON.stringify(results, null, 2));
