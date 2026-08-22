/**
 * Scrape MD product pages using Playwright (more stable than puppeteer).
 * Uses a single browser with fresh contexts per product.
 */
import { chromium } from 'playwright';
import fs from 'fs';

const sitemapXml = fs.readFileSync('/tmp/md-gb-sitemap.xml', 'utf-8');
const allUrls = [...sitemapXml.matchAll(/<loc>([^<]+)<\/loc>/g)].map(m => m[1]);

// Better gender filtering - check against known men's categories
const catSitemapXml = fs.readFileSync('/tmp/md-gb-cat.xml', 'utf-8');
const menCatPatterns = [...catSitemapXml.matchAll(/<loc>https:\/\/www\.massimodutti\.com\/gb\/men\/([^<]+)<\/loc>/g)].map(m => m[1]);
console.log(`Men's categories: ${menCatPatterns.length}`);
menCatPatterns.slice(0, 5).forEach(c => console.log(`  /gb/men/${c}`));

// Use men's category keywords to filter products
const menCatKeywords = new Set();
for (const cat of menCatPatterns) {
  // Extract keywords from the category path: "trousers/jeans/wide-leg-n4827"
  const parts = cat.replace(/-n\d+$/, '').split('/');
  parts.forEach(p => menCatKeywords.add(p));
}

console.log(`Men's category keywords: ${[...menCatKeywords].length}`);
console.log([...menCatKeywords].slice(0, 20).join(', '));

// Build inclusion list from men's category keywords + common terms
const inclusionKeywords = new Set([...menCatKeywords, 
  'shirt', 'trouser', 'jean', 'polo', 'blazer', 'jacket', 
  'coat', 'shoe', 'loafer', 'trainer', 'sneaker', 'belt', 
  'wallet', 'sweater', 'jumper', 'cardigan', 't-shirt', 
  'vest', 'short', 'bermuda', 'swim', 'trunk', 'pant',
  'boot', 'derby', 'espadrille', 'sandal', 'backpack', 'bag',
  'hat', 'cap', 'beanie', 'sunglass', 'tie', 
  'knit', 'pique', 'suit', 'linen', 'cotton', 'merino',
  'wool', 'cable-knit', 'crew-neck', 'v-neck',
  'overshirt', 'parka', 'bomber', 'denim', 'chino',
  'jogger', 'slim-fit', 'regular-fit', 'tapered',
  'sweatshirt', 'hoodie', 'gilet', 'waistcoat',
  'backpack', 'holdall', 'duffle',
]);

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

const menUrls = allUrls.filter(url => {
  const slug = url.toLowerCase();
  for (const kw of womenExclude) if (slug.includes(kw)) return false;
  for (const kw of inclusionKeywords) if (slug.includes(kw)) return true;
  return false;
});

// Deduplicate
const seen = new Set();
const uniqueMen = [];
for (const url of menUrls) {
  const id = url.match(/l(\d+)/)?.[1];
  if (!seen.has(id)) { seen.add(id); uniqueMen.push(url); }
}

console.log(`\nMen's products from sitemap: ${uniqueMen.length}`);

// Exclude already-cataloged and already-scraped
const catalogContent = fs.readFileSync('./data/shopCatalogItems.ts', 'utf-8');
const existingIds = new Set();
const idMatches = catalogContent.matchAll(/https:\/\/static\.massimodutti\.net\/assets\/public\/[0-9a-f\/]+\/(\d+)-o1\/\1-o1\.jpg/g);
for (const m of idMatches) existingIds.add(m[1]);

let prevResults = [];
try { prevResults = JSON.parse(fs.readFileSync('/tmp/md-scraped-products.json', 'utf-8')); } catch {}
const alreadyScraped = new Set(prevResults.map(r => r.productId));

const newUrls = uniqueMen.filter(url => {
  const slugId = url.match(/l(\d+)/)?.[1];
  for (const eid of existingIds) if (eid.startsWith(slugId)) return false;
  for (const sid of alreadyScraped) if (sid.startsWith(slugId)) return false;
  return true;
});

console.log(`New products to scrape: ${newUrls.length}`);

// Focus on summer items first
const summerPriority = ['linen', 'cotton', 'polo', 'bermuda', 'short', 't-shirt',
  'loafer', 'espadrille', 'trainer', 'sandal', 'belt', 'shirt', 'trouser', 'pant'];

const priorityUrls = newUrls.filter(u => summerPriority.some(p => u.toLowerCase().includes(p)));
const remainingUrls = newUrls.filter(u => !summerPriority.some(p => u.toLowerCase().includes(p)));

// Sort: priority first, then rest
const orderedUrls = [...priorityUrls, ...remainingUrls];

console.log(`Priority: ${priorityUrls.length}, Remaining: ${remainingUrls.length}`);

const browser = await chromium.launch({ headless: true });
const results = [...prevResults];
const maxProducts = Math.min(30, orderedUrls.length);

for (let i = 0; i < maxProducts; i++) {
  const url = orderedUrls[i];
  const slug = url.split('/').pop();
  
  console.log(`\n[${i+1}/${maxProducts}] ${slug}`);
  
  const context = await browser.newContext({
    userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    viewport: { width: 1280 + Math.floor(Math.random() * 200), height: 900 + Math.floor(Math.random() * 100) },
  });
  
  try {
    const page = await context.newPage();
    page.setDefaultTimeout(45000);
    
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 45000 });
    await page.waitForTimeout(2000);
    
    const title = await page.title();
    console.log(`  Title: "${title}"`);
    
    if (!title || title.includes('Access Denied') || title.includes('Just a moment')) {
      console.log('  ⚠ Blocked');
      await context.close();
      continue;
    }
    
    // Extract category from title
    const catMatch = title.match(/·\s*([^·]+?)\s*\|/);
    const category = catMatch ? catMatch[1].trim() : '';
    
    // Check for women's categories
    if (/Dresses|Skirts|Blouses|Lingerie|Nightwear/i.test(category)) {
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
      console.log(`  ✅ ${productId} (${category})`);
      
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
        category, allCdnUrls: data.cdnUrls, verified,
      });
    } else {
      console.log(`  ⚠ No CDN data. URLs: ${data.cdnUrls.length}`);
      data.cdnUrls.slice(0, 3).forEach(u => console.log(`    ${u}`));
    }
    
    await context.close();
    await new Promise(r => setTimeout(r, 2000 + Math.random() * 3000));
    
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
