/**
 * Use puppeteer-extra stealth to access MD product pages from the sitemap.
 * Extract CDN image URLs from the rendered page.
 */
import puppeteer from 'puppeteer-extra';
import StealthPlugin from 'puppeteer-extra-plugin-stealth';
import fs from 'fs';

puppeteer.use(StealthPlugin());

// Parse the sitemap to get men's product URLs
const sitemapXml = fs.readFileSync('/tmp/md-gb-sitemap.xml', 'utf-8');
const allUrls = [...sitemapXml.matchAll(/<loc>([^<]+)<\/loc>/g)].map(m => m[1]);

// Filter for men's products (exclude women's keywords)
const womenKeywords = ['dress', 'skirt', 'blouse', 'bra', 'heel', 'bikini', 'thong', 'top', 
  'handbag', 'earring', 'necklace', 'tie', 'scarf', 'jumpsuit', 'bodysuit',
  'women', 'woman', 'female', 'bride', 'makeup', 'lingerie', 'nightie',
  'gown', 'stiletto', 'pumps', 'flat-sandal', 'high-heel', 'wedge'];

const menKeywords = ['shirt', 'trouser', 'jean', 'polo', 'blazer', 'suit', 'jacket', 
  'coat', 'shoe', 'trainer', 'loafer', 'sneaker', 'belt', 'wallet', 'sweater',
  'jumper', 'cardigan', 't-shirt', 'vest', 'short', 'swim', 'trunk',
  'boot', 'derby', 'espadrille', 'sandal', 'backpack', 'bag', 'briefcase',
  'hat', 'cap', 'beanie', 'sunglass', 'tie', 'men'];

const menUrls = allUrls.filter(url => {
  const slug = url.toLowerCase();
  // Exclude women's keywords
  for (const kw of womenKeywords) {
    if (slug.includes(kw) && !slug.includes('swim')) return false;
  }
  // Must have at least one men's keyword
  for (const kw of menKeywords) {
    if (slug.includes(kw)) return true;
  }
  return false;
});

console.log(`Total URLs in sitemap: ${allUrls.length}`);
console.log(`Filtered men's URLs: ${menUrls.length}`);

// Pick a sample of men's products to scrape
// Focus on categories the user wants: linen shirts, linen pants, polos
const wantedKeywords = ['linen-trouser', 'linen-pants', 'linen-shirt', 'linen-polo', 'cotton-polo', 
  'polo-shirt', 'linen-short', 'bermuda', 'linen-blazer'];

const wantedUrls = menUrls.filter(url => {
  const slug = url.toLowerCase();
  return wantedKeywords.some(kw => slug.includes(kw));
});

console.log(`\nWanted products: ${wantedUrls.length}`);
wantedUrls.slice(0, 30).forEach(url => {
  console.log(`  ${url.split('/').pop()}`);
});

// Now try to access product pages with puppeteer stealth
const browser = await puppeteer.launch({
  headless: true,
  args: ['--no-sandbox', '--disable-setuid-sandbox'],
});

const page = await browser.newPage();
await page.setUserAgent('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36');
await page.setViewport({ width: 1280, height: 900 });
page.setDefaultTimeout(30000);

const results = [];

for (const url of wantedUrls.slice(0, 20)) {
  const slug = url.split('/').pop();
  console.log(`\n=== ${slug} ===`);
  
  try {
    await page.goto(url, { waitUntil: 'networkidle0', timeout: 30000 });
    
    const title = await page.title();
    console.log(`  Title: "${title}"`);
    
    if (title.includes('Access Denied') || !title) {
      console.log('  Blocked');
      continue;
    }
    
    await new Promise(r => setTimeout(r, 2000));
    
    // Extract CDN images
    const cdnUrls = await page.evaluate(() => {
      const results = [];
      const imgs = document.querySelectorAll('img');
      imgs.forEach(img => {
        const src = img.getAttribute('src') || img.getAttribute('data-src') || '';
        if (src.includes('static.massimodutti.net')) {
          results.push(src);
        }
      });
      
      // Also check background images and meta tags
      const metas = document.querySelectorAll('meta[property="og:image"]');
      metas.forEach(m => results.push(m.getAttribute('content')));
      
      // Check inline JSON data
      const scripts = document.querySelectorAll('script');
      scripts.forEach(s => {
        const text = s.textContent || '';
        const matches = text.match(/static\\.massimodutti\\.net[^"'"']+/g);
        if (matches) results.push(...matches);
      });
      
      return results;
    });
    
    if (cdnUrls.length > 0) {
      console.log(`  Found ${cdnUrls.length} CDN URLs`);
      cdnUrls.slice(0, 5).forEach(u => console.log(`    ${u}`));
      results.push({ url, cdnUrls });
    } else {
      console.log('  No CDN URLs found');
      
      // Save screenshot and HTML for debugging
      await page.screenshot({ path: `/tmp/md-sitemap-${slug.slice(0, 40)}.png` });
      const html = await page.content();
      const cdnMatches = html.match(/static\\.massimodutti\\.net[^"'"']+/g) || [];
      console.log(`  HTML CDN matches: ${cdnMatches.length}`);
      cdnMatches.slice(0, 3).forEach(u => console.log(`    ${u}`));
    }
    
  } catch (err) {
    console.log(`  Error: ${err.message.slice(0, 100)}`);
  }
}

console.log(`\n\n=== RESULTS ===`);
console.log(`Products scraped: ${results.length}`);
for (const r of results) {
  console.log(`${r.url.split('/').pop()}: ${r.cdnUrls.length} images`);
}

fs.writeFileSync('/tmp/md-sitemap-results.json', JSON.stringify(results, null, 2));
await browser.close();
