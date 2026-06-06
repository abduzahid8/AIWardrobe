import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';

const BASE_URL = 'https://www.uniqlo.com/us/en';

const CATEGORIES = [
  { name: 'Tops', url: '/men/tops' },
  { name: 'T-Shirts', url: '/men/tops/t-shirts' },
  { name: 'Sweatshirts & Hoodies', url: '/men/tops/sweatshirts-and-hoodies' },
  { name: 'Shirts & Polos', url: '/men/shirts-and-polos' },
  { name: 'Polos', url: '/men/shirts-and-polos/polo-shirts' },
  { name: 'Sweaters', url: '/men/sweaters' },
  { name: 'Bottoms', url: '/men/bottoms' },
  { name: 'Jeans', url: '/men/bottoms/jeans' },
  { name: 'Shorts', url: '/men/bottoms/shorts' },
  { name: 'Outerwear', url: '/men/outerwear-and-blazers' },
  { name: 'Accessories', url: '/men/accessories-and-shoes' },
  { name: 'Innerwear', url: '/men/innerwear' },
];

async function scrapePage(page, url) {
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });

  const selectors = [
    'a.product-tile__link[href*="/products/"]',
    'a.fr-ec-product-tile[href*="/products/"]',
  ];

  let found = false;
  for (const sel of selectors) {
    try {
      await page.waitForSelector(sel, { timeout: 15000 });
      found = true;
      break;
    } catch {
      // try next selector
    }
  }

  if (!found) {
    await page.waitForTimeout(5000);
  }

  await page.waitForTimeout(3000);

  const products = await page.evaluate(() => {
    const items = [];
    const seenHrefs = new Set();

    const linkSelectors = [
      'a.product-tile__link[href*="/products/"]',
      'a.fr-ec-product-tile[href*="/products/"]',
    ];

    for (const sel of linkSelectors) {
      const links = document.querySelectorAll(sel);
      for (const link of links) {
        const href = link.href || '';
        if (!href || href.includes('onetrust') || href.includes('cookie')) continue;

        const baseHref = href.split('?')[0];
        if (seenHrefs.has(baseHref)) continue;
        seenHrefs.add(baseHref);

        const isMainLayout = link.classList.contains('product-tile__link');
        const isFrLayout = link.classList.contains('fr-ec-product-tile');

        let mainImgSrc = '';
        let mainImgAlt = '';
        let price = '';
        let name = '';
        let productId = '';

        if (isMainLayout) {
          const tile = link.querySelector('.product-tile');
          if (!tile) continue;

          const imgs = tile.querySelectorAll('img');
          for (const img of imgs) {
            const src = img.src || '';
            if (src.includes('/item/') && src.includes('_3x4')) {
              if (!mainImgSrc) {
                mainImgSrc = src;
                mainImgAlt = img.alt || '';
              }
            }
          }

          if (!mainImgSrc) continue;

          const contentArea = tile.querySelector('.product-tile__content-area');
          if (contentArea) {
            const priceEls = contentArea.querySelectorAll('.typography');
            for (const el of priceEls) {
              const text = el.textContent?.trim() || '';
              if (/^\$[\d,]+\.?\d*/.test(text)) {
                price = text;
                break;
              }
            }
          }

          name = mainImgAlt;

          const m = href.match(/\/products\/E(\d{6})/);
          if (m) productId = m[1];
        } else if (isFrLayout) {
          const imgs = link.querySelectorAll('img');
          for (const img of imgs) {
            const src = img.src || '';
            if (src.includes('imagesgoods') && src.includes('_3x4')) {
              if (!mainImgSrc) {
                mainImgSrc = src;
                mainImgAlt = img.alt || '';
              }
            }
          }

          if (!mainImgSrc) continue;

          const nameEl = link.querySelector('h3[class*="product-tile-horizontal"], h3[class*="product-tile"]');
          if (nameEl) {
            name = nameEl.textContent?.trim() || '';
          }
          if (!name) {
            name = mainImgAlt;
          }

          const priceEl = link.querySelector('.fr-ec-price-text');
          if (priceEl) {
            price = priceEl.textContent?.trim() || '';
          }

          const idAttr = link.getAttribute('id') || '';
          const m = idAttr.match(/E?(\d{6})/);
          if (m) productId = m[1];
          if (!productId) {
            const m2 = href.match(/\/products\/E(\d{6})/);
            if (m2) productId = m2[1];
          }
        }

        if (!productId) {
          const m = mainImgSrc.match(/imagesgoods\/(\d{6})/);
          if (m) productId = m[1];
        }
        if (!productId) continue;

        const cleanImageUrl = mainImgSrc.split('?')[0];
        const cleanProductUrl = href.split('?')[0];

        items.push({
          productId,
          name: name || '',
          price: price || '',
          imageUrl: cleanImageUrl,
          productUrl: cleanProductUrl,
        });
      }
    }

    return items;
  });

  return products;
}

function deduplicate(products) {
  const seen = new Map();
  for (const p of products) {
    if (!p.productId) continue;
    if (!p.name && !p.price) continue;

    const existing = seen.get(p.productId);
    if (!existing) {
      seen.set(p.productId, p);
    } else {
      if (!existing.price && p.price) {
        seen.set(p.productId, p);
      } else if (!existing.name && p.name) {
        seen.set(p.productId, p);
      }
    }
  }
  return Array.from(seen.values());
}

async function main() {
  console.log('Launching browser...');
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    userAgent:
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
  });
  const page = await context.newPage();

  let allProducts = [];

  for (const cat of CATEGORIES) {
    const url = `${BASE_URL}${cat.url}`;
    console.log(`\nScraping ${cat.name} (${url})...`);
    const products = await scrapePage(page, url);
    console.log(`  Found ${products.length} raw products`);
    allProducts.push(...products.map((p) => ({ ...p, category: cat.name })));
  }

  const deduped = deduplicate(allProducts);
  deduped.sort((a, b) => a.productId.localeCompare(b.productId));

  console.log('\n=== SUMMARY ===');
  console.log(`Total raw: ${allProducts.length}`);
  console.log(`Total unique: ${deduped.length}`);

  // Save to JSON
  const outputPath = path.join(process.cwd(), 'data', 'uniqlo-products.json');
  fs.writeFileSync(outputPath, JSON.stringify(deduped, null, 2), 'utf-8');
  console.log(`\nSaved to ${outputPath}`);

  await browser.close();
}

main().catch((err) => {
  console.error('Error:', err);
  process.exit(1);
});
