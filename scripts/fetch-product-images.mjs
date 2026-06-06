#!/usr/bin/env node
/**
 * Fetches product images from Zara and Massimo Dutti for Summer 2026 men's collection.
 * Uses Jina reader API to bypass bot protection.
 * 
 * Usage: node scripts/fetch-product-images.mjs
 */
import https from 'https';
import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ASSETS = path.join(__dirname, '..', 'assets', 'images');

// Category → garmentType mapping
const GARMENT_TYPES = {
  'shirt': 'upper_body',
  't-shirt': 'upper_body',
  'polo': 'upper_body',
  'blazer': 'upper_body',
  'jacket': 'upper_body',
  'knitwear': 'upper_body',
  'jumper': 'upper_body',
  'trouser': 'lower_body',
  'pants': 'lower_body',
  'jeans': 'lower_body',
  'short': 'lower_body',
  'shoe': 'shoes',
  'sneaker': 'shoes',
  'loafer': 'shoes',
};

function inferGarmentType(name) {
  const n = name.toLowerCase();
  const types = [
    ['trouser', 'lower_body'], ['pants', 'lower_body'], ['jeans', 'lower_body'],
    ['short', 'lower_body'], ['sneaker', 'shoes'], ['loafer', 'shoes'],
    ['shoe', 'shoes'], ['boot', 'shoes'], ['blazer', 'upper_body'],
    ['jacket', 'upper_body'], ['coat', 'upper_body'], ['knit', 'upper_body'],
    ['jumper', 'upper_body'], ['sweater', 'upper_body'], ['hoodie', 'upper_body'],
    ['sweatshirt', 'upper_body'], ['shirt', 'upper_body'], ['polo', 'upper_body'],
    ['t-shirt', 'upper_body'], ['tee', 'upper_body'], ['tie', 'accessory'],
    ['belt', 'accessory'], ['bag', 'accessory'],
  ];
  for (const [kw, gt] of types) {
    if (n.includes(kw)) return gt;
  }
  return 'upper_body';
}

function jinaFetch(url) {
  return new Promise((resolve, reject) => {
    const jinaUrl = `https://r.jina.ai/${url}`;
    https.get(jinaUrl, {
      headers: { 'User-Agent': 'Mozilla/5.0' },
      timeout: 20000,
    }, (res) => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => resolve(data));
    }).on('error', reject);
  });
}

function downloadImage(url, destPath) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(destPath);
    const proto = url.startsWith('https') ? https : http;
    proto.get(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Referer': 'https://www.zara.com/',
        'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
      },
      timeout: 10000,
    }, (response) => {
      if (response.statusCode !== 200) {
        file.close();
        fs.unlinkSync(destPath);
        reject(new Error(`HTTP ${response.statusCode}`));
        return;
      }
      response.pipe(file);
      file.on('finish', () => { file.close(); resolve(destPath); });
    }).on('error', (err) => {
      file.close();
      if (fs.existsSync(destPath)) fs.unlinkSync(destPath);
      reject(err);
    });
  });
}

async function processZaraProduct(sku, nameOverride) {
  const url = `https://www.zara.com/us/en/${sku}.html`;
  console.log(`\n📦 ZARA: ${sku}`);
  
  try {
    const html = await jinaFetch(url);
    
    // Extract product name
    let name = '';
    const nameMatch = html.match(/# ([^\n]+?)(?:\s*[|-]|\n|$)/);
    if (nameMatch) name = nameMatch[1].trim().replace(/ - ZARA.*$/, '');
    if (!name) name = nameOverride || sku;
    
    // Extract main product image (-p suffix)
    const imgMatch = html.match(/https:\/\/static\.zara\.net\/assets\/public\/[^"'\s) ]+-p\/[^"'\s) ]+\.jpg/);
    const imageUrl = imgMatch ? imgMatch[0] : '';
    
    if (!imageUrl) {
      console.log(`  ✗ No image found for ${sku}`);
      return null;
    }
    
    console.log(`  Name: ${name}`);
    console.log(`  Image: ${imageUrl.substring(0, 80)}...`);
    
    // Download image
    const filename = `zara-${sku}-p.jpg`;
    const destPath = path.join(ASSETS, filename);
    try {
      await downloadImage(imageUrl, destPath);
      const stats = fs.statSync(destPath);
      console.log(`  ✓ Downloaded: ${filename} (${(stats.size / 1024).toFixed(1)}KB)`);
    } catch (err) {
      console.log(`  ✗ Download failed: ${err.message}`);
      return null;
    }
    
    return { sku, name, imageUrl, filename };
  } catch (err) {
    console.log(`  ✗ Error: ${err.message}`);
    return null;
  }
}

async function main() {
  console.log('🚀 Fetching ZARA Men Summer 2026 product images\n');
  
  const products = [
    // ── Linen Shirts & Polos ─────────────────────────────────────
    { id: '100-linen-polo-shirt-p02634252', name: '100% Linen Polo Shirt' },
    { id: '100-linen-regular-fit-shirt-p03090110', name: '100% Linen Regular Fit Shirt' },
    { id: '100-linen-relaxed-fit-shirt-p05070904', name: '100% Linen Relaxed Fit Shirt' },
    { id: 'regular-fit-100-linen-shirt-p01063410', name: 'Regular Fit 100% Linen Shirt' },
    { id: 'linen---cotton-shirt-p01063412', name: 'Linen & Cotton Shirt' },
    { id: 'relaxed-fit-100-linen-shirt-with-pleated-pockets-p01195264', name: 'Relaxed Fit Linen Shirt with Pockets' },
    { id: 'cotton-linen-blend-polo-shirt-p01820325', name: 'Cotton Linen Blend Polo Shirt' },
    { id: '100-linen-polo-shirt-p02634253', name: '100% Linen Polo Shirt' },
    { id: 'regular-fit-100-linen-shirt-p01957102', name: 'Regular Fit 100% Linen Shirt' },
    { id: '100-linen-relaxed-fit-shirt-p04120220', name: '100% Linen Relaxed Fit Shirt' },
    { id: 'textured-linen---cotton-shirt-p07545180', name: 'Textured Linen & Cotton Shirt' },
    
    // ── Polos & Knitwear ────────────────────────────────────────
    { id: 'knit-cotton-linen-blend-polo-p03920678', name: 'Knit Cotton Linen Blend Polo' },
    { id: 'knit-textured-polo-shirt-p03332410', name: 'Knit Textured Polo Shirt' },
    { id: 'cotton-silk-knit-polo-shirt-p05755434', name: 'Cotton Silk Knit Polo Shirt' },
    { id: 'regular-fit-linen-knit-polo-p02893441', name: 'Regular Fit Linen Knit Polo' },
    { id: 'hemp-cotton-knit-polo-p02142432', name: 'Hemp Cotton Knit Polo' },
    { id: 'knit-cotton-polo-shirt-p09598431', name: 'Knit Cotton Polo Shirt' },
    { id: 'linen---cotton-cardigan-p06674449', name: 'Linen & Cotton Cardigan' },
    { id: 'zip-up-cardigan-p09598423', name: 'Zip-Up Cardigan' },
    { id: 'regular-fit-cotton-linen-jumper-p09598467', name: 'Regular Fit Cotton Linen Jumper' },
    
    // ── Blazers ─────────────────────────────────────────────────
    { id: '100-linen-double-breasted-blazer-p04632333', name: '100% Linen Double Breasted Blazer' },
    { id: '100-linen-double-breasted-suit-blazer-p04286333', name: '100% Linen Suit Blazer' },
    
    // ── Trousers ────────────────────────────────────────────────
    { id: '100-linen-relaxed-fit-pants-p02634254', name: '100% Linen Relaxed Fit Pants' },
    { id: '100-linen-relaxed-fit-pants-p05070902', name: '100% Linen Relaxed Fit Pants' },
    { id: 'linen-cotton-blend-pleated-suit-pants-p04553594', name: 'Linen Cotton Pleated Suit Pants' },
    { id: 'relaxed-fit-cotton---linen-pants-p04470460', name: 'Relaxed Fit Cotton Linen Pants' },
    { id: 'relaxed-fit-pleated-pants-p00706922', name: 'Relaxed Fit Pleated Pants' },
    { id: 'baggy-fit-jeans-p04048407', name: 'Baggy Fit Jeans' },
    
    // ── Shorts ──────────────────────────────────────────────────
    { id: '100-linen-relaxed-fit-shorts-p05070903', name: '100% Linen Relaxed Fit Shorts' },
    { id: 'regular-fit-100-linen-cargo-shorts-p01063441', name: 'Regular Fit Linen Cargo Shorts' },
    { id: 'baggy-fit-jorts-p00541441', name: 'Baggy Fit Jorts' },
    
    // ── Shoes ───────────────────────────────────────────────────
    { id: 'casual-leather-loafers-p12613720', name: 'Casual Leather Loafers' },
    { id: 'leather-penny-loafers-p12617720', name: 'Leather Penny Loafers' },
    { id: 'leather-tassel-loafers-p12632720', name: 'Leather Tassel Loafers' },
    { id: 'leather-penny-espadrilles-p12626720', name: 'Leather Penny Espadrilles' },
    { id: 'dress-penny-loafers-p12628720', name: 'Dress Penny Loafers' },
    { id: 'chunky-sole-sneakers-p12393820', name: 'Chunky Sole Sneakers' },
    { id: 'retro-style-sneakers-p12225720', name: 'Retro Style Sneakers' },
    { id: 'barefoot-leather-sneaker-p12242720', name: 'Barefoot Leather Sneaker' },
    { id: 'monochrome-chunky-sneakers-p12215520', name: 'Monochrome Chunky Sneakers' },
    { id: 'retro-metallic-sneakers-p12202820', name: 'Retro Metallic Sneakers' },
    
    // ── Jackets ─────────────────────────────────────────────────
    { id: '100-linen-zip-jacket-p00706848', name: '100% Linen Zip Jacket' },
    { id: 'faux-leather-bomber-jacket-p03918412', name: 'Faux Leather Bomber Jacket' },
    { id: 'technical-bomber-jacket-p04302460', name: 'Technical Bomber Jacket' },
    { id: 'relaxed-fit-denim-jacket-p03991488', name: 'Relaxed Fit Denim Jacket' },
    { id: 'relaxed-fit-washed-cotton-jacket-p02634205', name: 'Relaxed Fit Washed Cotton Jacket' },
    
    // ── Swimwear ────────────────────────────────────────────────
    { id: 'short-structured-swimsuit-p08574409', name: 'Short Structured Swim Trunks' },
    { id: 'seersucker-swim-trunks-p08574410', name: 'Seersucker Swim Trunks' },
    { id: 'mid-length-textured-swimsuit-p08574457', name: 'Mid-Length Textured Swimsuit' },
    { id: 'long-floral-print-swim-trunks-p00495457', name: 'Long Floral Print Swim Trunks' },
    
    // ── Accessories ─────────────────────────────────────────────
    { id: 'braided-leather-belt-p02823406', name: 'Braided Leather Belt' },
    { id: 'leather-belt-p02823400', name: 'Leather Belt' },
    { id: 'embossed-leather-belt-p05919302', name: 'Embossed Leather Belt' },
    { id: 'washed-cotton-cap-p05875300', name: 'Washed Cotton Cap' },
    { id: 'washed-bucket-hat-p04988409', name: 'Washed Bucket Hat' },
    { id: 'foldable-sports-backpack-25l-p13266620', name: 'Foldable Sports Backpack' },
    { id: 'woven-shoulder-bag-p13630720', name: 'Woven Shoulder Bag' },
    
    // ── Jeans ───────────────────────────────────────────────────
    { id: 'basic-slim-fit-jeans-p00774454', name: 'Basic Slim Fit Jeans' },
  ];
  
  const results = [];
  
  for (const product of products) {
    const result = await processZaraProduct(product.id, product.name);
    if (result) {
      results.push({
        ...result,
        brand: 'ZARA',
        garmentType: inferGarmentType(result.name),
        sourceUrl: `https://www.zara.com/us/en/${product.id}.html`,
      });
    }
    // Small delay between requests
    await new Promise(r => setTimeout(r, 1000));
  }
  
  // Save results
  const outputPath = path.join(__dirname, '..', 'data', 'zara-products-fetched.json');
  fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));
  console.log(`\n✅ Results saved to ${outputPath}`);
  console.log(`   ${results.length} products fetched successfully`);
}

main().catch(console.error);
