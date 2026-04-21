#!/usr/bin/env node
/**
 * scripts/seed-shop-catalog.js
 *
 * Uploads local shop images to Supabase Storage and seeds the shop_catalog table.
 *
 * Usage:
 *   SUPABASE_URL=https://xxx.supabase.co \
 *   SUPABASE_SERVICE_KEY=eyJ... \
 *   node scripts/seed-shop-catalog.js
 *
 * Requirements:
 *   npm install @supabase/supabase-js  (already in package.json)
 *
 * Run this ONCE after applying supabase/migrations/005_shop_catalog.sql.
 * Safe to re-run — uses upsert so existing rows are updated, not duplicated.
 */

const fs   = require('fs');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');

const SUPABASE_URL         = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY;

if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
    console.error('❌  Set SUPABASE_URL and SUPABASE_SERVICE_KEY env vars first.');
    process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);
const BUCKET   = 'shop-catalog';
const ROOT     = path.join(__dirname, '..');

// ── Catalog definition ──────────────────────────────────────────────────────
// Each entry mirrors a row in the shop_catalog table.
// localImage is resolved relative to the project root.
const CATALOG = [
    {
        id: 'shop-inspo-1',
        brand: 'ZARA',
        name: 'Oversized Blazer',
        price: 129.00,
        currency: 'USD',
        category: 'tops',
        garment_type: 'upper_body',
        description: 'Relaxed oversized blazer with structured shoulders',
        sort_order: 10,
        localImage: 'pictures/shop/image copy.png',
    },
    {
        id: 'shop-inspo-2',
        brand: 'ZARA',
        name: 'Wide Leg Trousers',
        price: 89.90,
        currency: 'USD',
        category: 'bottoms',
        garment_type: 'lower_body',
        description: 'Wide-leg trousers with a relaxed silhouette',
        sort_order: 20,
        localImage: 'pictures/shop/image copy 2.png',
    },
    {
        id: 'shop-inspo-3',
        brand: 'ZARA',
        name: 'Structured Jacket',
        price: 69.90,
        currency: 'USD',
        category: 'tops',
        garment_type: 'upper_body',
        description: 'Structured cropped jacket with button closure',
        sort_order: 30,
        localImage: 'pictures/shop/image copy 3.png',
    },
    {
        id: 'shop-inspo-4',
        brand: 'ZARA',
        name: 'Slim Fit Jeans',
        price: 15.90,
        currency: 'USD',
        category: 'bottoms',
        garment_type: 'lower_body',
        description: 'Classic slim-fit jeans in mid-wash denim',
        sort_order: 40,
        localImage: 'pictures/shop/image copy 4.png',
    },
    {
        id: 'shop-inspo-5',
        brand: 'ZARA',
        name: 'Ribbed Knit Top',
        price: 35.90,
        currency: 'USD',
        category: 'tops',
        garment_type: 'upper_body',
        description: 'Fine ribbed knit top with crew neck',
        sort_order: 50,
        localImage: 'pictures/shop/image copy 5.png',
    },
    {
        id: 'shop-inspo-6',
        brand: 'ZARA',
        name: 'Leather Ankle Boots',
        price: 99.90,
        currency: 'USD',
        category: 'shoes',
        garment_type: 'shoes',
        description: 'Leather ankle boots with block heel',
        sort_order: 60,
        localImage: 'pictures/shop/image copy 6.png',
    },
    {
        id: 'shop-inspo-7',
        brand: 'ZARA',
        name: 'Satin Mini Dress',
        price: 59.90,
        currency: 'USD',
        category: 'dresses',
        garment_type: 'dresses',
        description: 'Satin mini dress with thin shoulder straps',
        sort_order: 70,
        localImage: 'pictures/shop/image.png',
    },
    {
        id: 'shop-inspo-8',
        brand: 'ZARA',
        name: 'Brown Pants',
        price: 79.90,
        currency: 'USD',
        category: 'bottoms',
        garment_type: 'lower_body',
        description: 'Tailored brown trousers with straight leg',
        sort_order: 80,
        localImage: 'pictures/shop/Brown-pants-with_line.png',
    },
    {
        id: 'shop-inspo-9',
        brand: 'ZARA',
        name: 'Brown Loafers',
        price: 89.90,
        currency: 'USD',
        category: 'shoes',
        garment_type: 'shoes',
        description: 'Classic brown leather loafers',
        sort_order: 90,
        localImage: 'pictures/shop/Brown_loafers.png.png',
    },
    {
        id: 'shop-inspo-10',
        brand: 'ZARA',
        name: 'Grey Loafers',
        price: 95.90,
        currency: 'USD',
        category: 'shoes',
        garment_type: 'shoes',
        description: 'Premium grey leather loafers',
        sort_order: 100,
        localImage: 'pictures/shop/Grey_loafers_loropiana.png',
    },
    {
        id: 'shop-inspo-11',
        brand: 'ZARA',
        name: 'High Waist Trousers',
        price: 69.90,
        currency: 'USD',
        category: 'bottoms',
        garment_type: 'lower_body',
        description: 'High-waist white trousers with pleated front',
        sort_order: 110,
        localImage: 'pictures/shop/highweist_trousers_whte.png',
    },
];

// ── Helpers ────────────────────────────────────────────────────────────────

function mimeType(filePath) {
    const ext = path.extname(filePath).toLowerCase();
    if (ext === '.jpg' || ext === '.jpeg') return 'image/jpeg';
    if (ext === '.webp') return 'image/webp';
    return 'image/png';
}

async function uploadImage(item) {
    const localPath = path.join(ROOT, item.localImage);
    if (!fs.existsSync(localPath)) {
        console.warn(`  ⚠  Image not found, skipping upload: ${item.localImage}`);
        return null;
    }

    const fileBuffer  = fs.readFileSync(localPath);
    const storagePath = `items/${item.id}${path.extname(item.localImage)}`;

    const { error } = await supabase.storage
        .from(BUCKET)
        .upload(storagePath, fileBuffer, {
            contentType: mimeType(item.localImage),
            upsert: true,
        });

    if (error) {
        console.error(`  ✗  Upload failed for ${item.id}:`, error.message);
        return null;
    }

    const { data: urlData } = supabase.storage.from(BUCKET).getPublicUrl(storagePath);
    return urlData.publicUrl;
}

// ── Main ───────────────────────────────────────────────────────────────────

async function main() {
    console.log(`\n🚀  Seeding shop_catalog → ${SUPABASE_URL}\n`);

    const rows = [];

    for (const item of CATALOG) {
        process.stdout.write(`  ↑  ${item.id}  ${item.name} … `);

        const imageUrl = await uploadImage(item);
        if (!imageUrl) {
            console.log('skipped (no image).');
            continue;
        }

        const { localImage, ...rest } = item;
        rows.push({ ...rest, image_url: imageUrl });

        console.log('done.');
    }

    if (rows.length === 0) {
        console.log('\n⚠  No rows to insert.');
        return;
    }

    const { error } = await supabase
        .from('shop_catalog')
        .upsert(rows, { onConflict: 'id' });

    if (error) {
        console.error('\n❌  DB upsert failed:', error.message);
        process.exit(1);
    }

    console.log(`\n✅  Seeded ${rows.length} items successfully.\n`);
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
