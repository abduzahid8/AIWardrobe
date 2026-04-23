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
        brand: 'Classic',
        name: 'Oversized Blazer',
        price: 129.00,
        currency: 'USD',
        category: 'tops',
        garment_type: 'upper_body',
        description: 'Relaxed oversized blazer with structured shoulders',
        sort_order: 10,
        imageUrl: 'https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=600&q=80',
    },
    {
        id: 'shop-inspo-2',
        brand: 'Classic',
        name: 'Wide Leg Trousers',
        price: 89.90,
        currency: 'USD',
        category: 'bottoms',
        garment_type: 'lower_body',
        description: 'Wide-leg trousers with a relaxed silhouette',
        sort_order: 20,
        imageUrl: 'https://images.unsplash.com/photo-1624378439575-d8705ad7ae80?w=600&q=80',
    },
    {
        id: 'shop-inspo-3',
        brand: 'Classic',
        name: 'Structured Jacket',
        price: 69.90,
        currency: 'USD',
        category: 'tops',
        garment_type: 'upper_body',
        description: 'Structured cropped jacket with button closure',
        sort_order: 30,
        imageUrl: 'https://images.unsplash.com/photo-1551028919-ac66c5f85b4f?w=600&q=80',
    },
    {
        id: 'shop-inspo-4',
        brand: 'Classic',
        name: 'Slim Fit Jeans',
        price: 15.90,
        currency: 'USD',
        category: 'bottoms',
        garment_type: 'lower_body',
        description: 'Classic slim-fit jeans in mid-wash denim',
        sort_order: 40,
        imageUrl: 'https://images.unsplash.com/photo-1542272604-787c3835535d?w=600&q=80',
    },
    {
        id: 'shop-inspo-5',
        brand: 'Classic',
        name: 'Ribbed Knit Top',
        price: 35.90,
        currency: 'USD',
        category: 'tops',
        garment_type: 'upper_body',
        description: 'Fine ribbed knit top with crew neck',
        sort_order: 50,
        imageUrl: 'https://images.unsplash.com/photo-1618354691373-d851c5c3a990?w=600&q=80',
    },
    {
        id: 'shop-inspo-6',
        brand: 'Classic',
        name: 'Leather Ankle Boots',
        price: 99.90,
        currency: 'USD',
        category: 'shoes',
        garment_type: 'shoes',
        description: 'Leather ankle boots with block heel',
        sort_order: 60,
        imageUrl: 'https://images.unsplash.com/photo-1449505278894-297fdb3edbc1?w=600&q=80',
    },
    {
        id: 'shop-inspo-7',
        brand: 'Classic',
        name: 'Satin Mini Dress',
        price: 59.90,
        currency: 'USD',
        category: 'dresses',
        garment_type: 'dresses',
        description: 'Satin mini dress with thin shoulder straps',
        sort_order: 70,
        imageUrl: 'https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=600&q=80',
    },
    {
        id: 'shop-inspo-8',
        brand: 'Classic',
        name: 'Brown Pants',
        price: 79.90,
        currency: 'USD',
        category: 'bottoms',
        garment_type: 'lower_body',
        description: 'Tailored brown trousers with straight leg',
        sort_order: 80,
        imageUrl: 'https://images.unsplash.com/photo-1473966968600-fa801b869a1a?w=600&q=80',
    },
    {
        id: 'shop-inspo-9',
        brand: 'Classic',
        name: 'Brown Loafers',
        price: 89.90,
        currency: 'USD',
        category: 'shoes',
        garment_type: 'shoes',
        description: 'Classic brown leather loafers',
        sort_order: 90,
        imageUrl: 'https://images.unsplash.com/photo-1638247025967-b4e38f787b76?w=600&q=80',
    },
    {
        id: 'shop-inspo-10',
        brand: 'Classic',
        name: 'Grey Loafers',
        price: 95.90,
        currency: 'USD',
        category: 'shoes',
        garment_type: 'shoes',
        description: 'Premium grey leather loafers',
        sort_order: 100,
        imageUrl: 'https://images.unsplash.com/photo-1560769629-975ec94e6a86?w=600&q=80',
    },
    {
        id: 'shop-inspo-11',
        brand: 'Classic',
        name: 'High Waist Trousers',
        price: 69.90,
        currency: 'USD',
        category: 'bottoms',
        garment_type: 'lower_body',
        description: 'High-waist white trousers with pleated front',
        sort_order: 110,
        imageUrl: 'https://images.unsplash.com/photo-1506629082955-511b1aa562c8?w=600&q=80',
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
        
        const { imageUrl, ...rest } = item;
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
