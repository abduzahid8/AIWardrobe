#!/usr/bin/env node
/**
 * scripts/add-massimo-dutti-2026.js
 *
 * Seeds the 10 new Massimo Dutti 2026 Summer collection menswear items
 * into the public.shop_catalog Supabase table.
 *
 * Usage:
 *   SUPABASE_SERVICE_KEY=your_service_role_key node scripts/add-massimo-dutti-2026.js
 */

const { createClient } = require('@supabase/supabase-js');
const dotenv = require('dotenv');
const path = require('path');

// Load environment variables from root .env
dotenv.config({ path: path.join(__dirname, '../.env') });

const SUPABASE_URL = process.env.EXPO_PUBLIC_SUPABASE_URL || 'https://fyqpifmrsftsfqibhwhy.supabase.co';
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_KEY;

if (!SUPABASE_KEY) {
    console.error('❌ Error: SUPABASE_SERVICE_KEY (service_role key) environment variable is required.');
    console.error('Please run:');
    console.error('  SUPABASE_SERVICE_KEY=your_service_role_key node scripts/add-massimo-dutti-2026.js');
    process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

const ITEMS = [
    {
        id: 'classic-m-shirt-12',
        brand: 'Massimo Dutti',
        name: '100% Linen Stand-Collar Shirt',
        price: 89.90,
        currency: 'USD',
        category: 'tops',
        garment_type: 'upper_body',
        description: 'Summer 2026 - Premium fluid 100% linen shirt with a stand collar and regular fit in sand beige',
        image_url: 'https://massimodutti.com/content/dam/massimodutti/Men/2026/shirts/1200x1500/0142_241_712.jpg',
        is_active: true,
        sort_order: 1000
    },
    {
        id: 'classic-m-shirt-13',
        brand: 'Massimo Dutti',
        name: 'Fluid Linen Regular Fit Shirt',
        price: 79.90,
        currency: 'USD',
        category: 'tops',
        garment_type: 'upper_body',
        description: 'Summer 2026 - Lightweight regular-fit shirt in 100% breathable linen, featuring a classic spread collar in navy',
        image_url: 'https://massimodutti.com/content/dam/massimodutti/Men/2026/shirts/1200x1500/0145_245_401.jpg',
        is_active: true,
        sort_order: 1001
    },
    {
        id: 'classic-m-shirt-14',
        brand: 'Massimo Dutti',
        name: 'Striped Linen Summer Shirt',
        price: 89.90,
        currency: 'USD',
        category: 'tops',
        garment_type: 'upper_body',
        description: 'Summer 2026 - Breathable 100% linen long-sleeve shirt in thin off-white and blue stripe',
        image_url: 'https://massimodutti.com/content/dam/massimodutti/Men/2026/shirts/1200x1500/0148_248_305.jpg',
        is_active: true,
        sort_order: 1002
    },
    {
        id: 'classic-m-blazer-06',
        brand: 'Massimo Dutti',
        name: 'Unstructured Linen Blazer',
        price: 249.00,
        currency: 'USD',
        category: 'tops',
        garment_type: 'upper_body',
        description: 'Summer 2026 - Premium unstructured regular-fit linen blazer with notch lapels and patch pockets in natural stone',
        image_url: 'https://massimodutti.com/content/dam/massimodutti/Men/2026/blazers/1200x1500/2065_335_710.jpg',
        is_active: true,
        sort_order: 1003
    },
    {
        id: 'classic-m-blazer-07',
        brand: 'Massimo Dutti',
        name: 'Fluid Double-Breasted Linen Blazer',
        price: 299.00,
        currency: 'USD',
        category: 'tops',
        garment_type: 'upper_body',
        description: 'Summer 2026 - Soft double-breasted linen blazer with peak lapels and a relaxed, elegant silhouette in sage green',
        image_url: 'https://massimodutti.com/content/dam/massimodutti/Men/2026/blazers/1200x1500/2068_338_502.jpg',
        is_active: true,
        sort_order: 1004
    },
    {
        id: 'classic-m-trouser-11',
        brand: 'Massimo Dutti',
        name: 'Fluid Linen Wide-Leg Trousers',
        price: 129.00,
        currency: 'USD',
        category: 'bottoms',
        garment_type: 'lower_body',
        description: 'Summer 2026 - Relaxed wide-leg trousers in 100% fluid linen with pressed creases in sand beige',
        image_url: 'https://massimodutti.com/content/dam/massimodutti/Men/2026/trousers/1200x1500/1032_422_712.jpg',
        is_active: true,
        sort_order: 1005
    },
    {
        id: 'classic-m-trouser-12',
        brand: 'Massimo Dutti',
        name: 'Slim Fit Linen Chinos',
        price: 99.90,
        currency: 'USD',
        category: 'bottoms',
        garment_type: 'lower_body',
        description: 'Summer 2026 - Classic slim-fit chinos crafted from a lightweight linen-cotton blend in navy blue',
        image_url: 'https://massimodutti.com/content/dam/massimodutti/Men/2026/trousers/1200x1500/1035_425_401.jpg',
        is_active: true,
        sort_order: 1006
    },
    {
        id: 'classic-m-short-03',
        brand: 'Massimo Dutti',
        name: 'Linen Drawstring Shorts',
        price: 79.90,
        currency: 'USD',
        category: 'bottoms',
        garment_type: 'lower_body',
        description: 'Summer 2026 - Relaxed-fit linen shorts with an elasticated drawstring waist in off-white',
        image_url: 'https://massimodutti.com/content/dam/massimodutti/Men/2026/shorts/1200x1500/1040_430_001.jpg',
        is_active: true,
        sort_order: 1007
    },
    {
        id: 'classic-m-shoe-12',
        brand: 'Massimo Dutti',
        name: 'Suede Slip-On Loafer',
        price: 149.00,
        currency: 'USD',
        category: 'shoes',
        garment_type: 'shoes',
        description: 'Summer 2026 - Split suede slip-on penny loafers with ultra-flexible sole and unlined construction in tobacco brown',
        image_url: 'https://massimodutti.com/content/dam/massimodutti/Men/2026/shoes/1200x1500/1275_675_710.jpg',
        is_active: true,
        sort_order: 1008
    },
    {
        id: 'classic-m-shoe-13',
        brand: 'Massimo Dutti',
        name: 'Suede Espadrilles',
        price: 119.00,
        currency: 'USD',
        category: 'shoes',
        garment_type: 'shoes',
        description: 'Summer 2026 - Casual split suede espadrilles with classic braided jute midsole and rubber outsole in sand beige',
        image_url: 'https://massimodutti.com/content/dam/massimodutti/Men/2026/shoes/1200x1500/1278_680_712.jpg',
        is_active: true,
        sort_order: 1009
    }
];

async function run() {
    console.log(`\n🚀 Seeding ${ITEMS.length} Massimo Dutti Summer 2026 items into remote Supabase at ${SUPABASE_URL}...`);
    
    const { data, error } = await supabase
        .from('shop_catalog')
        .upsert(ITEMS, { onConflict: 'id' });

    if (error) {
        console.error('❌ Database insertion failed:', error.message);
        process.exit(1);
    }

    console.log(`\n✅ Success! ${ITEMS.length} items have been successfully upserted into Supabase's shop_catalog table!\n`);
}

run().catch(err => {
    console.error('Unexpected error:', err);
    process.exit(1);
});
