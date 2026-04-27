
const SUPABASE_URL = 'https://fyqpifmrsftsfqibhwhy.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ5cXBpZm1yc2Z0c2ZxaWJod2h5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQzMDYyOTgsImV4cCI6MjA3OTg4MjI5OH0.dydnFn3lqub7qMo9uFfn5yUyY4Wr_eQPnsbvHWHwMTk';

async function callStep(payload) {
    const response = await fetch(`${SUPABASE_URL}/functions/v1/mannequin-tryon`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'apikey': SUPABASE_ANON_KEY,
            'Authorization': `Bearer ${SUPABASE_ANON_KEY}`
        },
        body: JSON.stringify(payload)
    });
    return await response.json();
}

async function runFullTest() {
    console.log('🚀 Starting Full 4-Step Virtual Try-On Sequence Test...\n');
    
    const items = [
        { label: 'top', type: 'upper_body', url: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg' }, // Blue Shirt
        { label: 'layer', type: 'upper_body', url: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/448034/item/usgoods_09_448034_3x4.jpg' }, // Navy Blazer
        { label: 'pants', type: 'lower_body', url: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/450251/item/usgoods_35_450251_3x4.jpg' }, // Brown Chinos
        { label: 'shoes', type: 'shoes', url: 'https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/484330/item/goods_32_484330_3x4.jpg' }  // White Sneakers
    ];

    let currentMannequin = items[0].url; // Start with the first item's image as the "base" or use a blank one if you had it. 
    // Usually it starts with a bare mannequin. I'll use the shirt as a starting point for simplicity in this test script.
    let outfitSoFar = '';

    for (let i = 0; i < items.length; i++) {
        const item = items[i];
        console.log(`\n--- Step ${i + 1}/4: ${item.label.toUpperCase()} ---`);
        console.log(`Processing ${item.label}...`);
        
        const result = await callStep({
            mannequin_image: currentMannequin,
            garment: {
                image: item.url,
                label: item.label,
                type:  item.type
            },
            step: i + 1,
            total: items.length,
            outfit_so_far: outfitSoFar
        });

        if (result.success) {
            console.log(`✅ Success (Mode: ${result.mode})`);
            console.log(`Outfit: ${result.outfit_so_far.slice(0, 100)}...`);
            currentMannequin = result.resultUrl;
            outfitSoFar = result.outfit_so_far;
        } else {
            console.error(`❌ FAILED at step ${i + 1}:`, result.error);
            return;
        }
    }

    console.log('\n✨ FULL SEQUENCE COMPLETED SUCCESSFULLY! ✨');
    console.log('Final Image Result String Length:', currentMannequin.length);
}

runFullTest();
