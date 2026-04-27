
const fs = require('fs');
const path = require('path');

const SUPABASE_URL = 'https://fyqpifmrsftsfqibhwhy.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ5cXBpZm1yc2Z0c2ZxaWJod2h5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQzMDYyOTgsImV4cCI6MjA3OTg4MjI5OH0.dydnFn3lqub7qMo9uFfn5yUyY4Wr_eQPnsbvHWHwMTk';

async function callStep(payload) {
    const res = await fetch(`${SUPABASE_URL}/functions/v1/mannequin-tryon`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'apikey': SUPABASE_ANON_KEY, 'Authorization': `Bearer ${SUPABASE_ANON_KEY}` },
        body: JSON.stringify(payload)
    });
    return res.json();
}

async function runTestAndSave() {
    const garments = [
        { label: 'top',   type: 'upper_body', url: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg' },
        { label: 'layer', type: 'upper_body', url: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/448034/item/usgoods_09_448034_3x4.jpg' },
        { label: 'pants', type: 'lower_body', url: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/450251/item/usgoods_35_450251_3x4.jpg' },
        { label: 'shoes', type: 'shoes',      url: 'https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/484330/item/goods_32_484330_3x4.jpg' },
    ];

    let outfitSoFar = '{}';
    let lastResult = null;

    console.log('Running test and saving artifacts...');

    for (let i = 0; i < garments.length; i++) {
        const g = garments[i];
        console.log(`Step ${i+1}/4: ${g.label}`);
        lastResult = await callStep({
            mannequin_image: garments[0].url,
            garment: { image: g.url, label: g.label, type: g.type },
            step: i + 1,
            total: garments.length,
            outfit_so_far: outfitSoFar,
        });
        outfitSoFar = lastResult.outfit_so_far;
    }

    console.log('Final Result Status:', lastResult.success);
    if (lastResult && lastResult.resultUrl) {
        console.log(`✅ Final resultUrl received (length: ${lastResult.resultUrl.length})`);
        const base64Data = lastResult.resultUrl.replace(/^data:image\/\w+;base64,/, "");
        const buffer = Buffer.from(base64Data, 'base64');
        const outputPath = '/Users/zohidvohidjonov/Desktop/AIWardrobe/scratch/test_result.png';
        fs.writeFileSync(outputPath, buffer);
        console.log(`✅ Final image SAVED to: ${outputPath}`);
    } else {
        console.log('❌ No resultUrl in lastResult:', JSON.stringify(lastResult, null, 2));
    }
}

runTestAndSave();
