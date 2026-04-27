
const fs = require('fs');
const SUPABASE_URL = 'https://fyqpifmrsftsfqibhwhy.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ5cXBpZm1yc2Z0c2ZxaWJod2h5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQzMDYyOTgsImV4cCI6MjA3OTg4MjI5OH0.dydnFn3lqub7qMo9uFfn5yUyY4Wr_eQPnsbvHWHwMTk';

async function testFullOutfit() {
    console.log('--- Testing Full Outfit Generation (2 Steps) ---');
    
    const mannequinImage = fs.readFileSync('assets/images/mannequin_front.png', { encoding: 'base64' });
    const topImage = 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg'; // White T
    const pantsImage = 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg'; // Just reuse the same for test stability

    let outfitSoFar = '{}';

    // Step 1: Top
    console.log('Step 1: Sending Top...');
    const res1 = await fetch(`${SUPABASE_URL}/functions/v1/mannequin-tryon`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'apikey': SUPABASE_ANON_KEY,
            'Authorization': `Bearer ${SUPABASE_ANON_KEY}`
        },
        body: JSON.stringify({
            mannequin_image: `data:image/png;base64,${mannequinImage}`,
            garment: { image: topImage, label: 'top', type: 'upper_body' },
            step: 1,
            total: 2,
            outfit_so_far: outfitSoFar
        })
    });
    const d1 = await res1.json();
    if (!d1.success) { console.error('Step 1 failed:', d1.error); return; }
    outfitSoFar = d1.outfit_so_far;
    console.log('Step 1 Success. Mode:', d1.mode);

    // Step 2: Pants
    console.log('Step 2: Sending Pants (Final)...');
    const res2 = await fetch(`${SUPABASE_URL}/functions/v1/mannequin-tryon`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'apikey': SUPABASE_ANON_KEY,
            'Authorization': `Bearer ${SUPABASE_ANON_KEY}`
        },
        body: JSON.stringify({
            mannequin_image: `data:image/png;base64,${mannequinImage}`,
            garment: { image: pantsImage, label: 'pants', type: 'lower_body' },
            step: 2,
            total: 2,
            outfit_so_far: outfitSoFar
        })
    });
    const d2 = await res2.json();
    if (!d2.success) { console.error('Step 2 failed:', d2.error); return; }
    
    console.log('Overall Success! Final Mode:', d2.mode);
    const resultUrl = d2.resultUrl;
    
    if (resultUrl.startsWith('data:')) {
        const base64Data = resultUrl.split(',')[1];
        fs.writeFileSync('scratch/overall_outfit.png', base64Data, 'base64');
        console.log('Overall result saved to scratch/overall_outfit.png');
    }
}

testFullOutfit();
