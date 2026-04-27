
// Test the fixed "collect then generate once" fallback with the EXACT garments from the user's screenshot
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

async function runTest() {
    // Mirroring the user's 4 chosen garments:
    // 1. White oversized T-shirt
    // 2. Black relaxed comfort jacket (zip-up)
    // 3. Cream/off-white linen regular fit pants
    // 4. Dark brown leather loafers
    const garments = [
        { label: 'top',   type: 'upper_body', url: 'https://static.zara.net/assets/public/0a35/e48e/56654285b0b6/a54a1d5ea78b/01854800250-p/01854800250-p.jpg' },
        { label: 'layer', type: 'upper_body', url: 'https://static.zara.net/assets/public/7b2e/6bae/7a6747c58c41/0a0f8491c8bd/07792301800-p/07792301800-p.jpg' },
        { label: 'pants', type: 'lower_body', url: 'https://static.zara.net/assets/public/3dfb/f0b6/b4b44cbbb41c/aec80c72fc5e/09039051712-p/09039051712-p.jpg' },
        { label: 'shoes', type: 'shoes',      url: 'https://static.zara.net/assets/public/53c8/4a0c/21e0421889d0/86b64d29d5d6/12100110700-p/12100110700-p.jpg' },
    ];

    // Fallback to Uniqlo items if Zara CDN blocks
    const fallbackGarments = [
        { label: 'top',   type: 'upper_body', url: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg' },
        { label: 'layer', type: 'upper_body', url: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/448034/item/usgoods_09_448034_3x4.jpg' },
        { label: 'pants', type: 'lower_body', url: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/450251/item/usgoods_35_450251_3x4.jpg' },
        { label: 'shoes', type: 'shoes',      url: 'https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/484330/item/goods_32_484330_3x4.jpg' },
    ];

    const slots = fallbackGarments; // Using reliable Uniqlo CDN
    const total = slots.length;
    let currentMannequin = slots[0].url; // placeholder - in real app this is the mannequin base
    let outfitSoFar = '{}';

    console.log('🚀 Testing fixed 4-step pipeline: collect descriptions → generate ONCE at end\n');

    for (let i = 0; i < slots.length; i++) {
        const g = slots[i];
        const isLast = i === slots.length - 1;
        const expectedMode = isLast ? 'flux-dev-final' : 'collecting';
        
        console.log(`\n▶ Step ${i + 1}/${total}: ${g.label.toUpperCase()} [expected mode: ${expectedMode}]`);
        
        const start = Date.now();
        const result = await callStep({
            mannequin_image: currentMannequin,
            garment: { image: g.url, label: g.label, type: g.type },
            step: i + 1,
            total,
            outfit_so_far: outfitSoFar,
        });
        const elapsed = ((Date.now() - start) / 1000).toFixed(1);

        if (!result.success) {
            console.error(`❌ FAILED: ${result.error}`);
            return;
        }

        const modeOk = result.mode === expectedMode;
        console.log(`${modeOk ? '✅' : '⚠️'} Mode: ${result.mode} (${elapsed}s)`);
        
        if (result.outfit_so_far) {
            const outfit = JSON.parse(result.outfit_so_far);
            console.log(`   Collected so far:`, Object.keys(outfit).join(', '));
        }

        if (result.mode !== 'collecting') {
            currentMannequin = result.resultUrl;
            console.log(`   Image size: ${(result.resultUrl.length / 1024).toFixed(0)} KB (base64)`);
        } else {
            console.log(`   (No image generated yet — same mannequin passed forward)`);
        }
        
        outfitSoFar = result.outfit_so_far || outfitSoFar;
    }

    console.log('\n✨ TEST COMPLETE');
    console.log('Final image length:', currentMannequin.length, 'chars');
    console.log('Steps 1-3 should be "collecting", Step 4 should be "flux-dev-final"');
}

runTest();
