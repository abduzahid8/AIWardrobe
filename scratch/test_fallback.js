
const SUPABASE_URL = 'https://fyqpifmrsftsfqibhwhy.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ5cXBpZm1yc2Z0c2ZxaWJod2h5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQzMDYyOTgsImV4cCI6MjA3OTg4MjI5OH0.dydnFn3lqub7qMo9uFfn5yUyY4Wr_eQPnsbvHWHwMTk';

async function callEdgeFunction(payload) {
    const response = await fetch(`${SUPABASE_URL}/functions/v1/mannequin-tryon`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'apikey': SUPABASE_ANON_KEY,
            'Authorization': `Bearer ${SUPABASE_ANON_KEY}`
        },
        body: JSON.stringify(payload)
    });
    const data = await response.json();
    return data;
}

async function runTest() {
    console.log('=== Sequential Try-On Fallback Test ===\n');
    
    const shirtUrl   = 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg';
    const chestUrl   = 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/448034/item/usgoods_09_448034_3x4.jpg';
    const mannequinUrl = shirtUrl;  // placeholder bare mannequin

    // Step 1 — Top
    console.log('--- Step 1/2: Top ---');
    const step1 = await callEdgeFunction({
        mannequin_image: mannequinUrl,
        garment: { image: shirtUrl, label: 'top', type: 'upper_body' },
        step: 1,
        total: 2,
        outfit_so_far: '',
    });
    
    console.log('Mode:', step1.mode ?? '(none)');
    console.log('Success:', step1.success);
    console.log('outfit_so_far returned:', step1.outfit_so_far ?? '(none)');
    if (step1.success) {
        console.log('✅ Step 1 PASSED — got resultUrl (' + ((step1.resultUrl || '').length) + ' chars)');
    } else {
        console.log('❌ Step 1 FAILED:', step1.error);
        return;
    }

    // Step 2 — Layer
    console.log('\n--- Step 2/2: Layer ---');
    const step2 = await callEdgeFunction({
        mannequin_image: step1.resultUrl,
        garment: { image: chestUrl, label: 'layer', type: 'upper_body' },
        step: 2,
        total: 2,
        outfit_so_far: step1.outfit_so_far ?? '',
    });

    console.log('Mode:', step2.mode ?? '(none)');
    console.log('Success:', step2.success);
    console.log('outfit_so_far returned:', step2.outfit_so_far ?? '(none)');
    if (step2.success) {
        console.log('✅ Step 2 PASSED — got resultUrl (' + ((step2.resultUrl || '').length) + ' chars)');
    } else {
        console.log('❌ Step 2 FAILED:', step2.error);
    }

    console.log('\n=== All steps complete ===');
}

runTest();
