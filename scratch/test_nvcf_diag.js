
// Direct test of NVCF asset upload API with the actual NVIDIA key from app_config
const SUPABASE_URL = 'https://fyqpifmrsftsfqibhwhy.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ5cXBpZm1yc2Z0c2ZxaWJod2h5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQzMDYyOTgsImV4cCI6MjA3OTg4MjI5OH0.dydnFn3lqub7qMo9uFfn5yUyY4Wr_eQPnsbvHWHwMTk';

async function testNVCFDiagnostic() {
    console.log('Testing NVCF via edge function diagnostic mode...');
    const res = await fetch(`${SUPABASE_URL}/functions/v1/mannequin-tryon`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'apikey': SUPABASE_ANON_KEY,
            'Authorization': `Bearer ${SUPABASE_ANON_KEY}`
        },
        body: JSON.stringify({ diagnostic: true })
    });
    const data = await res.json();
    console.log('Auth check:', JSON.stringify(data));
}

async function testNVCFAsset() {
    console.log('\nTesting Step 1 (TOP) directly to see PATH 1 logs...');
    const shirtUrl = 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg';
    const res = await fetch(`${SUPABASE_URL}/functions/v1/mannequin-tryon`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'apikey': SUPABASE_ANON_KEY,
            'Authorization': `Bearer ${SUPABASE_ANON_KEY}`
        },
        body: JSON.stringify({
            mannequin_image: shirtUrl,
            garment: { image: shirtUrl, label: 'top', type: 'upper_body' },
            step: 1,
            total: 1, // single step so no collecting, forces PATH 1 or PATH 2 final
            outfit_so_far: '{}'
        })
    });
    const data = await res.json();
    console.log('Mode:', data.mode);
    console.log('Success:', data.success);
    console.log('Error:', data.error ?? '(none)');
    if (data.resultUrl) {
        console.log('Result length:', data.resultUrl.length);
    }
}

testNVCFDiagnostic().then(() => testNVCFAsset());
