
const fs = require('fs');
const SUPABASE_URL = 'https://fyqpifmrsftsfqibhwhy.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ5cXBpZm1yc2Z0c2ZxaWJod2h5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQzMDYyOTgsImV4cCI6MjA3OTg4MjI5OH0.dydnFn3lqub7qMo9uFfn5yUyY4Wr_eQPnsbvHWHwMTk';

async function testTryOn() {
    console.log('Testing mannequin-tryon edge function...');
    
    const mannequinImage = fs.readFileSync('assets/images/mannequin_front.png', { encoding: 'base64' });
    const garmentImage = 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg'; // White T-shirt
    
    const payload = {
        mannequin_image: `data:image/png;base64,${mannequinImage}`,
        garment: {
            image: garmentImage,
            label: 'top',
            type: 'upper_body'
        },
        step: 1,
        total: 1,
        outfit_so_far: '{}'
    };

    try {
        const res = await fetch(`${SUPABASE_URL}/functions/v1/mannequin-tryon`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'apikey': SUPABASE_ANON_KEY,
                'Authorization': `Bearer ${SUPABASE_ANON_KEY}`
            },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (!data.success) {
            console.error('API Error:', data.error);
            return;
        }

        console.log('API Success! Mode:', data.mode);
        const resultUrl = data.resultUrl;
        
        if (resultUrl.startsWith('data:')) {
            const base64Data = resultUrl.split(',')[1];
            fs.writeFileSync('scratch/tryon_result.png', base64Data, 'base64');
            console.log('Result saved to scratch/tryon_result.png');
        } else {
            console.log('Result URL is not a data URI:', resultUrl);
        }
    } catch (e) {
        console.error('Network Error:', e.message);
    }
}

testTryOn();
