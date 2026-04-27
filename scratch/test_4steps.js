
const fs = require('fs');
const SUPABASE_URL = 'https://fyqpifmrsftsfqibhwhy.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ5cXBpZm1yc2Z0c2ZxaWJod2h5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQzMDYyOTgsImV4cCI6MjA3OTg4MjI5OH0.dydnFn3lqub7qMo9uFfn5yUyY4Wr_eQPnsbvHWHwMTk';

async function test4StepOutfit() {
    console.log('--- Testing 4-Step Full Outfit (Layer, Top, Pants, Shoes) ---');
    
    const mannequinImage = fs.readFileSync('assets/images/mannequin_front.png', { encoding: 'base64' });
    
    const items = [
        { label: 'layer', type: 'upper_body', image: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg', desc: 'Navy Jacket' },
        { label: 'top',   type: 'upper_body', image: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg', desc: 'White Polo' },
        { label: 'pants', type: 'lower_body', image: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg', desc: 'Black Pants' },
        { label: 'shoes', type: 'shoes',      image: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg', desc: 'Shoes' }
    ];

    let outfitSoFar = '{}';
    let currentMannequin = `data:image/png;base64,${mannequinImage}`;

    for (let i = 0; i < items.length; i++) {
        const item = items[i];
        const step = i + 1;
        console.log(`Step ${step}/${items.length}: Sending ${item.label} (${item.desc})...`);
        
        try {
            console.log(`[test] waiting 2s...`);
            await new Promise(r => setTimeout(r, 2000));

            const res = await fetch(`${SUPABASE_URL}/functions/v1/mannequin-tryon`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'apikey': SUPABASE_ANON_KEY,
                    'Authorization': `Bearer ${SUPABASE_ANON_KEY}`
                },
                body: JSON.stringify({
                    mannequin_image: currentMannequin,
                    garment: { image: item.image, label: item.label, type: item.type },
                    step: step,
                    total: items.length,
                    outfit_so_far: outfitSoFar
                })
            });

            if (!res.ok) {
                console.error(`Step ${step} HTTP Error: ${res.status} ${res.statusText}`);
                const text = await res.text();
                console.error('Response:', text.slice(0, 500));
                return;
            }

            const data = await res.json();
            if (!data.success) {
                console.error(`Step ${step} Logic Error:`, data.error);
                return;
            }

            console.log(`Step ${step} Success. Mode: ${data.mode}`);
            outfitSoFar = data.outfit_so_far;
            
            // If the mode is 'hf-kontext', the resultUrl is the edited mannequin.
            // If the mode is 'collecting', it's just the original mannequin.
            // If it's 'flux-dev-final', it's the final result.
            if (data.resultUrl && data.resultUrl.startsWith('data:')) {
                currentMannequin = data.resultUrl;
            }

            if (step === items.length) {
                const base64Data = data.resultUrl.split(',')[1];
                fs.writeFileSync('scratch/full_4step_outfit.png', base64Data, 'base64');
                console.log('--- ALL DONE ---');
                console.log('Final result saved to scratch/full_4step_outfit.png');
            }
        } catch (e) {
            console.error(`Step ${step} fetch error:`, e.message);
            return;
        }
    }
}

test4StepOutfit();
