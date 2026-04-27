
const fs = require('fs');

const REPLICATE_TOKEN = process.env.REPLICATE_TOKEN || 'YOUR_REPLICATE_TOKEN_HERE';
const NVIDIA_KEY = process.env.NVIDIA_KEY || 'YOUR_NVIDIA_KEY_HERE';
const SUPABASE_URL = process.env.SUPABASE_URL || 'YOUR_SUPABASE_URL_HERE';
const SUPABASE_ANON_KEY = process.env.SUPABASE_ANON_KEY || 'YOUR_SUPABASE_ANON_KEY_HERE';

const SHIRT_IMAGE = 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg';

async function testReplicateCatVTON() {
    console.log('--- Testing Replicate CatVTON (Virtual Try-On on Mannequin) ---');
    
    const mannequinImage = fs.readFileSync('assets/images/mannequin_front.png', { encoding: 'base64' });
    const mannequinDataUri = `data:image/png;base64,${mannequinImage}`;

    // Use IDM-VTON model on Replicate - best virtual try-on model
    const res = await fetch('https://api.replicate.com/v1/predictions', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${REPLICATE_TOKEN}`,
            'Content-Type': 'application/json',
            'Prefer': 'wait'
        },
        body: JSON.stringify({
            // IDM-VTON: best model for virtual try-on
            version: 'c871bb9b046607b680449ecbae55fd8c6d945e0a1948644bf2361b3d021d3ff4',
            input: {
                human_img: mannequinDataUri,
                garm_img: SHIRT_IMAGE,
                garment_des: 'white polo shirt',
                is_checked: true,
                is_checked_crop: false,
                denoise_steps: 30,
                seed: 42
            }
        })
    });

    const data = await res.json();
    console.log('Status:', res.status);
    
    if (!res.ok) {
        console.error('Error:', JSON.stringify(data));
        return;
    }
    
    console.log('Prediction status:', data.status);
    console.log('Prediction ID:', data.id);

    if (data.status === 'succeeded') {
        const outputUrl = Array.isArray(data.output) ? data.output[1] : data.output;
        console.log('Output URL:', outputUrl);
        if (outputUrl) {
            const imgRes = await fetch(outputUrl);
            const buf = Buffer.from(await imgRes.arrayBuffer());
            fs.writeFileSync('scratch/replicate_vton_result.png', buf);
            console.log('Saved to scratch/replicate_vton_result.png');
        }
    } else {
        console.log('Prediction response:', JSON.stringify(data, null, 2));
    }
}

testReplicateCatVTON().catch(console.error);
