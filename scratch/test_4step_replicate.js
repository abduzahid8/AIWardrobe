
const fs = require('fs');

const REPLICATE_TOKEN = process.env.REPLICATE_TOKEN || 'YOUR_REPLICATE_TOKEN_HERE';
// IDM-VTON model version on Replicate
const VTON_VERSION = 'c871bb9b046607b680449ecbae55fd8c6d945e0a1948644bf2361b3d021d3ff4';

const items = [
    { label: 'top',   image: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg', desc: 'white polo shirt' },
    { label: 'pants', image: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg', desc: 'dark slim-fit trousers' },
    { label: 'layer', image: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg', desc: 'navy blue blazer jacket' },
    { label: 'shoes', image: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg', desc: 'white canvas sneakers' },
];

async function applyGarmentToMannequin(humanImgB64OrUrl, garmentImgUrl, desc) {
    const isDataUri = humanImgB64OrUrl.startsWith('data:');
    const humanImg = isDataUri ? humanImgB64OrUrl : humanImgB64OrUrl;
    
    const res = await fetch('https://api.replicate.com/v1/predictions', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${REPLICATE_TOKEN}`,
            'Content-Type': 'application/json',
            'Prefer': 'wait=60'
        },
        body: JSON.stringify({
            version: VTON_VERSION,
            input: {
                human_img: humanImg,
                garm_img: garmentImgUrl,
                garment_des: desc,
                is_checked: true,
                is_checked_crop: false,
                denoise_steps: 25,
                seed: 42
            }
        })
    });

    const data = await res.json();
    if (!res.ok || data.status === 'failed') {
        throw new Error(`Replicate failed: ${JSON.stringify(data.error ?? data)}`);
    }
    
    // If still processing, poll
    if (data.status !== 'succeeded') {
        console.log('  Polling...', data.id);
        return await pollPrediction(data.id);
    }
    
    const outputUrl = Array.isArray(data.output) ? data.output[1] || data.output[0] : data.output;
    if (!outputUrl) throw new Error('No output URL from Replicate');
    return outputUrl;
}

async function pollPrediction(id, maxRetries = 20) {
    for (let i = 0; i < maxRetries; i++) {
        await new Promise(r => setTimeout(r, 3000));
        const res = await fetch(`https://api.replicate.com/v1/predictions/${id}`, {
            headers: { 'Authorization': `Bearer ${REPLICATE_TOKEN}` }
        });
        const data = await res.json();
        if (data.status === 'succeeded') {
            const outputUrl = Array.isArray(data.output) ? data.output[1] || data.output[0] : data.output;
            return outputUrl;
        }
        if (data.status === 'failed') throw new Error(`Prediction failed: ${data.error}`);
        console.log(`  Still processing... (${i + 1}/${maxRetries})`);
    }
    throw new Error('Polling timed out');
}

async function urlToBase64DataUri(url) {
    const res = await fetch(url);
    const buf = Buffer.from(await res.arrayBuffer());
    const mime = res.headers.get('content-type') || 'image/jpeg';
    return `data:${mime};base64,${buf.toString('base64')}`;
}

async function full4StepTryOn() {
    console.log('=== Full 4-Item Sequential Try-On (Replicate IDM-VTON) ===\n');
    
    // Start with mannequin
    const mannequinB64 = `data:image/png;base64,${fs.readFileSync('assets/images/mannequin_front.png', 'base64')}`;
    let currentImage = mannequinB64;

    for (let i = 0; i < items.length; i++) {
        const item = items[i];
        console.log(`Step ${i + 1}/4: Applying ${item.label} ("${item.desc}")...`);
        
        const outputUrl = await applyGarmentToMannequin(currentImage, item.image, item.desc);
        console.log(`  ✅ Done. Output: ${outputUrl}`);
        
        // Convert URL result to base64 for next step
        if (i < items.length - 1) {
            console.log('  Converting to base64 for next step...');
            currentImage = await urlToBase64DataUri(outputUrl);
        } else {
            // Last step - save result
            const imgRes = await fetch(outputUrl);
            const buf = Buffer.from(await imgRes.arrayBuffer());
            fs.writeFileSync('scratch/full_outfit_replicate.png', buf);
            console.log('\n=== ALL DONE ===');
            console.log('Final result saved to scratch/full_outfit_replicate.png');
        }
    }
}

full4StepTryOn().catch(console.error);
