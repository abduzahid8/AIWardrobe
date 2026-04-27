const fs = require('fs');

const NVIDIA_KEY = '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';
const KONTEXT_URL = 'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev';

async function callKontext(mannequinBuf, prompt) {
    const b64 = mannequinBuf.toString('base64');
    const dataUri = `data:image/png;base64,${b64}`;
    console.log(`[kontext] Sending mannequin as JSON data URI...`);

    const res = await fetch(KONTEXT_URL, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${NVIDIA_KEY}`,
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        },
        body: JSON.stringify({
            prompt,
            image: dataUri,
            width: 1024,
            height: 1024,
            steps: 28,
            cfg_scale: 3.5,
            seed: 42,
        })
    });

    const text = await res.text();
    console.log(`[kontext] HTTP ${res.status}: ${text.slice(0, 400)}`);
    if (!res.ok) return null;

    const data = JSON.parse(text);
    const b64out = data.artifacts?.[0]?.base64 || data.image || data.b64_json;
    return b64out ? (b64out.startsWith('data:') ? b64out : `data:image/png;base64,${b64out}`) : null;
}

async function test() {
    console.log('=== Testing FLUX.1-Kontext with JSON & data URI ===\n');

    const mannBuf = fs.readFileSync('assets/images/mannequin_front.png');

    const prompt = 
        `The image shows a pristine white headless fashion mannequin on a white studio background. ` +
        `Dress the mannequin with ALL of the following garments simultaneously: ` +
        `(1) A crisp white polo shirt on the torso ` +
        `(2) Dark navy slim-fit chino trousers on the legs ` +
        `(3) A structured navy blue blazer jacket layered over the shirt ` +
        `(4) Clean white low-top canvas sneakers on the feet. ` +
        `The clothes must look perfectly steamed, photorealistic, and perfectly fitted to the mannequin body. ` +
        `Keep the white seamless studio background and headless mannequin exactly as-is.`;

    const result = await callKontext(mannBuf, prompt);

    if (result) {
        const b64 = result.split(',')[1] || result;
        fs.writeFileSync('scratch/kontext_4item_result.png', Buffer.from(b64, 'base64'));
        console.log('\n✅ SUCCESS! Saved to scratch/kontext_4item_result.png');
    } else {
        console.log('\n❌ No result from Kontext.');
    }
}

test().catch(e => console.error('FATAL:', e.message));
