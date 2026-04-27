const fs = require('fs');

const NVIDIA_KEY = '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';
const NVIDIA_URL = 'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux_1-kontext-dev';

async function test() {
    console.log("Testing with FormData and composite...");
    // Let's create a dummy combined image (just using mannequin for both sides is fine for test API hit)
    const mannBuf = fs.readFileSync('assets/images/mannequin_front.png');
    
    // In Node.js we don't have OffscreenCanvas easily, so we just use the mannequin image twice
    // For this test, any proper PNG buffer works
    const imageBlob = new Blob([mannBuf], { type: 'image/png' });
    
    const form = new FormData();
    form.append('prompt', 'This image contains two panels side by side: LEFT = a mannequin, RIGHT = a garment. Dress the mannequin in the LEFT panel with the garment on the RIGHT. Preserve mannequin.');
    form.append('image', imageBlob, 'composite.png');
    form.append('width', '800'); // the original code used 800x1328
    form.append('height', '1328');
    form.append('steps', '28');
    form.append('cfg_scale', '3.5');
    form.append('seed', '42');

    const res = await fetch(NVIDIA_URL, {
      method: 'POST',
      headers: { Authorization: `Bearer ${NVIDIA_KEY}`, Accept: 'application/json' },
      body: form,
    });

    const text = await res.text();
    console.log(`[test] HTTP ${res.status}: ${text.slice(0, 300)}`);
}

test().catch(console.error);
