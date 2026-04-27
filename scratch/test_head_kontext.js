const fs = require('fs');

const NVIDIA_KEY = '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';
const NVIDIA_URL = 'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev';

async function test() {
    console.log("Testing with FormData and dot version URL...");
    const mannBuf = fs.readFileSync('assets/images/mannequin_front.png');
    const imageBlob = new Blob([mannBuf], { type: 'image/png' });
    
    const form = new FormData();
    form.append('prompt', 'Dress the mannequin with a red jacket.');
    form.append('image', imageBlob, 'composite.png');
    form.append('width', '1024');
    form.append('height', '1024');
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
