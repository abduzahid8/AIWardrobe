const fs = require('fs');

const NVIDIA_KEY = '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';
const NVIDIA_URL = 'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext';

async function callNvidiaKontext() {
    const mannBuf = fs.readFileSync('assets/images/mannequin_front.png');
    const imageBlob = new Blob([new Uint8Array(mannBuf)], { type: 'image/png' });
    
    const form = new FormData();
    form.append('prompt', 'Dress the mannequin.');
    form.append('image', imageBlob, 'composite.jpg');
    form.append('width', '800');
    form.append('height', '1328');

    console.log("Sending composite to NVIDIA...");
    
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), 60000); 

    try {
        const res = await fetch(NVIDIA_URL, {
            method: 'POST',
            headers: { Authorization: `Bearer ${NVIDIA_KEY}`, Accept: 'application/json' },
            body: form,
            signal: controller.signal
        });
        clearTimeout(id);
        const text = await res.text();
        console.log(`[test] HTTP ${res.status}`);
        console.log(text.slice(0, 300));
    } catch(e) {
        console.error("Fetch failed:", e.message);
    }
}

callNvidiaKontext().catch(console.error);
