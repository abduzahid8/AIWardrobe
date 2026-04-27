const fs = require('fs');

async function testNVIDIA() {
    const NVIDIA_KEY = "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";
    
    // Create a 512x512 mock image for upload
    const base64Pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
    const blob = new Blob([Buffer.from(base64Pixel, 'base64')], {type: 'image/png'});
    
    const form = new FormData();
    form.append('prompt', 'test');
    form.append('image', blob, 'image.png');
    form.append('width', '512');
    form.append('height', '512');
    form.append('steps', '28');

    const res = await fetch('https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev', {
        method: 'POST',
        headers: { Authorization: `Bearer ${NVIDIA_KEY}`, Accept: 'application/json' },
        body: form
    });
    
    const text = await res.text();
    console.log("NVIDIA STATUS:", res.status);
    console.log("NVIDIA RESPONSE:", text);
}
testNVIDIA();
