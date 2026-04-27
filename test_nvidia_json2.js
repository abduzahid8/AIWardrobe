const fs = require('fs');

async function testNVIDIA() {
    const NVIDIA_KEY = "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";
    
    // Create a robust 512x512 mock image buffer
    const canvas = require('canvas');
    const cvs = canvas.createCanvas(512, 512);
    const ctx = cvs.getContext('2d');
    ctx.fillStyle = 'red';
    ctx.fillRect(0, 0, 512, 512);
    const b64 = cvs.toDataURL('image/png'); // "data:image/png;base64,..."

    const payload = {
        prompt: "test",
        image: b64,
        steps: 28,
        width: 512,
        height: 512
    };

    const res = await fetch('https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev', {
        method: 'POST',
        headers: { 
            'Authorization': `Bearer ${NVIDIA_KEY}`, 
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify(payload)
    });
    
    console.log("STATUS:", res.status);
    console.log("TEXT:", await res.text());
}
testNVIDIA();
