const fs = require('fs');
const path = require('path');

async function testNVIDIA() {
    const NVIDIA_KEY = "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";
    
    const imgBuf = fs.readFileSync(path.join(__dirname, 'assets/images/mannequin_front.png'));
    const b64 = "data:image/png;base64," + imgBuf.toString('base64');

    const payload = {
        prompt: "test",
        image: b64,
        steps: 28,
        width: 1024,
        height: 1024
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
    console.log("TEXT:", (await res.text()).substring(0, 500));
}
testNVIDIA();
