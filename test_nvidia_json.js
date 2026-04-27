const fs = require('fs');

async function testNVIDIA() {
    const NVIDIA_KEY = "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";
    
    // A realistic 10x10 base64 transparent PNG
    const b64 = "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mNkYPhfz0AEYBxVSF+FAAhKDveksOjmAAAAAElFTkSuQmCC";

    const payload = {
        prompt: "test",
        image: b64,
        steps: 28
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
