const fs = require('fs');

async function testNVIDIA() {
    const NVIDIA_KEY = "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";
    
    // 512x512 transparent pixel or similar
    const b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

    const payload = {
        prompt: "test",
        image: `data:image/png;base64,${b64}`,
        num_inference_steps: 28,
        guidance_scale: 4.0,
        seed: 42,
        height: 768,
        width: 768
    };

    console.log("Sending JSON to NVIDIA HYPHEN...");
    const res = await fetch('https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux-1-kontext-dev', {
        method: 'POST',
        headers: { 
            'Authorization': `Bearer ${NVIDIA_KEY}`, 
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    });
    
    console.log("STATUS:", res.status);
    console.log("TEXT:", (await res.text()).substring(0, 500));
}
testNVIDIA();
