const fs = require('fs');
const NVIDIA_KEY = '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';
async function run() {
    const assetId = 'd9d8de92-0c75-475d-b792-8e4057c55eea'; // Already uploaded MANNEQUIN
    
    console.log("Calling Kontext with dot version...");
    const res = await fetch('https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${NVIDIA_KEY}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            prompt: "Dress this mannequin in a red jacket.",
            image: assetId,
            width: 1024, height: 1024, steps: 28, cfg_scale: 3.5, seed: 42
        })
    });
    console.log("Status:", res.status);
    const body = await res.text();
    console.log("Response:", body.slice(0, 300));
}
run();
