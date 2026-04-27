const fs = require('fs');

async function testNVIDIA() {
    const NVIDIA_KEY = "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";
    
    const form = new FormData();
    form.append('prompt', 'test');
    form.append('image', new Blob([Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFElEQVQIW2NkYGD4z8DAwMgAI0AMCKcAQSEzCygAAAAASUVORK5CYII=", "base64")], {type: "image/png"}), "test.png");

    const res = await fetch('https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev', {
        method: 'POST',
        headers: { Authorization: `Bearer ${NVIDIA_KEY}`, Accept: 'application/json' },
        body: form
    });
    
    console.log("STATUS:", res.status);
    console.log("TEXT:", await res.text());
}
testNVIDIA();
