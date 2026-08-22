import 'dotenv/config';
const KEY = process.env.NVIDIA_API_KEY_FLUX_1;
const urls = [
  'https://ai.api.nvidia.com/v1/cv/briaai/rmbg-1.4',
  'https://ai.api.nvidia.com/v1/cv/briaai/rmbg-2.0',
  'https://ai.api.nvidia.com/v1/cv/bria/rmbg',
  'https://ai.api.nvidia.com/v1/cv/nvidia/sam2',
  'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-schnell',
  'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-dev',
  'https://ai.api.nvidia.com/v1/genai/stabilityai/sdxl',
  'https://ai.api.nvidia.com/v1/cv/nvidia/nv-grounding-dino',
  'https://ai.api.nvidia.com/v1/cv/nvidia/segformer',
  'https://ai.api.nvidia.com/v1/cv/nvidia/nv-clip',
  'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-canny-dev',
  'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-depth-dev',
];
for (const url of urls) {
  try {
    const r = await fetch(url, {
      method: 'POST',
      headers: { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json' },
      body: '{}',
      signal: AbortSignal.timeout(6000),
    });
    const txt = await r.text();
    console.log(`${r.status}  ${url}  -> ${txt.slice(0, 110)}`);
  } catch (e) {
    console.log(`ERR    ${url}  -> ${e.message}`);
  }
}
