import 'dotenv/config';
const KEY = process.env.NVIDIA_API_KEY_FLUX_1;
const TINY_PNG_DATAURI =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNiAAAABgADNjd8qAAAAABJRU5ErkJggg==';

const candidates = [
  'https://ai.api.nvidia.com/v1/vlm/google/paligemma',
  'https://integrate.api.nvidia.com/v1/chat/completions',
  'https://ai.api.nvidia.com/v1/vlm/microsoft/phi-4-multimodal-instruct',
  'https://ai.api.nvidia.com/v1/vlm/google/gemma-3-27b-it',
  'https://ai.api.nvidia.com/v1/vlm/google/gemma-3n-e4b-it',
  'https://ai.api.nvidia.com/v1/vlm/google/gemma-3n-e2b-it',
  'https://ai.api.nvidia.com/v1/vlm/nvidia/llama-3.1-nemotron-nano-vl-8b-v1',
];

const payloads = {
  default: {
    model: 'google/paligemma',
    messages: [
      { role: 'user', content: [{ type: 'text', text: `segment shirt <image>` }, { type: 'image_url', image_url: { url: TINY_PNG_DATAURI } }] },
    ],
  },
};

for (const url of candidates) {
  try {
    const body = { ...payloads.default };
    // Match model name to URL slug
    const slug = url.split('/').slice(-1)[0];
    if (slug && slug !== 'completions') {
      // Try with provider prefix
      const segs = url.split('/');
      const provider = segs[segs.length - 2];
      body.model = `${provider}/${slug}`;
    }
    const r = await fetch(url, {
      method: 'POST',
      headers: { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(15000),
    });
    const txt = await r.text();
    console.log(`${r.status}  ${url}\n   model=${body.model}\n   -> ${txt.slice(0, 200)}\n`);
  } catch (e) {
    console.log(`ERR    ${url}  -> ${e.message}\n`);
  }
}
