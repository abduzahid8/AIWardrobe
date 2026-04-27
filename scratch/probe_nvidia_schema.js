// Discover the request schema for FLUX.1-Kontext-dev on NVIDIA NIM hosted.
const KEY = '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';
const URL = 'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev';
const TINY_PNG_B64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=';
const dataUri = `data:image/png;base64,${TINY_PNG_B64}`;

const tries = [
  { label: 'empty', body: {} },
  { label: 'prompt only', body: { prompt: 'test' } },
  { label: 'prompt + image', body: { prompt: 'test', image: dataUri } },
  { label: 'prompt + image + steps', body: { prompt: 'test', image: dataUri, steps: 10 } },
  { label: 'prompt + image + cfg_scale', body: { prompt: 'test', image: dataUri, cfg_scale: 4.0 } },
  { label: 'prompt + image_url', body: { prompt: 'test', image_url: dataUri } },
  { label: 'prompt + input_image', body: { prompt: 'test', input_image: dataUri } },
  { label: 'OPTIONS schema', method: 'OPTIONS' },
  { label: 'GET docs', method: 'GET' },
];

(async () => {
  for (const t of tries) {
    try {
      const opts = {
        method: t.method ?? 'POST',
        headers: {
          Authorization: `Bearer ${KEY}`,
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
      };
      if (opts.method === 'POST') opts.body = JSON.stringify(t.body ?? {});
      const res = await fetch(URL, opts);
      const txt = await res.text();
      console.log(`[${t.label}] ${res.status}`);
      console.log(`   ${txt.slice(0, 500).replace(/\s+/g, ' ')}`);
    } catch (e) {
      console.log(`[${t.label}] ERR ${e.message}`);
    }
  }
})();
