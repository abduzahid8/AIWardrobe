// Probe NVIDIA hosted endpoints to find the correct one for FLUX.1-Kontext-dev.
const KEY = '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';
const TINY_PNG_B64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=';
const dataUri = `data:image/png;base64,${TINY_PNG_B64}`;

const urls = [
  'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux-1-kontext-dev',
  'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux_1-kontext-dev',
  'https://ai.api.nvidia.com/v1/infer/black-forest-labs/flux_1-kontext-dev',
  'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev',
  'https://integrate.api.nvidia.com/v1/genai/black-forest-labs/flux_1-kontext-dev',
  'https://integrate.api.nvidia.com/v1/infer/black-forest-labs/flux_1-kontext-dev',
  'https://integrate.api.nvidia.com/v1/images/generations',
];

(async () => {
  for (const url of urls) {
    try {
      const isOpenAI = url.endsWith('/images/generations');
      const body = isOpenAI
        ? { model: 'black-forest-labs/flux_1-kontext-dev', prompt: 'test', n: 1, response_format: 'b64_json', image: dataUri }
        : { prompt: 'test', image: dataUri, num_inference_steps: 1, seed: 0 };
      const res = await fetch(url, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${KEY}`,
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify(body),
      });
      const txt = await res.text();
      console.log(`${res.status} ${url}`);
      console.log(`     ${txt.slice(0, 220).replace(/\s+/g, ' ')}`);
    } catch (e) {
      console.log(`ERR  ${url} -> ${e.message}`);
    }
  }
})();
