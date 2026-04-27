// Hypothesis: image=data:<mime>;example_id,<INDEX> referencing
// NVCF-INPUT-ASSET-REFERENCES header (comma-separated assetIds).
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

const KEY = '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';
const INVOKE = 'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev';
const ASSETS = 'https://api.nvcf.nvidia.com/v2/nvcf/assets';

async function uploadAsset(filePath, contentType) {
  const r1 = await fetch(ASSETS, {
    method: 'POST',
    headers: { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json', accept: 'application/json' },
    body: JSON.stringify({ contentType, description: 'mannequin' }),
  });
  const { assetId, uploadUrl } = await r1.json();
  const buf = fs.readFileSync(filePath);
  const r2 = await fetch(uploadUrl, {
    method: 'PUT',
    headers: { 'Content-Type': contentType, 'x-amz-meta-nvcf-asset-description': 'mannequin' },
    body: buf,
  });
  if (!r2.ok) throw new Error('PUT ' + r2.status);
  return assetId;
}

(async () => {
  const assetId = await uploadAsset(path.join(ROOT, 'assets/images/mannequin_front.png'), 'image/png');
  console.log('assetId:', assetId);

  const res = await fetch(INVOKE, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${KEY}`,
      Accept: 'application/json',
      'Content-Type': 'application/json',
      'NVCF-INPUT-ASSET-REFERENCES': assetId,
    },
    body: JSON.stringify({
      prompt: 'Add a small red baseball cap on the head of the figure.',
      image: 'data:image/png;example_id,0',  // <-- INDEX, not the UUID
      aspect_ratio: 'match_input_image',
      steps: 30,
      cfg_scale: 3.5,
      seed: 42,
    }),
  });
  const txt = await res.text();
  console.log(`status: ${res.status}`);
  console.log(`body head: ${txt.slice(0, 300).replace(/\s+/g, ' ')}`);
  if (res.ok) {
    const j = JSON.parse(txt);
    const b64 = j.artifacts?.[0]?.base64 || j.image;
    if (b64) {
      const out = path.join(__dirname, 'nvidia_real_result.jpg');
      fs.writeFileSync(out, Buffer.from(b64.startsWith('data:') ? b64.split(',')[1] : b64, 'base64'));
      console.log('SAVED ->', out);
    }
  }
})().catch((e) => { console.error('FATAL', e); process.exit(1); });
