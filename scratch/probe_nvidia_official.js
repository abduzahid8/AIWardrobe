// Replicate the official NVIDIA sample exactly, then retry with uploaded asset.
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

const KEY = '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';
const INVOKE = 'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev';
const ASSETS = 'https://api.nvcf.nvidia.com/v2/nvcf/assets';

async function call(label, body, extraHeaders = {}) {
  const res = await fetch(INVOKE, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${KEY}`,
      Accept: 'application/json',
      'Content-Type': 'application/json',
      ...extraHeaders,
    },
    body: JSON.stringify(body),
  });
  const txt = await res.text();
  console.log(`[${label}] ${res.status}`);
  console.log(`   ${txt.slice(0, 400).replace(/\s+/g, ' ')}`);
  if (res.ok) {
    try {
      const j = JSON.parse(txt);
      console.log('   keys:', Object.keys(j));
      const b64 = j.artifacts?.[0]?.base64 || j.image;
      if (b64) {
        const out = path.join(__dirname, `nvidia_${label.replace(/\W+/g, '_')}.png`);
        const raw = b64.startsWith('data:') ? b64.split(',')[1] : b64;
        fs.writeFileSync(out, Buffer.from(raw, 'base64'));
        console.log('   saved ->', out);
        return true;
      }
    } catch {}
  }
  return false;
}

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
  // 1. Exact official sample
  await call('official_sample', {
    prompt: 'Now the mouse is holding pizza instead',
    image: 'data:image/png;example_id,0',
    aspect_ratio: 'match_input_image',
    steps: 30,
    cfg_scale: 3.5,
    seed: 0,
  });

  // 2. With uploaded asset, mimicking the official param shape
  const assetId = await uploadAsset(path.join(ROOT, 'assets/images/mannequin_front.png'), 'image/png');
  console.log('uploaded:', assetId);

  await call(
    'uploaded_no_header',
    {
      prompt: 'Add a small red baseball cap on the head.',
      image: `data:image/png;example_id,${assetId}`,
      aspect_ratio: 'match_input_image',
      steps: 30,
      cfg_scale: 3.5,
      seed: 0,
    },
  );

  await call(
    'uploaded_with_header',
    {
      prompt: 'Add a small red baseball cap on the head.',
      image: `data:image/png;example_id,${assetId}`,
      aspect_ratio: 'match_input_image',
      steps: 30,
      cfg_scale: 3.5,
      seed: 0,
    },
    { 'NVCF-INPUT-ASSET-REFERENCES': assetId },
  );
})().catch((e) => { console.error('FATAL', e); process.exit(1); });
