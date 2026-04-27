// Final attempt: try multipart/form-data, alternate models, and inspect 500 error details.
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

const KEY = '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';
const ASSETS = 'https://api.nvcf.nvidia.com/v2/nvcf/assets';

async function uploadAsset(filePath, contentType) {
  const r1 = await fetch(ASSETS, {
    method: 'POST',
    headers: { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json', accept: 'application/json' },
    body: JSON.stringify({ contentType, description: 'tryon' }),
  });
  const { assetId, uploadUrl } = await r1.json();
  const buf = fs.readFileSync(filePath);
  const r2 = await fetch(uploadUrl, {
    method: 'PUT',
    headers: { 'Content-Type': contentType, 'x-amz-meta-nvcf-asset-description': 'tryon' },
    body: buf,
  });
  if (!r2.ok) throw new Error('PUT failed ' + r2.status);
  return assetId;
}

const ENDPOINTS = [
  'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev',
  'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-pro',
  'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-max',
];

async function tryEndpoint(url, assetId, fileBuf) {
  console.log(`\n--- ${url} ---`);
  const dataUri = `data:image/png;example_id,${assetId}`;

  // Variant A: JSON body with asset_id data URI + NVCF header
  let res = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${KEY}`,
      'Content-Type': 'application/json',
      Accept: 'application/json',
      'NVCF-INPUT-ASSET-REFERENCES': assetId,
    },
    body: JSON.stringify({ prompt: 'Add a small red baseball cap.', image: dataUri }),
  });
  console.log(`[json+asset] ${res.status}`);
  console.log(`   ${(await res.text()).slice(0, 250).replace(/\s+/g, ' ')}`);

  // Variant B: multipart/form-data with raw file
  const form = new FormData();
  form.append('prompt', 'Add a small red baseball cap.');
  form.append('image', new Blob([fileBuf], { type: 'image/png' }), 'image.png');
  res = await fetch(url, {
    method: 'POST',
    headers: { Authorization: `Bearer ${KEY}`, Accept: 'application/json' },
    body: form,
  });
  console.log(`[multipart] ${res.status}`);
  console.log(`   ${(await res.text()).slice(0, 250).replace(/\s+/g, ' ')}`);
}

(async () => {
  const filePath = path.join(ROOT, 'assets/images/mannequin_front.png');
  const fileBuf = fs.readFileSync(filePath);
  const assetId = await uploadAsset(filePath, 'image/png');
  console.log('assetId:', assetId, 'size:', fileBuf.length);

  for (const url of ENDPOINTS) {
    await tryEndpoint(url, assetId, fileBuf);
  }
})().catch((e) => { console.error('FATAL', e); process.exit(1); });
