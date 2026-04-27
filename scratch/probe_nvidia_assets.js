// Probe full NVIDIA NVCF Asset Upload + Flux Kontext invocation flow.
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

const KEY = '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';
const INVOKE = 'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev';
const ASSETS = 'https://api.nvcf.nvidia.com/v2/nvcf/assets';

async function uploadAsset(filePath, contentType, description) {
  // 1) request presigned URL
  const r1 = await fetch(ASSETS, {
    method: 'POST',
    headers: { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json', accept: 'application/json' },
    body: JSON.stringify({ contentType, description }),
  });
  if (!r1.ok) throw new Error(`assets POST ${r1.status}: ${await r1.text()}`);
  const { assetId, uploadUrl } = await r1.json();
  console.log('assetId:', assetId);

  // 2) PUT bytes
  const buf = fs.readFileSync(filePath);
  const r2 = await fetch(uploadUrl, {
    method: 'PUT',
    headers: {
      'Content-Type': contentType,
      'x-amz-meta-nvcf-asset-description': description,
    },
    body: buf,
  });
  if (!r2.ok) throw new Error(`PUT upload ${r2.status}: ${await r2.text()}`);
  console.log('upload OK', buf.length, 'bytes');
  return assetId;
}

async function invoke(assetId, prompt) {
  // Try a few ways of referencing the asset.
  const variants = [
    {
      label: 'image=assetId in body',
      body: { prompt, image: assetId },
      headers: {},
    },
    {
      label: 'image=assetId, NVCF header',
      body: { prompt, image: assetId },
      headers: { 'NVCF-INPUT-ASSET-REFERENCES': assetId },
    },
    {
      label: 'image=<example_id> placeholder, NVCF header',
      body: { prompt, image: `<${assetId}>` },
      headers: { 'NVCF-INPUT-ASSET-REFERENCES': assetId },
    },
  ];

  for (const v of variants) {
    const res = await fetch(INVOKE, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${KEY}`,
        'Content-Type': 'application/json',
        Accept: 'application/json',
        ...v.headers,
      },
      body: JSON.stringify(v.body),
    });
    const txt = await res.text();
    console.log(`[${v.label}] ${res.status}`);
    console.log(`   ${txt.slice(0, 400).replace(/\s+/g, ' ')}`);
    if (res.ok) {
      console.log('   SUCCESS, full body length:', txt.length);
      try {
        const j = JSON.parse(txt);
        console.log('   keys:', Object.keys(j));
      } catch {}
      return v;
    }
  }
  return null;
}

(async () => {
  const img = path.join(ROOT, 'assets/images/mannequin_front.png');
  const assetId = await uploadAsset(img, 'image/png', 'mannequin');
  await invoke(assetId, 'Dress this mannequin in a plain white t-shirt.');
})().catch((e) => { console.error('FATAL', e); process.exit(1); });
