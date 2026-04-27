// Test the working format `data:<mime>;example_id,<assetId>` more carefully.
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

const KEY = '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';
const INVOKE = 'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev';
const ASSETS = 'https://api.nvcf.nvidia.com/v2/nvcf/assets';

async function uploadAsset(filePath, contentType, description) {
  const r1 = await fetch(ASSETS, {
    method: 'POST',
    headers: { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json', accept: 'application/json' },
    body: JSON.stringify({ contentType, description }),
  });
  if (!r1.ok) throw new Error(`assets POST ${r1.status}: ${await r1.text()}`);
  const { assetId, uploadUrl } = await r1.json();
  const buf = fs.readFileSync(filePath);
  const r2 = await fetch(uploadUrl, {
    method: 'PUT',
    headers: { 'Content-Type': contentType, 'x-amz-meta-nvcf-asset-description': description },
    body: buf,
  });
  if (!r2.ok) throw new Error(`PUT upload ${r2.status}: ${await r2.text()}`);
  console.log('uploaded', buf.length, 'bytes assetId=', assetId);
  return assetId;
}

async function tryInvoke(label, headers, body) {
  const res = await fetch(INVOKE, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${KEY}`,
      'Content-Type': 'application/json',
      Accept: 'application/json',
      ...headers,
    },
    body: JSON.stringify(body),
  });
  const txt = await res.text();
  console.log(`[${label}] ${res.status}`);
  console.log(`   ${txt.slice(0, 400).replace(/\s+/g, ' ')}`);
  if (res.ok) {
    try {
      const j = JSON.parse(txt);
      console.log('   JSON keys:', Object.keys(j));
      if (j.image) {
        const out = path.join(__dirname, 'nvidia_kontext_result.png');
        const b64 = j.image.startsWith('data:') ? j.image.split(',')[1] : j.image;
        fs.writeFileSync(out, Buffer.from(b64, 'base64'));
        console.log('   Saved result ->', out);
      }
      if (j.artifacts?.[0]?.base64) {
        const out = path.join(__dirname, 'nvidia_kontext_artifact.png');
        fs.writeFileSync(out, Buffer.from(j.artifacts[0].base64, 'base64'));
        console.log('   Saved artifact ->', out);
      }
    } catch (e) { console.log('   parse err', e.message); }
    return true;
  }
  return false;
}

(async () => {
  const img = path.join(ROOT, 'assets/images/mannequin_front.png');
  const assetId = await uploadAsset(img, 'image/png', 'mannequin');
  const dataUri = `data:image/png;example_id,${assetId}`;

  // Variant 1: include the NVCF-INPUT-ASSET-REFERENCES header
  let ok = await tryInvoke(
    'with NVCF header',
    { 'NVCF-INPUT-ASSET-REFERENCES': assetId },
    { prompt: 'Add a small red baseball cap on the head of the figure.', image: dataUri },
  );
  if (ok) return;

  // Variant 2: no header
  ok = await tryInvoke(
    'no extra header',
    {},
    { prompt: 'Add a small red baseball cap on the head of the figure.', image: dataUri },
  );
  if (ok) return;

  // Variant 3: with explicit aspect_ratio
  ok = await tryInvoke(
    'with aspect_ratio 1:1',
    { 'NVCF-INPUT-ASSET-REFERENCES': assetId },
    { prompt: 'Add a small red baseball cap on the head of the figure.', image: dataUri, aspect_ratio: '1:1' },
  );
  if (ok) return;

  // Variant 4: with seed
  ok = await tryInvoke(
    'with seed',
    { 'NVCF-INPUT-ASSET-REFERENCES': assetId },
    { prompt: 'Add a small red baseball cap on the head of the figure.', image: dataUri, seed: 42 },
  );
})().catch((e) => { console.error('FATAL', e); process.exit(1); });
