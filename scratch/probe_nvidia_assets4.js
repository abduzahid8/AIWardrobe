// Try alternate asset endpoints under ai.api.nvidia.com.
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

const KEY = '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';
const INVOKE = 'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev';

const assetEndpoints = [
  'https://ai.api.nvidia.com/v1/assets',
  'https://ai.api.nvidia.com/v2/assets',
  'https://ai.api.nvidia.com/v1/genai/assets',
  'https://api.nvcf.nvidia.com/v2/nvcf/assets',
];

async function probe() {
  for (const url of assetEndpoints) {
    try {
      const r = await fetch(url, {
        method: 'POST',
        headers: { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json', accept: 'application/json' },
        body: JSON.stringify({ contentType: 'image/png', description: 'test' }),
      });
      const t = await r.text();
      console.log(`POST ${url} -> ${r.status}`);
      console.log(`   ${t.slice(0, 250).replace(/\s+/g, ' ')}`);
    } catch (e) {
      console.log(`ERR POST ${url} -> ${e.message}`);
    }
  }
}

async function fullFlow() {
  // Use NVCF asset upload, then verify the asset is reachable from the genai endpoint by including
  // BOTH the data-uri and an explicit list-of-assets header.
  const assetsUrl = 'https://api.nvcf.nvidia.com/v2/nvcf/assets';
  const r1 = await fetch(assetsUrl, {
    method: 'POST',
    headers: { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json', accept: 'application/json' },
    body: JSON.stringify({ contentType: 'image/png', description: 'mannequin' }),
  });
  const j1 = await r1.json();
  console.log('asset created:', j1.assetId);

  const buf = fs.readFileSync(path.join(ROOT, 'assets/images/mannequin_front.png'));
  const r2 = await fetch(j1.uploadUrl, {
    method: 'PUT',
    headers: { 'Content-Type': 'image/png', 'x-amz-meta-nvcf-asset-description': 'mannequin' },
    body: buf,
  });
  console.log('upload status:', r2.status, r2.statusText);

  // Confirm asset is listed
  const list = await fetch(assetsUrl, { headers: { Authorization: `Bearer ${KEY}` } });
  const listJson = await list.json();
  console.log('asset count:', (listJson.assets || []).length);

  // Wait a moment and invoke
  await new Promise((r) => setTimeout(r, 1500));

  const dataUri = `data:image/png;example_id,${j1.assetId}`;
  const variants = [
    { label: 'no header, just dataUri', headers: {} },
    { label: 'NVCF-INPUT-ASSET-REFERENCES', headers: { 'NVCF-INPUT-ASSET-REFERENCES': j1.assetId } },
    { label: 'nvcf-asset-input-reference (lowercase singular)', headers: { 'nvcf-asset-input-reference': j1.assetId } },
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
      body: JSON.stringify({ prompt: 'Add a small red baseball cap on the head.', image: dataUri }),
    });
    const txt = await res.text();
    console.log(`[${v.label}] ${res.status}`);
    console.log(`   ${txt.slice(0, 300).replace(/\s+/g, ' ')}`);
    if (res.ok) {
      try {
        const j = JSON.parse(txt);
        console.log('   keys:', Object.keys(j));
      } catch {}
      break;
    }
  }
}

(async () => {
  console.log('=== probe asset endpoints ===');
  await probe();
  console.log('\n=== full flow w/ NVCF assets ===');
  await fullFlow();
})().catch((e) => { console.error('FATAL', e); process.exit(1); });
