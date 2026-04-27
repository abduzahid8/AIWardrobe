// Try the asset-id data URI variants the parser expects.
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

(async () => {
  const img = path.join(ROOT, 'assets/images/mannequin_front.png');
  const assetId = await uploadAsset(img, 'image/png', 'mannequin');

  const variants = [
    `data:image/png;example_id,${assetId}`,
    `data:image/png;asset_id,${assetId}`,
    `data:image/png;assetId,${assetId}`,
    `data:image/png;example,${assetId}`,
    assetId,
  ];

  for (const imgValue of variants) {
    const res = await fetch(INVOKE, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${KEY}`,
        'Content-Type': 'application/json',
        Accept: 'application/json',
        'NVCF-INPUT-ASSET-REFERENCES': assetId,
      },
      body: JSON.stringify({ prompt: 'Add a small red hat on top of the figure.', image: imgValue }),
    });
    const txt = await res.text();
    console.log(`[${imgValue.slice(0, 60)}] ${res.status}`);
    console.log(`   ${txt.slice(0, 350).replace(/\s+/g, ' ')}`);
    if (res.ok) {
      try {
        const j = JSON.parse(txt);
        console.log('   keys:', Object.keys(j));
        if (j.image) console.log('   image len:', j.image.length);
        if (j.artifacts) console.log('   artifacts[0] keys:', Object.keys(j.artifacts[0] || {}));
      } catch {}
      break;
    }
  }
})().catch((e) => { console.error('FATAL', e); process.exit(1); });
