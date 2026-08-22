import 'dotenv/config';
import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';

const NVIDIA_KONTEXT_URL = 'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev';
const NVIDIA_KEY = process.env.NVIDIA_API_KEY_FLUX_1 || process.env.NVIDIA_API_KEY;

if (!NVIDIA_KEY) {
  console.error('Missing NVIDIA_API_KEY_FLUX_1 or NVIDIA_API_KEY in environment');
  process.exit(1);
}

function toDataUri(buffer, contentType = 'image/png') {
  return `data:${contentType};base64,${buffer.toString('base64')}`;
}

async function uploadNvidiaInputAsset({ imageBuffer, contentType, description }) {
  const createRes = await fetch('https://api.nvcf.nvidia.com/v2/nvcf/assets', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${NVIDIA_KEY}`,
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify({ contentType, description }),
  });

  if (!createRes.ok) {
    throw new Error(`NVIDIA asset create failed (${createRes.status}): ${await createRes.text()}`);
  }

  const createData = await createRes.json();
  const assetId = createData?.assetId;
  const uploadUrl = createData?.uploadUrl;
  if (!assetId || !uploadUrl) {
    throw new Error('NVIDIA asset create returned no assetId/uploadUrl');
  }

  const putRes = await fetch(uploadUrl, {
    method: 'PUT',
    headers: {
      'Content-Type': contentType,
      'x-amz-meta-nvcf-asset-description': description,
    },
    body: imageBuffer,
  });

  if (!putRes.ok) {
    throw new Error(`NVIDIA asset upload failed (${putRes.status}): ${await putRes.text()}`);
  }

  return assetId;
}

async function buildComposite() {
  const mannequinPath = path.join(process.cwd(), 'assets/images/mannequin_front.png');
  const garmentPath = path.join(process.cwd(), 'assets/images/basic_cardigan.png');

  const mannequinBuf = fs.readFileSync(mannequinPath);
  const garmentBuf = fs.readFileSync(garmentPath);

  const [left, right] = await Promise.all([
    sharp(mannequinBuf).resize(768, 1024, { fit: 'contain', background: { r: 255, g: 255, b: 255, alpha: 1 } }).png().toBuffer(),
    sharp(garmentBuf).resize(768, 1024, { fit: 'contain', background: { r: 255, g: 255, b: 255, alpha: 1 } }).png().toBuffer(),
  ]);

  return sharp({
    create: { width: 1536, height: 1024, channels: 4, background: { r: 255, g: 255, b: 255, alpha: 1 } },
  })
    .composite([
      { input: left, left: 0, top: 0 },
      { input: right, left: 768, top: 0 },
    ])
    .png()
    .toBuffer();
}

function buildPrompt() {
  return [
    'The left half shows a fixed headless mannequin already dressed in a base composite and the right half shows the exact product garment.',
    'Improve only garment fit, drape, hems, sleeve settling, and overlap so it looks naturally worn in real life.',
    'Preserve the mannequin identity, pose, body shape, headless neck form, arms, legs, lighting, shadows, framing, and camera exactly.',
    'Do not introduce any human model, face, hair, hands, extra limbs, background objects, text, or crop changes.',
    'Do not replace the mannequin with a real person and do not invent new garment pieces.',
    'Focus edits on the upper torso and sleeves.',
    'Keep the exact garment color, silhouette, material, and design details from the product image.',
    'Output only the dressed mannequin on a clean white background.',
  ].join(' ');
}

async function main() {
  const compositeBuffer = await buildComposite();
  fs.mkdirSync(path.join(process.cwd(), 'scripts/out'), { recursive: true });
  fs.writeFileSync(path.join(process.cwd(), 'scripts/out/kontext-input.png'), compositeBuffer);

  const meta = await sharp(compositeBuffer).metadata();
  const ratio = !meta.width || !meta.height ? '9:16' : meta.width / meta.height > 1.2 ? '16:9' : meta.width / meta.height > 0.9 ? '1:1' : '9:16';
  const assetId = await uploadNvidiaInputAsset({
    imageBuffer: compositeBuffer,
    contentType: 'image/png',
    description: 'aiwardrobe-kontext-refine-test',
  });

  console.log('Uploaded asset:', assetId);

  const res = await fetch(NVIDIA_KONTEXT_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${NVIDIA_KEY}`,
      'Content-Type': 'application/json',
      Accept: 'application/json',
      'NVCF-INPUT-ASSET-REFERENCES': assetId,
    },
    body: JSON.stringify({
      prompt: buildPrompt(),
      image: `data:image/png;example_id,${assetId}`,
      aspect_ratio: ratio,
    }),
  });

  const text = await res.text();
  console.log('HTTP', res.status);
  console.log(text.slice(0, 1000));

  if (!res.ok) {
    console.error('Kontext request failed for asset', assetId);
    process.exit(1);
  }

  const data = JSON.parse(text);
  const output = data.artifacts?.[0]?.base64 || data.image || data.output?.image;
  if (!output) {
    console.error('No output image found in response');
    process.exit(1);
  }

  const dataUri = output.startsWith('data:') ? output : toDataUri(Buffer.from(output, 'base64'));
  fs.writeFileSync(path.join(process.cwd(), 'scripts/out/kontext-output.png'), Buffer.from(dataUri.split(',')[1], 'base64'));
  console.log('Saved scripts/out/kontext-output.png');
}

main().catch((err) => {
  console.error('FATAL:', err?.message || err);
  process.exit(1);
});
