import 'dotenv/config';
import fs from 'node:fs/promises';
import sharp from 'sharp';

const KEY = process.env.NVIDIA_API_KEY_FLUX_1;
if (!KEY) { console.error('no key'); process.exit(1); }

// Tiny 4x4 white PNG
const tinyPng = await sharp({
  create: { width: 4, height: 4, channels: 3, background: { r: 255, g: 255, b: 255 } },
}).png().toBuffer();

// Register asset
const reg = await fetch('https://api.nvcf.nvidia.com/v2/nvcf/assets', {
  method: 'POST',
  headers: { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json', Accept: 'application/json' },
  body: JSON.stringify({ contentType: 'image/png', description: 't' }),
});
const { assetId, uploadUrl } = await reg.json();
console.log('assetId:', assetId);

const put = await fetch(uploadUrl, {
  method: 'PUT',
  headers: { 'Content-Type': 'image/png', 'x-amz-meta-nvcf-asset-description': 't' },
  body: tinyPng,
});
console.log('PUT status:', put.status);

const FLUX = 'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev';
const variants = [
  ['image=public_url', { prompt: 'hi', image: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg' }],
  ['image=public_url+ar', { prompt: 'hi', image: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg', aspect_ratio: 'match_input_image' }],
];

for (const [name, body] of variants) {
  const r = await fetch(FLUX, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${KEY}`,
      'Content-Type': 'application/json',
      Accept: 'application/json',
      'NVCF-INPUT-ASSET-REFERENCES': assetId,
    },
    body: JSON.stringify(body),
  });
  const text = await r.text();
  console.log(`[${name}] ${r.status}: ${text.slice(0, 220)}`);
}
