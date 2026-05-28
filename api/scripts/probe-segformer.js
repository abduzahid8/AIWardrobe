import 'dotenv/config';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  AutoModel,
  AutoProcessor,
  RawImage,
  env,
} from '@huggingface/transformers';
import sharp from 'sharp';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
env.cacheDir = path.join(process.cwd(), 'cache', 'hf');

const MODEL_ID = 'mattmdjaga/segformer_b2_clothes';

const url = 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg';
const r = await fetch(url);
const srcBuf = Buffer.from(await r.arrayBuffer());

const model = await AutoModel.from_pretrained(MODEL_ID);
const processor = await AutoProcessor.from_pretrained(MODEL_ID);
console.log('model.config.id2label:', model.config?.id2label);

const { data: rgbaData, info } = await sharp(srcBuf).resize(1024, 1024, { fit: 'inside' }).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
const W = info.width; const H = info.height;
const rgb = Buffer.alloc(W*H*3);
for (let i=0,j=0;i<W*H;i++,j+=3) { rgb[j]=rgbaData[i*4]; rgb[j+1]=rgbaData[i*4+1]; rgb[j+2]=rgbaData[i*4+2]; }
const rawImage = new RawImage(new Uint8Array(rgb), W, H, 3);
const inputs = await processor(rawImage);
console.log('processor inputs keys:', Object.keys(inputs));
console.log('pixel_values dims:', inputs.pixel_values?.dims);

const output = await model({ pixel_values: inputs.pixel_values });
console.log('output keys:', Object.keys(output));
const logits = output.logits;
console.log('logits dims:', logits.dims);

const dims = logits.dims;
const numClasses = dims[1]; const lh = dims[2]; const lw = dims[3];
const ld = logits.data;
console.log(`logits: ${numClasses} classes, ${lh}x${lw}, total bytes ${ld.length}`);

// histogram of argmax classes
const counts = new Array(numClasses).fill(0);
for (let i = 0; i < lh * lw; i++) {
  let bestC = 0, bestV = -Infinity;
  for (let c = 0; c < numClasses; c++) {
    const v = ld[c * lh * lw + i];
    if (v > bestV) { bestV = v; bestC = c; }
  }
  counts[bestC]++;
}
console.log('class histogram:', counts.map((n, i) => `${i}=${n}`).join(' '));
