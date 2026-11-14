import test from 'node:test';
import assert from 'node:assert/strict';
import sharp from 'sharp';
import { createHash } from 'node:crypto';
import {
  prepareCloudVisualMedia, cloudVisualMediaClass,
  CLOUD_VISUAL_MAX_INPUT_BYTES, CLOUD_VISUAL_MAX_OUTPUT_BYTES,
} from './cloud_visual_media.mjs';
import { verifiedVisualImage } from './visual_critic_hosted.mjs';

async function example(format, { width = 800, height = 600 } = {}) {
  const image=sharp({ create: {
    width,height,channels:4,background:{r:180,g:55,b:45,alpha:.6},
  }});
  if (format === 'png') return image.png().toBuffer();
  if (format === 'webp') return image.webp({ quality:90 }).toBuffer();
  if (format === 'avif') return image.avif({quality:60}).toBuffer();
  if (format === 'jpeg') return image.jpeg({quality:90}).toBuffer();
  throw new Error('invalid_fixture_format');
}

test('real png, webp, jpeg and avif photos become bounded decoder-valid JPEG bytes', async()=>{
  for (const format of ['png','webp','jpeg','avif']) {
    const input=await example(format);
    assert.equal(cloudVisualMediaClass(input),format==='avif'?'heif_avif':format);
    const jpeg=await prepareCloudVisualMedia(input);
    assert.ok(Buffer.isBuffer(jpeg),format);
    assert.ok(jpeg.length<CLOUD_VISUAL_MAX_OUTPUT_BYTES,format);
    assert.equal(cloudVisualMediaClass(jpeg),'jpeg');
    const outputMeta=await sharp(jpeg).metadata();
    assert.ok(outputMeta.width<=1024 && outputMeta.height<=1024);
    assert.equal(outputMeta.format,'jpeg');
    assert.equal(outputMeta.hasAlpha,false);
    const item=verifiedVisualImage({
      bytes:jpeg,sha256:createHash('sha256').update(jpeg).digest('hex'),
    });
    assert.equal(item.mimeType,'image/jpeg');
    assert.equal(Buffer.from(item.data,'base64').equals(jpeg),true);
  }
});

test('large but bounded webp resizes to 1024 edge and 512KiB image cap', async()=>{
  const input=await example('webp',{width:3600,height:2400});
  assert.ok(input.length<=CLOUD_VISUAL_MAX_INPUT_BYTES);
  const jpeg=await prepareCloudVisualMedia(input);
  assert.ok(jpeg && jpeg.length<=CLOUD_VISUAL_MAX_OUTPUT_BYTES);
  const meta=await sharp(jpeg).metadata();
  assert.ok(meta.width<=1024 && meta.height<=1024);
});

test('input cap, invalid bytes, vector and tiny/icon images fail closed', async()=>{
  assert.equal(await prepareCloudVisualMedia(null),null);
  assert.equal(await prepareCloudVisualMedia(Buffer.alloc(128,0)),null);
  assert.equal(await prepareCloudVisualMedia(Buffer.alloc(CLOUD_VISUAL_MAX_INPUT_BYTES+1)),null);
  assert.equal(await prepareCloudVisualMedia(Buffer.from('<svg width="300" height="300"></svg>'.repeat(10))),null);
  assert.equal(await prepareCloudVisualMedia(Buffer.alloc(130,0x89)),null);
  assert.equal(await prepareCloudVisualMedia(await example('png',{width:32,height:32})),null);
  assert.equal(await prepareCloudVisualMedia(await example('jpeg'),{maxBytes:1}),null);
});

test('untrusted seller EXIF/metadata is not included in normalized inference JPEG',async()=>{
  const source=await sharp({
    create:{width:640,height:480,channels:3,background:'#263648'},
  }).jpeg({quality:85}).withMetadata({orientation:6,exif:{
    IFD0:{ImageDescription:'UNTRUSTED-SELLER-EXIF-MESSAGE'},
  }}).toBuffer();
  const result=await prepareCloudVisualMedia(source);
  assert.ok(result);
  const metadata=await sharp(result).metadata();
  assert.equal(metadata.orientation,undefined);
  assert.equal(metadata.exif,undefined);
  assert.equal(metadata.width,480);
  assert.equal(metadata.height,640);
  assert.ok(!result.includes(Buffer.from('UNTRUSTED-SELLER-EXIF-MESSAGE')));
});
