import test from 'node:test';
import assert from 'node:assert/strict';
import sharp from 'sharp';
import {buildPhotoEditorialContactSheet,PHOTO_CONTACT_SHEET_VERSION}
  from './photo_editor_contact_sheet.mjs';

const slots=['top','outerwear','bottom','shoes'];
const looks=Array.from({length:5},(_,i)=>({
  id:'private-user-identifier-'+i,
  items:slots.filter((slot)=>i%2===0||slot!=='outerwear')
    .map((macroCategory,j)=>({
      id:'PRIVATE_SKU_SECRET_'+i+'_'+j,
      macroCategory,imageUrl:'https://catalog-photos.example/'+i+'/'+j+'.jpg',
      name:'PRIVATE USER GARMENT',
    })),
}));
const jpeg=await sharp({create:{width:120,height:130,channels:3,background:'#657c8a'}})
  .jpeg({quality:75}).toBuffer();

test('one exact original-photo collage contains real garments in fixed four-slot L rows',async()=>{
  let calls=0,auth=0;
  const sheet=await buildPhotoEditorialContactSheet(looks,{
    resolvePhoto:async()=>{calls++;return {bytes:jpeg};},
    authorizePhoto:url=>{auth++;return url.startsWith('https://catalog-photos.example/');},
  });
  assert.equal(PHOTO_CONTACT_SHEET_VERSION,'s6-real-photo-portfolio-sheet-v1');
  assert.ok(sheet,'all preapproved original merchant photographs must be representable');
  assert.deepEqual(sheet.lookTokens,['L00','L01','L02','L03','L04']);
  assert.equal(calls,18,'all 4+3+4+3+4 whole-outfit photos, never invented layers');
  assert.equal(auth,18,'each independent source image re-authorized');
  const meta=await sharp(sheet.bytes).metadata();
  assert.equal(meta.format,'jpeg');
  assert.equal(meta.width,sheet.width);
  assert.equal(meta.height,sheet.height);
  assert.ok(sheet.bytes.length<540000);
});
test('a missing or unlicensed source photograph can NEVER become visual inference evidence',async()=>{
  let fetches=0;
  const unauthorized=await buildPhotoEditorialContactSheet(looks,{
    resolvePhoto:async()=>{fetches++;return {bytes:jpeg};},
    authorizePhoto:url=>!url.includes('/1/'),
  });
  assert.equal(unauthorized,null);
  assert.equal(fetches,0,'do not fetch any media after eligibility fails');
  const missing=await buildPhotoEditorialContactSheet(looks,{
    resolvePhoto:async item=>item.imageUrl.endsWith('/2/0.jpg')
      ?null:{bytes:jpeg},
    authorizePhoto:()=>true,
  });
  assert.equal(missing,null);
  assert.equal(await buildPhotoEditorialContactSheet(looks.slice(0,1),{
    resolvePhoto:async()=>({bytes:jpeg}),authorizePhoto:()=>true,
  }),null,'a single outfit is not a collection and cannot imply a missing peer');
});
test('contact sheet media preparation has a hard finite deadline',async()=>{
  const before=Date.now();
  const sheet=await buildPhotoEditorialContactSheet(looks,{
    resolvePhoto:async()=>new Promise(()=>{}),
    authorizePhoto:()=>true,deadlineMs:300,
  });
  assert.equal(sheet,null);
  assert.ok(Date.now()-before<1600,'no CDN stall can exhaust a 72s styling request');
});


test('exact two source-approved looks create a REAL non-fabricated 2-row photo collage',async()=>{
  let reads=0;
  const sheet=await buildPhotoEditorialContactSheet(looks.slice(0,2),{
    resolvePhoto:async()=>{reads++;return {bytes:jpeg};},
    authorizePhoto:url=>url.startsWith('https://catalog-photos.example/'),
  });
  assert.ok(sheet,'the fashion editor must see TWO approved outfit photographs');
  assert.deepEqual(sheet.lookTokens,['L00','L01']);
  assert.equal(reads,7,'four real photos plus three real photos (optional layer absent)');
  const meta=await sharp(sheet.bytes).metadata();
  assert.equal(meta.format,'jpeg');
  assert.equal(meta.height,34+160*2,'exactly two image rows, no invented third');
  assert.ok(sheet.bytes.length<540000);
});

test('large EIGHT-look real-photograph portfolio builds within the same bounded mobile edit window without 12 separate SVG canvases',async()=>{
  const realLookSlots=['top','outerwear','bottom','shoes'];
  const allEight=Array.from({length:8},(_,i)=>({
    id:'PRIVATE_REAL_ORIGINAL_'+i,
    items:realLookSlots.map((slot,j)=>({
      id:'SECRET_PRODUCT_'+i+'_'+j,macroCategory:slot,
      imageUrl:'https://catalog-photos.example/true-product-'+i+'-'+j+'.jpg',
    })),
  }));
  const largeJpeg=await sharp({create:{width:640,height:840,
    channels:3,background:'#9eaabc'}}).jpeg({quality:88}).toBuffer();
  let sourceReads=0;
  const started=Date.now();
  const sheet=await buildPhotoEditorialContactSheet(allEight,{
    resolvePhoto:async()=>{sourceReads++;return {bytes:largeJpeg};},
    authorizePhoto:url=>url.startsWith('https://catalog-photos.example/'),
    deadlineMs:3800,
  });
  assert.ok(sheet,'CPU montage must finish inside the REAL stage PHOTO jury deadline');
  assert.equal(sourceReads,32,'every one of 32 real original source images must contribute');
  assert.equal(sheet.lookTokens.length,8);
  const info=await sharp(sheet.bytes).metadata();
  assert.equal(info.height,34+8*160);
  assert.equal(info.width,44+160*4);
  assert.ok(sheet.bytes.length<540000);
  assert.ok(Date.now()-started<3800,
    '8-look JPEG montage must complete before source-authorized editorial jury times out');
});
