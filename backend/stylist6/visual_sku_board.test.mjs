import test from 'node:test';
import assert from 'node:assert/strict';
import sharp from 'sharp';
import {buildVisualSkuBoard,VISUAL_SKU_BOARD_VERSION} from './visual_sku_board.mjs';

const roles=['top','outerwear','bottom','shoes'];
const pool=roles.flatMap((slot,j)=>Array.from({length:8},(_,i)=>({
  token:'P'+String(j*8+i).padStart(3,'0'),slot,
  item:{id:'PRIVATE_PRODUCT_'+slot+i,isShopItem:true,macroCategory:slot,
    imageUrl:'https://catalog.example/image/'+slot+i+'.jpg'},
})));
const photo=await sharp({create:{width:145,height:175,
  channels:3,background:'#ded3c3'}}).jpeg({quality:72}).toBuffer();
test('vision composer gets a single board of ACTUAL authorized product-image bytes and opaque tokens',async()=>{
  let fetches=0,auth=0;
  const board=await buildVisualSkuBoard(pool,{
    authorizePhoto:()=>{auth++;return true;},
    resolvePhoto:async()=>{fetches++;return {bytes:photo};},
  });
  assert.equal(VISUAL_SKU_BOARD_VERSION,'s6-original-photo-merchandise-board-v1');
  assert.ok(board?.bytes?.length>1000);
  assert.equal(board?.photoCount,32);
  assert.equal(fetches,32);
  assert.equal(auth,32);
  assert.equal(new Set(board.seenTokens).size,32);
  const meta=await sharp(board.bytes).metadata();
  assert.equal(meta.format,'jpeg');
  assert.equal(meta.width,board.width);
  assert.equal(meta.height,board.height);
  assert.ok(board.bytes.length<650000);
});
test('unavailable/unlicensed original merchant photos NEVER enter external vision inference',async()=>{
  let fetched=0,reason='';
  const candidate=await buildVisualSkuBoard(pool,{
    authorizePhoto:url=>!url.endsWith('outerwear0.jpg')&&!url.endsWith('outerwear1.jpg')
      &&!url.endsWith('outerwear2.jpg'),
    resolvePhoto:async()=>{fetched++;return {bytes:photo};},
    onStatus:x=>{reason=x;},
  });
  assert.equal(candidate,null);
  assert.equal(reason,'insufficient_photo_inventory');
  assert.equal(fetched,29,'never fetch a user-unlicensed image');
});
test('hanging external photo cannot prevent fallback or invent an image board',async()=>{
  const at=Date.now();let reason='';
  const image=await buildVisualSkuBoard(pool,{
    authorizePhoto:()=>true,resolvePhoto:async()=>new Promise(()=>{}),
    deadlineMs:350,onStatus:x=>{reason=x;},
  });
  assert.equal(image,null);
  assert.equal(reason,'deadline');
  assert.ok(Date.now()-at<1300);
});
