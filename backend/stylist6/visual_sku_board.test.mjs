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
  assert.ok(board?.photoCount>=24&&board.photoCount<=32);
  assert.equal(fetches,32);
  assert.equal(auth,32);
  assert.equal(new Set(board.seenTokens).size,board.photoCount);
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


test('visual fashion board finishes on per-slot photo QUORUM without waiting for stalled shops',async()=>{
  const began=Date.now();let status='';
  const board=await buildVisualSkuBoard(pool,{
    authorizePhoto:()=>true,
    resolvePhoto:async item=>/[67]$/.test(item.id)
      ?new Promise(()=>{}):{bytes:photo},
    deadlineMs:1300,onStatus:x=>{status=x;},
  });
  assert.equal(status,'ready');
  assert.ok(board?.photoCount>=24);
  assert.equal(board?.seenTokens.length,24,
    'exactly six original merchant photos per garment role are enough');
  assert.ok(Date.now()-began<1000,
    'slow irrelevant merchants must never block a valid creative decision');
  assert.equal(new Set(board.seenTokens).size,24);
});

test('three real image choices per role allow a genuine complete visual fashion design',async()=>{
  const start=Date.now();const selected=[];
  const sheet=await buildVisualSkuBoard(pool,{
    minimumPerSlot:3,
    deadlineMs:1400,
    authorizePhoto:()=>true,
    resolvePhoto:async item=>{
      const n=Number(item.id.slice(-1));
      if(n>=3)return new Promise(()=>{});
      return {bytes:photo};
    },
    onStatus:x=>selected.push(x),
  });
  assert.equal(selected.at(-1),'ready');
  assert.equal(sheet.photoCount,12);
  assert.ok(Date.now()-start<1000,
    'twelve genuine photographs should not wait for 20 blocked source CDNs');
  const counts=roles.map(slot=>pool.filter(p=>
    p.slot===slot&&sheet.seenTokens.includes(p.token)).length);
  assert.deepEqual(counts,[3,3,3,3]);
});
