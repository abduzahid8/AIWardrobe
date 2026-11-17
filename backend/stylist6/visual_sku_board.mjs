/**
 * Real product-photograph contact board for autonomous AI fashion COMPOSITION.
 * Each product shows only its opaque P-token and the garment-role column.
 * No merchant URLs, user identifiers, source SKU names or annotations are
 * embedded in the image. Independent source-photo authorization is required.
 *
 * A photo-less product is excluded from the visual model's allowed token set.
 * It can still be used by the legacy source-bound text composer if the
 * entire visual-board path is unavailable. Nothing here approves an outfit.
 */
import sharp from 'sharp';

export const VISUAL_SKU_BOARD_VERSION='s6-original-photo-merchandise-board-v1';
const SLOTS=['top','outerwear','bottom','shoes'];
const COL_W=172,ROW_H=128,HEADER=38,PHOTO_W=144,PHOTO_H=100;
const MAX_MERCHANT_IMAGE_BYTES=512*1024;
const MAX_BOARD_IMAGE_BYTES=650_000;
const X=COL_W*SLOTS.length;

const tokenOk=x=>typeof x==='string'&&/^P\d{1,4}$/.test(x);
const safeUrl=x=>typeof x==='string'&&/^https:\/\//.test(x);

export async function buildVisualSkuBoard(pool,{
  resolvePhoto,authorizePhoto,deadlineMs=4000,signal,onStatus,
}={}){
  const note=x=>{try{onStatus?.(x);}catch{}};
  if(!Array.isArray(pool)||pool.length<24||pool.length>48||
     typeof resolvePhoto!=='function'||typeof authorizePhoto!=='function'||
     signal?.aborted){note('invalid_input');return null;}
  const rows=new Map(SLOTS.map(slot=>[slot,[]]));
  const uniqueTokens=new Set(),uniqueProducts=new Set();
  for(const value of pool){
    if(!SLOTS.includes(value?.slot)||!tokenOk(value?.token)||
       !value?.item||!value.item.isShopItem||
       uniqueTokens.has(value.token)||uniqueProducts.has(String(value.item.id||''))||
       rows.get(value.slot).length>=12){note('invalid_input');return null;}
    uniqueTokens.add(value.token);
    uniqueProducts.add(String(value.item.id||''));
    rows.get(value.slot).push(value);
  }
  const raw=Array.from(rows.values());
  if(raw.some(a=>a.length<6)){note('insufficient_inventory');return null;}
  const all=raw.flat();
  const controller=new AbortController();
  const ms=Math.max(350,Math.min(5000,Math.floor(deadlineMs)));
  const timer=setTimeout(()=>controller.abort(),ms);
  const combined=signal?AbortSignal.any([signal,controller.signal]):controller.signal;
  const untilAbort=new Promise((_,reject)=>{
    if(combined.aborted)reject(Error('visual_board_deadline'));
    else combined.addEventListener('abort',()=>reject(Error('visual_board_deadline')),{once:true});
  });
  try{
    const cells=await Promise.race([Promise.allSettled(all.map(async value=>{
      const url=value.item.imageUrl||value.item.image_url;
      if(!safeUrl(url)||!authorizePhoto(url))return null;
      const media=await resolvePhoto(value.item);
      if(!media||!Buffer.isBuffer(media.bytes)||
        media.bytes.length<128||media.bytes.length>MAX_MERCHANT_IMAGE_BYTES)return null;
      const bytes=await sharp(media.bytes,{limitInputPixels:4_200_000})
        .resize(PHOTO_W,PHOTO_H,{fit:'contain',background:'#f8f9fb'})
        .jpeg({quality:58}).toBuffer();
      return {token:value.token,slot:value.slot,bytes};
    })),untilAbort]);
    if(combined.aborted){note('deadline');return null;}
    const successful=new Map(cells.flatMap((result,i)=>result.status==='fulfilled'&&result.value
      ?[[all[i].token,result.value]]:[]));
    const permitted=new Set(successful.keys());
    if(raw.some(items=>items.filter(x=>permitted.has(x.token)).length<6)){
      note('insufficient_photo_inventory');return null;
    }
    // A single plain SVG overlays role labels and opaque tokens. Do NOT
    // generate one full-sheet SVG per token (expensive on Render CPU).
    const rowCount=Math.max(...raw.map(x=>x.length));
    const height=HEADER+rowCount*ROW_H;
    const overlay=[],labels=[];
    for(let col=0;col<SLOTS.length;col++){
      const label=SLOTS[col].toUpperCase();
      labels.push('<text x="'+(col*COL_W+10)+'" y="25">'+label+'</text>');
      for(let i=0;i<raw[col].length;i++){
        const value=raw[col][i];
        const matched=successful.get(value.token);
        if(!matched)continue;
        overlay.push({input:matched.bytes,
          left:col*COL_W+12,top:HEADER+i*ROW_H+4});
        labels.push('<text x="'+(col*COL_W+12)+'" y="'+
          (HEADER+i*ROW_H+ROW_H-8)+'">'+value.token+'</text>');
      }
    }
    const svg=Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="'+
      X+'" height="'+height+'"><g fill="#1b2635" font-family="Arial" '+
      'font-size="13" font-weight="700">'+labels.join('')+'</g></svg>');
    overlay.push({input:svg,left:0,top:0});
    const bytes=await Promise.race([
      sharp({create:{width:X,height,channels:3,background:'#eef0f4'}})
        .composite(overlay).jpeg({quality:67}).toBuffer(),untilAbort]);
    if(combined.aborted){note('deadline');return null;}
    if(bytes.length>MAX_BOARD_IMAGE_BYTES){note('oversized');return null;}
    note('ready');
    return {bytes,mimeType:'image/jpeg',version:VISUAL_SKU_BOARD_VERSION,
      seenTokens:[...permitted],width:X,height,
      photoCount:permitted.size};
  }catch{
    note(combined.aborted?'deadline':'media_unavailable');
    return null;
  }finally{clearTimeout(timer);controller.abort();}
}
