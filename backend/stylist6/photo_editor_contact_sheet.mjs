/**
 * Independently photo-approved outfit candidates -> one visual editorial board.
 *
 * Each L-row shows real, previously authorized and independently evaluated
 * SOURCE PHOTOGRAPHS in a consistent top / layer / bottom / shoes grid.
 * The only labels rendered are opaque L-tokens and fixed slot names. No user
 * IDs, merchant URLs, garment names, product IDs or private text are embedded.
 *
 * A collection-level VLM can now compare real material/color/proportion and
 * architectural variety rather than judging the SAME text-only summaries.
 * All media comes from the trusted request-scoped visual-critic media cache;
 * missing/unauthorized bytes never become fictional photography.
 */
import sharp from 'sharp';

export const PHOTO_CONTACT_SHEET_VERSION='s6-real-photo-portfolio-sheet-v1';
export const PHOTO_CONTACT_SLOTS=['top','outerwear','bottom','shoes'];
const MAX_LOOKS=8;
// One fairly small real-photo editorial board per request, not a catalog PDF.
// The Render free CPU took >2.6s for the older 740px MozJPEG montage.
const CELL_W=160,CELL_H=160,HEADER=34,LABEL_W=44;
const WIDTH=LABEL_W+CELL_W*PHOTO_CONTACT_SLOTS.length;
const MAX_BYTES=540_000;
// MozJPEG runs expensive second-pass compression on a shared CPU and was
// timing out under live Render. Default libvips JPEG is much faster.
const jpegOptions={quality:68};

const svgText=(text,x,y,{size=15,bold=false,height}={})=>Buffer.from(
  `<svg width="${WIDTH}" height="${height}" xmlns="http://www.w3.org/2000/svg">
    <text x="${x}" y="${y}" font-family="Arial,Helvetica,sans-serif"
      font-size="${size}" font-weight="${bold?'700':'500'}" fill="#263141">${text}</text>
  </svg>`
);

/** No photography is fabricated. A failed fetch, stale media or missing
 * authorization returns null; the existing photo-evidence text editor takes
 * over rather than silently deleting a candidate or claiming to see an image.
 */
export async function buildPhotoEditorialContactSheet(looks,{
  resolvePhoto,authorizePhoto,deadlineMs=2200,signal,onStatus,
}={}){
  const note=status=>{try{onStatus?.(status);}catch{}};
  if(!Array.isArray(looks)||looks.length<3||looks.length>MAX_LOOKS||
    typeof resolvePhoto!=='function'||typeof authorizePhoto!=='function'||
    signal?.aborted){note('invalid_input');return null;}
  const sources=looks.map((look,index)=>{
    if(!Array.isArray(look?.items))return null;
    const seen=new Set();
    const cells=PHOTO_CONTACT_SLOTS.map(slot=>{
      const item=look.items.find(x=>x?.macroCategory===slot);
      if(!item)return slot==='outerwear'?null:false;
      const url=item.imageUrl||item.image_url;
      if(seen.has(slot)||typeof url!=='string'||!authorizePhoto(url))return false;
      seen.add(slot);
      return item;
    });
    return cells.includes(false)||['top','bottom','shoes'].some((s,i)=>
      !cells[PHOTO_CONTACT_SLOTS.indexOf(s)])?null:{token:'L'+String(index).padStart(2,'0'),cells};
  });
  if(sources.some(x=>!x)){note('unauthorized_or_missing');return null;}
  const controller=new AbortController();
  const timer=setTimeout(()=>controller.abort(),Math.min(4600,Math.max(300,deadlineMs)));
  const combined=signal?AbortSignal.any([signal,controller.signal]):controller.signal;
  const abortPromise=new Promise((_,reject)=>{
    if(combined.aborted)reject(Error('contact_deadline'));
    else combined.addEventListener('abort',()=>reject(Error('contact_deadline')),{once:true});
  });
  try{
    const height=HEADER+sources.length*CELL_H;
    const overlays=[];
    const labels=[
      svgText('TOP',LABEL_W+10,23,{size:14,bold:true,height}),
      svgText('LAYER',LABEL_W+CELL_W+10,23,{size:14,bold:true,height}),
      svgText('BOTTOM',LABEL_W+CELL_W*2+10,23,{size:14,bold:true,height}),
      svgText('SHOES',LABEL_W+CELL_W*3+10,23,{size:14,bold:true,height}),
    ].map(input=>({input,left:0,top:0}));
    overlays.push(...labels);
    const cells=await Promise.race([Promise.all(sources.flatMap((row,i)=>row.cells.map(async(item,j)=>{
      if(!item)return null;
      const asset=await resolvePhoto(item);
      if(!asset||!Buffer.isBuffer(asset.bytes)||asset.bytes.length<128||
        asset.bytes.length>512*1024)throw Error('missing_or_oversized_photo');
      const image=await sharp(asset.bytes,{limitInputPixels:4_200_000})
        .resize(CELL_W-10,CELL_H-27,{fit:'contain',background:'#f8fafc'})
        .jpeg({quality:63}).toBuffer();
      return {input:image,left:LABEL_W+j*CELL_W+5,top:HEADER+i*CELL_H+18};
    }))),abortPromise]);
    for(const cell of cells)if(cell)overlays.push(cell);
    if(combined.aborted){note('deadline');return null;}
    for(let i=0;i<sources.length;i++){
      overlays.push({input:svgText(sources[i].token,4,HEADER+i*CELL_H+33,
        {size:15,bold:true,height}),left:0,top:0});
      const missingLayer=sources[i].cells[1]===null;
      if(missingLayer)overlays.push({input:svgText('none',
        LABEL_W+CELL_W+12,HEADER+i*CELL_H+90,{size:13,height}),left:0,top:0});
    }
    const png=sharp({create:{width:WIDTH,height,channels:3,background:'#f0f3f7'}})
      .composite(overlays).jpeg(jpegOptions);
    const result=await Promise.race([png.toBuffer(),abortPromise]);
    if(combined.aborted){note('deadline');return null;}
    if(result.length>MAX_BYTES){note('oversized');return null;}
    note('ready');
    return {bytes:result,mimeType:'image/jpeg',width:WIDTH,height,
      lookTokens:sources.map(x=>x.token),
      version:PHOTO_CONTACT_SHEET_VERSION};
  }catch(error){
    note(combined.aborted?'deadline':
      error?.message==='missing_or_oversized_photo'?'source_missing':'photo_processing');
    return null;
  }finally{clearTimeout(timer);controller.abort();}
}
