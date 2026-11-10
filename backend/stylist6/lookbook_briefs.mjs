/**
 * Experimental lookbook-to-brief adapter (Stylist 6).
 *
 * Consumes manually-reviewed full-outfit blueprints, not Instagram data or
 * copyright-unverified raw photo files. Does not pretend to train a model or
 * guarantee the reference is fashionable. True image comparison is delegated
 * to the (still unimplemented) authenticated visual critic.
 *
 * Production inclusion requires media SHA256 verification and rights review.
 */
export const LOOKBOOK_BRIEFS_VERSION = 'lookbook-briefs-v1';

const normalize=(value)=>String(value||'').toLowerCase().trim()
  .replace(/[^a-z0-9_-]+/g,'_').replace(/[-_]+/g,'_');
const has=(source,term)=>String(source||'').toLowerCase().includes(term);
const unique=(values)=>[...new Set(values.filter(Boolean))];
function toGarmentSubtype(slot, label) {
  const v=String(label||'').toLowerCase();
  if(slot==='top'){
    if(v.includes('t-shirt')||v.includes('tee '))return ['t-shirt'];
    if(v.includes('polo'))return ['polo'];
    if(/turtleneck|roll-neck|mockneck|high-neck|crewneck knit|sweater|knit/.test(v))
      return ['fine-knit'];
    if(v.includes('shirt'))return ['shirt','dress-shirt'];
    return ['unknown'];
  }
  if(slot==='outerwear'){
    if(!label)return [];
    if(v.includes('cardigan'))return ['cardigan'];
    if(v.includes('blazer'))return ['blazer'];
    if(v.includes('coat'))return ['coat'];
    if(v.includes('overshirt')||v.includes('chore'))return ['overshirt','jacket'];
    if(v.includes('bomber'))return ['bomber'];
    if(v.includes('puffer')||v.includes('vest')||v.includes('gilet')||v.includes('quilted'))
      return ['jacket'];
    return ['jacket','overshirt'];
  }
  if(slot==='bottom'){
    if(v.includes('jeans')||v.includes('denim'))return ['jeans'];
    if(v.includes('chino'))return ['chino'];
    if(v.includes('tailored')||v.includes('pleated')||v.includes('trousers'))
      return ['tailored-trouser','trouser'];
    return ['trouser'];
  }
  if(slot==='shoes'){
    if(v.includes('loafer'))return ['loafer'];
    if(v.includes('chelsea'))return ['chelsea','boot'];
    if(v.includes('chukka')||v.includes('ankle boot'))return ['boot'];
    if(v.includes('sneaker'))return ['minimal-sneaker','sneaker'];
    if(v.includes('derby'))return ['derby'];
    return ['loafer','derby'];
  }
  return [];
}
function normalizeStyle(value) {
  const v=normalize(value);
  if(['old_money','summer_old_money','ivy_preppy'].includes(v))return 'old_money';
  if(['semi_classic','business_casual','smart_casual'].includes(v))return 'smart_casual';
  return v;
}
function matchStrength(ref,request) {
  const style=normalizeStyle(request.style), refStyles=(ref.styles||[]).map(normalizeStyle);
  const occasion=normalize(request.occasion),refOccasions=(ref.occasions||[]).map(normalize);
  const season=normalize(request.season);
  const mainStyle=refStyles.includes(style)?2.0:(style==='classic'&&refStyles.includes('old_money')?0.9:
    style==='old_money'&&refStyles.includes('classic')?0.75:0);
  const occasionScore=refOccasions.includes(occasion)?1.7:
    occasion==='work'&&refOccasions.includes('business_meeting')?1.1:
    occasion==='date_night'&&refOccasions.includes('smart_casual')?0.45:0;
  const seasonScore=season && (ref.seasons||[]).includes(season)?1.3:0;
  const layerPenalty=season==='summer'&&ref.layer_policy==='required'?-1.2:0;
  return mainStyle+occasionScore+seasonScore+layerPenalty;
}
function distance(a,b){
  const roles=['top','outerwear','bottom','shoes'];
  const fam=(r,s)=>toGarmentSubtype(s,r.garments?.[s])[0]||'absent';
  const roleDif=roles.filter(slot=>fam(a,slot)!==fam(b,slot)).length/roles.length;
  const aP=new Set(a.palette||[]),bP=new Set(b.palette||[]);
  const colors=[...aP,...bP];
  const palDif=colors.length?1-[...aP].filter(x=>bP.has(x)).length/new Set(colors).size:0;
  return 0.75*roleDif+0.25*palDif;
}
function isSafeRef(ref){
  return !!ref&&/^rt_\d{3}$/.test(ref.id)&&
    ref.complete_outfit_collage===true &&
    ref.fit_on_body_verified===false &&
    ref.garments&&typeof ref.garments==='object' &&
    ['top','bottom','shoes'].every(slot=>typeof ref.garments[slot]==='string'&&ref.garments[slot].length) &&
    ['required','absent'].includes(ref.layer_policy) &&
    (ref.layer_policy==='absent'||typeof ref.garments.outerwear==='string') &&
    Array.isArray(ref.palette)&&Array.isArray(ref.seasons)&&Array.isArray(ref.occasions);
}
/**
 * Produce experiment-only briefs with source traceability and deliberate
 * garment-architecture diversity. IDs and labels are never product SKUs.
 * This DOES NOT establish the quality of the references or enable training.
 */
function checkIndex(index){
  if(!index||index.dataset_id!=='image-teaste-2026-11-10' ||
      index.source!=='user_uploaded_flat_lay_reference_archive' ||
      !Array.isArray(index.images))throw Error('invalid_reference_index');
  if(index.images.length!==46||new Set(index.images.map(x=>x.id)).size!==46)
    throw Error('incomplete_reference_collection');
}
function toBrief(index,ref){
  return {
    id:ref.id,
    name:String(ref.title||ref.id).slice(0,140),
    referenceId:ref.id, // offloaded media vault resolves validated hash privately
    roles:{
      top:toGarmentSubtype('top',ref.garments.top),
      outerwear:toGarmentSubtype('outerwear',ref.garments.outerwear),
      bottom:toGarmentSubtype('bottom',ref.garments.bottom),
      shoes:toGarmentSubtype('shoes',ref.garments.shoes),
    },
    layerPolicy:ref.layer_policy,
    palette:unique(ref.palette.map(normalize)),
    silhouette:String(ref.silhouette||'balanced').slice(0,48),
    referenceEvidence:{sourceIndex:index.dataset_id,refId:ref.id,method:'manual_outfit_collage_review',
      confidence:'medium',isVisualImageComparison:false},
  };
}
/**
 * Exact single-reference brief for coverage checks and isolated visual
 * pilots. Same mapping as referenceFirstBriefs, without relevance ranking.
 */
export function briefForReferenceId(index,refId){
  checkIndex(index);
  const ref=index.images.find(x=>x&&x.id===refId);
  if(!ref||!isSafeRef(ref))throw Error('reference_not_usable');
  return toBrief(index,ref);
}
export function referenceFirstBriefs(index,request,limit=8){
  checkIndex(index);
  if(!request||typeof request.style!=='string'||typeof request.occasion!=='string')
    throw Error('invalid_stylist6_request');
  if(!Number.isInteger(limit)||limit<6||limit>12)throw Error('invalid_reference_limit');
  const pool=index.images.filter(isSafeRef);
  if(pool.length<limit)throw Error('insufficient_reviewed_references');
  const ranked=pool.map((ref)=>({ref,relevance:matchStrength(ref,request)}))
    .sort((a,b)=>b.relevance-a.relevance||a.ref.id.localeCompare(b.ref.id));
  const chosen=[];
  while(chosen.length<limit) {
    const next=ranked.filter(x=>!chosen.some(r=>r.id===x.ref.id))
      .map(x=>({ref:x.ref,score:x.relevance+0.95*(chosen.length?
        Math.min(...chosen.map(c=>distance(c,x.ref))):0.5)}))
      .sort((a,b)=>b.score-a.score||a.ref.id.localeCompare(b.ref.id))[0];
    if(!next)break;
    chosen.push(next.ref);
  }
  if(chosen.length<6)throw Error('insufficient_reference_diversity');
  return chosen.map(ref=>toBrief(index,ref));
}
