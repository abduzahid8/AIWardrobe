/**
 * Real user-supplied archive: semantic fashion inspiration WITHOUT transmitting
 * original copyrighted/permission-unverified reference photographs.
 *
 * The 46 manual captions are approximate observations, not a vetted
 * professional taste benchmark or SKU/size/weather/merchant claim. The model
 * uses them to THINK about alternative whole-outfit concepts before making
 * its independent decisions from current, source-admitted merchandise.
 * No photo bytes, URLs, creator identity, filenames or private ID leave here.
 */
import referenceArchive from '../../benchmarks/stylist6/references/image_teaste_46.json' with {type:'json'};

export const EDITORIAL_TEXT_MOODBOARDS_VERSION='s6-user-reference-text-architecture-v1';
const SLOTS=['top','outerwear','bottom','shoes'];
const clean=(x,max=92)=>typeof x==='string'
  ?x.replace(/https?:\/\/\S+|\b[\w.%+-]+@[\w.-]+\.[a-z]{2,}\b/gi,'')
    .replace(/[\u0000-\u001f<>{}]/g,' ').replace(/\s+/g,' ').trim().slice(0,max):'';
const standardized=x=>clean(x,60).toLowerCase().replace(/[\s-]+/g,'_');
const suffix=x=>clean(x,75).toLowerCase().split(/[\s-]+/).at(-1)||'';
const words=x=>new Set(clean(x,75).toLowerCase().split(/[^a-z]+/g)
  .filter(token=>token.length>=4));
const overlap=(a,b)=>{
  const aa=words(a),bb=words(b);
  if(!aa.size||!bb.size)return 0;
  let shared=0;for(const v of aa)if(bb.has(v))shared++;
  return shared/(aa.size+bb.size-shared);
};

/**
 * Retrieve several distinct garment RELATIONSHIPS from the owner's manually
 * labeled lookbook. This is an inspiration prior, not a deterministic outfit
 * template. The true composer may ignore a sample if the live source SKUs,
 * user's explicitly excluded colors, weather or occasion cannot support it.
 */
export function editorialTextMoodboards({
  style,occasion,preferences={},limit=3,
}={}){
  const s=standardized(style),o=standardized(occasion);
  if(!s||!o||!Number.isInteger(limit)||limit<1||limit>4)return [];
  const disallowed=new Set((Array.isArray(preferences?.avoidColors)
    ?preferences.avoidColors:[]).map(standardized).filter(Boolean));
  const candidates=(Array.isArray(referenceArchive?.images)
    ?referenceArchive.images:[]).flatMap(row=>{
    const occasions=Array.isArray(row?.occasions)
      ?row.occasions.map(standardized):[];
    const styles=Array.isArray(row?.styles)?row.styles.map(standardized):[];
    // Use the exact user-requested occasion. An actual-date look is not
    // interchangeable with a workout, travel or wedding recommendation.
    if(!occasions.includes(o))return [];
    const garments=Object.fromEntries(SLOTS.map(slot=>[slot,
      clean(row?.garments?.[slot],74)]));
    if(SLOTS.some(slot=>garments[slot].length<6))return [];
    const palette=(Array.isArray(row?.palette)?row.palette:[])
      .map(x=>standardized(x)).filter(x=>x&&x.length<=24).slice(0,6);
    if(palette.some(x=>disallowed.has(x)) ||
      SLOTS.some(slot=>[...disallowed].some(c=>
        clean(garments[slot]).toLowerCase().split(/[^a-z]+/).includes(c))))
      return [];
    let styleScore=styles.includes(s)?10:0;
    if(!styleScore&&s==='classic'&&styles.includes('old_money'))styleScore=4;
    if(!styleScore&&s==='classic'&&styles.includes('smart_casual'))styleScore=2;
    if(!styleScore&&s==='semi_classic'&&
      styles.some(x=>['classic','old_money','smart_casual'].includes(x)))styleScore=3;
    if(!styleScore)return [];
    return [{styleScore,palette,garments,
      roleSignature:suffix(garments.outerwear),
      styleContext:styles.includes(s)?'matched':'adjacent'}];
  });
  if(!candidates.length)return [];
  const selected=[],remaining=[...candidates];
  while(selected.length<limit&&remaining.length){
    remaining.sort((a,b)=>{
      const quality=x=>{
        const sameLayer=selected.some(y=>y.roleSignature===x.roleSignature);
        const similarities=selected.map(y=>
          SLOTS.reduce((sum,slot)=>sum+overlap(y.garments[slot],x.garments[slot]),0)/4);
        const closest=similarities.length?Math.max(...similarities):0;
        return x.styleScore-(sameLayer?16:0)-closest*12;
      };
      return quality(b)-quality(a)||
        a.roleSignature.localeCompare(b.roleSignature)||
        a.garments.top.localeCompare(b.garments.top);
    });
    selected.push(remaining.shift());
  }
  return selected.map(x=>({
    // Human labels indicate an illustrated *idea*, never physical products.
    roleDescriptions:x.garments,paletteIdeas:x.palette,
    inspirationStyle:x.styleContext,
  }));
}
