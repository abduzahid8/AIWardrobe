/**
 * Privacy-minimal PHOTO critic training journal, not a personal taste store.
 *
 * Can persist only anonymized feature HMACs + independent original-photo
 * outcomes on source-authentic FOUR-SKU catalog-only outfits. Zero user IDs,
 * request IDs, merchant names, item IDs, image URLs, prompts or photo bytes.
 *
 * The staging server owns the HMAC key, rights policy, RLS-protected write
 * and time budget. The training dataset is NOT human-labelled fashion taste;
 * humans must still blind-review any resulting model before promotion.
 */
import {createHmac} from 'node:crypto';
import {outfitRelationshipFeatures} from './learned_outfit_preference.mjs';

export const PHOTO_REWARD_JOURNAL_VERSION='s6-photo-reward-journal-v1';
const VALID_REASONS=new Set([
  'approved','model_veto','below_quality_floor',
  'style_mismatch','occasion_mismatch','formality_mismatch',
]);
const SLOTS=['top','outerwear','bottom','shoes'];
// Security/consent boundary, NOT a styling rule: never persist custom
// requests, free-text occasions, company names or personal information as
// even pseudonymous context. These are the app's fixed UI codes only.
const SAFE_STYLES=new Set([
  'old_money','semi_classic','minimalist','casual','classic','business_casual',
]);
const SAFE_OCCASIONS=new Set(['work','casual','date','formal','sport','travel']);
// These are the EXACT three values from the app's typed preference schema.
// This is a privacy context discriminator, NOT a fashion decision rule.
const SAFE_FITS=new Set(['balanced','loose','fitted']);
const safeContext=(context)=>{
  const style=String(context?.style||'').trim().toLowerCase();
  const occasion=String(context?.occasion||'').trim().toLowerCase();
  const fit=String(context?.fitPreference||'balanced').trim().toLowerCase();
  return SAFE_STYLES.has(style)&&SAFE_OCCASIONS.has(occasion)&&SAFE_FITS.has(fit)
    ?{style,occasion,fit}:null;
};
const validQuality=q=>Number.isFinite(q)&&q>=0&&q<=1;
const hash=(key,type,value)=>createHmac('sha256',key)
  .update(PHOTO_REWARD_JOURNAL_VERSION+'\0'+type+'\0'+value).digest('hex');

/** Privacy boundary for generalized PHOTO data, never a fashion rule.
 * Fit 'balanced' with empty preference arrays is the actual app default;
 * it must NOT silently prevent all real staging catalog QA examples.
 */
export function anonymousPhotoJournalAllowed({sourceMode,requiredIds,preferences}={}){
  if(sourceMode!=='catalog'||!Array.isArray(requiredIds)||requiredIds.length>0)
    return false;
  const p=preferences&&typeof preferences==='object'&&!Array.isArray(preferences)
    ?preferences:{};
  const personalArrays=['avoidColors','favoriteColors','likedPatterns',
    'dislikedPatterns','favoriteBrands','avoidBrands','primaryOccasions',
    'styleGoals'];
  if(personalArrays.some(k=>Array.isArray(p[k])&&p[k].length>0))return false;
  if(['bodyType','workDressCode'].some(k=>typeof p[k]==='string'&&p[k].length>0))
    return false;
  if(p.prefersSustainable===true)return false;
  // Isolate the three real app fit modes in separate anonymized context
  // hashes; do not collapse the personal loose/fitted choice into neutral.
  if(p.fitPreference!==undefined&&p.fitPreference!==null&&
      !SAFE_FITS.has(String(p.fitPreference).trim().toLowerCase()))
    return false;
  const known=new Set([...personalArrays,'bodyType','workDressCode',
    'prefersSustainable','fitPreference','priceRange','stylePersonality']);
  if(Object.entries(p).some(([k,v])=>!known.has(k)&&v!==undefined&&v!==null))
    return false;
  return true;
}

/** Bounded, anonymized journal inputs ONLY from actual independent photo
 * reviews. The helper never fetches source images or connects to a DB.
 * Context hashes are deterministic *inside* the owner's server secret,
 * never across different deployments or user-provided salts.
 */
export function compilePhotoRewardExamples({
  looks,decisions,critiques,featuresOf,context,secret,
}={}){
  if(!Array.isArray(looks)||!Array.isArray(decisions)||
      !Array.isArray(critiques)||typeof featuresOf!=='function'||
      typeof secret!=='string'||secret.length<32||
      !safeContext(context))
    return [];
  const safe=safeContext(context);
  const contextHash=hash(secret,'context',
    safe.style+'\0'+safe.occasion+'\0'+safe.fit);
  const available=new Map(looks.slice(0,8).filter(look=>
    typeof look?.id==='string'&&
    Array.isArray(look.items)&&look.items.length===4&&
    SLOTS.every(slot=>look.items.filter(it=>it?.macroCategory===slot).length===1)&&
    look.items.every(it=>it.isShopItem===true&&
      typeof it.id==='string'&&it.id.length>0)&&
    new Set(look.items.map(it=>it.id)).size===4).map(look=>[look.id,look]));
  const seen=new Set(),byCritique=new Map();
  for(const c of critiques.slice(0,8)){
    if(!available.has(c?.candidateId)||typeof c.reject!=='boolean'||seen.has(c.candidateId)){
      if(typeof c?.candidateId==='string')byCritique.delete(c.candidateId);
      continue;
    }
    seen.add(c.candidateId);
    if(['visualCohesion','styleFit','occasionFit','formalityFit','confidence']
      .every(k=>validQuality(c[k])))byCritique.set(c.candidateId,c);
  }
  const rows=[],fingerprints=new Set();
  for(const d of decisions.slice(0,8)){
    if(!available.has(d?.candidateId)||!VALID_REASONS.has(d?.reason)||
        !validQuality(d.quality)||!byCritique.has(d.candidateId))continue;
    const c=byCritique.get(d.candidateId);
    if(d.reason==='approved'&&c.reject!==false)continue;
    if(d.reason==='model_veto'&&c.reject!==true)continue;
    const look=available.get(d.candidateId);
    let tokens;
    try{tokens=outfitRelationshipFeatures(featuresOf(look.items));}catch{continue}
    if(!Array.isArray(tokens)||tokens.length<12||tokens.length>180)continue;
    const featureHashes=[...new Set(tokens.map(t=>hash(secret,'feature',t)))].sort();
    const grade=Math.round(d.quality*10000)/10000;
    const outcome=d.reason==='approved'?1:-1;
    const fingerprint=hash(secret,'sample',
      contextHash+'\0'+featureHashes.join(',')+'\0'+outcome+'\0'+grade);
    if(fingerprints.has(fingerprint))continue;
    fingerprints.add(fingerprint);
    rows.push({
      fingerprint,
      context_hash:contextHash,
      feature_hashes:featureHashes,
      photo_quality:grade,
      photo_outcome:outcome,
      photo_reason:d.reason,
      evidence_version:PHOTO_REWARD_JOURNAL_VERSION,
    });
  }
  return rows.slice(0,8);
}
