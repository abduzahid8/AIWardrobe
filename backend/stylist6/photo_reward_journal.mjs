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
// This is a privacy boundary, NEVER a styling rule. The independent
// PHOTO judge may legitimately have evaluated a personalized request:
// do not mislabel that approval as universal aesthetic truth. Instead
// separate every supported non-identifying preference profile using a
// keyed, one-way HMAC context. No raw preference, free-text brief, body
// attribute, user ID, URL or merchant/SKU identity is persisted.
const SAFE_STYLES=new Set([
  'old_money','semi_classic','minimalist','casual','classic','business_casual',
]);
const SAFE_OCCASIONS=new Set(['work','casual','date','formal','sport','travel']);
const SAFE_FITS=new Set(['balanced','loose','fitted']);
const PHOTO_PREF_ARRAYS=new Set([
  'avoidColors','favoriteColors','likedPatterns','dislikedPatterns',
  'favoriteBrands','avoidBrands','primaryOccasions','styleGoals',
]);
const PHOTO_PREF_TEXT=new Set([
  'bodyType','workDressCode','stylePersonality',
]);
const PHOTO_PREF_BOOLEAN=new Set(['prefersSustainable']);
const PHOTO_PREF_BUDGET=new Set(['priceRange','maxShopBudget']);
const PHOTO_PREF_KNOWN=new Set([
  ...PHOTO_PREF_ARRAYS,...PHOTO_PREF_TEXT,...PHOTO_PREF_BOOLEAN,
  ...PHOTO_PREF_BUDGET,'fitPreference','budgetRatesId',
]);
const PHOTO_BUDGET_KEYS=new Set(['min','max','amount','currency']);
const safeString=v=>{
  if(typeof v!=='string'||v.length>100||
    /[\x00-\x1f<>@]/.test(v)||/:\/\//.test(v))return null;
  const norm=v.trim().toLowerCase();
  return norm.length?norm:null;
};
function normalizedPhotoPrefs(input,fitOverride){
  if(input!==undefined&&input!==null&&
      (typeof input!=='object'||Array.isArray(input)))return null;
  const p=input||{},out={};
  const fit=safeString(p.fitPreference??fitOverride??'balanced');
  if(!fit||!SAFE_FITS.has(fit))return null;
  out.fitPreference=fit;
  for(const key of Object.keys(p).sort()){
    const value=p[key];
    if(!PHOTO_PREF_KNOWN.has(key)){
      if(value!==undefined&&value!==null)return null;
      continue;
    }
    if(key==='fitPreference'||key==='budgetRatesId'||
        value===undefined||value===null)continue;
    if(PHOTO_PREF_ARRAYS.has(key)){
      if(!Array.isArray(value)||value.length>20)return null;
      const vals=value.map(safeString);
      if(vals.some(x=>x===null))return null;
      if(vals.length)out[key]=[...new Set(vals)].sort();
    }else if(PHOTO_PREF_TEXT.has(key)){
      const val=safeString(value);
      if(!val)return null;
      out[key]=val;
    }else if(PHOTO_PREF_BOOLEAN.has(key)){
      if(typeof value!=='boolean')return null;
      if(value)out[key]=true;
    }else if(PHOTO_PREF_BUDGET.has(key)){
      if(!value||typeof value!=='object'||Array.isArray(value)||
          Object.keys(value).length>4)return null;
      const budget={};
      for(const bkey of Object.keys(value).sort()){
        if(!PHOTO_BUDGET_KEYS.has(bkey))return null;
        const item=value[bkey];
        if(item===undefined||item===null)continue;
        if(bkey==='currency'){
          const str=safeString(item);
          if(!str||!/^[a-z]{3}$/.test(str))return null;
          budget[bkey]=str;
        }else{
          if(typeof item!=='number'||!Number.isFinite(item)||
              item<0||item>1e12)return null;
          budget[bkey]=Math.round(item*100)/100;
        }
      }
      if(Object.keys(budget).length)out[key]=budget;
    }
  }
  return out;
}
const safeContext=context=>{
  // Match the APP's canonical UI style/occasion codes, NOT arbitrary
  // free-text hints. "Old Money", "old-money" and "old_money" are one
  // scenario; "Date night" must not silently become an untrainable label.
  const style=safeString(context?.style)?.replace(/[\s-]+/g,'_');
  const rawOccasion=safeString(context?.occasion);
  const occasion=['date night','date_night'].includes(rawOccasion)
    ?'date':rawOccasion;
  const preferences=normalizedPhotoPrefs(context?.preferences,
    context?.fitPreference);
  return style&&occasion&&preferences&&
    SAFE_STYLES.has(style)&&SAFE_OCCASIONS.has(occasion)
    ?{style,occasion,preferences}:null;
};
const validQuality=q=>Number.isFinite(q)&&q>=0&&q<=1;
const hash=(key,type,value)=>createHmac('sha256',key)
  .update(PHOTO_REWARD_JOURNAL_VERSION+'\0'+type+'\0'+value).digest('hex');
/** The ONLY stored representation of personalized PHOTO context. Keyed,
 * bounded, stable across requests with equivalent preferences and opaque
 * across distinct profiles. Never emit its unhashed canonical source.
 */
export function photoRewardContextHash(context,secret){
  if(typeof secret!=='string'||secret.length<32)return null;
  const normalized=safeContext(context);
  return normalized?hash(secret,'context',JSON.stringify(normalized)):null;
}
/** Accept actual styled catalog PHOTO verdicts, but never without a
 * contextual HMAC isolation guarantee. This is NOT authorization to read
 * a personal wardrobe or copy original image bytes.
 */
export function anonymousPhotoJournalAllowed({sourceMode,requiredIds,preferences}={}){
  return sourceMode==='catalog'&&Array.isArray(requiredIds)&&
    requiredIds.length===0&&normalizedPhotoPrefs(preferences)!==null;
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
      typeof secret!=='string'||secret.length<32)
    return [];
  const contextHash=photoRewardContextHash(context,secret);
  if(!contextHash)return [];
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
