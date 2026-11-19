/**
 * Cross-request PHOTO fashion reward trained only on authentic anonymous
 * independent whole-outfit photo outcomes. Unlike owner preference learning,
 * these labels are AI photo-critic supervision, never called human taste.
 *
 * Input is HMAC-hashed source-grounded four-slot relations from an RLS private
 * short-lived staging table: no merchant SKUs, owners, photographs or prompts.
 * A PAIRWISE learned model may help decide which real outfit to independently
 * PHOTO-review first. It can NEVER approve an outfit, relax stock/source
 * feasibility or replace human blinded fashion evaluation.
 *
 * The model ABSTAINS on insufficient or uncalibrated independent evidence.
 */
import {createHash,createHmac} from 'node:crypto';
import {outfitRelationshipFeatures} from './learned_outfit_preference.mjs';
import {PHOTO_REWARD_JOURNAL_VERSION} from './photo_reward_journal.mjs';
export const PHOTO_REWARD_HISTORY_VERSION='s6-photo-history-pairwise-v1';
const DIGEST=/^[0-9a-f]{64}$/;
const MAX_ROWS=240,MAX_FEATURES=180,MIN_FEATURES=12;
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const neutral=(status,extra={})=>({version:PHOTO_REWARD_HISTORY_VERSION,
  status,usable:false,examples:0,positive:0,negative:0,holdoutWins:0,
  holdoutPairs:0,weights:new Map(),...extra});
const score=(weights,features)=>features.reduce((sum,f)=>sum+(weights.get(f)||0),0);
const gradient=difference=>1/(1+Math.exp(clamp(difference,-30,30)));
const sha=s=>createHash('sha256').update(s).digest('hex');
const hmac=(secret,type,value)=>createHmac('sha256',secret)
  .update(PHOTO_REWARD_JOURNAL_VERSION+'\0'+type+'\0'+value).digest('hex');
const SAFE_STYLES=new Set(['old_money','semi_classic','minimalist','casual',
  'classic','business_casual']);
const SAFE_OCCASIONS=new Set(['work','casual','date','formal','sport','travel']);
const SAFE_FITS=new Set(['balanced','loose','fitted']);
/** Match the journal's exact HMAC domain without exposing the shared secret
 * or accepting arbitrary user-provided free-text outfit scenarios. */
export function photoHistoryContextHash(context,secret){
  if(typeof secret!=='string'||secret.length<32)return null;
  const style=String(context?.style||'').trim().toLowerCase();
  const occasion=String(context?.occasion||'').trim().toLowerCase();
  const fit=String(context?.fitPreference||'balanced').trim().toLowerCase();
  return SAFE_STYLES.has(style)&&SAFE_OCCASIONS.has(occasion)&&SAFE_FITS.has(fit)
    ?hmac(secret,'context',style+'\0'+occasion+'\0'+fit):null;
}
/** Extract ONLY source-grounded outfit relations already available to the
 * fashion engine; emit stable keyed fingerprints, not raw item identities. */
export function photoHistoryFeatureHashes(items,featuresOf,secret){
  if(typeof secret!=='string'||secret.length<32||
      !Array.isArray(items)||items.length!==4||typeof featuresOf!=='function')
    return null;
  let tokens;
  try{tokens=outfitRelationshipFeatures(featuresOf(items));}catch{return null}
  return Array.isArray(tokens)&&tokens.length>=12&&tokens.length<=180
    ?[...new Set(tokens.map(t=>hmac(secret,'feature',t)))].sort():null;
}
function evidence(row,contextHash){
  if(!row||typeof row!=='object'||Array.isArray(row)||
      row.context_hash!==contextHash||row.evidence_version!=='s6-photo-reward-journal-v1'||
      !DIGEST.test(String(row.fingerprint||''))||
      !Array.isArray(row.feature_hashes)||
      row.feature_hashes.length<MIN_FEATURES||row.feature_hashes.length>MAX_FEATURES||
      row.feature_hashes.some(x=>typeof x!=='string'||!DIGEST.test(x))||
      ![-1,1].includes(row.photo_outcome)||
      !Number.isFinite(Number(row.photo_quality))||
      Number(row.photo_quality)<0||Number(row.photo_quality)>1)
    return null;
  const features=[...new Set(row.feature_hashes)].sort();
  if(features.length<MIN_FEATURES)return null;
  const key=sha(features.join(','));
  return {key,features,label:row.photo_outcome};
}
function fit(positive,negative){
  const weights=new Map();
  for(let epoch=0;epoch<18;epoch++){
    const lr=0.055/(1+0.12*epoch);
    for(const p of positive)for(const n of negative){
      const delta=new Map(),d=gradient(score(weights,p.features)-score(weights,n.features));
      for(const f of p.features)delta.set(f,(delta.get(f)||0)+1);
      for(const f of n.features)delta.set(f,(delta.get(f)||0)-1);
      for(const [key,v] of delta){
        if(!v)continue;
        const current=weights.get(key)||0;
        const next=current*(1-0.001*lr)+lr*d*v;
        if(Number.isFinite(next))weights.set(key,clamp(next,-3,3));
      }
    }
  }
  return weights;
}
/** The holdout is chosen before fitting and by OUTFIT FEATURE group,
 * not by per-request row ID: repeated catalogue aliases cannot leak labels.
 */
export function trainPhotoRewardHistory(rows,contextHash){
  if(!DIGEST.test(String(contextHash||''))||!Array.isArray(rows))
    return neutral('unavailable');
  const grouped=new Map();
  for(const row of rows.slice(0,MAX_ROWS)){
    const x=evidence(row,contextHash);
    if(!x)continue;
    const old=grouped.get(x.key);
    if(old&&old.label!==x.label)grouped.set(x.key,{conflict:true});
    else if(!old)grouped.set(x.key,x);
  }
  const unique=[...grouped.values()].filter(x=>!x.conflict);
  const pos=unique.filter(x=>x.label===1).sort((a,b)=>a.key.localeCompare(b.key));
  const neg=unique.filter(x=>x.label===-1).sort((a,b)=>a.key.localeCompare(b.key));
  const count={examples:pos.length+neg.length,positive:pos.length,negative:neg.length};
  // This is a STATISTICAL evidence floor, never a fashion category rule.
  if(pos.length<6||neg.length<10)return neutral('insufficient_real_photo_history',count);
  const split=group=>{
    const held=new Set([Math.floor(group.length/4),Math.floor(group.length*3/4)]);
    return {
      hold:group.filter((_,i)=>held.has(i)),
      train:group.filter((_,i)=>!held.has(i)),
    };
  };
  const a=split(pos),b=split(neg);
  const w=fit(a.train,b.train);
  const comparisons=a.hold.flatMap(p=>b.hold.map(n=>score(w,p.features)-score(w,n.features)));
  const wins=comparisons.filter(x=>x>0.04).length;
  // Require 4/4 held-out correct pairwise original-photo decisions;
  // a nice looking synthetic model is still not independently calibrated.
  if(wins!==comparisons.length)return neutral('uncalibrated_history',
    {...count,holdoutWins:wins,holdoutPairs:comparisons.length});
  return {version:PHOTO_REWARD_HISTORY_VERSION,
    status:'heldout_photo_validated',usable:true,...count,
    holdoutWins:wins,holdoutPairs:comparisons.length,
    weights:fit(pos,neg)};
}
/** Returns a source-independent teacher PRIOR, not an approval verdict. */
export function scorePhotoRewardHistory(model,featureHashes){
  if(!model?.usable||!Array.isArray(featureHashes)||
      featureHashes.length<MIN_FEATURES||featureHashes.length>MAX_FEATURES||
      featureHashes.some(x=>typeof x!=='string'||!DIGEST.test(x)))return null;
  const distinct=[...new Set(featureHashes)];
  if(distinct.length<MIN_FEATURES)return null;
  const margin=score(model.weights,distinct);
  return Number.isFinite(margin)?clamp(Math.tanh(margin/4),-1,1):null;
}
