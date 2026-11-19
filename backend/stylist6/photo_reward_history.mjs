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
import {PHOTO_REWARD_JOURNAL_VERSION,photoRewardContextHash} from './photo_reward_journal.mjs';
export const PHOTO_REWARD_HISTORY_VERSION='s6-photo-history-pairwise-v3-crossvalidated';
const DIGEST=/^[0-9a-f]{64}$/;
const PHOTO_REJECTION_REASONS=new Set([
  'model_veto','below_quality_floor','style_mismatch',
  'occasion_mismatch','formality_mismatch',
]);
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
/** EXACT SAME protected photo-context digest as the writer: the reader
 * must not mix fashion quality judgments across different real preference
 * contexts or accidentally train on another user's explicit exclusions.
 */
export function photoHistoryContextHash(context,secret){
  return photoRewardContextHash(context,secret);
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
      (row.photo_outcome===1
        ?row.photo_reason!=='approved'
        :!PHOTO_REJECTION_REASONS.has(row.photo_reason))||
      row.photo_quality===null||row.photo_quality===undefined||
      !Number.isFinite(Number(row.photo_quality))||
      Number(row.photo_quality)<0||Number(row.photo_quality)>1)
    return null;
  const features=[...new Set(row.feature_hashes)].sort();
  if(features.length<MIN_FEATURES)return null;
  const key=sha(features.join(','));
  return {key,features,label:row.photo_outcome,
    quality:Number(row.photo_quality)};
}
/**
 * Actual-photo contrastive learning, not a catalog/subtype rule.
 *
 * The old classifier gave every one-off feature of a single photograph the
 * same authority as a garment RELATIONSHIP repeated across independent
 * photos. That memorized SKU accidents instead of transferable style.
 *
 * Only SHARED visual relations in the TRAINING fold can move weights;
 * unseen held-out features never participate in fitting the frequency
 * prior. The source-independent original-photo QUALITY gap controls how
 * strongly two complete photographed outfits are compared. Ratings are
 * from the PHOTO critic, never from the owner's explicit likes.
 */
function fit(positive,negative){
  const all=[...positive,...negative];
  const frequency=new Map();
  for(const look of all)for(const feature of look.features)
    frequency.set(feature,(frequency.get(feature)||0)+1);
  // Statistical support shrinkage, not a garment ban: one-off exact source
  // quirks supply zero training weight, repeated evidence increasingly so.
  const reliability=new Map([...frequency].map(([key,count])=>
    [key,clamp((count-1)/3,0,1)]));
  const pairs=positive.flatMap(p=>negative.map(n=>({
    p,n,confidence:clamp(0.5+Math.max(0,p.quality-n.quality)*2,0.5,1.5),
  })));
  // A failed photographed outfit may still be BETTER than another failed
  // outfit. Learn that degree of genuine visual quality without ever
  // treating either rejection as a pass. This helps with the common
  // 1-approval / many-rejections staging batches.
  for(const group of [positive,negative])
    for(let i=0;i<group.length;i++)for(let j=i+1;j<group.length;j++){
      const p=group[i],n=group[j],gap=Math.abs(p.quality-n.quality);
      if(gap<0.08)continue;
      pairs.push({p:p.quality>n.quality?p:n,
        n:p.quality>n.quality?n:p,
        confidence:Math.min(0.22,gap*0.55)});
    }
  const weights=new Map();
  for(let epoch=0;epoch<18;epoch++){
    const lr=0.055/(1+0.12*epoch);
    for(const {p,n,confidence} of pairs){
      const delta=new Map(),d=confidence*
        gradient(score(weights,p.features)-score(weights,n.features));
      for(const f of p.features)delta.set(f,(delta.get(f)||0)+1);
      for(const f of n.features)delta.set(f,(delta.get(f)||0)-1);
      for(const [key,v] of delta){
        const support=reliability.get(key)||0;
        if(!v||!support)continue;
        const current=weights.get(key)||0;
        const next=current*(1-0.003*lr)+lr*d*v*support;
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
  // Statistical evidence floor, not a garment/style rule. With only four
  // holdout comparisons the old 100%-correct requirement repeatedly
  // abstained despite real 4-fold PHOTO ordering accuracy near 80% on
  // 42 independently reviewed Work outfits. That four-pair estimate was
  // overly noisy. The wider evidence check below still NEVER grants a
  // fashion approval: it only permits ordering UNREVIEWED PHOTO auditions.
  // The other two style/occasion contexts must validate separately.
  if(pos.length<8||neg.length<12||count.examples<24)
    return neutral('insufficient_real_photo_history',count);
  // Group by complete LOOK FEATURE identity (not request or row order).
  // Each independent held-out PHOTO outfit appears in one and only one
  // test fold; cross-fitting cannot see the inspected outfit's judgment.
  // Sorting by a hashed semantic signature also makes shuffled Supabase
  // history rows generate EXACTLY the same model and accuracy receipt.
  const folds=4;
  let wins=0,pairs=0,weakFold=false;
  for(let fold=0;fold<folds;fold++){
    const trainPos=pos.filter((_,i)=>i%folds!==fold);
    const trainNeg=neg.filter((_,i)=>i%folds!==fold);
    const heldPos=pos.filter((_,i)=>i%folds===fold);
    const heldNeg=neg.filter((_,i)=>i%folds===fold);
    if(trainPos.length<6||trainNeg.length<9||
        heldPos.length<2||heldNeg.length<2){
      weakFold=true;
      continue;
    }
    const weights=fit(trainPos,trainNeg);
    let foldWins=0,foldPairs=0;
    for(const p of heldPos)for(const n of heldNeg){
      foldPairs++;
      if(score(weights,p.features)-score(weights,n.features)>0.04)
        foldWins++;
    }
    wins+=foldWins;pairs+=foldPairs;
    // Four-fold cross-validation must not conceal a collapsed independent
    // holdout group behind a favorable aggregate. 60% is a guard against
    // a near-random fold, NOT a claim of human preference correctness.
    if(foldWins/foldPairs<0.60)weakFold=true;
  }
  // The pairs share outfits; they are not independent human votes. Require
  // sufficient independent PHOTO looks as above, full fold coverage and
  // clear, repeatable out-of-fold ranking advantage before using the prior.
  const validated=!weakFold&&pairs>=24&&wins/pairs>=0.75;
  if(!validated)return neutral('uncalibrated_history',
    {...count,holdoutWins:wins,holdoutPairs:pairs});
  return {version:PHOTO_REWARD_HISTORY_VERSION,
    status:'heldout_photo_validated',usable:true,...count,
    holdoutWins:wins,holdoutPairs:pairs,
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
