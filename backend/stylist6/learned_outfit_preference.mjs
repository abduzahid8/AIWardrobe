/**
 * Personal four-garment preference learning, not a collection of fashion bans.
 *
 * Supervision is ONLY the wearer's explicit signed outfit feedback. Inferred
 * photo-critic scores, click-throughs and unlabeled catalog merchandise are
 * never silently promoted to human taste. No user IDs, product identifiers,
 * images, brand names, free-text notes or raw photos enter model features.
 *
 * The pairwise logistic learner sees anonymous outfit-level semantic facts,
 * including cross-slot construction/color/texture interactions. Weights are
 * fitted from REAL positive-versus-negative preferences, not hand-written
 * rules ("never use a cardigan"). No reliable balanced labels -> ABSTAIN.
 *
 * Pure/bounded/in-memory per request; no persistent global user model.
 */
export const LEARNED_OUTFIT_PREFERENCE_VERSION='s6-owner-pairwise-v1';
const SLOTS=4,FIELDS=5,MAX_ROWS=60,MAX_TOKEN=48,MAX_FEATURES=180;
const MIN_PER_CLASS=6,HELD_OUT_PER_CLASS=2;
const NON_EVIDENCE=new Set(['unknown','neutral','missing','none','']);
const clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));
const validWord=x=>{
  if(typeof x!=='string'||x.length>MAX_TOKEN)return null;
  const s=x.normalize('NFKC').trim().toLowerCase().replace(/\s+/g,'_');
  return /^[a-z0-9][a-z0-9_-]{0,47}$/.test(s)&&!NON_EVIDENCE.has(s)?s:null;
};
export function normalizedOutfitLearningFeatures(value){
  if(!Array.isArray(value)||value.length!==SLOTS)return null;
  let observed=0;
  const out=[];
  for(const slot of value){
    if(slot===null){out.push(null);continue}
    if(!Array.isArray(slot)||slot.length!==FIELDS)return null;
    const f=slot.map(x=>x===null||x===undefined?null:validWord(x));
    if(slot.some((x,i)=>x!==null&&x!==undefined&&f[i]===null&&!NON_EVIDENCE.has(
      typeof x==='string'?x.trim().toLowerCase():'')))return null;
    if(f[0])observed++;
    out.push(f);
  }
  return observed>=3?out:null;
}
/** Raw garment category, silhouette, color, texture and pattern plus ALL
 * pairwise slot relations. No stylist-authored compatibility weights. */
export function outfitRelationshipFeatures(raw){
  const matrix=normalizedOutfitLearningFeatures(raw);
  if(!matrix)return null;
  const tokens=new Set();
  for(let s=0;s<SLOTS;s++){
    const row=matrix[s];if(!row)continue;
    for(let f=0;f<FIELDS;f++)
      if(row[f])tokens.add('u'+s+':'+f+'='+row[f]);
    for(let t=s+1;t<SLOTS;t++){
      const other=matrix[t];if(!other)continue;
      for(let f=0;f<FIELDS;f++)if(row[f])
        for(let g=0;g<FIELDS;g++)if(other[g])
          tokens.add('p'+s+t+':'+f+g+'='+row[f]+'|'+other[g]);
    }
  }
  return tokens.size>=12&&tokens.size<=MAX_FEATURES
    ?[...tokens].sort():null;
}
const rating=x=>x===1?1:x===-1?-1:null;
function feedbackExample(row,now){
  if(!row||typeof row!=='object'||Array.isArray(row)||
      rating(row.rating)===null)return null;
  const raw=row.look_signature??row.lookSignature;
  if(!raw||raw.version!==1)return null;
  const tokens=outfitRelationshipFeatures(raw.features);
  if(!tokens)return null;
  const when=Date.parse(row.created_at??row.createdAt);
  if(!Number.isFinite(when)||when>now+86400000||now-when>365*86400000)
    return null;
  return {tokens,label:row.rating,key:JSON.stringify(tokens)};
}
const dot=(weights,tokens)=>tokens.reduce((sum,t)=>sum+(weights.get(t)||0),0);
const sigmoidMinus=margin=>1/(1+Math.exp(clamp(margin,-35,35)));
function fit(positives,negatives){
  const weights=new Map();
  for(let epoch=0;epoch<24;epoch++){
    // Deterministic, no random order/seed, no external API or training IDs.
    const lr=0.075/(1+epoch*0.12);
    for(const pos of positives)
      for(const neg of negatives){
        const margin=dot(weights,pos.tokens)-dot(weights,neg.tokens);
        const gradient=sigmoidMinus(margin);
        const diff=new Map();
        for(const t of pos.tokens)diff.set(t,(diff.get(t)||0)+1);
        for(const t of neg.tokens)diff.set(t,(diff.get(t)||0)-1);
        for(const [key,delta] of diff){
          if(!delta)continue;
          const updated=(weights.get(key)||0)*(1-lr*0.018)+lr*gradient*delta;
          if(Number.isFinite(updated))weights.set(key,clamp(updated,-3,3));
        }
      }
  }
  return weights;
}
/**
 * A personal model is only released when previously unseen held-out rated
 * looks are correctly ranked against each other. No labels => neutral. This
 * tests comparison quality, not whether an individual clothing rule is met.
 */
export function trainOwnerOutfitPreference(rows,now=Date.now()){
  const inactive={version:LEARNED_OUTFIT_PREFERENCE_VERSION,status:'untrained',
    usable:false,examples:0,holdoutPairs:0,holdoutWins:0,weights:new Map()};
  if(!Array.isArray(rows)||!Number.isFinite(now))return inactive;
  const bySignature=new Map();
  for(const row of rows.slice(0,MAX_ROWS)){
    const x=feedbackExample(row,now);if(!x)continue;
    const previous=bySignature.get(x.key);
    if(previous&&previous.label!==x.label)bySignature.set(x.key,{conflict:true});
    else if(!previous)bySignature.set(x.key,x);
  }
  const unique=[...bySignature.values()].filter(x=>!x.conflict);
  const positive=unique.filter(x=>x.label===1).sort((a,b)=>a.key.localeCompare(b.key));
  const negative=unique.filter(x=>x.label===-1).sort((a,b)=>a.key.localeCompare(b.key));
  const examples=positive.length+negative.length;
  if(!examples)return inactive;
  if(positive.length<MIN_PER_CLASS||negative.length<MIN_PER_CLASS)
    return {...inactive,examples,status:'insufficient_human_comparisons'};
  // Keep both ends of the deterministic semantic spectrum in validation;
  // taking the lexicographically first two samples would confound the
  // holdout with a single garment/color family, producing false conclusions.
  const split=group=>{
    const holdIndex=new Set([Math.floor(group.length/4),
      Math.floor(group.length*3/4)]);
    return {
      held:group.filter((_,i)=>holdIndex.has(i)),
      train:group.filter((_,i)=>!holdIndex.has(i)),
    };
  };
  const pos=split(positive),neg=split(negative);
  const holdP=pos.held,holdN=neg.held;
  const weights=fit(pos.train,neg.train);
  let wins=0;
  for(const p of holdP)for(const n of holdN)
    if(dot(weights,p.tokens)-dot(weights,n.tokens)>0.05)wins++;
  const total=holdP.length*holdN.length;
  if(wins!==total)return {...inactive,examples,holdoutPairs:total,
    holdoutWins:wins,status:'uncalibrated_on_holdout'};
  // Retrain on all approved explicit examples only AFTER holdout passes.
  const finalWeights=fit(positive,negative);
  return {version:LEARNED_OUTFIT_PREFERENCE_VERSION,status:'validated_human_preferences',
    usable:true,examples,holdoutPairs:total,holdoutWins:wins,weights:finalWeights};
}
/** Bounded [-1,+1] signed model opinion; zero is true abstention. */
export function learnedOutfitPreferenceScore(model,stylingFeatures){
  if(!model?.usable)return 0;
  const tokens=outfitRelationshipFeatures(stylingFeatures);
  if(!tokens)return 0;
  const raw=dot(model.weights,tokens);
  return Number.isFinite(raw)?clamp(Math.tanh(raw/4),-1,1):0;
}
