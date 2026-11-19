/**
 * In-request photographic decision learning — SOURCE image *evidence* is the
 * original independent four-item PHOTO critic's verdict. The learned reward
 * uses ONLY grounded semantic observations of the actual product options,
 * NEVER user feedback, original JPEG bytes, personal IDs or merchant URLs.
 *
 * Real photo-approved outfits are positive *teacher examples*, not proof of
 * wearer preference. The outcome can only order which authentic unreviewed
 * combinations to ask the SAME independent photo critic to examine next.
 * No score can approve a photo, bypass a hard source gate or ban garment types.
 *
 * Small-sample protection: insufficient or non-generalizing photo examples
 * make the teacher ABSTAIN; an uncalibrated in-request model does NOT replace
 * a human fashion professional or the original-photo judge.
 */
import {outfitRelationshipFeatures,normalizedOutfitLearningFeatures} from './learned_outfit_preference.mjs';

export const PHOTO_FEEDBACK_TEACHER_VERSION='s6-photo-pairwise-teacher-v1';
const clamp=(v,min,max)=>Math.max(min,Math.min(max,v));
// These reasons already came from the INDEPENDENT photographic review.
// An NVIDIA verdict can be reject=false but still fail the unchanged
// original-photo quality/style/occasion gate. Those are useful NEGATIVE
// fashion comparisons, not missing evidence or fabricated vetoes.
// User-requested color exclusion and low-confidence verdicts are NOT
// generalized into learned aesthetic preferences.
const PHOTO_NEGATIVE_REASONS=new Set([
  'model_veto','below_quality_floor','style_mismatch',
  'occasion_mismatch','formality_mismatch',
]);
const similarityFeatures=(items,featuresOf)=>{
  if(!Array.isArray(items)||items.length!==4||typeof featuresOf!=='function')return null;
  try {return outfitRelationshipFeatures(featuresOf(items));}catch{return null;}
};
const dot=(weights,features)=>features.reduce((sum,x)=>sum+(weights.get(x)||0),0);
const difference=(weights,pos,neg)=>dot(weights,pos.features)-dot(weights,neg.features);
const marginGradient=margin=>1/(1+Math.exp(clamp(margin,-30,30)));
function fitPairs(pairs){
  const w=new Map();
  // No manually specified outfit/category/color weights. Each observed
  // comparison comes from the original-photo judge and the correct source
  // context, or the original-photo judge's own graded quality within its
  // rejected group. The fitted weights are local to this ONE request.
  for(let epoch=0;epoch<18;epoch++){
    const step=0.06/(1+epoch*0.16);
    for(const [p,n] of pairs){
      const gradient=marginGradient(difference(w,p,n));
      const diff=new Map();
      for(const x of p.features)diff.set(x,(diff.get(x)||0)+1);
      for(const x of n.features)diff.set(x,(diff.get(x)||0)-1);
      for(const [k,v] of diff){
        if(!v)continue;
        const updated=(w.get(k)||0)*(1-0.005*step)+step*gradient*v;
        if(Number.isFinite(updated))w.set(k,clamp(updated,-2,2));
      }
    }
  }
  return w;
}
function fit(positives,negatives){
  return fitPairs(positives.flatMap(p=>negatives.map(n=>[p,n])));
}
const sharedPhotoFeatures=(a,b)=>{
  let same=0;
  for(let slot=0;slot<4;slot++)for(let field=0;field<5;field++)
    if(a.matrix[slot]?.[field]&&a.matrix[slot][field]===b.matrix[slot]?.[field])
      same++;
  return same;
};
// A frequently observed REAL scenario: one approved and four rejected
// complete PHOTO looks. Do not discard useful graded critic evidence merely
// because we lack a SECOND positive photo. Learn a ranking within the
// *rejected group* and only use it when a matched, unseen PHOTO quality
// comparison is correctly predicted. This cannot APPROVE a rejected look.
function trainGradedPhotoTeacher(positive,negative){
  if(positive.length<1||negative.length<4)return null;
  const available=negative.filter(x=>Number.isFinite(x.quality));
  if(available.length<4)return null;
  const pairs=[];
  for(let i=0;i<available.length;i++)for(let j=i+1;j<available.length;j++){
    const x=available[i],y=available[j];
    const gap=Math.abs(x.quality-y.quality);
    const shared=sharedPhotoFeatures(x,y);
    if(gap<0.075||shared<7)continue;
    const [better,worse]=x.quality>y.quality?[x,y]:[y,x];
    pairs.push({better,worse,gap,shared});
  }
  // Prefer the most comparable *unseen* photographic pair with meaningful
  // quality separation; the selection never depends on learned predictions.
  pairs.sort((a,b)=>b.shared-a.shared||b.gap-a.gap||
    (a.better.key+a.worse.key).localeCompare(b.better.key+b.worse.key));
  const held=pairs[0];
  if(!held)return null;
  const remaining=negative.filter(x=>x!==held.better&&x!==held.worse);
  if(remaining.length<2)return null;
  const comparisons=[
    ...positive.flatMap(p=>remaining.map(n=>[p,n])),
    ...remaining.flatMap((a,i)=>remaining.slice(i+1).flatMap(b=>{
      if(!Number.isFinite(a.quality)||!Number.isFinite(b.quality)||
          Math.abs(a.quality-b.quality)<0.075)return [];
      return [a.quality>b.quality?[a,b]:[b,a]];
    })),
  ];
  if(comparisons.length<2)return null;
  const weights=fitPairs(comparisons);
  if(difference(weights,held.better,held.worse)<=0)
    return {status:'uncalibrated_on_photos',usable:false};
  const full=[
    ...positive.flatMap(p=>negative.map(n=>[p,n])),
    ...negative.flatMap((a,i)=>negative.slice(i+1).flatMap(b=>
      Number.isFinite(a.quality)&&Number.isFinite(b.quality)&&
      Math.abs(a.quality-b.quality)>=0.075
        ?[a.quality>b.quality?[a,b]:[b,a]]:[])),
  ];
  return {status:'graded_photo_validated',usable:true,weights:fitPairs(full)};
}
export function trainPhotoFeedbackTeacher({looks,decisions,critiques,featuresOf}={}){
  const inactive=(status,extra={})=>({version:PHOTO_FEEDBACK_TEACHER_VERSION,
    status,usable:false,examples:0,positive:0,negative:0,
    weights:new Map(),...extra});
  if(!Array.isArray(looks)||!Array.isArray(decisions)||
    !Array.isArray(critiques)||typeof featuresOf!=='function')return inactive('unavailable');
  const byId=new Map(looks.slice(0,8).filter(x=>typeof x?.id==='string')
    .map(x=>[x.id,x]));
  const byCritique=new Map(critiques.slice(0,8).filter(x=>
    typeof x?.candidateId==='string'&&typeof x.reject==='boolean'&&
    ['visualCohesion','styleFit','occasionFit','formalityFit','confidence']
      .every(k=>typeof x[k]==='number'&&Number.isFinite(x[k])&&x[k]>=0&&x[k]<=1))
    .map(x=>[x.candidateId,x]));
  const signatures=new Map();
  for(const d of decisions.slice(0,8)){
    if(typeof d?.candidateId!=='string'||!byId.has(d.candidateId))continue;
    const critique=byCritique.get(d.candidateId);
    const positive=d.reason==='approved'&&critique?.reject===false;
    const negative=PHOTO_NEGATIVE_REASONS.has(d.reason)&&!!critique&&
      // An explicit model_veto must actually be a model rejection; other
      // negative reasons come from the authenticated selector's real-photo
      // numerical/occasion gates even if the provider reject flag was false.
      (d.reason!=='model_veto'||critique.reject===true);
    if(!positive&&!negative)continue;
    const look=byId.get(d.candidateId);
    let raw=null;
    try{raw=featuresOf(look.items);}catch{continue;}
    const features=outfitRelationshipFeatures(raw);
    const matrix=normalizedOutfitLearningFeatures(raw);
    if(!features||!matrix)continue;
    const key=JSON.stringify(features),label=positive?1:-1;
    const previous=signatures.get(key);
    if(previous&&previous.label!==label)signatures.set(key,{conflict:true});
    else if(!previous)signatures.set(key,{features,matrix,label,key,
      quality:Number.isFinite(d.quality)&&d.quality>=0&&d.quality<=1
        ?d.quality:null});
  }
  const unique=[...signatures.values()].filter(x=>!x.conflict);
  const positive=unique.filter(x=>x.label===1).sort((a,b)=>a.key.localeCompare(b.key));
  const negative=unique.filter(x=>x.label===-1).sort((a,b)=>a.key.localeCompare(b.key));
  const count={examples:positive.length+negative.length,positive:positive.length,
    negative:negative.length};
  if(positive.length<2||negative.length<3){
    const graded=trainGradedPhotoTeacher(positive,negative);
    if(graded?.usable)return {version:PHOTO_FEEDBACK_TEACHER_VERSION,
      ...count,...graded};
    if(graded)return inactive(graded.status,count);
    return inactive('insufficient_independent_photos',count);
  }
  // Select a matched held-out pair: two original-photo verdicts that
  // shared the most SOURCE-GROUNDED product traits BEFORE the decision.
  // This tests whether a learned four-piece RELATIONSHIP predicts a
  // photo critic's decision rather than memorizing arbitrary garment types.
  // A small run without comparable outfits must ABSTAIN.
  const matched=positive.flatMap(p=>negative.map(n=>({
    p,n,shared:sharedPhotoFeatures(p,n),
  }))).sort((a,b)=>b.shared-a.shared||
    (a.p.key+a.n.key).localeCompare(b.p.key+b.n.key))[0];
  if(!matched||matched.shared<7)
    return inactive('no_comparable_photo_pair',count);
  const holdP=matched.p,holdN=matched.n;
  const weights=fit(positive.filter(x=>x!==holdP),negative.filter(x=>x!==holdN));
  const heldOutMargin=difference(weights,holdP,holdN);
  if(heldOutMargin<=0)
    return inactive('uncalibrated_on_photos',{...count,heldOutMargin});
  return {version:PHOTO_FEEDBACK_TEACHER_VERSION,status:'photo_validated',
    usable:true,...count,weights:fit(positive,negative)};
}
export function photoTeacherScore(teacher,look,featuresOf){
  if(!teacher?.usable||!look||typeof featuresOf!=='function')return null;
  const features=similarityFeatures(look.items,featuresOf);
  if(!features)return null;
  const score=dot(teacher.weights,features);
  return Number.isFinite(score)?clamp(Math.tanh(score/4),-1,1):null;
}
/** Sort ONLY unapproved real source candidates. Never produce an admitted look. */
export function photoTaughtRescueOrder(looks,teacher,featuresOf){
  if(!Array.isArray(looks))return [];
  if(!teacher?.usable)return [...looks];
  return [...looks].sort((a,b)=>{
    const x=photoTeacherScore(teacher,a,featuresOf);
    const y=photoTeacherScore(teacher,b,featuresOf);
    // An unknown photographic signature is not assumed to be bad or good;
    // retain source ordering rather than minting an unsupported score.
    return x===null||y===null?0:y-x;
  });
}
