import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {
  PHOTO_REWARD_HISTORY_VERSION,
  trainPhotoRewardHistory,scorePhotoRewardHistory,
  photoHistoryContextHash,photoHistoryFeatureHashes,
} from './photo_reward_history.mjs';
import {compilePhotoRewardExamples} from './photo_reward_journal.mjs';

const digest=x=>createHash('sha256').update(x).digest('hex');
const CONTEXT=digest('classic\0work');
// Both individual colors occur on BOTH sides. Only a CROSS-ROLE
// relationship distinguishes photographed good vs bad whole outfits.
const observed=(good,i)=>[
  ...Array.from({length:17},(_,j)=>digest('all_garments'+j)),
  digest('shirt_color='+(i%2?'navy':'cream')),
  digest('jacket_color='+(good?(i%2?'grey':'brown'):(i%2?'brown':'grey'))),
  digest('shirt_JACKET_pair='+(good
    ?(i%2?'navy_grey':'cream_brown')
    :(i%2?'navy_brown':'cream_grey'))),
  digest('shoe_profile='+(i%4)),
  digest('item_variation='+i),
].sort();
const sample=(good,i)=>({
  fingerprint:digest((good?'yes':'no')+i),
  context_hash:CONTEXT,
  feature_hashes:observed(good,i),
  photo_quality:good?.83:.52,
  photo_outcome:good?1:-1,
  photo_reason:good?'approved':'below_quality_floor',
  evidence_version:'s6-photo-reward-journal-v1',
});
const evidence=()=>Array.from({length:14},(_,i)=>[sample(true,i),sample(false,i)]).flat();

test('zero history, one thumb or all-negative original-photo critiques ABSTAIN',()=>{
  const model=trainPhotoRewardHistory([],CONTEXT);
  assert.equal(model.version,PHOTO_REWARD_HISTORY_VERSION);
  assert.equal(model.usable,false);
  assert.equal(model.examples,0);
  assert.equal(scorePhotoRewardHistory(model,observed(true,100)),null);
  const negative=Array.from({length:35},(_,i)=>sample(false,i));
  assert.equal(trainPhotoRewardHistory(negative,CONTEXT).usable,false);
});

test('whole-look relationship reward generalizes to unseen product features without color bans',()=>{
  const model=trainPhotoRewardHistory(evidence(),CONTEXT);
  assert.equal(model.status,'heldout_photo_validated');
  assert.equal(model.usable,true);
  assert.equal(model.holdoutPairs,4);
  assert.equal(model.holdoutWins,4);
  const good=scorePhotoRewardHistory(model,observed(true,111));
  const bad=scorePhotoRewardHistory(model,observed(false,111));
  assert.ok(good>bad,{good,bad});
  assert.ok(Math.abs(good)<=1&&Math.abs(bad)<=1);
  const colors=new Set(evidence().map(x=>x.feature_hashes.find(t=>
    t===digest('jacket_color=grey')||t===digest('jacket_color=brown'))));
  assert.deepEqual(colors,new Set([digest('jacket_color=brown'),digest('jacket_color=grey')]));
});

test('training is order-invariant and remains scoped to trusted context',()=>{
  const a=evidence(),b=[...a].reverse();
  const first=trainPhotoRewardHistory(a,CONTEXT);
  const reordered=trainPhotoRewardHistory(b,CONTEXT);
  assert.equal(first.status,reordered.status);
  for(let i=0;i<14;i++)for(const yes of [true,false])
    assert.equal(scorePhotoRewardHistory(first,observed(yes,i)),
      scorePhotoRewardHistory(reordered,observed(yes,i)));
  assert.equal(trainPhotoRewardHistory(a,digest('other style')).usable,false);
  assert.equal(trainPhotoRewardHistory(a.map(row=>({
    ...row,evidence_version:'human_thumbs'})),CONTEXT).examples,0);
});

test('contradictory labels of same profile are ignored; no implicit approvals from numeric scores',()=>{
  const rows=evidence();
  const contradict=rows.map(row=>({
    ...row,fingerprint:digest('different'+row.fingerprint),photo_outcome:-row.photo_outcome,
  }));
  const model=trainPhotoRewardHistory([...rows,...contradict],CONTEXT);
  assert.equal(model.usable,false);
  assert.equal(model.examples,0);
  const fake=rows.map(row=>({...row,photo_outcome:0,photo_quality:1}));
  assert.equal(trainPhotoRewardHistory(fake,CONTEXT).usable,false);
});

test('no raw merchant descriptions or images qualify as private scored features',()=>{
  const good=trainPhotoRewardHistory(evidence(),CONTEXT);
  for(const corrupt of [
    ['shop://PRIVATE_SKU'],
    ['https://merchant.example/photo.jpg'],
    Array.from({length:20},()=>digest('repeated_value')),
    Array.from({length:181},(_,i)=>digest('too_many'+i)),
    null,
  ])assert.equal(scorePhotoRewardHistory(good,corrupt),null);
  const withoutUserIds=evidence().map(row=>({
    ...row,customerId:'PRIVATE_OWNER',
    merchantUrl:'https://secret.example/shop',
  }));
  const other=trainPhotoRewardHistory(withoutUserIds,CONTEXT);
  assert.equal(other.usable,true);
  assert.doesNotMatch(JSON.stringify([...other.weights.keys()]),/PRIVATE|secret|owner/);
});

test('cross-request hash scorer EXACTLY matches independently judged journal evidence',()=>{
  const secret='isolated-staging-photo-reward-hmac-secret-test-only';
  const context={style:'classic',occasion:'work'};
  const slots=['top','outerwear','bottom','shoes'];
  const items=slots.map((macroCategory,i)=>({
    macroCategory,id:'MERCHANT_SKU_'+i,isShopItem:true,
    facts:[
      ['shirt','regular','cream','cotton','plain'],
      ['blazer','tailored','brown','wool','plain'],
      ['trousers','straight','grey','wool','plain'],
      ['loafer','regular','brown','leather','plain'],
    ][i],
  }));
  const look={id:'SOURCE_LOOK_PRIVATE',items};
  const featuresOf=pieces=>pieces.map(x=>x.facts);
  const source=compilePhotoRewardExamples({
    secret,context,looks:[look],
    decisions:[{candidateId:look.id,reason:'approved',quality:.87}],
    critiques:[{candidateId:look.id,reject:false,
      visualCohesion:.9,styleFit:.9,occasionFit:.9,formalityFit:.9,
      confidence:.9}],featuresOf,
  });
  assert.equal(source.length,1);
  assert.equal(photoHistoryContextHash(context,secret),source[0].context_hash);
  const looseContext={...context,fitPreference:'loose'};
  const looseRows=compilePhotoRewardExamples({
    secret,context:looseContext,looks:[look],
    decisions:[{candidateId:look.id,reason:'approved',quality:.87}],
    critiques:[{candidateId:look.id,reject:false,
      visualCohesion:.9,styleFit:.9,occasionFit:.9,formalityFit:.9,
      confidence:.9}],featuresOf,
  });
  assert.equal(looseRows.length,1);
  assert.equal(photoHistoryContextHash(looseContext,secret),
    looseRows[0].context_hash);
  assert.notEqual(looseRows[0].context_hash,source[0].context_hash,
    'do not mix relaxed/loose wearers with balanced-fit training decisions');
  assert.deepEqual(photoHistoryFeatureHashes(items,featuresOf,secret),
    source[0].feature_hashes);
  assert.equal(photoHistoryContextHash({style:'classic',occasion:'private graduation'},secret),null);
  assert.equal(photoHistoryFeatureHashes(items,featuresOf,'weak'),null);
});
