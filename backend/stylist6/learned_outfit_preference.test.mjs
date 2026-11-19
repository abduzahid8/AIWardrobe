import test from 'node:test';
import assert from 'node:assert/strict';
import {
  LEARNED_OUTFIT_PREFERENCE_VERSION,
  outfitRelationshipFeatures,normalizedOutfitLearningFeatures,
  trainOwnerOutfitPreference,learnedOutfitPreferenceScore,
} from './learned_outfit_preference.mjs';
import {
  feedbackTasteModel,evaluateFeedbackLook,FEEDBACK_TASTE_LOOK_CAP,
} from './feedback_taste.mjs';

const NOW=Date.parse('2026-11-19T05:00:00Z');
// Every individual COLOR occurs equally often in both classes. A fixed rule
// like "brown blazers are good" CANNOT solve this fixture: only the learned
// four-slot RELATIONSHIPS distinguish the two outcomes.
const wardrobe=(approved,i)=>[
  ['shirt','regular',i%2?'navy':'cream','cotton','plain'],
  ['jacket','tailored',approved?(i%2?'grey':'brown'):(i%2?'brown':'grey'),'wool','plain'],
  ['trouser','straight','taupe','wool','plain'],
  ['shoe-'+i,'regular','black','leather','plain'],
];
const thumb=(approved,i)=>({
  user_id:'PRIVATE_OWNER_ID_NEVER_A_FEATURE',
  request_id:'PRIVATE_REQUEST_ID_NEVER_A_FEATURE',
  rating:approved?1:-1,
  created_at:new Date(NOW).toISOString(),
  look_signature:{version:1,features:wardrobe(approved,i)},
});
const training=()=>Array.from({length:8},(_,i)=>[thumb(true,i),thumb(false,i)]).flat();
const freshItems=()=>['top','outerwear','bottom','shoes'].map((macroCategory,i)=>
  ({id:'UNSEEN_SOURCE_SKU_'+i,macroCategory}));

test('no user labels means no taste claims or learned preferences',()=>{
  const m=trainOwnerOutfitPreference([],NOW);
  assert.equal(m.version,LEARNED_OUTFIT_PREFERENCE_VERSION);
  assert.equal(m.usable,false);
  assert.equal(m.examples,0);
  assert.equal(learnedOutfitPreferenceScore(m,wardrobe(true,0)),0);
  assert.equal(feedbackTasteModel([],NOW).active,false);
});

test('one like, one dislike or 200 identical likes do not invent fashion rules',()=>{
  for(const rows of [
    [thumb(true,0)],
    [thumb(true,0),thumb(false,1)],
    Array.from({length:200},()=>thumb(true,0)),
    Array.from({length:8},(_,i)=>thumb(true,i)),
  ]){
    const m=trainOwnerOutfitPreference(rows,NOW);
    assert.equal(m.usable,false);
    assert.equal(learnedOutfitPreferenceScore(m,wardrobe(true,0)),0);
  }
});

test('learned outfit relationships distinguish combinations WITHOUT garment/color bans',()=>{
  const rows=training();
  const model=trainOwnerOutfitPreference(rows,NOW);
  assert.equal(model.version,LEARNED_OUTFIT_PREFERENCE_VERSION);
  assert.equal(model.usable,true);
  assert.equal(model.status,'validated_human_preferences');
  assert.equal(model.examples,16);
  assert.equal(model.holdoutPairs,4);
  assert.equal(model.holdoutWins,4);
  const good=learnedOutfitPreferenceScore(model,wardrobe(true,111));
  const bad=learnedOutfitPreferenceScore(model,wardrobe(false,111));
  assert.ok(good>0&&bad<0,{good,bad});
  assert.ok(good-bad>.15,{good,bad});
  assert.ok(Math.abs(good)<=1&&Math.abs(bad)<=1);
  assert.ok(model.weights.has('p01:22=cream|brown'));
  assert.ok(model.weights.has('p01:22=navy|grey'));
  assert.ok(model.weights.has('p01:22=cream|grey'));
  assert.ok(model.weights.has('p01:22=navy|brown'));
  // First-order item colors alone cannot explain the *same-color* positive
  // and negative examples. Both colors stay valid and purchasable.
  assert.deepEqual(new Set(rows.map(r=>r.look_signature.features[1][2])),
    new Set(['brown','grey']));
});

test('learned preferences affect the REAL fashion scorer softly across unseen SKUs',()=>{
  const model=feedbackTasteModel(training(),NOW);
  assert.equal(model.learnedPreference.usable,true);
  const a=evaluateFeedbackLook(freshItems(),'unseen-concept',model,wardrobe(true,90));
  const b=evaluateFeedbackLook(freshItems(),'unseen-concept',model,wardrobe(false,90));
  assert.ok(a.adjustment>0,{a});
  assert.ok(b.adjustment<0,{b});
  assert.ok(a.adjustment-b.adjustment>0.003);
  assert.ok(Math.abs(a.adjustment)<=.02);
  assert.ok(Math.abs(b.adjustment)<=.02);
  assert.ok(Math.abs(a.adjustment)<FEEDBACK_TASTE_LOOK_CAP);
  assert.equal(a.likedItems,0);
  assert.deepEqual(a.matches,[{type:'feedback_style',value:'liked'}]);
});

test('training and scores do not change with row order or user/merchant IDs',()=>{
  const rows=training(),other=rows.map((r,i)=>({
    ...r,user_id:'UNRELATED_USER_'+i,request_id:'UNRELATED_REQUEST_'+i,
    fakeMerchantProductId:'SECRET_SOURCE_'+i,
  }));
  const a=trainOwnerOutfitPreference(rows,NOW);
  const b=trainOwnerOutfitPreference(other.reverse(),NOW);
  assert.equal(a.status,'validated_human_preferences');
  assert.equal(b.status,a.status);
  for(let i=0;i<12;i++)for(const g of [true,false])
    assert.equal(learnedOutfitPreferenceScore(a,wardrobe(g,i)),
      learnedOutfitPreferenceScore(b,wardrobe(g,i)));
  assert.doesNotMatch(JSON.stringify([...b.weights.keys()]),
    /UNRELATED_USER|UNRELATED_REQUEST|SECRET_SOURCE/);
});

test('contradictory ratings of identical looks cannot manufacture evidence',()=>{
  const contradictory=training().flatMap(row=>[row,{...row,rating:-row.rating}]);
  const model=trainOwnerOutfitPreference(contradictory,NOW);
  assert.equal(model.usable,false);
  assert.equal(model.examples,0);
});

test('unknown, missing, malformed, stale and invented image facts abstain',()=>{
  const trained=trainOwnerOutfitPreference(training(),NOW);
  const bad=[
    null,[],[wardrobe(true,0)[0]],
    [{photoUrl:'https://private.invalid/image.jpg'},null,null,null],
    Array.from({length:4},()=>['unknown','unknown','unknown','unknown','unknown']),
    Array.from({length:4},()=>['https://user-private.invalid/secret','regular','white','wool','plain']),
  ];
  for(const features of bad){
    assert.equal(outfitRelationshipFeatures(features),null);
    assert.equal(learnedOutfitPreferenceScore(trained,features),0);
  }
  assert.equal(normalizedOutfitLearningFeatures(wardrobe(true,7)).length,4);
  assert.equal(trainOwnerOutfitPreference(training(),NOW+366*86400000).usable,false);
  assert.equal(trainOwnerOutfitPreference(
    training().map(row=>({...row,rating:'approved_by_photo_critic'})),NOW).usable,false);
});
