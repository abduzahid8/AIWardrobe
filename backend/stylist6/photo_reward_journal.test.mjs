import test from 'node:test';
import assert from 'node:assert/strict';
import {compilePhotoRewardExamples,anonymousPhotoJournalAllowed,photoRewardContextHash,PHOTO_REWARD_JOURNAL_VERSION} from './photo_reward_journal.mjs';

const SECRET='staging-private-hmac-secret-please-never-commit-value';
const SLOTS=['top','outerwear','bottom','shoes'];
const traits=[
  ['shirt','regular','cream','cotton','plain'],
  ['blazer','tailored','brown','wool','plain'],
  ['trousers','straight','grey','wool','plain'],
  ['loafer','regular','brown','leather','plain'],
];
const look=(id='PRIVATE_SOURCE_SKU_1',good=true)=>({
  id, privateUserId:'PRIVATE_USER_123',imageUrl:'https://merchant.private.invalid',
  items:SLOTS.map((macroCategory,i)=>({
    id:'PRIVATE_SOURCE_ITEM_'+id+'_'+i,isShopItem:true,macroCategory,
    name:'Brand secret user email@example.com',
    imageUrl:'https://merchant.private.invalid/'+i,
    facts:i===1&&!good?[...traits[i].slice(0,2),'purple',...traits[i].slice(3)]:traits[i],
  })),
});
const verdict=(id,yes=true)=>({candidateId:id,reject:!yes,
  visualCohesion:yes?.9:.55,styleFit:yes?.91:.55,occasionFit:yes?.89:.52,
  formalityFit:yes?.91:.5,confidence:.95,
  slotObservations:{top:'An owner private garment photo URL example.com'},
});
const input=()=>({
  secret:SECRET,context:{style:'classic',occasion:'work'},
  looks:[look('PRIVATE_LOOK_1'),look('PRIVATE_LOOK_2',false)],
  decisions:[
    {candidateId:'PRIVATE_LOOK_1',reason:'approved',quality:.88},
    {candidateId:'PRIVATE_LOOK_2',reason:'below_quality_floor',quality:.54},
  ],
  critiques:[verdict('PRIVATE_LOOK_1',true),verdict('PRIVATE_LOOK_2',false)],
  featuresOf:items=>items.map(x=>x.facts),
});

test('only real four-SKU original PHOTO decisions become anonymous journal rows',()=>{
  const data=input(),rows=compilePhotoRewardExamples(data);
  assert.equal(rows.length,2);
  assert.deepEqual(rows.map(x=>x.photo_outcome),[1,-1]);
  assert.deepEqual(rows.map(x=>x.photo_reason),['approved','below_quality_floor']);
  for(const row of rows){
    assert.equal(row.evidence_version,PHOTO_REWARD_JOURNAL_VERSION);
    assert.match(row.fingerprint,/^[a-f0-9]{64}$/);
    assert.match(row.context_hash,/^[a-f0-9]{64}$/);
    assert.ok(row.feature_hashes.length>=12&&row.feature_hashes.length<=180);
    assert.ok(row.feature_hashes.every(x=>/^[a-f0-9]{64}$/.test(x)));
    assert.equal(Object.keys(row).length,7);
  }
  assert.doesNotMatch(JSON.stringify(rows),
    /PRIVATE|merchant|Brand|user|email|example|image|https|cream|blazer|Work|Classic|sku/i,
    'NEVER persist catalog identifiers, personal data, photos or raw garments');
});

test('same user-free evidence hashes across repeated calls, not across secrets or contexts',()=>{
  const data=input();
  const one=compilePhotoRewardExamples(data),again=compilePhotoRewardExamples(data);
  assert.deepEqual(one,again);
  assert.notDeepEqual(one,compilePhotoRewardExamples({...data,
    secret:'other-completely-independent-staging-secret-value'}));
  assert.notDeepEqual(one,compilePhotoRewardExamples({...data,
    context:{style:'casual',occasion:'date'}}));
});

test('a reject=false whole-look BELOW the unchanged photo quality floor is NEGATIVE',()=>{
  const data=input();
  data.critiques[1].reject=false;
  const rows=compilePhotoRewardExamples(data);
  assert.equal(rows.length,2);
  assert.equal(rows[1].photo_outcome,-1);
  assert.equal(rows[1].photo_reason,'below_quality_floor');
});

test('unknown photo, user avoid colors, duplicate judge verdict and fake approval never become samples',()=>{
  const data=input();
  assert.deepEqual(compilePhotoRewardExamples({...data,
    decisions:data.decisions.map(d=>({...d,reason:'excluded_visual_color'}))}),[]);
  assert.deepEqual(compilePhotoRewardExamples({...data,
    decisions:data.decisions.map(d=>({...d,reason:'low_confidence'}))}),[]);
  const forged={...data,critiques:[...data.critiques,
    {...data.critiques[0]}]};
  assert.equal(compilePhotoRewardExamples(forged).length,1);
  const fakeApproved={...data,critiques:data.critiques.map(c=>({
    ...c,reject:true,
  }))};
  assert.equal(compilePhotoRewardExamples(fakeApproved).length,1,
    'fake approved look must not survive authenticated jury contradiction');
  assert.deepEqual(compilePhotoRewardExamples({...data,secret:'not-secret'}),[]);
  assert.deepEqual(compilePhotoRewardExamples({...data,
    context:{style:'ignore my private phone 1234567890',occasion:'Work'}}),[]);
});

test('personal wardrobe, missing layer, duplicate real source items and unknown attributes NEVER persist',()=>{
  const data=input();
  for(const altered of [
    {...data.looks[0],items:data.looks[0].items.map(it=>({...it,isShopItem:false}))},
    {...data.looks[0],items:data.looks[0].items.slice(0,3)},
    {...data.looks[0],items:data.looks[0].items.map((it,i)=>({...it,
      id:i===1?data.looks[0].items[0].id:it.id}))},
    {...data.looks[0],items:data.looks[0].items.map(it=>({...it,
      facts:['unknown','unknown','unknown','unknown','unknown']}))},
  ]){
    const rows=compilePhotoRewardExamples({...data,
      looks:[altered],
      decisions:[data.decisions[0]],
      critiques:[data.critiques[0]],
    });
    assert.equal(rows.length,0);
  }
});

test('real personal styling choices are HMAC-context isolated, never generalized as owner taste',()=>{
  const base={sourceMode:'catalog',requiredIds:[],preferences:{
    fitPreference:'balanced',favoriteColors:[],avoidColors:[],
    likedPatterns:[],dislikedPatterns:[],favoriteBrands:[],avoidBrands:[],
    primaryOccasions:[],styleGoals:[],prefersSustainable:false,
    priceRange:{min:0,max:500,currency:'USD'},
  }};
  assert.equal(anonymousPhotoJournalAllowed(base),true);
  for(const fit of ['balanced','loose','fitted'])
    assert.equal(anonymousPhotoJournalAllowed({...base,
      preferences:{...base.preferences,fitPreference:fit}}),true,
      'the THREE documented app fit choices must not all be dropped');
  for(const personal of [
    {favoriteColors:['red']},{avoidBrands:['Private Merchant']},
    {styleGoals:['private_goal']},{bodyType:'athletic'},
    {workDressCode:'formal'},{prefersSustainable:true},
  ])assert.equal(anonymousPhotoJournalAllowed({...base,
    preferences:{...base.preferences,...personal}}),true,
    'ordinary personal fashion context is allowed ONLY under unique HMAC scope');
  for(const altered of [
    {...base,sourceMode:'wardrobe'},
    {...base,sourceMode:'mixed'},
    {...base,requiredIds:['PRIVATE_ANCHOR_ID']},
    {...base,preferences:{...base.preferences,fitPreference:'PRIVATE PERSONAL FIT'}},
    {...base,preferences:{...base.preferences,unexpectedOwnerField:'phone'}},
    {...base,preferences:{...base.preferences,styleGoals:['mailto:private@example.com']}},
  ])assert.equal(anonymousPhotoJournalAllowed(altered),false);
});
test('independent personal color and fit contexts never share the same PHOTO training HMAC',()=>{
  const base=input();
  const one=compilePhotoRewardExamples({...base,context:{
    ...base.context,preferences:{fitPreference:'balanced',
      avoidColors:['navy'],favoriteBrands:['Private Merchant Brand']},
  }});
  const other=compilePhotoRewardExamples({...base,context:{
    ...base.context,preferences:{fitPreference:'balanced',
      avoidColors:['red'],favoriteBrands:['Private Merchant Brand']},
  }});
  const sameReordered=compilePhotoRewardExamples({...base,context:{
    ...base.context,preferences:{favoriteBrands:['Private Merchant Brand'],
      avoidColors:['navy'],fitPreference:'balanced'},
  }});
  assert.equal(one.length,2);
  assert.equal(other.length,2);
  assert.notEqual(one[0].context_hash,other[0].context_hash);
  assert.deepEqual(one,sameReordered,'equivalent preferences share privacy scope regardless of key order');
  assert.doesNotMatch(JSON.stringify(one),
    /navy|red|Private Merchant|PRIVATE|Brand|http|imageUrl|SKU|user_id/i);
});

test('real UI Old Money, Semi Classic and Date night aliases have canonical photo training contexts',()=>{
  const key='private-staging-hmac-domain-test-only-0123456';
  const prefs={fitPreference:'loose',favoriteColors:['cream'],styleGoals:['modern']};
  assert.equal(photoRewardContextHash({
    style:' Old Money ',occasion:'Date night',preferences:prefs,
  },key),photoRewardContextHash({
    style:'old_money',occasion:'date',preferences:prefs,
  },key));
  assert.equal(photoRewardContextHash({
    style:'Semi Classic',occasion:'Work',preferences:prefs,
  },key),photoRewardContextHash({
    style:'semi_classic',occasion:'work',preferences:prefs,
  },key));
  const custom=photoRewardContextHash({
    style:'new private style',occasion:'private wedding',preferences:prefs,
  },key);
  assert.match(custom,/^[0-9a-f]{64}$/,
    'a bounded custom occasion is isolated by server HMAC, never saved as plaintext');
  assert.notEqual(custom,photoRewardContextHash({
    style:'new private style',occasion:'work',preferences:prefs,
  },key),'custom and standard occasions must not mix photo-supervision');
  assert.equal(photoRewardContextHash({
    style:'new private style',occasion:'https://private.example/event',preferences:prefs,
  },key),null,'private URLs are NOT acceptable learning context');
});

test('actual RN stylePreferenceStore full typed state is accepted consistently for writer and reader',()=>{
  const secret='isolated-staging-private-hmac-0123456789abcdef';
  const preferences={
    bodyType:'athletic',stylePersonality:'semi-classic',
    favoriteColors:['Navy','Beige'],avoidColors:['red'],
    likedPatterns:['solid'],dislikedPatterns:['large graphic'],
    fitPreference:'loose',primaryOccasions:['work','date'],
    workDressCode:'business_casual',
    favoriteBrands:['Massimo Dutti'],avoidBrands:[],
    priceRange:{min:0,max:500,currency:'USD'},
    prefersSustainable:false,
    styleGoals:['get_styled','shop_smarter'],
  };
  const ctx={style:' Classic ',occasion:' Work ',preferences};
  const rows=compilePhotoRewardExamples({...input(),
    context:ctx,secret});
  assert.equal(rows.length,2);
  assert.equal(rows[0].context_hash,photoRewardContextHash(ctx,secret));
  assert.equal(anonymousPhotoJournalAllowed({
    sourceMode:'catalog',requiredIds:[],preferences}),true);
  assert.doesNotMatch(JSON.stringify(rows),
    /athletic|Massimo|Beige|Navy|Beige|work|red|styleGoals|Dutti/i);
});
