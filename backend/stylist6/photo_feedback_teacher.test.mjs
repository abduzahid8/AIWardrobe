import test from 'node:test';
import assert from 'node:assert/strict';
import {
  PHOTO_FEEDBACK_TEACHER_VERSION,trainPhotoFeedbackTeacher,
  photoTeacherScore,photoTaughtRescueOrder,
} from './photo_feedback_teacher.mjs';

// Balanced color occurrence: cannot get this right by imposing a global
// preference for black/brown jackets. Only TOP-LAYER outfit relations differ.
const factual=(good,i)=>[
 ['shirt','regular',i%2?'navy':'cream','cotton','plain'],
 ['blazer','tailored',good?(i%2?'grey':'brown'):(i%2?'brown':'grey'),'wool','plain'],
 ['trousers','straight','taupe','wool','plain'],
 ['shoe-'+i,'regular','black','leather','plain'],
];
const makeLook=(positive,i)=>({id:'PRIVATE_SOURCE_LOOK_'+(positive?'p':'n')+'_'+i,
  items:factual(positive,i).map(facts=>({facts}))});
const photo=(look,positive)=>({candidateId:look.id,reject:!positive,
  visualCohesion:positive?.87:.42,styleFit:positive?.9:.47,
  occasionFit:positive?.91:.55,formalityFit:positive?.9:.56,
  confidence:.92});
const input=(positives=4,negatives=4)=>{
 const positivesLooks=Array.from({length:positives},(_,i)=>makeLook(true,i));
 const negativesLooks=Array.from({length:negatives},(_,i)=>makeLook(false,i));
 const looks=[...positivesLooks,...negativesLooks];
 return {looks,decisions:looks.map(x=>({candidateId:x.id,
   reason:x.id.includes('_p_')?'approved':'below_quality_floor'})),
 critiques:looks.map(x=>photo(x,x.id.includes('_p_'))),featuresOf:items=>items.map(x=>x.facts)};
};

test('real photo teacher ABSTAINS with missing labels or one approved look',()=>{
 const nothing=trainPhotoFeedbackTeacher({});
 assert.equal(nothing.version,PHOTO_FEEDBACK_TEACHER_VERSION);
 assert.equal(nothing.usable,false);
 const sparse=trainPhotoFeedbackTeacher(input(1,6));
 assert.equal(sparse.usable,false);
 assert.equal(sparse.status,'insufficient_independent_photos');
 assert.equal(photoTeacherScore(sparse,makeLook(true,12),input().featuresOf),null);
});

test('independent PHOTO labels teach four-slot relationships, not garment bans',()=>{
 const evidence=input(4,4),learner=trainPhotoFeedbackTeacher(evidence);
 assert.equal(learner.examples,8);
 assert.equal(learner.positive,4);
 assert.equal(learner.negative,4);
 assert.equal(learner.status,'photo_validated');
 assert.equal(learner.usable,true);
 const good=photoTeacherScore(learner,makeLook(true,12),evidence.featuresOf);
 const bad=photoTeacherScore(learner,makeLook(false,12),evidence.featuresOf);
 assert.ok(Number.isFinite(good)&&Number.isFinite(bad));
 assert.ok(good>bad,{good,bad});
 assert.ok(learner.weights.has('p01:22=cream|brown'));
 assert.ok(learner.weights.has('p01:22=cream|grey'));
 // The critic is still the ONLY original-photo aesthetic approval authority.
 const pool=[makeLook(false,12),makeLook(true,12)];
 const ranked=photoTaughtRescueOrder(pool,learner,evidence.featuresOf);
 assert.deepEqual(ranked.map(x=>x.id),[pool[1].id,pool[0].id]);
 assert.equal(pool[0].id,makeLook(false,12).id,'source menu stays unchanged');
});

test('ONE photo-approved look can still learn relative photographic quality across rejected looks',()=>{
  // Real mobile symptom: 1/5 photo-approved, while other four have usable
  // photographic quality and occasion scores but did not reach final floor.
  // None of those rejected items can be published without a fresh jury.
  const evidence=input(1,4);
  const actual={...evidence,
    decisions:evidence.decisions.map((row,i)=>({
      ...row,quality:i===0?.91:i%2?.65:.44,
    })),
    critiques:evidence.critiques.map((row,i)=>({
      ...row,reject:false,
      visualCohesion:i===0?.92:i%2?.65:.44,
    })),
  };
  const model=trainPhotoFeedbackTeacher(actual);
  assert.equal(model.positive,1);
  assert.equal(model.negative,4,
    'quality floor failures are genuine photographic NEGATIVE examples even if model reject=false');
  assert.equal(model.examples,5);
  assert.equal(model.status,'graded_photo_validated');
  assert.equal(model.usable,true);
  const offered=[
    makeLook(false,88),makeLook(false,89),
  ];
  const ranked=photoTaughtRescueOrder(offered,model,actual.featuresOf);
  assert.deepEqual(ranked.map(x=>x.id),[offered[0].id,offered[1].id],
    'teacher orders real unapproved SKU candidates but never declares either approved');
  assert.ok(photoTeacherScore(model,offered[0],actual.featuresOf)>
    photoTeacherScore(model,offered[1],actual.featuresOf));
});

test('explicit avoided garment colors and ambiguous no-photo verdicts are NEVER generalized into fashion taste',()=>{
  const evidence=input(1,4);
  const rows=evidence.decisions.map((row,i)=>({
    ...row,reason:i===0?'approved':'excluded_visual_color',
    quality:i===0?.9:.4,
  }));
  const teacher=trainPhotoFeedbackTeacher({...evidence,decisions:rows});
  assert.equal(teacher.negative,0);
  assert.equal(teacher.usable,false);
  assert.equal(teacher.status,'insufficient_independent_photos');
  const ambiguity=trainPhotoFeedbackTeacher({...evidence,
    decisions:rows.map(row=>({...row,reason:'low_confidence'}))});
  assert.equal(ambiguity.examples,0);
});

test('photo-critic labels are NOT owner taste, and judge conflicts are inert',()=>{
 const evidence=input(4,4);
 const fake={...evidence,critiques:evidence.critiques.map(x=>({...x,reject:!x.reject}))};
 const untrusted=trainPhotoFeedbackTeacher(fake);
 assert.equal(untrusted.usable,false);
 assert.equal(untrusted.positive,0,
   'a reversed photo verdict cannot create a fictitious approval');
 assert.equal(untrusted.negative,4,
   'a real independent below-floor PHOTO quality decision stays learnable');
 const contradictory={...evidence,decisions:evidence.decisions.map(d=>({
   ...d,reason:d.reason==='approved'?'model_veto':'approved',
 }))};
 assert.equal(trainPhotoFeedbackTeacher(contradictory).usable,false,
   'a forged PASS in diagnostics cannot override the independent photo rejection');
});

test('no identity/photo bytes or user text are features and no artificial PASS occurs',()=>{
 const original=input(4,4);
 const variant={...original,looks:original.looks.map((x,i)=>({...x,
   id:x.id,owner:'PRIVATE_OWNER_'+i,originalPhoto:'https://private.invalid/'+i,
   items:x.items.map(y=>({...y,merchantToken:'PRIVATE_'+i,imageUrl:'https://private.invalid'})),
 }))};
 const m=trainPhotoFeedbackTeacher(variant);
 assert.equal(m.usable,true);
 assert.doesNotMatch(JSON.stringify([...m.weights.keys()]),
   /PRIVATE_|private\.invalid|imageUrl|originalPhoto|owner/);
 assert.deepEqual(photoTaughtRescueOrder([],m,variant.featuresOf),[]);
 assert.deepEqual(photoTaughtRescueOrder(original.looks,{usable:false},original.featuresOf),original.looks);
});
