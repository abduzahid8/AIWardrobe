import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DIRECTION_VERSION,
  directionDifference,
  labelCreativeDirections,
  selectCreativeDirectionDeck,
} from './creative_directions.mjs';

const slot = (subtype,color,volume='regular',texture='smooth',pattern='solid') =>
  [subtype,volume,color,texture,pattern];
const SHIRT = [
  slot('shirt','blue','slim'),
  null,
  slot('tailored-trouser','navy','slim'),
  slot('loafer','brown'),
];
const FRESH = [
  slot('shirt','cream','slim'),
  null,
  slot('tailored-trouser','grey','slim'),
  slot('loafer','brown'),
];
const EXPLORE = [
  slot('fine-knit','cream','regular','knit'),
  null,
  slot('chino','olive','wide','textured'),
  slot('chelsea','burgundy'),
];
function look(id,quality,features,keys=[id+'-top',id+'-bottom',id+'-shoes'],briefId='brief-'+id) {
  return {
    id,briefId,features,
    items:keys.map(k=>({id:k})),fashion:{score:quality},
  };
}
function distance(a,b) {
  const weights=[.35,.20,.20,.15,.10];
  let possible=0,changed=0;
  for(let s=0;s<4;s++){
    const x=a[s],y=b[s];
    if(!x||!y)continue;
    for(let j=0;j<5;j++){
      if(!x[j]||!y[j]||x[j]==='unknown'||y[j]==='unknown')continue;
      possible+=weights[j];
      if(x[j]!==y[j])changed+=weights[j];
    }
  }
  return possible>=.7?changed/possible:0;
}
const metrics={
  featuresOf:look=>look.features,
  featureDistance:distance,
  overlap:(a,b)=>a.items.filter(x=>b.items.some(y=>y.id===x.id)).length,
};

test('three directions: closest style, fresh interpretation, genuine discovery',()=>{
  const pool=[
    look('signature',.94,SHIRT),
    look('bland',.93,SHIRT),
    look('fresh',.91,FRESH),
    look('explore',.89,EXPLORE),
    look('off-quality',.50,[
      slot('graphic-tee','neon-red','wide'),
      null,slot('shorts','neon-yellow'),slot('sneaker','neon-green')
    ]),
  ];
  const selected=selectCreativeDirectionDeck(pool,metrics);
  assert.equal(selected.diagnostics.directionVersion,DIRECTION_VERSION);
  assert.deepEqual(selected.looks.map(x=>x.id),['signature','fresh','explore']);
  assert.deepEqual(selected.directions.map(x=>x.id),['signature','refresh','discovery']);
  assert.equal(selected.diagnostics.qualified,3);
  assert.ok(selected.directions[2].evidence.structuralSlots.length>=1);
  assert.ok(selected.directions[2].evidence.distance>=.24);
  assert.equal(selected.diagnostics.qualityEligible,4);
});

test('onboarding can be empty: discovery uses evidence, not invented preferences',()=>{
  const pool=[look('s',.93,SHIRT),look('f',.90,FRESH),look('d',.87,EXPLORE)];
  const a=selectCreativeDirectionDeck(pool,metrics);
  const b=selectCreativeDirectionDeck([...pool].reverse(),metrics);
  assert.deepEqual(a.looks.map(x=>x.id),b.looks.map(x=>x.id));
  assert.deepEqual(a.directions.map(x=>x.id),['signature','refresh','discovery']);
  assert.ok(!('preferences' in a.directions[2]));
});

test('mere replacement SKUs and one-color variations do not claim discovery',()=>{
  const onlyColor= [
    SHIRT,
    [slot('shirt','white','slim'),null,SHIRT[2],SHIRT[3]],
    [SHIRT[0],null,SHIRT[2],slot('loafer','black')],
  ].map((features,i)=>look('color-'+i,.91-i*.01,features));
  const result=selectCreativeDirectionDeck(onlyColor,metrics);
  assert.equal(result.looks.length,3);
  assert.equal(result.directions[0].id,'signature');
  assert.ok(result.directions.slice(1).every(x=>x.id==='alternative'));
  assert.ok(result.directions.slice(1).every(x=>x.qualified===false));
});

test('identical or shared physical products cannot be recycled to fake diversity',()=>{
  const one=look('one',.94,SHIRT,['anchor','pants-1','shoes-1']);
  const overlap=look('other',.93,FRESH,['anchor','pants-1','shoes-2']);
  const valid=look('valid',.90,EXPLORE,['anchor','pants-2','shoes-3']);
  const result=selectCreativeDirectionDeck([one,overlap,valid],metrics);
  assert.deepEqual(result.looks.map(x=>x.id),['one','valid']);
  assert.equal(result.looks.length,2);
  assert.equal(result.diagnostics.qualified<=2,true);
});

test('unknown measurements and locked-anchor feature gaps create no fake novelty',()=>{
  const unknown=[slot('shirt','unknown','unknown','unknown','unknown'),null,
    slot('trouser','unknown','unknown','unknown','unknown'),slot('loafer','unknown','unknown','unknown','unknown')];
  const diff=directionDifference(look('a',.9,unknown),look('b',.8,unknown),
    metrics.featuresOf,metrics.featureDistance);
  assert.equal(diff.distance,0);
  assert.deepEqual(diff.changedSlots,[]);
  assert.deepEqual(labelCreativeDirections([
    look('a',.9,unknown),look('b',.8,unknown),
  ],metrics).map(x=>x.id),['signature','alternative']);
});

test('genuine changes in a mandatory layer count as styling architecture',()=>{
  const layered=[SHIRT[0],slot('blazer','navy'),SHIRT[2],SHIRT[3]];
  const base=look('a',.91,SHIRT),candidate=look('b',.90,layered);
  const diff=directionDifference(base,candidate,metrics.featuresOf,metrics.featureDistance);
  assert.ok(diff.structuralSlots.includes('outerwear'));
  assert.ok(diff.distance>=.3);
});

test('low quality or missing candidate metadata cannot force a third direction',()=>{
  const candidates=[look('s',.94,SHIRT),look('f',.91,FRESH),
    look('danger',.67,EXPLORE),{id:'bad',items:[{id:'bad'}]}];
  const result=selectCreativeDirectionDeck(candidates,metrics);
  assert.equal(result.looks.length,2);
  assert.ok(result.looks.every(x=>x.id!=='danger'&&x.id!=='bad'));
  assert.equal(result.directions[1].id,'refresh');
});

test('third direction must be an editorial-quality discovery, not just a different look',()=>{
  const scored=(id,rank,features,editorial)=>{
    const result=look(id,rank,features);
    result.fashion.editorial={score:editorial};
    return result;
  };
  const pool=[
    scored('anchor',.95,SHIRT,.92),
    scored('refresh',.94,FRESH,.89),
    scored('flashy-but-poor',.935,EXPLORE,.63),
    scored('thoughtful',.91,[
      slot('fine-knit','taupe','regular','knit'),null,
      slot('chino','brown','regular','textured'),slot('loafer','oxblood')
    ],.88),
  ];
  const result=selectCreativeDirectionDeck(pool,metrics);
  assert.deepEqual(result.looks.map(x=>x.id),['anchor','refresh','thoughtful']);
  assert.deepEqual(result.directions.map(x=>x.id),['signature','refresh','discovery']);
  assert.equal(result.diagnostics.qualified,3);
});

test('an eligible but weaker third look is not mislabelled Discovery',()=>{
  const scored=(id,rank,features,quality)=>{
    const result=look(id,rank,features);
    result.fashion.editorial={score:quality};
    return result;
  };
  const result=selectCreativeDirectionDeck([
    scored('anchor',.95,SHIRT,.91),
    scored('refresh',.93,FRESH,.87),
    scored('weak-different',.92,EXPLORE,.65),
  ],metrics);
  assert.equal(result.looks.length,2);
  assert.deepEqual(result.directions.map(x=>x.id),['signature','refresh']);
  assert.deepEqual(labelCreativeDirections([
    scored('anchor',.95,SHIRT,.91),
    scored('weak-different',.92,EXPLORE,.65),
  ],metrics).map(x=>x.id),['signature','alternative']);
});
