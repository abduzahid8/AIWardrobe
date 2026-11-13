import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DIRECTION_VERSION,
  brandVarietyAdjustment,
  directionDifference,
  footwearVarietyAdjustment,
  lowerBodyPairRepeatAdjustment,
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

test('footwear variety favors distinct physical shoes without blocking sparse catalogs',()=>{
  const make=(id,shoe)=>({
    id,items:[{id:id+'-top',macroCategory:'top'}, {id:shoe,macroCategory:'shoes'}],
    fashion:{score:.90},
  });
  const base=make('base','black-derby');
  const same=make('same','black-derby');
  const different=make('different','brown-loafer');
  assert.equal(footwearVarietyAdjustment(same,[base],metrics.overlap),-.035);
  assert.equal(footwearVarietyAdjustment(different,[base],metrics.overlap),.012);
  assert.equal(footwearVarietyAdjustment(base,[],metrics.overlap),0);
  assert.equal(footwearVarietyAdjustment({id:'missing',items:[{id:'top',macroCategory:'top'}]},[base],metrics.overlap),0);
});

test('within comparable quality, creative second pick prefers non-reused footwear',()=>{
  const withShoe=(id,quality,features,shoeId)=>{
    const result=look(id,quality,features,[id+'-top',id+'-bottom',shoeId]);
    result.items[0].macroCategory='top';
    result.items[1].macroCategory='bottom';
    result.items[2].macroCategory='shoes';
    return result;
  };
  const lead=withShoe('lead',.94,SHIRT,'shoes-one');
  const repeated=withShoe('repeated',.922,FRESH,'shoes-one');
  const rotated=withShoe('rotated',.917,FRESH,'shoes-two');
  const result=selectCreativeDirectionDeck([lead,repeated,rotated],metrics);
  assert.equal(result.looks[0].id,'lead');
  assert.equal(result.looks[1].id,'rotated');
  assert.ok(result.directions[1].id==='refresh');
});

test('owner QA: near-tie must prefer another trouser SKU when shoe is held constant',()=>{
  const features=[
    slot('fine-knit','cream','regular','knit'), null,
    slot('tailored-trouser','navy','slim'),slot('loafer','brown'),
  ];
  const make=(id,quality,shape,bottom,shoe)=>{
    const result=look(id,quality,shape,
      [id+'-top',bottom,shoe]);
    result.items.forEach((item,i)=>{item.macroCategory=['top','bottom','shoes'][i]});
    return result;
  };
  const lead=make('lead',.950,SHIRT,'navy-trouser','brown-loafer');
  const repeated=make('repeated',.927,features,'navy-trouser','brown-loafer');
  const alternate=make('alternate',.922,features,'other-navy-trouser','brown-loafer');
  assert.equal(lowerBodyPairRepeatAdjustment(repeated,[lead],metrics.overlap),-.025);
  assert.equal(lowerBodyPairRepeatAdjustment(alternate,[lead],metrics.overlap),0);
  assert.equal(lowerBodyPairRepeatAdjustment(lead,[],metrics.overlap),0);
  // Both upper-body refreshes lack distinct Discovery evidence; the
  // selector must not invent a third qualified direction merely to fill
  // a slot. It should still select the genuinely varied trouser.
  assert.deepEqual(selectCreativeDirectionDeck([lead,repeated,alternate],metrics)
    .looks.map(x=>x.id),['lead','alternate']);
  // No hard ban: the adjustment itself is finite, not a failed hard
  // gate. A two-look catalog may still return only the signature when
  // editorial discovery is too weak; do not manufacture a third look.
  const sparse=selectCreativeDirectionDeck([lead,repeated],metrics);
  assert.deepEqual(sparse.looks.map(x=>x.id),['lead']);
  assert.equal(sparse.diagnostics.repeatedLowerPairs,0);
});

test('pair-repeat evidence is neutral if only one lower slot repeats or metadata missing',()=>{
  const make=(id,bottom,shoe)=>({
    id,items:[{id:'top-'+id,macroCategory:'top'},
      ...(bottom?[{id:bottom,macroCategory:'bottom'}]:[]),
      ...(shoe?[{id:shoe,macroCategory:'shoes'}]:[])],
  });
  const lead=make('a','bottom-a','shoe-a');
  assert.equal(lowerBodyPairRepeatAdjustment(make('b','bottom-b','shoe-a'),[lead],metrics.overlap),0);
  assert.equal(lowerBodyPairRepeatAdjustment(make('b','bottom-a','shoe-b'),[lead],metrics.overlap),0);
  assert.equal(lowerBodyPairRepeatAdjustment(make('b',null,'shoe-a'),[lead],metrics.overlap),0);
  assert.equal(lowerBodyPairRepeatAdjustment(null,[lead],metrics.overlap),0);
});

// ---------------------------------------------------------------------------
// Deck brand-monotony tie-break (s6-three-directions-editorial-v4). Same
// bounded-magnitude convention as the footwear-variety tie-break: a look
// whose whole brand set was already drawn pays at most -0.035 (scaled by the
// repeat share), a new brand set earns +0.012; brand-less looks stay neutral.
// ---------------------------------------------------------------------------

test('brand variety adjustment is proportional, bounded and neutral without brand evidence',()=>{
  const branded=(id,quality,brands)=>({
    id,items:brands.map((brand,index)=>({id:id+'-item-'+index,brand})),fashion:{score:quality},
  });
  const lead=branded('lead',.94,['alpha','beta']);
  // Full brand-set repeat pays exactly the footwear-repeat magnitude.
  assert.equal(brandVarietyAdjustment(branded('same',.92,['beta','alpha']),[lead]),-.035);
  // Partial repeat scales with the repeat share.
  assert.equal(brandVarietyAdjustment(branded('half',.92,['alpha','gamma']),[lead]),-.0175);
  // An all-new brand set earns the same nudge as a new physical shoe.
  assert.equal(brandVarietyAdjustment(branded('new',.92,['gamma','delta']),[lead]),.012);
  // No earlier looks, or no known brands on either side: neutral, never invented.
  assert.equal(brandVarietyAdjustment(branded('fresh',.92,['alpha']),[]),0);
  assert.equal(brandVarietyAdjustment(branded('unbranded',.92,['','  ',null]),[lead]),0);
  assert.equal(brandVarietyAdjustment(branded('named',.92,['alpha']),
    [branded('anon',.93,['',''])]),0);
  // Alias-normalized house names count as the same brand.
  assert.equal(brandVarietyAdjustment(branded('alias',.92,[' ALPHA ','beta']),[lead]),-.035);
});

test('within comparable quality a deck that repeats one brand house loses the near-tie',()=>{
  const branded=(id,quality,features,brands)=>{
    const result=look(id,quality,features,[id+'-top',id+'-bottom',id+'-shoes']);
    result.items.forEach((item,index)=>{
      item.macroCategory=['top','bottom','shoes'][index];
      item.brand=brands[index];
    });
    return result;
  };
  const lead=branded('lead',.94,SHIRT,['alpha','alpha','alpha']);
  const repeated=branded('repeated',.922,FRESH,['alpha','alpha','alpha']);
  const varied=branded('varied',.917,FRESH,['gamma','alpha','alpha']);
  // Quality gap .922 vs .917 is inside the tie-break window: the deck that
  // repeats the same brand set everywhere must lose to the varied look.
  const result=selectCreativeDirectionDeck([lead,repeated,varied],metrics);
  assert.equal(result.looks[1].id,'varied');
  // But a clear quality winner is never overridden: a candidate .018 above
  // the varied alternative still takes the refresh slot despite paying the
  // full same-brand penalty, and the highest-quality look always keeps the
  // signature card (the tie-break never applies to the lead).
  const clearWinner=branded('clear',.935,FRESH,['alpha','alpha','alpha']);
  const clearResult=selectCreativeDirectionDeck([lead,clearWinner,varied],metrics);
  assert.equal(clearResult.looks[0].id,'lead');
  assert.equal(clearResult.looks[1].id,'clear');
  const topResult=selectCreativeDirectionDeck([lead,repeated,
    branded('stronger',.95,FRESH,['alpha','alpha','alpha'])],metrics);
  assert.equal(topResult.looks[0].id,'stronger');
  // A single-brand catalog pays a constant offset: relative order is intact.
  const mono=selectCreativeDirectionDeck([
    branded('m-lead',.94,SHIRT,['alpha','alpha','alpha']),
    branded('m-second',.922,FRESH,['alpha','alpha','alpha']),
  ],metrics);
  assert.equal(mono.looks[1].id,'m-second');
});
