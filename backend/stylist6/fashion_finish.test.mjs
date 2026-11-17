import test from 'node:test';
import assert from 'node:assert/strict';
import { evaluateFashionFinish, FASHION_FINISH_VERSION } from './fashion_finish.mjs';

const ctx = {sourceMode:'catalog',style:'classic',occasion:'Work',request:'',preferences:{},weather:{temp:18}};
const garment=(slot,subtype,color,{texture='smooth',materials=[],pattern='solid',casualSole=false}={})=>({
  item:{id:`${slot}-${color}`,macroCategory:slot,name:`${color} ${subtype}`,color},
  d:{slot,subtype,color,texture,materials,pattern,casualSole},
});
const evaluate=(rows, context=ctx, requiredIds=[])=>evaluateFashionFinish({
  items:rows.map(r=>r.item),descriptors:rows.map(r=>r.d),context,requiredIds,
});
const knit=[
  garment('top','fine-knit','cream',{texture:'knit',materials:['cashmere']}),
  garment('outerwear','cardigan','beige',{texture:'knit'}),
  garment('bottom','chino','brown'),
];

test('earth-tone knit with grounded suede loafers outranks unsupported black footwear',()=>{
  const brown=evaluate([...knit,garment('shoes','loafer','brown',{texture:'suede',materials:['suede']})]);
  const black=evaluate([...knit,garment('shoes','loafer','black')]);
  assert.equal(brown.version,FASHION_FINISH_VERSION);
  assert.equal(brown.active,true);
  assert.ok(brown.paletteHarmony>black.paletteHarmony);
  assert.ok(brown.textureCoordination>black.textureCoordination);
  assert.ok(brown.adjustment>black.adjustment+.02);
  assert.ok(black.evidence.includes('unanchored_black_footwear_with_earth_palette'));
});

test('black footwear is supported by charcoal or navy garment evidence',()=>{
  const outfit=[
    garment('top','dress-shirt','white'),
    garment('outerwear','blazer','navy'),
    garment('bottom','tailored-trouser','charcoal'),
    garment('shoes','derby','black'),
  ];
  const finish=evaluate(outfit);
  assert.ok(finish.paletteHarmony>0);
  assert.ok(finish.shoeAppropriateness>0);
  assert.ok(finish.silhouetteBalance>0);
  assert.ok(finish.evidence.includes('dark_footwear_color_anchor'));
  assert.ok(!finish.evidence.includes('unanchored_black_footwear_with_earth_palette'));
});

test('explicit black shoe preference and requested sneakers are not overridden',()=>{
  const dark=evaluate([...knit,garment('shoes','loafer','black')],{...ctx,request:'Black shoes, please'});
  assert.ok(!dark.evidence.includes('unanchored_black_footwear_with_earth_palette'));
  const sprint=evaluate([...knit,garment('shoes','sport-sneaker','brown')],{...ctx,request:'Wear sneakers'});
  assert.equal(sprint.shoeAppropriateness,0);
  const favorites=evaluate([...knit,garment('shoes','loafer','black')],{
    ...ctx,preferences:{favoriteColors:['black']},
  });
  assert.ok(!favorites.evidence.includes('unanchored_black_footwear_with_earth_palette'));
});

test('unlayered plain business uniform is only a small tie-breaker unless requested or hot',()=>{
  const plain=[garment('top','dress-shirt','white'),
    garment('bottom','tailored-trouser','navy'),garment('shoes','oxford','black')];
  assert.equal(evaluate(plain).genericOfficePenalty,-.009);
  assert.equal(evaluate(plain,{...ctx,request:'No blazer or jacket'}).genericOfficePenalty,0);
  assert.equal(evaluate(plain,{...ctx,weather:{temp:33}}).genericOfficePenalty,0);
  assert.equal(evaluate(plain,{...ctx,weather:{temp:33}}).active,true);
});

test('unknown colors and fabric evidence cannot manufacture palettes or shoe support',()=>{
  const unknown=[garment('top','unknown','unknown'),garment('bottom','unknown','unknown'),
    garment('shoes','unknown','unknown')];
  const finish=evaluate(unknown);
  assert.equal(finish.adjustment,0);
  assert.deepEqual(finish.evidence,[]);
  assert.equal(finish.paletteHarmony,0);
  assert.equal(finish.shoeAppropriateness,0);
});

test('no changes to wardrobe, mixed, casual, sport, or pinned-anchor requests',()=>{
  const rows=[...knit,garment('shoes','loafer','brown',{texture:'suede'})];
  for(const context of [{...ctx,sourceMode:'wardrobe'},
    {...ctx,sourceMode:'mixed'},
    {...ctx,style:'casual'},
    {...ctx,occasion:'Workout'}]) {
    const result=evaluate(rows,context);
    assert.equal(result.active,false);
    assert.equal(result.adjustment,0);
  }
  const anchor=evaluate(rows,ctx,['top-cream']);
  assert.equal(anchor.active,false);
  assert.equal(anchor.adjustment,0);
});

test('bounded scoring cannot rescue a decisively worse candidate',()=>{
  const candidates=[
    [...knit,garment('shoes','loafer','brown',{texture:'suede',materials:['suede']})],
    [...knit,garment('shoes','loafer','black')],
    [garment('top','shirt','red'),garment('outerwear','blazer','yellow'),
     garment('bottom','trouser','burgundy'),garment('shoes','oxford','black')],
  ];
  for(const rows of candidates) {
    const x=evaluate(rows);
    assert.ok(x.adjustment>=-.045 && x.adjustment<=.038);
    assert.ok(Number.isFinite(x.adjustment));
  }
  const gap=.2;
  const a=evaluate(candidates[0]).adjustment;
  const b=evaluate(candidates[2]).adjustment;
  assert.ok(gap>Math.abs(a-b));
});
