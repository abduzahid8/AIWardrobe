import test from 'node:test';
import assert from 'node:assert/strict';
import { EDITORIAL_QUALITY_VERSION, evaluateEditorialQuality, editorialDirectionEligible } from './editorial_quality.mjs';

const context = {style:'minimalist',occasion:'date night',dressCode:'smart',weather:null,preferences:{}};
const polished = {dressCode:.93,style:.91,footwear:.96,proportion:.86,palette:.87,texture:.83,layer:.7};
const underdressed = {dressCode:.44,style:.49,footwear:.30,proportion:.64,palette:.60,texture:.7,layer:.7};
const shoe=(vivid,color='red')=>({slot:'shoes',subtype:'sneaker',color,vividFootwear:vivid});

test('whole-look quality penalizes an occasion/register mismatch, despite creative difference', () => {
  const good=evaluateEditorialQuality({dimensions:polished,context,descriptors:[shoe(false)]});
  const poor=evaluateEditorialQuality({dimensions:underdressed,context,descriptors:[shoe(true)]});
  assert.equal(good.version,EDITORIAL_QUALITY_VERSION);
  assert.ok(good.score>.83);
  assert.ok(poor.score < good.score-.20);
  assert.ok(good.adjustment > poor.adjustment);
  assert.ok(poor.evidence.includes('weak_occasion_style_or_silhouette_relationship'));
  assert.ok(poor.evidence.includes('unrequested_statement_footwear_in_quiet_style'));
  assert.ok(good.adjustment<=.04&&poor.adjustment>=-.08);
});

test('quiet-style footwear penalty is bounded and waived by an explicit favorite color', () => {
  const plain=evaluateEditorialQuality({dimensions:polished,context,descriptors:[shoe(false)]});
  const vivid=evaluateEditorialQuality({dimensions:polished,context,descriptors:[shoe(true)]});
  const chosen=evaluateEditorialQuality({dimensions:polished,
    context:{...context,preferences:{favoriteColors:['Red']}},descriptors:[shoe(true)]});
  assert.ok(vivid.score<plain.score);
  assert.equal(chosen.score,plain.score);
});

test('no extrapolation from unavailable weather or missing metadata',()=>{
  const unknown=evaluateEditorialQuality({context,descriptors:[shoe(true)]});
  assert.equal(unknown.score,null);
  assert.equal(unknown.adjustment,0);
  assert.deepEqual(unknown.evidence,[]);
  const noShoe=evaluateEditorialQuality({dimensions:polished,context});
  assert.ok(Number.isFinite(noShoe.score));
  assert.ok(!noShoe.evidence.includes('unrequested_statement_footwear_in_quiet_style'));
});

test('two prominent statement garments are a soft caution only in quiet styles',()=>{
  const descriptors=[{slot:'top',pattern:'graphic',patternProminence:'prominent'},
    {slot:'outerwear',pattern:'print',patternProminence:'prominent'}];
  const quiet=evaluateEditorialQuality({dimensions:polished,context,descriptors,hasLayer:true});
  const expressive=evaluateEditorialQuality({dimensions:polished,
    context:{...context,style:'streetwear'},descriptors,hasLayer:true});
  assert.ok(quiet.score<expressive.score);
  assert.ok(quiet.evidence.includes('multiple_statement_pieces_in_quiet_style'));
});

test('discovery requires absolute and relative editorial quality; refresh and fallbacks have bounded latitude',()=>{
  const make=(quality)=>({fashion:{editorial:{score:quality}}});
  assert.equal(editorialDirectionEligible(make(.91),make(.65),'discovery'),false);
  assert.equal(editorialDirectionEligible(make(.91),make(.82),'discovery'),false);
  assert.equal(editorialDirectionEligible(make(.91),make(.87),'discovery'),true);
  assert.equal(editorialDirectionEligible(make(.91),make(.84),'refresh'),true);
  assert.equal(editorialDirectionEligible(make(.91),make(.77),'alternative'),true);
  assert.equal(editorialDirectionEligible(make(.91),make(.65),'alternative'),false);
  assert.equal(editorialDirectionEligible(make(.70),make(.68),'discovery'),false);
  assert.equal(editorialDirectionEligible(make(.90),{fashion:{}},'discovery'),false);
  assert.equal(editorialDirectionEligible({fashion:{}},{fashion:{}},'discovery'),true);
});
