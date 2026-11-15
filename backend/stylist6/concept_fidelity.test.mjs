import test from 'node:test';
import assert from 'node:assert/strict';
import { evaluateConceptFidelity, verifiedPaletteMatch, CONCEPT_FIDELITY_VERSION } from './concept_fidelity.mjs';
const item=(id,slot,subtype,color)=>({id,name:`${color} ${subtype} ${slot}`,macroCategory:slot,subtype,color});
const sharp=[item('shirt','top','dress-shirt','white'),item('jacket','outerwear','blazer','navy'),
  item('trouser','bottom','tailored-trouser','grey'),item('shoes','shoes','derby','black')];
const leisure=[item('shirt','top','t-shirt','white'),item('jacket','outerwear','cardigan','navy'),
  item('trouser','bottom','jeans','blue'),item('shoes','shoes','sneaker','black')];
const office={id:'precise-office',roles:{top:['dress-shirt'],outerwear:['blazer'],
  bottom:['tailored-trouser'],shoes:['derby']},palette:['white','navy','grey','black']};
test('full outfit role and palette fidelity separates a tailored concept from an off-brief capsule',()=>{
  const tailored=evaluateConceptFidelity(sharp,office),off=evaluateConceptFidelity(leisure,office);
  assert.equal(tailored.version,CONCEPT_FIDELITY_VERSION);
  assert.equal(tailored.roleFit,1);
  assert.equal(off.roleFit,0);
  assert.ok(tailored.adjustment > off.adjustment+.06);
  assert.deepEqual(off.mismatchedSlots,['top','outerwear','bottom','shoes']);
});
test('identical garments are scored against their actual concept, not only the user occasion',()=>{
  const relaxed={...office,id:'soft-knit',roles:{...office.roles,top:['t-shirt'],outerwear:['cardigan'],bottom:['jeans'],shoes:['sneaker']}};
  assert.ok(evaluateConceptFidelity(leisure,relaxed).score > evaluateConceptFidelity(leisure,office).score);
});
test('locked user pieces and absent palette evidence never create fabricated mismatches',()=>{
  const anchored=evaluateConceptFidelity(leisure,office,{requiredIds:leisure.map(x=>x.id)});
  assert.equal(anchored.score,null);
  assert.equal(anchored.adjustment,0);
  const partial=evaluateConceptFidelity([sharp[0],{...sharp[1],color:'unknown',name:'blazer'}],office);
  assert.ok(partial.score > 0);
  assert.ok(partial.adjustment<=.03 && partial.adjustment>=-.05);
});
test('owned wardrobe and absent concepts can keep neutral ranking independently',()=>{
  const x=evaluateConceptFidelity(sharp,null);
  assert.equal(x.score,null);
  assert.equal(x.adjustment,0);
});


test('verified color tokens recover real navy, white, grey hues lost in coarse tone bins',()=>{
  assert.equal(verifiedPaletteMatch({name:'Navy pure wool blazer',color:'cool-dark'},['navy','white']),true);
  assert.equal(verifiedPaletteMatch({name:'White cotton shirt',color:'light'},['navy','white']),true);
  assert.equal(verifiedPaletteMatch({name:'Grey tailored trousers',color:'mid'},['navy','white']),false);
  assert.equal(verifiedPaletteMatch({name:'No color provided',color:'cool-dark'},['navy','white']),null);
  assert.equal(verifiedPaletteMatch({name:'Black leather derby',color:'dark'},['navy']),false);
});
