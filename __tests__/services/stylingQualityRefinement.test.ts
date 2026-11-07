import {garmentContextReason, scoreGarmentForContext, officeRefinementPenalty, compareUnbudgetedValue} from '../../src/domain/garmentContext';
import {describeGarment} from '../../src/domain/garmentDescriptor';
import {outfitCoherence} from '../../src/domain/outfitCoherence';
import {stylingContext} from '../../src/domain/stylingContext';
import {styleJudgeReason, detectSlotMismatch} from '../../src/domain/occasionRules';
import {fourSlotOutfitReason} from '../../src/domain/outfitStructure';

const work = stylingContext({style:'old_money', occasion:'Work'});
const formal = stylingContext({style:'classic', occasion:'Formal'});
const casual = stylingContext({style:'old_money', occasion:'Casual'});
const item = (name:string, macroCategory:string, extra:Record<string,unknown>={}) => ({id:macroCategory, name, macroCategory, ...extra});
const look = () => [
  item('White long-sleeve cotton poplin dress shirt','top'),
  item('Navy regular fit smooth wool blazer','outerwear'),
  item('Grey regular fit tailored wool trousers','bottom'),
  item('Brown leather penny loafers with slim leather outsole','shoes'),
];

test.each(['Short-sleeve cotton shirt','Short Sleeved Oxford Shirt','Shortsleeve poplin dress shirt'])(
  'strict Formal rejects explicit sleeve evidence: %s; Work/Casual remain possible', name => {
    const top=item(name,'top'), parts=look(); parts[0]=top;
    expect(garmentContextReason(top, formal)).toBe('formal_requires_long_sleeve_dress_base');
    expect(styleJudgeReason(parts,'classic','Formal')).not.toBeNull();
    expect(garmentContextReason(top, work)).toBeNull();
    expect(garmentContextReason(top, casual)).toBeNull();
    expect(fourSlotOutfitReason(parts)).toBeNull();
  });
test('unknown sleeve length stays unknown; long sleeves remain suitable',()=>{
  for(const name of ['White cotton poplin shirt','White long-sleeve dress shirt'])
    expect(garmentContextReason(item(name,'top'),formal)).toBeNull();
});
test('casual outsole construction outweighs a dress/penny marketing label; rubber alone does not',()=>{
  const chunky=item('Leather dress penny loafers with chunky rubber sole','shoes');
  const thin=item('Leather dress penny loafers with slim rubber sole','shoes');
  expect(describeGarment(chunky).casualSole).toBe(true);
  expect(describeGarment(item('Chunky rubber-soled penny loafers','shoes')).casualSole).toBe(true);
  expect(describeGarment(chunky).formality).not.toBe('formal');
  expect(garmentContextReason(chunky,formal)).toBe('formal_requires_dress_footwear');
  expect(describeGarment(thin).casualSole).toBe(false);
  expect(garmentContextReason(thin,formal)).toBeNull();
  expect(scoreGarmentForContext(thin,work)).toBeGreaterThan(scoreGarmentForContext(chunky,work));
  expect(officeRefinementPenalty(chunky,casual)).toBe(0);
});
test('the real imported Flextride row uses verified model evidence without mutating the catalog',()=>{
  const rows=require('../../data/shop_catalog_geox_shoes_20261030.json');
  const row=rows.find((r:any)=>r.id==='geox-u66mac0009bc9999');
  const shoe=Object.freeze({...row,macroCategory:'shoes',isShopItem:true});
  const before=JSON.stringify(shoe);
  expect(describeGarment(shoe).casualSole).toBe(true);
  expect(describeGarment(shoe).evidence).toContain('catalog:verified-comfort-sole');
  expect(garmentContextReason(shoe,work)).toBeNull();
  expect(garmentContextReason(shoe,formal)).toBe('formal_requires_dress_footwear');
  expect(scoreGarmentForContext(look()[3],work)).toBeGreaterThan(scoreGarmentForContext(shoe,work));
  expect(JSON.stringify(shoe)).toBe(before);
});
test('technical leisure and sport stripes lose office rank, without banning dress stripes or performance wool',()=>{
  const leisure=item('Technical leisure zip blazer','outerwear');
  const sport=item('Bold rugby striped cotton shirt','top',{pattern:'solid'});
  const pinstripe=item('Fine pinstriped cotton dress shirt','top');
  const tailored=item('Performance stretch wool tailored blazer','outerwear');
  expect(describeGarment(sport).pattern).toBe('stripe');
  expect(officeRefinementPenalty(leisure,work)).toBeGreaterThan(0);
  expect(officeRefinementPenalty(sport,work)).toBeGreaterThan(0);
  expect(officeRefinementPenalty(pinstripe,work)).toBe(0);
  expect(officeRefinementPenalty(tailored,work)).toBe(0);
  expect(officeRefinementPenalty(leisure,casual)).toBe(0);
  expect(scoreGarmentForContext(tailored,work)).toBeGreaterThan(scoreGarmentForContext(leisure,work));
  const refined=look(), mixed=look(); mixed[0]=sport; mixed[1]=leisure;
  expect(outfitCoherence(mixed,work).valid).toBe(true);
  expect(outfitCoherence(refined,work).score).toBeGreaterThan(outfitCoherence(mixed,work).score);
});
test('whole-look ranking rewards compatible proportions and fabrics, keeping contrasts soft',()=>{
  const refined=look(), contrast=look();
  contrast[1]=item('Slim fit navy smooth wool blazer','outerwear');
  contrast[2]=item('Wide fit grey tailored linen trousers','bottom');
  const good=outfitCoherence(refined,work), bad=outfitCoherence(contrast,work);
  expect(bad.valid).toBe(true);
  expect(good.dimensions.proportion).toBeGreaterThan(bad.dimensions.proportion);
  expect(good.dimensions.texture).toBeGreaterThan(bad.dimensions.texture);
  expect(good.evidence).toContain('compatible_tailoring_fabrics');
});
test('competing office upper patterns reduce palette rank; an isolated dress stripe is retained',()=>{
  const parts=look(); parts[0]=item('Blue fine striped cotton dress shirt','top');
  const quiet=outfitCoherence(parts,work); parts[1]=item('Navy checked wool blazer','outerwear');
  const competing=outfitCoherence(parts,work);
  expect(competing.valid).toBe(true);
  expect(competing.dimensions.palette).toBeLessThan(quiet.dimensions.palette);
  expect(competing.evidence).toContain('competing_office_upper_patterns');
});
test('value breaks equivalent shop ranking ties in one currency/slot, without a budget or free-unknown assumption',()=>{
  const affordable={isShopItem:true,macroCategory:'outerwear',price:150,currency:'USD'};
  const premium={...affordable,price:900};
  expect(compareUnbudgetedValue(affordable,premium)).toBeLessThan(0);
  for(const other of [{...premium,currency:'UZS'},{...premium,price:undefined},{...premium,price:0},{...premium,isWardrobe:true}])
    expect(compareUnbudgetedValue(affordable,other)).toBe(0);
  expect(work.budget).toBeUndefined();
});
test('existing title-based guards catch real mislabeled snapshot rows without rewriting them',()=>{
  const rows=require('../../data/shop_catalog_additional_500.json');
  const row=rows.find((r:any)=>r.id==='selected-homme-16101349-bracken-1264359');
  expect(row.category).toBe('bottoms');
  expect(detectSlotMismatch(row.category,row.name,row.garment_type)).toBe('top');
  expect(fourSlotOutfitReason([...look().slice(0,2),{...row,macroCategory:'bottom'},look()[3]])).not.toBeNull();
});
