import {scoreGarmentForContext, garmentContextReason} from '../../src/domain/garmentContext';
import {stylingContext} from '../../src/domain/stylingContext';
import {wholeOutfitRegisterClash} from '../../src/domain/outfitCoherence';
import {rejectForStyleOccasion, passesStyleJudge, styleJudgeReason} from '../../src/domain/occasionRules';

const jacket={id:'jacket',name:'Leather trucker jacket',macroCategory:'outerwear'};
const blazer={id:'blazer',name:'Unlined tailored blazer',macroCategory:'outerwear'};
const look=[{id:'top',name:'Cotton shirt',macroCategory:'top'},jacket,
  {id:'bottom',name:'Tailored wool trousers',macroCategory:'bottom'},
  {id:'shoes',name:'Black leather oxford shoes',macroCategory:'shoes'}];

test('Work ranks a tailored layer above a trucker without banning the trucker on Casual',()=>{
 const work=stylingContext({style:'classic',occasion:'work'}),casual=stylingContext({style:'classic',occasion:'casual'});
 expect(scoreGarmentForContext(blazer,work)).toBeGreaterThan(scoreGarmentForContext(jacket,work));
 expect(scoreGarmentForContext(jacket,casual)).toBeGreaterThan(scoreGarmentForContext(jacket,work));
 expect(wholeOutfitRegisterClash(look,work)).toBe('workwear_jacket_with_office_dress_footwear');
 expect(wholeOutfitRegisterClash(look,casual)).toBeNull();
 expect(garmentContextReason(jacket,work)).toBe('work_requires_smart_layer');
 expect(garmentContextReason(jacket,casual)).toBeNull();
 expect(garmentContextReason(blazer,work)).toBeNull();
 expect(garmentContextReason({name:'Slim Shield T Shirt Evening Blue',macroCategory:'top'},work)).toBe('work_requires_smart_base');
});

test('Classic casual/travel can use minimal leather sneakers while Work/Formal and tailoring remain strict',()=>{
 const sneakers={id:'sneaker',name:'Minimal white leather sneakers',macroCategory:'shoes'};
 const casual=[{id:'polo',name:'Fine cotton knit polo',macroCategory:'top'},
  {id:'overshirt',name:'Cotton overshirt',macroCategory:'outerwear'},
  {id:'chino',name:'Beige cotton chinos',macroCategory:'bottom'},sneakers];
 for(const occasion of ['casual','travel'])expect(styleJudgeReason(casual,'classic',occasion)).toBeNull();
 for(const occasion of ['work','formal'])expect(rejectForStyleOccasion(sneakers,'classic',occasion)).not.toBeNull();
 expect(passesStyleJudge([...casual.slice(0,2),look[2],sneakers],'classic','travel')).toBe(false);
});

test('Formal recovery rejects checked bases competing with patterned tailoring, without banning checks globally',()=>{
 const checked={id:'base',name:'Gant X Yale Checked Poplin Shirt Crisp Blue',macroCategory:'top'};
 const stripedLayer={id:'layer',name:'Striped Double-breasted Suit Blazer',macroCategory:'outerwear',color:'grey'};
 const stripedBottom={id:'bottom',name:'Striped Regular Fit Suit Trousers',macroCategory:'bottom',color:'grey'};
 const shoes={id:'shoes',name:'Black leather oxford shoes',macroCategory:'shoes'};
 const parts=[checked,stripedLayer,stripedBottom,shoes];
 for(const style of ['casual','classic']) {
  expect(styleJudgeReason(parts,style,'formal')).toBe('formal_patterned_tailoring_requires_quiet_base');
  expect(styleJudgeReason([{...checked,name:'White poplin shirt'},...parts.slice(1)],style,'formal')).toBeNull();
 }
 expect(wholeOutfitRegisterClash(parts,stylingContext({style:'casual',occasion:'casual'}))).toBeNull();
 expect(wholeOutfitRegisterClash([checked,{...stripedLayer,name:'Wool suit blazer'},{...stripedBottom,name:'Wool pinstripe suit trousers'},shoes],stylingContext({style:'casual',occasion:'formal'}))).toBe('formal_patterned_tailoring_requires_quiet_base');
 expect(styleJudgeReason([checked,{...stripedLayer,name:'Wool suit blazer'},{...stripedBottom,name:'Wool suit trousers'},shoes],'casual','formal')).toBeNull();
});

test('Formal bases reject the selected Henley and unverified merino button-through shirts',()=>{
 for(const name of ['Slim-Fit Cotton Henley Shirt','Merino Wool Button-Through Shirt in Soft Grey Mouline']) {
  const base={id:'base',name,macroCategory:'top'};
  for(const style of ['casual','classic']) {
   expect(garmentContextReason(base,stylingContext({style,occasion:'formal'}))).toBe('formal_requires_dress_base');
   expect(garmentContextReason(base,stylingContext({style,occasion:'casual'}))).toBeNull();
  }
 }
 expect(garmentContextReason({name:'White cotton poplin dress shirt',macroCategory:'top'},stylingContext({style:'casual',occasion:'formal'}))).toBeNull();
});
