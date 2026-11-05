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
