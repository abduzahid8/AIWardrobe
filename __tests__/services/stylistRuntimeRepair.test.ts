import {scoreGarmentForContext} from '../../src/domain/garmentContext';
import {stylingContext} from '../../src/domain/stylingContext';
import {wholeOutfitRegisterClash} from '../../src/domain/outfitCoherence';

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
});
