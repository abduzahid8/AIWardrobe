import {detectSlotMismatch,isOxfordFootwearName,styleJudgeReason} from '../../features/outfit-generator/utils/occasionRules';
import {sanitizeGeneratedOutfitItemsDetailed} from '../../features/outfit-generator/utils/sanitizeGeneratedOutfit';

const footwear=['Bexford Leather Oxfords - Black','Leather Oxford','Cap-toe Oxford shoes','Suede Oxfords','Black leather brogue Oxfords'];
const shirts=['White Oxford shirt','Blue Oxford cotton button-down shirt','Oxford cloth shirt','Oxford short-sleeve shirt'];
test.each(footwear)('keeps %s in shoes and repairs a misfiled top',name=>{
 expect(isOxfordFootwearName(name)).toBe(true);
 expect(detectSlotMismatch('shoes',name,'shoes')).toBeNull();
 expect(detectSlotMismatch('footwear',name,'shoes')).toBeNull();
 expect(detectSlotMismatch('top',name,'upper_body')).toBe('shoes');
});
test.each(shirts)('keeps %s in tops and repairs a misfiled shoe',name=>{
 expect(isOxfordFootwearName(name)).toBe(false);
 expect(detectSlotMismatch('top',name,'shirts')).toBeNull();
 expect(detectSlotMismatch('shoes',name,'shoes')).toBe('top');
});
test('client sanitization preserves the owned Oxford ID and the separate Oxford shirt',()=>{
 const items=[{id:'layer',name:'Navy wool blazer',type:'blazer',macroCategory:'outerwear',image:'layer.jpg'},
 {id:'shirt',name:'White Oxford shirt',type:'shirt',macroCategory:'top',image:'shirt.jpg'},
 {id:'pants',name:'Grey wool trousers',type:'pants',macroCategory:'bottom',image:'pants.jpg'},
 {id:'owned-oxfords',name:'Bexford Leather Oxfords - Black',type:'shoes',macroCategory:'shoes',image:'oxfords.jpg',isWardrobe:true,isShopItem:false}];
 const result=sanitizeGeneratedOutfitItemsDetailed(items as any,items as any,{style:'old_money',prompt:'Date night'});
 expect(result.missingSlots).toEqual([]);expect(result.items).toHaveLength(4);
 expect(result.items.find(i=>i.macroCategory==='shoes')?.id).toBe('owned-oxfords');
 expect(result.items.find(i=>i.macroCategory==='top')?.id).toBe('shirt');
 expect(styleJudgeReason(result.items,'old_money','date',undefined,'owned-oxfords')).toBeNull();
});

test('coated jeans do not self-confirm a corrupt outerwear label through the word coat',()=>{
 expect(detectSlotMismatch('outerwear','Slim-Straight Leg Metallic Coated Jeans','outerwear')).toBe('bottom');
 expect(detectSlotMismatch('bottom','Slim-Straight Leg Metallic Coated Jeans','pants')).toBeNull();
 expect(detectSlotMismatch('outerwear','Double-breasted wool coat','outerwear')).toBeNull();
});
