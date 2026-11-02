import {rejectForStyleOccasion, styleJudgeReason} from '../../features/outfit-generator/utils/occasionRules';

const polo = {id:'owned-polo',name:'Polo Piquet 1',type:'tees',macroCategory:'top'};
test('recognizes a saved polo despite its broad tees category',()=>{
  expect(rejectForStyleOccasion(polo,'casual','Date night')).toBeNull();
  const parts=[polo,{name:'Cashmere cardigan',macroCategory:'outerwear'},
    {name:'Navy chino trousers',macroCategory:'bottom'},{name:'Brown leather loafers',macroCategory:'shoes'}];
  expect(styleJudgeReason(parts,'old_money','Date night',undefined,polo.id)).toBeNull();
});
test('still rejects a real tee sold by a polo brand',()=>{
  expect(rejectForStyleOccasion({...polo,name:'Polo Ralph Lauren cotton T-shirt'},'casual','Date night')).toBeTruthy();
});
test('retains the piqué polo and tailored blazer incompatibility',()=>{
  expect(styleJudgeReason([polo,{name:'Wool suit blazer',macroCategory:'outerwear'},
    {name:'Navy chino trousers',macroCategory:'bottom'},{name:'Brown leather loafers',macroCategory:'shoes'}],
    'old_money','Date night',undefined,polo.id)).toMatch(/piqu/);
});
test('keeps the work dress-code restriction for piqué polos',()=>{
  expect(rejectForStyleOccasion(polo,'casual','Work')).toMatch(/workwear/);
});

test('the final quality gate honors an owned jersey-trouser anchor while keeping companion and dress-code checks',()=>{
 const trousers={id:'jersey-anchor',name:'3D jersey knit trousers',type:'pants',macroCategory:'bottom'};
 const parts=[trousers,{name:'Navy wool blazer',macroCategory:'outerwear'},{name:'White cotton shirt',macroCategory:'top'},{name:'Brown leather loafers',macroCategory:'shoes'}];
 expect(styleJudgeReason(parts,'old_money','Date night',undefined,trousers.id)).toBeNull();
 expect(styleJudgeReason(parts,'old_money','Date night')).toMatch(/sporting trouser/);
 expect(styleJudgeReason(parts.map(i=>i===trousers?{...i,name:'Drawstring jogger trousers'}:i),'old_money','Date night',undefined,trousers.id)).toBeTruthy();
 expect(styleJudgeReason(parts.map(i=>i.macroCategory==='outerwear'?{...i,name:'Technical nylon track jacket'}:i),'old_money','Date night',undefined,trousers.id)).toBeTruthy();
});
