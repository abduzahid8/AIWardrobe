import {rejectForStyleOccasion, styleJudgeReason} from '../../features/outfit-generator/utils/occasionRules';

const polo = {id:'owned-polo',name:'Polo Piquet 1',type:'tees',macroCategory:'top'};
test('recognizes a saved polo despite its broad tees category',()=>{
  expect(rejectForStyleOccasion(polo,'casual','Date night')).toBeNull();
  const parts=[polo,{name:'Cashmere cardigan',macroCategory:'outerwear'},
    {name:'Navy chino trousers',macroCategory:'bottom'},{name:'Brown leather loafers',macroCategory:'shoes'}];
  expect(styleJudgeReason(parts,'old_money','Date night',undefined,polo.id)).toBeNull();
});
test('still rejects a real tee sold by a polo brand for classic styles',()=>{
  // Stylist 5 redesign: date is a 'smart' dress code, so a cotton tee passes
  // a casual date look; classic styles still veto basic tees in every occasion.
  expect(rejectForStyleOccasion({...polo,name:'Polo Ralph Lauren cotton T-shirt'},'casual','Date night')).toBeNull();
  expect(rejectForStyleOccasion({...polo,name:'Polo Ralph Lauren cotton T-shirt'},'old_money','Date night')).toBe('basic tees break classic discipline');
});
test('retains the piqué polo and tailored blazer incompatibility',()=>{
  expect(styleJudgeReason([polo,{name:'Wool suit blazer',macroCategory:'outerwear'},
    {name:'Navy chino trousers',macroCategory:'bottom'},{name:'Brown leather loafers',macroCategory:'shoes'}],
    'old_money','Date night',undefined,polo.id)).toMatch(/piqu/);
});
test('keeps the work dress-code restriction for tee-typed polos via smart-work gates',()=>{
  // Stylist 5 redesign: work is a 'smart' dress code, so a polo no longer
  // trips the formal 'not workwear' block; for classic styles the targeted
  // smart-work base gate in garmentContextReason keeps the restriction.
  expect(rejectForStyleOccasion(polo,'casual','Work')).toBeNull();
  expect(rejectForStyleOccasion(polo,'old_money','Work')).toBe('work_requires_smart_base');
});

test('the final quality gate honors an owned jersey-trouser anchor while keeping companion and dress-code checks',()=>{
 const trousers={id:'jersey-anchor',name:'3D jersey knit trousers',type:'pants',macroCategory:'bottom'};
 const parts=[trousers,{name:'Navy wool blazer',macroCategory:'outerwear'},{name:'White cotton shirt',macroCategory:'top'},{name:'Brown leather loafers',macroCategory:'shoes'}];
 expect(styleJudgeReason(parts,'old_money','Date night',undefined,trousers.id)).toBeNull();
 expect(styleJudgeReason(parts,'old_money','Date night')).toMatch(/sporting trouser/);
 expect(styleJudgeReason(parts.map(i=>i===trousers?{...i,name:'Drawstring jogger trousers'}:i),'old_money','Date night',undefined,trousers.id)).toBeTruthy();
 expect(styleJudgeReason(parts.map(i=>i.macroCategory==='outerwear'?{...i,name:'Technical nylon track jacket'}:i),'old_money','Date night',undefined,trousers.id)).toBeTruthy();
});
