import {isDressGarment} from '../../backend/supabase/functions/_shared/outfitStructure';
import {outfitChoices,resolveOutfitChoices} from '../../backend/supabase/functions/_shared/outfitChoices';
const items=['top','bottom','shoes','outerwear'].flatMap(macroCategory=>Array.from({length:6},(_,i)=>({id:`${macroCategory}-${i}`,name:macroCategory,macroCategory})));
test.each([false,true])('offers only complete validated diverse combinations (layered=%s)',layered=>{
 const choices=outfitChoices(items,layered,parts=>!parts.some(p=>p.id==='bottom-0'), isDressGarment);
 expect(choices.length).toBeGreaterThanOrEqual(5);
 for(const c of choices){expect(c.items).toHaveLength(layered?4:3);expect(c.items.map(x=>x.id)).not.toContain('bottom-0');expect(new Set(c.items.map(x=>x.macroCategory)).size).toBe(c.items.length);}
 for(const a of choices)for(const b of choices)if(a!==b)expect(a.items.filter(x=>b.items.some(y=>y.id===x.id)).length).toBeLessThanOrEqual(a.items.length-2);
});
test('never offers a rejected combination or invents a missing slot',()=>{
 expect(outfitChoices(items,true,()=>false, isDressGarment)).toEqual([]);expect(outfitChoices(items.filter(i=>i.macroCategory!=='shoes'),true,()=>true, isDressGarment)).toEqual([]);
});
test('locks requested IDs into every offered outfit',()=>{
 const choices=outfitChoices(items,true,()=>true, isDressGarment,['top-2']);expect(choices.length).toBeGreaterThan(0);expect(choices.every(c=>c.items.some(i=>i.id==='top-2'))).toBe(true);
});
test('dress replaces top and bottom, never shoes or required outerwear',()=>{
 const dress={id:'dress-1',name:'Cotton dress',macroCategory:'dress'};
 const choices=outfitChoices([dress,...items.filter(i=>['outerwear','shoes'].includes(i.macroCategory))],true,()=>true, isDressGarment);
 expect(choices.length).toBeGreaterThan(0);expect(choices.every(c=>c.items.length===3&&c.items.includes(dress))).toBe(true);
});
test('resolves only an explicit AI-selected whole option',()=>{
 const choices=outfitChoices(items,false,()=>true, isDressGarment);
 const result=resolveOutfitChoices([{description:'Selected by AI',items:[{id:choices[0].id}]}],choices);
 expect(result![0].items).toEqual(choices[0].items);expect(result![0].description).toBe('Selected by AI');
 expect(resolveOutfitChoices(null,choices)).toBeNull();expect(resolveOutfitChoices([{items:[{id:'invented'}]}],choices)).toBeNull();expect(resolveOutfitChoices([{items:[{id:items[0].id}]}],choices)).toBeNull();
});
test('duplicate and partial selections fail instead of silently substituting looks',()=>{
 const choices=outfitChoices(items,false,()=>true, isDressGarment),selection={items:[{id:choices[0].id}]};
 expect(resolveOutfitChoices([selection,selection],choices)).toBeNull();expect(resolveOutfitChoices([{items:[{id:choices[0].id},{id:choices[1].id}]}],choices)).toBeNull();
});
test('search stops at the bound even for a hostile validator/pool',()=>{
 const accepts=jest.fn(()=>false);outfitChoices(items,true,accepts, isDressGarment,[],24,10);expect(accepts).toHaveBeenCalledTimes(10);
});

test('dress shirts remain eligible tops alongside actual dresses',()=>{
 const shirt={id:'formal-shirt',name:'White dress shirt',macroCategory:'top'};
 const choices=outfitChoices([shirt,...items.filter(i=>i.macroCategory!=='top')],false,()=>true, isDressGarment);
 expect(choices.length).toBeGreaterThan(0);expect(choices.every(c=>c.items.includes(shirt)&&c.items.length===3)).toBe(true);
});
test('offers dress and separates alternatives together instead of starving dresses',()=>{
 const dress={id:'dress',name:'Cotton dress',macroCategory:'dress'};const choices=outfitChoices([...items,dress],false,()=>true, isDressGarment);
 expect(choices.some(c=>c.items.includes(dress))).toBe(true);expect(choices.some(c=>!c.items.includes(dress))).toBe(true);
});
