import {assignRotatingShoes,createShoeRotationPriority,compareRotationOrder,wasRecentlyShown} from '../../backend/supabase/functions/_shared/outfitRotation';
import {productIdentityKeys,countDistinctProducts,assembleDiverseDeck} from '../../backend/supabase/functions/_shared/outfitDiversity';
import {isDrivingFootwear,rejectForStyleOccasion} from '../../backend/supabase/functions/_shared/occasionRules';
const shoe=(i:number)=>({id:`shoe-${i}`,name:`Brown leather loafer edition ${i}`,macroCategory:'shoes',imageUrl:`https://catalog.invalid/${i}.jpg`});
const looks=(g:number)=>Array.from({length:3},(_,i)=>({items:[
 {id:`layer-${g}-${i}`,name:`Navy blazer ${g}-${i}`,macroCategory:'outerwear'},
 {id:`top-${g}-${i}`,name:`White cotton shirt ${g}-${i}`,macroCategory:'top'},
 {id:`bottom-${g}-${i}`,name:`Navy trousers ${g}-${i}`,macroCategory:'bottom'},shoe(i)],quality:1}));

test.each([3,4,5,6,7,10,50])('%i valid physical shoes achieve the mathematical minimum without a pinned shoe',n=>{
 const pool=Array.from({length:n},(_,i)=>shoe(i)),decks:{productKeys:string[]}[]=[],usage=new Map<string,number>(),run:string[][]=[];
 for(let g=0;g<8;g++){
  const previous=new Set(decks.at(-1)?.productKeys||[]),history=new Set(decks.flatMap(d=>d.productKeys));
  const priority=createShoeRotationPriority(history,previous,decks);
  const assigned=assignRotatingShoes(looks(g),pool,()=>true,priority,items=>items.some(i=>i.id==='shoe-0')?1:0.99);
  expect(assigned.deck).toHaveLength(3);
  const selected=assigned.deck.map(o=>o.items.find(i=>i.macroCategory==='shoes')!);
  const overlap=selected.filter(i=>wasRecentlyShown(i,previous)).length;
  expect(overlap).toBe(g?Math.max(0,6-n):0);
  expect(new Set(selected.map(i=>i.id)).size).toBe(3);
  run.push(selected.map(i=>i.id));selected.forEach(i=>usage.set(i.id,(usage.get(i.id)||0)+1));
  decks.push({productKeys:assigned.deck.flatMap(o=>o.items.flatMap(productIdentityKeys))});
 }
 expect(usage.size).toBe(Math.min(n,24));
 if(n===5){expect(run.some(ids=>!ids.includes('shoe-0'))).toBe(true);expect(Math.max(...usage.values())-Math.min(...usage.values())).toBeLessThanOrEqual(1);}
});
test('duplicate names or normalized photos cannot count as extra shoes',()=>{
 const pool=[shoe(0),shoe(1),shoe(2),{...shoe(0),id:'duplicate-name',imageUrl:'https://catalog.invalid/another.jpg'},
 {...shoe(1),id:'duplicate-photo',name:'Renamed leather shoe',imageUrl:'http://catalog.invalid/1.jpg?width=800'}];
 expect(countDistinctProducts(pool)).toBe(3);
 const previous=new Set([shoe(0),shoe(1),shoe(2)].flatMap(productIdentityKeys));
 const result=assignRotatingShoes(looks(1),pool,()=>true,createShoeRotationPriority(previous,previous,[{productKeys:[...previous]}]),()=>1);
 expect(result.feasibleMinimumOverlap).toBe(3);
});
test('a hard quality rejection can never be resurrected by rotation scheduling',()=>{
 const pool=[shoe(0),shoe(1),shoe(2),{...shoe(3),name:'Rugged lug combat boots'}];
 const previous=new Set(pool.slice(0,3).flatMap(productIdentityKeys));
 const result=assignRotatingShoes(looks(1),pool,items=>!items.some(i=>i.name.includes('combat')),createShoeRotationPriority(previous,previous,[]),()=>1);
 expect(result.deck.flatMap(o=>o.items).some(i=>i.id==='shoe-3')).toBe(false);expect(result.feasibleMinimumOverlap).toBe(3);
});
test('shoe overlap wins before freshness in other slots or a small quality advantage',()=>{
 const previous=new Set(productIdentityKeys(shoe(0))),history=new Set([...previous,'id:old-top']);
 const priority=createShoeRotationPriority(history,previous,[{productKeys:[...previous]}]);
 expect(compareRotationOrder(priority([shoe(1),{id:'old-top',macroCategory:'top'}]),priority([shoe(0),{id:'fresh-top',macroCategory:'top'}]))).toBeGreaterThan(0);
 const options=[{items:[shoe(0)],quality:1},{items:[shoe(1)],quality:.9}];
 const chosen=assembleDiverseDeck({candidates:options,itemsOf:o=>o.items,qualityOf:o=>o.quality,target:1,poolItems:options.flatMap(o=>o.items),usedKeys:new Set(),priorityOfDeck:deck=>priority(deck.flatMap(o=>o.items))});
 expect(chosen.deck[0].items[0].id).toBe('shoe-1');
});
test('structured penny loafers with a moccasin toe are not driving moccasins',()=>{
 const structured={name:'Lincoln penny loafer',macroCategory:'shoes',description:'Full-grain leather hand-stitched moccasin toe. Blake stitch construction. Premium leather outsoles.'};
 expect(isDrivingFootwear(structured)).toBe(false);expect(rejectForStyleOccasion(structured,'old_money','date')).toBeNull();
 expect(isDrivingFootwear({...structured,description:structured.description+' Driving sole with rubber nubs.'})).toBe(true);
 expect(isDrivingFootwear({name:'Soft moccasins with rubber nubs',macroCategory:'shoes'})).toBe(true);
});
test('catalog pre-ranking cannot cache a vaguely named shoe as clothing',()=>{
 const named={id:'vague',name:'Lincoln | Dark Brown',macroCategory:'shoes',imageUrl:'https://catalog.invalid/lincoln.jpg'};
 const previous=new Set(productIdentityKeys(named));
 const priority=createShoeRotationPriority(previous,previous,[{productKeys:[...previous]}]);
 const {macroCategory:_,...identityOnly}=named;
 priority([identityOnly]); // same partial identity used during catalog sorting
 expect(priority([named])[0]).toBe(-1);
 expect(compareRotationOrder(priority([shoe(9)]),priority([named]))).toBeGreaterThan(0);
});
test('vaguely named catalog shoes obey oldest-last-use and appearance tie-breaks after pre-ranking',()=>{
 const pool=['Chairman | Black','Lincoln | Black','Lincoln | Brown','Duke | Coffee','Saint | Cognac'].map((name,i)=>({...shoe(i),name}));
 const decks=[{productKeys:[...productIdentityKeys(pool[0]),...productIdentityKeys(pool[1])]},
  {productKeys:[...productIdentityKeys(pool[2]),...productIdentityKeys(pool[3]),...productIdentityKeys(pool[4])]}];
 const previous=new Set(decks[1].productKeys),history=new Set(decks.flatMap(d=>d.productKeys)),priority=createShoeRotationPriority(history,previous,decks);
 for(const p of pool){const {macroCategory:_,...partial}=p;priority([partial]);}
 const result=assignRotatingShoes(looks(2),pool,()=>true,priority,()=>1);
 const ids=result.deck.flatMap(o=>o.items.filter(p=>p.macroCategory==='shoes').map(p=>p.id));
 expect(ids).toContain('shoe-0');expect(ids).toContain('shoe-1');
 expect(result.feasibleMinimumOverlap).toBe(1);
});
