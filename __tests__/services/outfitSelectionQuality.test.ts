/** @jest-environment node */
import {assessOutfitQuality} from '../../src/domain/occasionRules';
import {describeGarment} from '../../src/domain/garmentDescriptor';
import {curatedOutfitShortlist, resolveOutfitChoices, parseRankedOutfitChoices} from '../../src/domain/outfitChoices';
import {assembleDiverseDeck, spreadSuitableProducts} from '../../src/domain/outfitDiversity';
const slots=['top','outerwear','bottom','shoes'];
const look=(names:string[],prefix='q')=>names.map((name,i)=>({id:`${prefix}-${i}`,name,macroCategory:slots[i]}));
const quality=(names:string[],occasion='Casual',style='casual')=>assessOutfitQuality(look(names),style,occasion).acceptance;

test.each([
 ['Plain cotton shirt','Navy technical blazer','Grey dress trousers','Black Derby shoes with chunky comfort soles'],
 ['Plain cotton shirt','Navy wool blazer','Grey formal trousers','Black sporty hybrid Derby shoes'],
 ['Plain cotton tee','Cotton overshirt','Drawstring jersey trousers','Suede driving loafers'],
 ['Plain cotton tee','Heavyweight winter wool coat','Lightweight linen trousers','Leather Chelsea boots'],
 ['Lightweight linen shirt','Heavy insulated down jacket','Cotton chinos','Leather Chelsea boots'],
])('whole-outfit incompatibility cannot pass even a casual floor: %s / %s / %s / %s',(...names)=>{
 expect(quality(names).passed).toBe(false);
});
test.each(['Work','Formal','Date night'])('washed denim with tailoring fails for %s',occasion=>{
 expect(quality(['Plain cotton shirt','Navy wool blazer','Heavily washed denim jeans','Brown leather penny loafers'],occasion).passed).toBe(false);
});
test.each([
 ['Plain cotton shirt','Navy technical blazer','Cotton chinos','Black sporty hybrid Derby shoes'],
 ['Plain fine cotton polo','Fine merino cardigan','Grey tailored wool trousers','Suede penny loafers with crepe soles'],
 ['Plain cotton shirt','Cotton overshirt','Drawstring linen trousers','Suede driving loafers'],
 ['Plain cotton tee','Cotton overshirt','Drawstring jersey trousers','Minimal leather sneakers'],
 ['Plain cotton tee','Denim jacket','Heavily washed denim jeans','Minimal leather sneakers'],
 ['Lightweight linen shirt','Fine merino cardigan','Cotton chinos','Brown leather loafers'],
 ['Plain cotton shirt','Navy wool blazer','Grey dress trousers','Black Oxford shoes with slim rubber soles'],
])('preserves casual and travel combinations: %s / %s / %s / %s',(...names)=>{
 for(const occasion of ['Casual','Travel'])expect(quality(names,occasion).passed).toBe(true);
});
test('construction classification recognizes drivers before marketing loafer labels',()=>{
 expect(describeGarment({name:'Suede driving loafers',macroCategory:'shoes'}).subtype).toBe('driver');
 expect(describeGarment({name:'Sporty hybrid Derby shoes',macroCategory:'shoes'}).soleProfile).toBe('sneaker');
 expect(describeGarment({name:'Derby sporty hybrid shoes',macroCategory:'shoes'}).soleProfile).toBe('sneaker');
});
const candidate=(n:number,score=0.94)=>({id:`look${n}`,score,items:look(['White cotton shirt','Navy wool blazer','Grey dress trousers','Brown leather loafers'],`p${n}`)});
const select=(candidates:ReturnType<typeof candidate>[],rankOf?:(look:ReturnType<typeof candidate>)=>number|undefined)=>assembleDiverseDeck({candidates,itemsOf:c=>c.items,qualityOf:c=>c.score,rankOf,usedKeys:new Set<string>(),target:3,maxSearchSteps:512});
test('shortlist retains the feasible seed and exposes nine approved alternatives without enumeration',()=>{
 const candidates=Array.from({length:24},(_,i)=>candidate(i));
 const seed=select(candidates).deck;
 const shortlist=curatedOutfitShortlist(candidates,seed,c=>c.score);
 expect(shortlist).toHaveLength(9);expect(new Set(shortlist.map(c=>c.id)).size).toBe(9);
 for(const c of seed)expect(shortlist).toContain(c);
 const ids=shortlist.map(c=>c.id).reverse();
 const ranked=resolveOutfitChoices(parseRankedOutfitChoices(JSON.stringify({rankedLookIds:ids})),shortlist)!;
 const ranks=new Map(ids.map((id,i)=>[shortlist.find(c=>c.id===id)!,i]));
 const selected=select(shortlist,c=>ranks.get(c));
 expect(selected.deck.map(c=>c.id).sort()).toEqual(ids.slice(0,3).sort());
 expect(selected.productDiversity.illegalProductReuseCount).toBe(0);
 expect(ranked).toHaveLength(9);
 expect(curatedOutfitShortlist(candidates.slice(0,2),[],c=>c.score)).toHaveLength(2);
});
test('NVIDIA preference cannot waive physical uniqueness or a material quality gap',()=>{
 const candidates=[candidate(0),candidate(1),candidate(2),candidate(3,0.5),candidate(4)];
 candidates[4].items[0]=candidates[0].items[0];
 const ranks=new Map(candidates.map((c,i)=>[c,[0,2,3,1,0][i]]));
 const selected=select(candidates,c=>ranks.get(c));
 expect(selected.deck.some(c=>c.id==='look3')).toBe(false);
 expect(selected.productDiversity.illegalProductReuseCount).toBe(0);
});
test('inferior brands cannot buy a place before item caps or in a complete deck',()=>{
 const rows=[{brand:'A',score:0.95},{brand:'A',score:0.94},{brand:'B',score:0.90}];
 expect(spreadSuitableProducts(rows,r=>r,r=>r.score)).toEqual(rows);
 const candidates=[candidate(0,0.95),candidate(1,0.94),candidate(2,0.93),candidate(3,0.92)];
 candidates.forEach((c,i)=>c.items.forEach(item=>Object.assign(item,{brand:i===3?'Unique Brand':'Repeated Brand'})));
 expect(select(candidates).deck.map(c=>c.id).sort()).toEqual(['look0','look1','look2']);
});
