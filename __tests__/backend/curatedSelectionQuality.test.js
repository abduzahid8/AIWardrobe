/** @jest-environment node */
const fs=require('fs'),path=require('path');
const harness=fs.readFileSync(path.join(__dirname,'outfitBudgetWire.test.js'),'utf8').split('const owned=')[0];
const load=new Function('require','__dirname','jest',harness+'\nreturn load;')(require,__dirname,jest);
const slots=['top','outerwear','bottom','shoes'];
const names=['White poplin shirt','Navy wool blazer','Grey dress trousers','Black leather Oxford shoes with slim sole'];
const rows=slots.flatMap((macroCategory,s)=>Array.from({length:6},(_,i)=>({
 id:`shop_curated-${s}-${i}`,name:`${names[s]} model ${i}`,macroCategory,type:macroCategory,
 description:`${names[s]} in regular fit`,color:['white','navy','grey','black'][s],
 imageUrl:`https://example.invalid/curated-${s}-${i}.jpg`,isShopItem:true,isWardrobe:false,price:50,currency:'USD'
})));
const request=()=>new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify({
 stylePreferences:'classic',occasion:'Work',prompt:'Work',limit:3,deckTarget:3,wardrobeItems:rows,useProvidedWardrobeOnly:true,
 preferences:{maxShopBudget:{amount:220,currency:'USD'}}
})});
test('real curated handler ranks every approved look in one call, then selects a unique ranked triple',async()=>{
 const {handler,fetchMock}=load([],null,null,null,null,'curated');
 let options,preferred;
 fetchMock.mockImplementation(async(_url,init)=>{
  const messages=JSON.parse(init.body).messages;
  options=JSON.parse(messages.at(-1).content);
  // The planner hands every approved candidate to one ranking call; the
  // displayed deck is three of them. Previously this was exactly nine.
  expect(options.looks.length).toBeGreaterThanOrEqual(3);
  // The ranking wire carries planner-normalized garment facets; raw product
  // descriptions are intentionally not part of the contract anymore.
  expect(options.products.every(p=>p.garment&&typeof p.garment.category==='string')).toBe(true);
  const reversed=[...options.looks].reverse();
  // Choose a feasible alternative triple, rather than just echoing the seed.
  for(const a of reversed)for(const b of reversed)for(const c of reversed){
   if(preferred||a===b||a===c||b===c)continue;
   const ids=[...a.items,...b.items,...c.items];
   if(new Set(ids).size===12)preferred=[a,b,c];
  }
  expect(preferred).toHaveLength(3);
  const ids=[...preferred,...options.looks.filter(l=>!preferred.includes(l))].map(l=>l.id);
  return new Response(JSON.stringify({choices:[{message:{content:JSON.stringify({rankedLookIds:ids})}}]}));
 });
 const response=await handler(request()),data=await response.json();
 expect({status:response.status,data}).toMatchObject({status:200,data:{success:true}});
 expect(fetchMock).toHaveBeenCalledTimes(1);
 expect(data.outfits).toHaveLength(3);
 // Final shoe rotation may substitute shoes to satisfy its existing objective;
 // NVIDIA's preferred clothing alternatives must survive the final deck gate.
 const clothing=items=>items.map(i=>typeof i==='string'?i:i.id).filter(id=>!id.startsWith('shop_curated-3-')).sort().join('|');
 expect(data.outfits.map(o=>clothing(o.items)).sort()).toEqual(preferred.map(o=>clothing(o.items)).sort());
 expect(new Set(data.outfits.flatMap(o=>o.items.map(i=>i.id))).size).toBe(12);
 for(const o of data.outfits){expect(o.qualityAssessment.acceptance.passed).toBe(true);expect(o.items.reduce((n,i)=>n+i.price,0)).toBeLessThanOrEqual(220);}
});
