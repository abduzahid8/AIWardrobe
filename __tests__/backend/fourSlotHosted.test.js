/** @jest-environment node */
const fs=require('fs'),path=require('path');
const harness=fs.readFileSync(path.join(__dirname,'outfitBudgetWire.test.js'),'utf8').split('const owned=')[0];
const oldSource=process.env.OUTFIT_WIRE_SOURCE;
process.env.OUTFIT_WIRE_SOURCE=process.env.FOUR_SLOT_HANDLER_SOURCE || path.resolve(__dirname,'../../backend/supabase/functions/generate-outfits/index.ts');
const load=new Function('require','__dirname','jest',harness+'\nreturn load;')(require,__dirname,jest);
if(oldSource===undefined)delete process.env.OUTFIT_WIRE_SOURCE;else process.env.OUTFIT_WIRE_SOURCE=oldSource;
const items=[['layer','outerwear','Lightweight linen blazer'],['shirt','top','White cotton shirt'],['pants','bottom','Navy chino trousers'],['shoes','shoes','Brown leather loafers']].map(([id,macroCategory,name])=>({id,macroCategory,name,type:macroCategory,isWardrobe:true,color:'neutral',imageUrl:`https://example.invalid/${id}.jpg`,currency:'USD'}));
const request=(rows,weather)=>new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify({stylePreferences:'classic',occasion:'Work',prompt:'Work',weather,limit:1,wardrobeItems:rows,useProvidedWardrobeOnly:true,wardrobeOnly:true})});
test.each([undefined,{temp:35,condition:'sunny'},{temp:5,condition:'snow'}])('hosted handler returns all four categories in %j',async weather=>{
 const rows=weather?.temp===5?items.map(i=>i.id==='layer'?{...i,name:'Wool overcoat'}:i.id==='shoes'?{...i,name:'Leather boots'}:i):items;
 const {handler}=load(rows.map(i=>i.id));const response=await handler(request(rows,weather));const data=await response.json();
 expect(response.status).toBe(200);expect(data.layered).toBe(true);expect(data.outfits.length).toBeGreaterThan(0);
 for(const outfit of data.outfits)expect(outfit.items.map(i=>i.macroCategory).sort()).toEqual(['bottom','outerwear','shoes','top']);
});
test('hosted handler rejects a wardrobe without a layer',async()=>{
 const {handler}=load(['shirt','pants','shoes']);const response=await handler(request(items.slice(1)));const data=await response.json();expect(response.ok).toBe(false);expect(data.outfits||[]).toEqual([]);
});

test('a summer no-jacket request keeps a light alternative layer and four slots',async()=>{
 const rows=items.map(i=>i.macroCategory==='outerwear'?{...i,name:'Lightweight linen cardigan'}:i.macroCategory==='shoes'?{...i,name:'Brown leather sneakers'}:i.macroCategory==='top'?{...i,name:'White cotton shirt'}:i);
 const {handler}=load(rows.map(i=>i.id));const body=await request(rows,{temp:30}).json();body.stylePreferences='casual';body.occasion='Casual';body.prompt='summer, no jacket';
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));const data=await response.json();
 expect(response.status).toBe(200);expect(data.layered).toBe(true);expect(data.outfits[0].items.map(i=>i.macroCategory).sort()).toEqual(['bottom','outerwear','shoes','top']);
 expect(data.outfits[0].items.find(i=>i.macroCategory==='outerwear').name).toBe('Lightweight linen cardigan');
});

test('no-jacket intent never permits a successful three-item outfit',async()=>{
 const rows=items.slice(1);const {handler}=load(rows.map(i=>i.id));const body=await request(rows,{temp:30}).json();body.stylePreferences='casual';body.occasion='Casual';body.prompt='summer, no jacket';
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(response.status).toBe(422);expect((await response.json()).outfits).toEqual([]);
});

test('a winter destination overrides local summer weather and rejects a blazer plus loafers',async()=>{
 const {handler,fetchMock}=load(items.map(i=>i.id));const body=await request(items,{temp:30}).json();body.prompt='Winter trip in snow';
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(response.status).toBe(422);expect(fetchMock).not.toHaveBeenCalled();expect((await response.json()).outfits).toEqual([]);
});

test('curated generation falls back to validated complete options during a provider outage',async()=>{
 const {handler,fetchMock}=load(items.map(i=>i.id),null,null,null,null,'curated');fetchMock.mockImplementation(async()=>new Response('{}',{status:503}));
 const response=await handler(request(items));const data=await response.json();
 expect(response.status).toBe(200);expect(data.source).toBe('deterministic_fallback');expect(data.outfits[0].items).toHaveLength(4);
});

test('a locked puffer overrides classic taste while all new slots remain valid',async()=>{
 const rows=items.map(i=>i.id==='layer'?{...i,name:'Black insulated puffer jacket'}:i.id==='shoes'?{...i,name:'Brown leather boots'}:i);
 const {handler}=load(rows.map(i=>i.id),null,null,null,null,'curated');const body=await request(rows,{temp:0}).json();body.occasion='Casual';body.prompt='Cold winter outfit around my puffer';body.anchorItemId='layer';
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));const data=await response.json();
 expect(response.status).toBe(200);expect(data.outfits.every(o=>o.items.some(i=>i.id==='layer'))).toBe(true);
});

const richItems = [
 ['Navy linen blazer','Beige cotton blazer','Grey wool blazer'],
 ['White cotton shirt','Blue oxford shirt','Cream linen shirt'],
 ['Navy chino trousers','Grey wool trousers','Beige cotton chinos'],
 ['Brown leather loafers','Black leather derby shoes','Burgundy leather oxford shoes'],
].flatMap((names,slotIndex)=>names.map((name,index)=>({id:`rich-${slotIndex}-${index}`,macroCategory:['outerwear','top','bottom','shoes'][slotIndex],name,type:['outerwear','shirt','trousers','shoes'][slotIndex],isWardrobe:true,color:['navy','white','navy','brown'][slotIndex],imageUrl:`https://example.invalid/rich-${slotIndex}-${index}.jpg`})));
async function stylistRequest(rows){const body=await request(rows).json();return new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify({...body,stylePreferences:'casual',occasion:'Casual',prompt:'Casual',limit:5,deckTarget:3})});}
test.each([1,2])('replaces a partial %i-card provider deck with three validated four-item cards',async count=>{
 const proposals=Array.from({length:count},(_,i)=>richItems.filter(row=>row.id.endsWith(`-${i}`)).map(row=>row.id));
 const {handler}=load(proposals);const response=await handler(await stylistRequest(richItems));const data=await response.json();
 expect(response.status).toBe(200);expect(data.outfits).toHaveLength(3);
 for(const outfit of data.outfits)expect(outfit.items.map(i=>i.macroCategory).sort()).toEqual(['bottom','outerwear','shoes','top']);
 const all=data.outfits.flatMap(o=>o.items.map(i=>i.id));expect(new Set(all).size).toBe(12);
});
test('never calls a one-look Personal Stylist deck successful',async()=>{
 const {handler}=load(items.map(i=>i.id));const response=await handler(await stylistRequest(items));const data=await response.json();
 expect(response.status).toBe(422);expect(data.code).toBe('incomplete_outfit_deck');expect(data.outfits).toEqual([]);
});

const rotationFixtures = ['outerwear','top','bottom','shoes'].flatMap((slot,s)=>Array.from({length:9},(_,i)=>({id:`rotation-${s}-${i}`,name:['Navy cotton blazer','White cotton shirt','Beige cotton chinos','Brown leather loafers'][s]+` edition ${i}`,type:slot,macroCategory:slot,color:['navy','white','beige','brown'][s],isWardrobe:true,imageUrl:`https://example.invalid/rotation-${s}-${i}.jpg`})));
test('three generations consume 36 distinct physical products from rich inventory, including nine shoes',async()=>{
 const {handler}=load(rotationFixtures.slice(0,4).map(i=>i.id),null,null,null,null,'curated');
 const keys=new Set(),ids=new Set(),generations=[];
 const {productIdentityKeys}=require('../../backend/supabase/functions/_shared/outfitDiversity');
 for(let generation=0;generation<3;generation++){
  const body=await (await stylistRequest(rotationFixtures)).json();body.excludeItemIds=[...ids];body.excludeProductKeys=[...keys];
  const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));const data=await response.json();
  expect(response.status).toBe(200);expect(data.outfits).toHaveLength(3);expect(data.rotation.recycledSlots).toEqual([]);
  const parts=data.outfits.flatMap(o=>o.items);expect(parts).toHaveLength(12);
  const currentKeys=new Set(parts.flatMap(productIdentityKeys));expect([...currentKeys].filter(key=>keys.has(key))).toEqual([]);
  for(const key of currentKeys)keys.add(key);for(const item of parts)ids.add(item.id);generations.push(parts);
 }
 expect(ids.size).toBe(36);expect(new Set(generations.flat().filter(i=>i.macroCategory==='shoes').map(i=>i.id)).size).toBe(9);
});

test('exhausted shoes recycle only after all fresh shoes are included, while other slots remain fresh',async()=>{
 const rows=rotationFixtures.filter(i=>i.macroCategory!=='shoes'||i.id==='rotation-3-0'||i.id==='rotation-3-1'||i.id==='rotation-3-2'||i.id==='rotation-3-3');
 const {handler}=load(rows.slice(0,4).map(i=>i.id),null,null,null,null,'curated');
 const old=rows.filter(i=>i.macroCategory==='shoes'&&i.id!=='rotation-3-3');const {productIdentityKeys}=require('../../backend/supabase/functions/_shared/outfitDiversity');
 const body=await (await stylistRequest(rows)).json();body.excludeItemIds=old.map(i=>i.id);body.excludeProductKeys=old.flatMap(productIdentityKeys);
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));const data=await response.json();
 expect(response.status).toBe(200);expect(data.outfits).toHaveLength(3);expect(data.rotation.recycledSlots).toEqual(['shoes']);
 expect(data.outfits.flatMap(o=>o.items).some(i=>i.id==='rotation-3-3')).toBe(true);
 expect(new Set(data.outfits.flatMap(o=>o.items).map(i=>i.id)).size).toBe(12);
});

test('a new catalog ID cannot resurrect a recently shown product by name or normalized photo',async()=>{
 const old=rotationFixtures.filter(i=>i.id.endsWith('-0'));
 const aliases=old.flatMap(i=>[{...i,id:'name-alias-'+i.id,name:i.name.toUpperCase(),imageUrl:`https://example.invalid/alias-${i.id}.jpg`},{...i,id:'photo-alias-'+i.id,name:i.name+' renamed',imageUrl:i.imageUrl+'?width=800'}]);
 const {handler}=load([],null,null,null,null,'curated');const {productIdentityKeys}=require('../../backend/supabase/functions/_shared/outfitDiversity');const history=new Set(old.flatMap(productIdentityKeys));
 const body=await (await stylistRequest([...rotationFixtures,...aliases])).json();body.excludeItemIds=old.map(i=>i.id);body.excludeProductKeys=[...history];
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));const data=await response.json();
 expect(response.status).toBe(200);expect(data.outfits).toHaveLength(3);
 expect(data.outfits.flatMap(o=>o.items).flatMap(productIdentityKeys).filter(key=>history.has(key))).toEqual([]);
});

test('deck boundaries avoid previous shoes and exhaust older last-use order in a five-shoe pool',async()=>{
 const rows=rotationFixtures.filter(i=>i.macroCategory!=='shoes'||Number(i.id.split('-').at(-1))<5);
 const {handler}=load([],null,null,null,null,'curated');const {productIdentityKeys}=require('../../backend/supabase/functions/_shared/outfitDiversity');const decks=[];
 for(let g=0;g<3;g++){
  const body=await(await stylistRequest(rows)).json();body.rotationDecks=decks;body.previousDeckProductKeys=decks.at(-1)?.productKeys||[];body.olderRecentProductKeys=decks.slice(0,-1).flatMap(d=>d.productKeys);body.excludeProductKeys=decks.flatMap(d=>d.productKeys);
  const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));const data=await response.json();expect(response.status).toBe(200);expect(data.outfits).toHaveLength(3);
  const parts=data.outfits.flatMap(o=>o.items),shoes=parts.filter(i=>i.macroCategory==='shoes');
  if(g===1){const previous=decks[0].shoeIds;expect(shoes.filter(i=>!previous.includes(i.id))).toHaveLength(2);expect(shoes.filter(i=>previous.includes(i.id))).toHaveLength(1);}
  if(g===2){const previous=decks[1].shoeIds;expect(shoes.filter(i=>previous.includes(i.id))).toHaveLength(1);}
  decks.push({requestId:`request-${g}`,productKeys:parts.flatMap(productIdentityKeys),shoeIds:shoes.map(i=>i.id)});
 }
});

test('five same-intent generations keep all physical products fresh in a rich pool',async()=>{
 const rows=['outerwear','top','bottom','shoes'].flatMap((slot,s)=>Array.from({length:15},(_,i)=>({id:`five-${s}-${i}`,name:['Navy cotton blazer','White cotton shirt','Beige cotton chinos','Brown leather loafers'][s]+` edition ${i}`,macroCategory:slot,type:slot,color:['navy','white','beige','brown'][s],isWardrobe:true,imageUrl:`https://example.invalid/five-${s}-${i}.jpg`})));
 const {handler}=load([],null,null,null,null,'curated'),decks=[];const {productIdentityKeys}=require('../../backend/supabase/functions/_shared/outfitDiversity');
 for(let g=0;g<5;g++){
  const body=await(await stylistRequest(rows)).json();body.rotationScope='fixture|wardrobe|casual|casual';body.generationRequestId=`five-request-${g}`;body.rotationDecks=decks;body.previousDeckProductKeys=decks.at(-1)?.productKeys||[];body.olderRecentProductKeys=decks.slice(0,-1).flatMap(d=>d.productKeys);body.excludeProductKeys=decks.flatMap(d=>d.productKeys);
  const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));const data=await response.json();expect(response.status).toBe(200);expect(data.outfits).toHaveLength(3);
  const products=data.outfits.flatMap(o=>o.items),keys=products.flatMap(productIdentityKeys),seen=new Set(decks.flatMap(d=>d.productKeys));expect(keys.filter(k=>seen.has(k))).toEqual([]);
  expect(data.previousDeckOverlapBySlot).toEqual({outerwear:0,top:0,bottom:0,shoes:0});expect(data.generationRequestId).toBe(body.generationRequestId);expect(data.rotationScope).toBe(body.rotationScope);
  decks.push({requestId:body.generationRequestId,productKeys:keys});
 }
});

test.each([3,4,5,6,10])('%i admitted shoes meet the minimum across five hosted generations without pinning a favorite',async n=>{
 const rows=['outerwear','top','bottom','shoes'].flatMap((slot,s)=>Array.from({length:s===3?n:15},(_,i)=>({id:`scheduled-${s}-${i}`,name:['Navy cotton blazer','White cotton shirt','Beige cotton chinos','Brown leather loafers'][s]+` edition ${i}`,macroCategory:slot,type:slot,color:['navy','white','beige','brown'][s],isWardrobe:true,imageUrl:`https://example.invalid/scheduled-${s}-${i}.jpg`})));
 const {handler}=load([],null,null,null,null,'curated'),decks=[],usage=new Map(),run=[];
 const {productIdentityKeys}=require('../../backend/supabase/functions/_shared/outfitDiversity');
 for(let g=0;g<5;g++){
  const body=await(await stylistRequest(rows)).json();body.rotationDecks=decks;body.previousDeckProductKeys=decks.at(-1)?.productKeys||[];body.olderRecentProductKeys=decks.slice(0,-1).flatMap(d=>d.productKeys);body.excludeProductKeys=decks.flatMap(d=>d.productKeys);
  const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));const data=await response.json();
  expect(response.status).toBe(200);expect(data.outfits).toHaveLength(3);expect(data.outfits.every(o=>o.items.length===4)).toBe(true);
  const products=data.outfits.flatMap(o=>o.items),shoes=products.filter(p=>p.macroCategory==='shoes'),previous=new Set(decks.at(-1)?.productKeys||[]);
  const overlaps=shoes.filter(p=>productIdentityKeys(p).some(k=>previous.has(k))).length;
  expect(overlaps).toBe(g?Math.max(0,6-n):0);expect(data.shoeRotationAudit.previousShoeOverlap).toBe(overlaps);
  for(const slot of ['outerwear','top','bottom'])expect(products.filter(p=>p.macroCategory===slot).some(p=>productIdentityKeys(p).some(k=>previous.has(k)))).toBe(false);
  run.push(shoes.map(p=>p.id));shoes.forEach(p=>usage.set(p.id,(usage.get(p.id)||0)+1));decks.push({requestId:`scheduled-${g}`,productKeys:products.flatMap(productIdentityKeys)});
 }
 expect(usage.size).toBe(n);if(n>3)expect(run.some(ids=>!ids.includes('scheduled-3-0'))).toBe(true);
});
