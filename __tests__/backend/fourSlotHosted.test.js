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

test('a summer no-jacket request returns exactly three real slots',async()=>{
 const rows=items.slice(1).map(i=>i.macroCategory==='shoes'?{...i,name:'Brown leather sneakers'}:i.macroCategory==='top'?{...i,name:'White cotton polo shirt'}:i);
 const {handler}=load(rows.map(i=>i.id));const body=await request(rows,{temp:30}).json();body.stylePreferences='casual';body.occasion='Casual';body.prompt='summer, no jacket';
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));const data=await response.json();
 expect(response.status).toBe(200);expect(data.layered).toBe(false);expect(data.outfits[0].items.map(i=>i.macroCategory).sort()).toEqual(['bottom','shoes','top']);
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
