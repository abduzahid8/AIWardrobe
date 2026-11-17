/** @jest-environment node */
const fs=require('fs'),path=require('path');
const harness=fs.readFileSync(path.join(__dirname,'outfitBudgetWire.test.js'),'utf8').split('const owned=')[0];
const load=new Function('require','__dirname','jest',harness+'\nreturn load;')(require,__dirname,jest);
const slots=['top','outerwear','bottom','shoes'];
const names=['White regular poplin dress shirt','Navy regular wool blazer','Grey straight tailored trousers','Brown leather penny loafers'];
const rows=slots.flatMap((macroCategory,s)=>Array.from({length:4},(_,i)=>({id:`shop_s3-${s}-${i}`,name:`${names[s]} model ${i}`,
  macroCategory,type:macroCategory,imageUrl:`https://fixture.invalid/${s}-${i}.jpg`,price:50,currency:'USD',isShopItem:true,isWardrobe:false})));
const catalog={rpc:()=>{const q={abortSignal:()=>q,then:resolve=>Promise.resolve({data:[],error:null}).then(resolve)};return q;}};
const ready=()=>load([],null,null,null,catalog);
const request=extra=>new Request('https://fixture.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify({
  stylePreferences:'classic',occasion:'Work',limit:3,deckTarget:3,wardrobeItems:rows,useProvidedWardrobeOnly:true,...extra})});
test('one failed provider call returns the prevalidated anchored, budgeted plan without shoe reconstruction',async()=>{
  const {handler,fetchMock}=ready();fetchMock.mockImplementation(async()=>new Response('{}',{status:503}));
  const anchor={...rows[0],id:'owned-anchor',isShopItem:false,isWardrobe:true};
  const response=await handler(request({anchorItemId:anchor.id,wardrobeItems:[anchor,...rows.filter(row=>row.macroCategory!=='top')],preferences:{maxShopBudget:{amount:160,currency:'USD'}}}));
  const data=await response.json();expect({status:response.status,code:data.code}).toEqual({status:200,code:undefined});
  expect(fetchMock).toHaveBeenCalledTimes(1);expect(data.source).toBe('deterministic_fallback');expect(data.outfits).toHaveLength(3);
  for(const look of data.outfits){expect(look.items.some(item=>item.id===anchor.id)).toBe(true);expect(look.items.map(item=>item.macroCategory).sort()).toEqual([...slots].sort());expect(look.items.filter(item=>item.isShopItem).reduce((n,item)=>n+item.price,0)).toBeLessThanOrEqual(160);}
  expect(new Set(data.outfits.flatMap(look=>look.items.filter(item=>item.id!==anchor.id).map(item=>item.id))).size).toBe(9);
  expect(data.diagnostics.counts.evaluations).toBeLessThanOrEqual(256);
});
test('display history and malformed ranking cannot turn an existing unique triple into a 422',async()=>{
  const {handler,fetchMock}=ready();fetchMock.mockImplementation(async()=>new Response(JSON.stringify({choices:[{message:{content:'{"rankedLookIds":["invented"]}'}}]})));
  const response=await handler(request({excludeItemIds:rows.map(row=>row.id)}));const data=await response.json();
  expect({status:response.status,code:data.code}).toEqual({status:200,code:undefined});expect(fetchMock).toHaveBeenCalledTimes(1);
  expect(new Set(data.outfits.flatMap(look=>look.items.map(item=>item.id))).size).toBe(12);
  expect(data.source).toBe('deterministic_fallback');
});
test('zero explicit budget and invalid authentication fail with a stage before provider work',async()=>{
  const {handler,fetchMock}=ready();const denied=await handler(request({preferences:{maxShopBudget:{amount:0,currency:'USD'}}}));
  const data=await denied.json();expect(denied.status).toBe(422);expect(typeof data.code).toBe('string');expect(typeof data.stage).toBe('string');expect(fetchMock).not.toHaveBeenCalled();
  const unauth=await handler(new Request('https://fixture.invalid',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}));
  expect(unauth.status).toBe(401);expect(await unauth.json()).toMatchObject({code:'unauthorized',stage:'admission'});
});
