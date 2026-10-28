/** @jest-environment node */
const {makeReferenceSnapshot}=require('../fixtures/budgetFx');
const fxCore=require('../../backend/supabase/functions/_shared/budgetFx');
const fs=require('fs'),path=require('path'),ts=require('typescript');
const entry=process.env.OUTFIT_WIRE_SOURCE || path.resolve(__dirname,'../../backend/supabase/functions/generate-outfits/index.ts');
const compile=source=>ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
function load(modelItems=['wardrobe-shirt','wardrobe-pants','wardrobe-shoes','wardrobe-layer'],authFailure=null,admission=null,paidTier=null,catalogQuery=null,selectionMode=null){
 let handler;
 const loadRates=jest.fn(async()=>makeReferenceSnapshot());
 const pinnedRates=jest.fn(async(_admin,id)=>{const snapshot=makeReferenceSnapshot();if(id!==snapshot.id)throw new fxCore.BudgetFxError();return snapshot;});
 const rateMock=jest.fn(async()=>admission);
 const fetchMock=jest.fn(async()=>new Response(JSON.stringify({choices:[{message:{content:JSON.stringify({outfits:Array.isArray(modelItems[0])?modelItems.map(items=>({items})):[{items:modelItems}]})}}]})));
 const catalogRpc=jest.fn((name,args)=>{if(catalogQuery.rpc)return catalogQuery.rpc(name,args);if(args.p_after_id)catalogQuery.gt('id',args.p_after_id);const result={abortSignal:()=>result,then:resolve=>Promise.resolve(catalogQuery).then(({data,error})=>({data:data?.map(row=>({...row,_budget_cost:Number(row.price)>0&&args.p_rates[row.currency]?String(Number(row.price)*args.p_rates[row.currency]):null})),error})).then(resolve)};return result;});
 const requireModule=name=>{
  if(name.includes('server.ts'))return {serve:fn=>{handler=fn;}};
  if(name.includes('supabase-js'))return {createClient:(_url,_key,options)=>({...paidTier?{from:()=>({select:()=>({eq:()=>({order:()=>({limit:async()=>({data:[{tier:paidTier,status:'active'}],error:null})})})})})}:{},...(catalogQuery?{rpc:catalogRpc,from:table=>{if(table==='shop_catalog')return catalogQuery;throw Error('Unexpected fixture table');}}:{}),auth:{getUser:async()=>authFailure?{data:{user:null},error:authFailure}:options?.global?.headers?.Authorization==='Bearer fixture-user'?{data:{user:{id:'fixture-user-id'}},error:null}:{data:{user:null},error:{status:401}}}})};
  if(name.includes('rateLimit'))return {enforceRateLimit:rateMock};
  if(name.includes('budgetFxCache'))return {currentBudgetRates:loadRates,budgetRatesById:pinnedRates};
  if(name.includes('budgetFx.ts'))return fxCore;
  const target={};const file=path.resolve(path.dirname(entry),name);
  new Function('require','exports',compile(fs.readFileSync(file,'utf8')))(child=>requireModule(child.startsWith('.')?path.resolve(path.dirname(file),child):child),target);return target;
 };
 const env={SUPABASE_URL:'https://example.invalid',SUPABASE_ANON_KEY:'fixture-anon',SUPABASE_SERVICE_ROLE_KEY:'fixture-service',NVIDIA_API_KEY:'fixture-nvidia',GEMINI_API_KEY:'fixture-gemini',NVIDIA_OUTFIT_SELECTION_MODE:selectionMode};
 const validation=new Function('require','exports','Deno','fetch',compile(fs.readFileSync(entry,'utf8'))+'\nreturn {filterValidOutfits,fetchShopItemsForAllSlotsEdge};')(requireModule,{}, {env:{get:name=>env[name]}},fetchMock);
 return {handler,fetchMock,rateMock,loadRates,pinnedRates,catalogRpc,...validation};
}
const owned=[['wardrobe-shirt','top','White cotton shirt'],['wardrobe-pants','bottom','Navy cotton trousers'],['wardrobe-shoes','shoes','Brown leather loafers'],['wardrobe-layer','outerwear','Beige linen blazer']].map(([id,macroCategory,name])=>({id,macroCategory,name,type:macroCategory,color:'neutral',isWardrobe:true,imageUrl:'https://example.invalid/image.jpg',currency:'USD'}));
const request=(budget,wardrobeItems=owned)=>new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify({prompt:'Create one casual outfit. Budget under $1000.',stylePreferences:'casual',occasion:'Everyday',weather:{temp:20},limit:1,wardrobeItems,useProvidedWardrobeOnly:true,wardrobeOnly:true,preferences:{maxShopBudget:budget}})});
test.each([250,0])('wire prompt preserves the selected %s USD cap despite larger prompt text',async amount=>{
 const {handler,fetchMock}=load();const response=await handler(request({amount,currency:'USD'}));
 expect(response.status).toBe(200);expect(fetchMock).toHaveBeenCalled();
 const messages=JSON.parse(fetchMock.mock.calls[0][1].body).messages;
 expect(messages.map(row=>row.content).join('\n')).toContain(`HARD BUDGET ${amount} USD`);
});
test('malformed explicit budget is rejected before any provider request',async()=>{
 const {handler,fetchMock}=load();const response=await handler(request({amount:-1,currency:'USD'}));
 expect(response.status).toBe(400);expect((await response.json()).error).toBe('invalid_outfit_budget');expect(fetchMock).not.toHaveBeenCalled();
});

test('an impossible shopping budget reports insufficient budget, not an empty wardrobe',async()=>{
 const {handler,fetchMock}=load();const shopping=owned.map(item=>({...item,id:'shop_'+item.id,isWardrobe:false,isShopItem:true,price:50}));
 const response=await handler(request({amount:1,currency:'USD'},shopping));const body=await response.json();
 expect(response.status).toBe(422);expect(body.code).toBe('no_items_within_budget');expect(body.outfits).toEqual([]);expect(fetchMock).not.toHaveBeenCalled();
});
test('an actually empty pool is distinguished from budget filtering',async()=>{
 const {handler,fetchMock}=load();const response=await handler(request(null,[]));
 expect(response.status).toBe(422);expect((await response.json()).code).toBe('no_matching_clothing');expect(fetchMock).not.toHaveBeenCalled();
});

test('multiple text currencies are compared by value rather than their raw numbers',async()=>{
 const {handler,fetchMock}=load();
 const input=await request(null).json();input.prompt='Create a casual outfit. Maximum $100 or £90.';
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(input)}));
 expect(response.status).toBe(200);
 const messages=JSON.parse(fetchMock.mock.calls[0][1].body).messages;
 expect(messages.map(row=>row.content).join('\n')).toContain('HARD BUDGET 100 USD');
});

test('the server does not return a casual look that the shared app style judge rejects',async()=>{
 const {handler}=load();
 const items=owned.map(item=>item.macroCategory==='top'?{...item,name:'Футболка SS BADGE 30S JERSEY CLASSIC TEE'}:item);
 const response=await handler(request({amount:250,currency:'USD'},items));
 expect(response.ok).toBe(false);const body=await response.json();expect(body.success).toBe(false);expect(body.outfits || []).toEqual([]);
});


test('returned product attributes come only from the selected authoritative rows',async()=>{
 const poisoned=owned.map(item=>({id:item.id,name:'Invented product',brand:'Invented brand',price:9999,currency:'GBP',imageUrl:'https://invented.invalid/image.jpg'}));
 const {handler}=load(poisoned);const response=await handler(request({amount:250,currency:'USD'}));
 expect(response.status).toBe(200);const body=await response.json();
 for(const item of body.outfits[0].items){
  const row=owned.find(row=>row.id===item.id);
  expect(item.name).toBe(row.name);expect(item.currency).toBe(row.currency);
  expect(item.price).toBeUndefined();expect(item.brand).not.toBe('Invented brand');expect(item.imageUrl).toBe(row.imageUrl);
 }
});

test('an anchor substitution cannot turn a valid budgeted selection into an over-budget response',async()=>{
 const {handler}=load();
 const items=owned.map(item=>item.macroCategory==='top'?item:{...item,isWardrobe:false,isShopItem:true,price:100});
 items.push({...owned[0],id:'shop_anchor',name:'White cotton knit polo',isWardrobe:false,isShopItem:true,price:200});
 const input=await request({amount:250,currency:'USD'},items).json();input.anchorItemId='shop_anchor';
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(input)}));
 expect(response.ok).toBe(false);expect((await response.json()).outfits || []).toEqual([]);
});

test('a complete pool whose cheapest possible look exceeds the cap is rejected before NVIDIA',async()=>{
 const {handler,fetchMock}=load();const items=owned.map(item=>({...item,isWardrobe:false,isShopItem:true,price:100}));
 const response=await handler(request({amount:250,currency:'USD'},items));const body=await response.json();
 expect(response.status).toBe(422);expect(body.code).toBe('no_outfit_within_budget');expect(body.minimumShopSpend).toBe(400);expect(body.outfits).toEqual([]);expect(fetchMock).not.toHaveBeenCalled();
});

test('unknown required pricing reports unavailable prices without claiming the budget is too low',async()=>{
 const {handler,fetchMock}=load();const items=owned.map(item=>({...item,isWardrobe:false,isShopItem:true,price:item.macroCategory==='top'?undefined:100}));
 const response=await handler(request({amount:250,currency:'USD'},items));
 expect(response.status).toBe(422);expect((await response.json()).code).toBe('unverified_shop_pricing');expect(fetchMock).not.toHaveBeenCalled();
});
test('a dress cannot bypass the required shirt and pants categories',async()=>{
 const {handler,fetchMock}=load();const items=owned.map(item=>({...item,name:item.macroCategory==='top'?'Plain midi dress':item.name,isWardrobe:false,isShopItem:true,price:100}));
 const response=await handler(request({amount:250,currency:'USD'},items));expect(response.status).toBe(422);expect((await response.json()).outfits||[]).toEqual([]);expect(fetchMock).not.toHaveBeenCalled();
});


test.each([undefined,'Bearer fixture-anon','Bearer invalid-token'])('provided clothing cannot bypass user authentication with %s',async authorization=>{
 const {handler,fetchMock}=load();const input=await request({amount:250,currency:'USD'}).json();
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{'Content-Type':'application/json',...(authorization?{Authorization:authorization}:{})},body:JSON.stringify(input)}));
 expect(response.status).toBe(401);expect((await response.json()).error).toBe('unauthorized');expect(fetchMock).not.toHaveBeenCalled();
});


test('an Auth outage fails closed without being reported as invalid credentials',async()=>{
 const {handler,fetchMock}=load(undefined,{status:503});const response=await handler(request({amount:250,currency:'USD'}));
 expect(response.status).toBe(503);expect((await response.json()).error).toBe('auth_unavailable');expect(fetchMock).not.toHaveBeenCalled();
});

// The staged hosted handler must return admission denial before any model call.
test('rate denial is returned before paid provider work',async()=>{
 const denial=new Response(JSON.stringify({error:'rate_limited'}),{status:429,headers:{'retry-after':'30'}});
 const {handler,fetchMock,rateMock}=load(undefined,null,denial);
 const response=await handler(request({amount:250,currency:'USD'}));
 expect(rateMock).toHaveBeenCalled();
 expect(response.status).toBe(429);
 expect(response.headers.get('retry-after')).toBe('30');
 expect(fetchMock).not.toHaveBeenCalled();
});
test('generation admission requires the server paid entitlement check',async()=>{
 const {handler,rateMock}=load();
 await handler(request({amount:-1,currency:'USD'}));
 expect(rateMock).toHaveBeenCalledWith(expect.any(Request),expect.objectContaining({feature:'generate-outfits',limit:10}));
 expect(rateMock.mock.calls[0][1].meterPaidWork).not.toBe(false);
});

test('mixed-currency infeasibility is measured before a provider request',async()=>{
 const {handler,fetchMock}=load();
 const shopping=owned.map((item,index)=>({...item,id:'shop_'+item.id,isWardrobe:false,isShopItem:true,
  price:index===0?100000:20,currency:index===0?'UZS':'USD'}));
 const response=await handler(request({amount:400000,currency:'UZS'},shopping));
 const body=await response.json();
 expect(response.status).toBe(422);expect(body.code).toBe('no_outfit_within_budget');
 expect(body.minimumShopSpend).toBeCloseTo(100000+3*20*makeReferenceSnapshot().uzsPerUnit.USD);expect(fetchMock).not.toHaveBeenCalled();
});

test.each([undefined,null,0,-1])('an entirely unpriced pool (%s) returns a pricing error before paid provider work',async price=>{
 const {handler,fetchMock}=load();const items=owned.map(item=>({...item,isWardrobe:false,isShopItem:true,price}));
 const response=await handler(request({amount:250,currency:'USD'},items));const body=await response.json();
 expect(response.status).toBe(422);expect(body.code).toBe('unverified_shop_pricing');expect(body.outfits).toEqual([]);expect(fetchMock).not.toHaveBeenCalled();
});
test('optional unpriced shop candidates do not block an available owned outfit',async()=>{
 const {handler,fetchMock}=load();const items=[...owned,{...owned[0],id:'shop_optional',isWardrobe:false,isShopItem:true,price:null}];
 const response=await handler(request({amount:250,currency:'USD'},items));
 expect(response.status).toBe(200);expect(fetchMock).toHaveBeenCalled();expect((await response.json()).outfits[0].items.every(item=>item.id!=='shop_optional')).toBe(true);
});
test('an unpriced locked item reports unavailable pricing even when another outfit is possible',async()=>{
 const {handler,fetchMock}=load();const anchor={...owned[0],id:'shop_unpriced_anchor',isWardrobe:false,isShopItem:true};
 const input=await request({amount:250,currency:'USD'},[...owned,anchor]).json();input.anchorItemId=anchor.id;
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(input)}));
 expect(response.status).toBe(422);expect((await response.json()).code).toBe('unverified_shop_pricing');expect(fetchMock).not.toHaveBeenCalled();
});

test('published EUR direction cannot admit a 100 EUR shop piece under a 100 USD cap',async()=>{
 const {handler,fetchMock,loadRates}=load();const items=owned.map(item=>({...item,isWardrobe:false,isShopItem:true,price:100,currency:'EUR'}));
 const r=await handler(request({amount:100,currency:'USD'},items));expect(r.status).toBe(422);expect((await r.json()).code).toBe('no_items_within_budget');expect(loadRates).toHaveBeenCalledTimes(1);expect(fetchMock).not.toHaveBeenCalled();
});
test('a forged client rate snapshot cannot override the server reference',async()=>{
 const {handler,loadRates}=load();const items=owned.map(item=>({...item,isWardrobe:false,isShopItem:true,price:100,currency:'EUR'}));
 const input=await request({amount:100,currency:'USD',fx:{snapshot:{uzsPerUnit:{EUR:1,USD:999999}}}},items).json();
 const r=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(input)}));expect(r.status).toBe(422);expect(loadRates).toHaveBeenCalledTimes(1);
});
test('the server pins a trusted snapshot ID and acknowledges it in successful output',async()=>{
 const {handler,pinnedRates,loadRates}=load(),input=await request({amount:250,currency:'USD'}).json();input.preferences.budgetRatesId=makeReferenceSnapshot().id;
 const r=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(input)}));
 expect(r.status).toBe(200);expect((await r.json()).budgetReference.id).toBe(input.preferences.budgetRatesId);expect(pinnedRates).toHaveBeenCalledTimes(1);expect(loadRates).not.toHaveBeenCalled();
});
test('an unknown snapshot ID fails before provider work',async()=>{
 const {handler,fetchMock}=load(),input=await request({amount:250,currency:'USD'}).json();input.preferences.budgetRatesId='fabricated';
 const r=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(input)}));expect(r.status).toBe(503);expect((await r.json()).error).toBe('budget_rates_unavailable');expect(fetchMock).not.toHaveBeenCalled();
});
test('an owned same-currency budget does not depend on a reference-rate request',async()=>{
 const {handler,loadRates,pinnedRates}=load();const r=await handler(request({amount:250,currency:'USD'}));expect(r.status).toBe(200);expect(loadRates).not.toHaveBeenCalled();expect(pinnedRates).not.toHaveBeenCalled();
});

test.each(['GET','DELETE','PUT'])('%s cannot trigger metered generation',async method=>{
 const {handler,rateMock,fetchMock}=load();const r=await handler(new Request('https://example.invalid',{method,headers:{Authorization:'Bearer fixture-user'}}));expect(r.status).toBe(405);expect(rateMock).not.toHaveBeenCalled();expect(fetchMock).not.toHaveBeenCalled();
});

test('a required EUR piece over the USD cap is rejected before provider work even with free remaining slots',async()=>{
 const {handler,fetchMock}=load();const items=owned.map(item=>item.macroCategory==='top'?{...item,isWardrobe:false,isShopItem:true,price:100,currency:'EUR'}:item);
 const r=await handler(request({amount:100,currency:'USD'},items)),body=await r.json();expect(r.status).toBe(422);expect(body.code).toBe('no_outfit_within_budget');expect(body.minimumShopSpend).toBeCloseTo(112.290032479,8);expect(fetchMock).not.toHaveBeenCalled();
});
test('outerwear cannot replace the mandatory base-top slot during budget feasibility checks',async()=>{
 const {handler}=load(['owned_light_layer','wardrobe-pants','wardrobe-shoes']);const items=owned.map(item=>item.macroCategory==='top'?{...item,isWardrobe:false,isShopItem:true,price:100,currency:'EUR'}:item);
 items.push({...owned[0],id:'owned_light_layer',name:'Navy lightweight cotton cardigan',macroCategory:'outerwear',type:'cardigan'});
 const r=await handler(request({amount:100,currency:'USD'},items));expect(r.status).toBe(422);expect((await r.json()).code).toBe('no_outfit_within_budget');
});


test('an owned base-top alternative prevents a false budget rejection',async()=>{
 const {handler}=load(['owned_free_top','wardrobe-pants','wardrobe-shoes']);const items=owned.map(item=>item.macroCategory==='top'?{...item,isWardrobe:false,isShopItem:true,price:100,currency:'EUR'}:item);
 items.push({...owned[0],id:'owned_free_top',name:'Light blue cotton shirt'});
 const r=await handler(request({amount:100,currency:'USD'},items));expect(r.status).toBe(200);
});

test('a disliked cheap tee cannot make the required-slot budget look feasible',async()=>{
 const {handler,fetchMock}=load(['shop_wardrobe-shirt','shop_wardrobe-pants','shop_wardrobe-shoes']);
 const shopping=owned.map(item=>({...item,id:'shop_'+item.id,isWardrobe:false,isShopItem:true,price:100}));
 shopping[0].name='White cotton knit polo shirt';
 shopping.push({...shopping[0],id:'shop_cheap_tee',name:'Red graphic T-shirt',color:'red',price:1});
 const input=await request({amount:250,currency:'USD'},shopping).json();input.stylePreferences='old_money';input.occasion='Work';input.prompt='Work';input.preferences.avoidColors=['red'];
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(input)}));
 const body=await response.json();
 expect(response.status).toBe(422);
 expect(body.code).toBe('no_outfit_within_budget');
 expect(body.minimumShopSpend).toBe(400);
 expect(fetchMock).not.toHaveBeenCalled();
});

test('correction explains the actual authoritative combination rejection instead of trusting model metadata',async()=>{
 const modelItems=[{id:'wardrobe-shirt'}, {id:'wardrobe-pants',name:'Cotton chinos'}, {id:'wardrobe-shoes',name:'Polished leather oxfords'}];
 const {handler,fetchMock}=load(modelItems);
 const items=owned.map(item=>item.macroCategory==='bottom'?{...item,name:'Wool suit trousers'}:item.macroCategory==='shoes'?{...item,name:'Brown suede loafers'}:item);
 const body=await request({amount:300,currency:'USD'},items).json();
 body.stylePreferences='old_money';body.occasion='date';body.prompt='Create one old-money date outfit. Budget under $300.';
 const result=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(result.ok).toBe(false);
 expect(fetchMock).toHaveBeenCalledTimes(2);
 const correction=JSON.parse(fetchMock.mock.calls[1][1].body).messages.find(message=>message.role==='user').content;
 expect(correction).toContain('AUTHORITATIVE VALIDATION FAILURES');
 expect(correction).toContain('suede stays in smart casual — suits take leather');
 expect(correction).not.toContain('Polished leather oxfords');
});

test('a correction that fixes the named combination error can return a validated complete outfit',async()=>{
 const {handler,fetchMock}=load();
 const items=owned.map(item=>item.macroCategory==='bottom'?{...item,name:'Wool suit trousers'}:item.macroCategory==='shoes'?{...item,name:'Brown suede loafers'}:item);
 items.push({...owned[2],id:'leather-shoes',name:'Brown leather penny loafers'});
 const resultFor=ids=>new Response(JSON.stringify({choices:[{message:{content:JSON.stringify({outfits:[{items:ids.map(id=>({id}))} ]})}}]}));
 fetchMock.mockResolvedValueOnce(resultFor(['wardrobe-shirt','wardrobe-pants','wardrobe-shoes','wardrobe-layer']))
   .mockResolvedValueOnce(resultFor(['wardrobe-shirt','wardrobe-pants','leather-shoes','wardrobe-layer']));
 const body=await request({amount:300,currency:'USD'},items).json();
 body.stylePreferences='old_money';body.occasion='date';body.prompt='Create one old-money date outfit. Budget under $300.';
 const result=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(result.status).toBe(200);
 expect((await result.json()).outfits[0].items.map(item=>item.id)).toEqual(['wardrobe-shirt','wardrobe-pants','leather-shoes','wardrobe-layer']);
 expect(fetchMock).toHaveBeenCalledTimes(2);
});

test('paid accounts use the selected NVIDIA provider without Gemini routing',async()=>{
 const {handler,fetchMock}=load(undefined,null,null,'premium');
 const response=await handler(request({amount:300,currency:'USD'}));
 expect(response.status).toBe(200);
 expect(fetchMock).toHaveBeenCalledTimes(1);
 expect(fetchMock.mock.calls[0][0]).toBe('https://integrate.api.nvidia.com/v1/chat/completions');
 expect((await response.json()).provider).toMatch(/^nvidia:/);
});

it.each(['Brown loafers with tie laces','Brown driving moccasins'])('keeps the explicit old-money footwear ban for %s',async name=>{
 const {handler}=load();
 const items=owned.map(item=>item.macroCategory==='shoes'?{...item,name}:item);
 const body=await request({amount:300,currency:'USD'},items).json();
 body.stylePreferences='old_money';body.occasion='date';body.prompt='Create one old-money date outfit. Budget under $300.';
 const result=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(result.ok).toBe(false);
 const data=await result.json();expect(data.success).toBe(false);expect(data.outfits || []).toEqual([]);
});

it.each([
 ['color','red','navy',{avoidColors:['red']}],
 ['pattern','pinstripe','plain',{dislikedPatterns:['pinstripe']}],
])('model %s cannot conceal an authoritative user-preference exclusion',(_field,canonicalValue,modelValue,preferences)=>{
 const {filterValidOutfits}=load();
 const source=owned.map(item=>item.macroCategory==='top'?{...item,[_field]:canonicalValue}:item);
 const itemMap=new Map(source.map(item=>[item.id,item]));
 const model={items:source.map(item=>({id:item.id,[_field]:modelValue}))};
 expect(filterValidOutfits([model],'casual',itemMap,true,preferences,'Everyday',{temp:20})).toEqual([]);
});

test('fabricated model metadata cannot reject otherwise allowed canonical items',()=>{
 const {filterValidOutfits}=load();
 const itemMap=new Map(owned.map(item=>[item.id,item]));
 const model={items:owned.map(item=>({id:item.id,name:'Red graphic hoodie',color:'red',pattern:'floral',macroCategory:'outerwear'}))};
 const result=filterValidOutfits([model],'casual',itemMap,true,{avoidColors:['red'],dislikedPatterns:['floral']},'Everyday',{temp:20});
 expect(result).toHaveLength(1);
 expect(result[0].items).toEqual(owned);
});

test('workout intent uses athletic guidance even when the saved style is old money',async()=>{
 const {handler,fetchMock}=load();
 const names={top:'Navy performance training shirt',bottom:'Navy performance training trousers',shoes:'White athletic running shoes'};
 const athletic=owned.map(item=>({...item,name:names[item.macroCategory]}));
 const body=await request({amount:300,currency:'USD'},athletic).json();
 body.stylePreferences='old_money';body.occasion='Gym workout';body.prompt='Create one complete outfit for a gym workout.';
 const result=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(result.status).toBe(200);
 const messages=JSON.parse(fetchMock.mock.calls[0][1].body).messages;
 const system=messages.find(message=>message.role==='system').content;
 const user=messages.find(message=>message.role==='user').content;
 expect(system).toContain('STYLE DIRECTION — SPORT');
 expect(system).toContain('performance');
 expect(system).not.toContain('CITY BANS');
 expect(system).not.toContain('Black only pairs with white');
 expect(user).toContain('Requested style: sport');
});

test('cold workout prompts require a warm performance layer without prescribing dress clothing',async()=>{
 const ids=['wardrobe-shirt','wardrobe-pants','wardrobe-shoes','training-jacket'];
 const {handler,fetchMock}=load(ids);
 const names={top:'Navy performance training shirt',bottom:'Navy performance training trousers',shoes:'White athletic running shoes'};
 const athletic=owned.map(item=>({...item,name:names[item.macroCategory]}));
 athletic.push({...owned[0],id:'training-jacket',macroCategory:'outerwear',name:'Warm performance training jacket',type:'jacket'});
 const body=await request({amount:300,currency:'USD'},athletic).json();
 body.stylePreferences='old_money';body.occasion='Gym workout';body.prompt='Create one complete outfit for a gym workout.';body.weather={temp:15,condition:'Clear'};
 const result=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(result.status).toBe(200);
 expect((await result.json()).outfits[0].items).toHaveLength(4);
 const user=JSON.parse(fetchMock.mock.calls[0][1].body).messages.find(message=>message.role==='user').content;
 expect(user).toContain('warm performance layers');
 expect(user).not.toContain('wool blazer');
});

test('workout guidance still rejects dress shoes instead of relaxing activity requirements',async()=>{
 const {handler}=load();
 const athletic=owned.map(item=>item.macroCategory==='top'?{...item,name:'Navy performance training shirt'}:item.macroCategory==='bottom'?{...item,name:'Navy performance training trousers'}:item);
 const body=await request({amount:300,currency:'USD'},athletic).json();body.stylePreferences='old_money';body.occasion='Gym workout';body.prompt='Create one complete outfit for a gym workout.';
 const result=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(result.ok).toBe(false);expect((await result.json()).outfits || []).toEqual([]);
});

test('missing an eligible athletic slot is rejected before provider work',async()=>{
 const {handler,fetchMock}=load();
 const names={top:'Navy performance training shirt',bottom:'Navy performance training trousers',shoes:'Brown leather loafers'};
 const body=await request({amount:400,currency:'USD'},owned.map(item=>({...item,name:names[item.macroCategory]}))).json();
 body.stylePreferences='sport';body.occasion='Gym workout';body.prompt='Create one gym outfit.';
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(response.status).toBe(422);expect((await response.json()).code).toBe('no_matching_clothing');expect(fetchMock).not.toHaveBeenCalled();
});

test('catalog activity filtering runs before per-slot trimming',async()=>{
 const {fetchShopItemsForAllSlotsEdge}=load();
 const data=[{id:'dress-top',name:'Cotton dress shirt',category:'tops',garment_type:'upper_body',image_url:'https://example.invalid/dress.jpg'},
 {id:'training-top',name:'Performance training shirt',category:'tops',garment_type:'upper_body',image_url:'https://example.invalid/training.jpg'}];
 const q={select:jest.fn(),eq:jest.fn(),limit:jest.fn(),or:jest.fn(),order:jest.fn(),gt:jest.fn(),abortSignal:jest.fn(),then:resolve=>Promise.resolve({data,error:null}).then(resolve)};
 for(const method of ['select','eq','limit','or','order','gt','abortSignal'])q[method].mockReturnValue(q);
 const result=await fetchShopItemsForAllSlotsEdge({from:()=>q},['top'],'sport',1);
 expect(result.map(item=>item.id)).toEqual(['shop_training-top']);
 expect(q.or).toHaveBeenCalledWith(expect.stringContaining('description.ilike.%training%'));
});

test('workout prompts exclude ineligible extra items while retaining a complete valid pool',async()=>{
 const {handler,fetchMock}=load();const names={top:'Navy performance training shirt',bottom:'Navy performance training trousers',shoes:'White athletic running shoes'};
 const athletic=owned.map(item=>({...item,name:names[item.macroCategory]}));athletic.push({...owned[0],id:'ineligible-dress-shirt',name:'Formal dress shirt'});
 const body=await request({amount:400,currency:'USD'},athletic).json();body.stylePreferences='sport';body.occasion='Gym workout';body.prompt='Create one gym outfit.';
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(response.status).toBe(200);
 const prompt=JSON.parse(fetchMock.mock.calls[0][1].body).messages.find(message=>message.role==='user').content;
 expect(prompt).not.toContain('ineligible-dress-shirt');
});

test('an activity-ineligible locked anchor is rejected rather than silently replaced',async()=>{
 const {handler,fetchMock}=load();const names={top:'Navy performance training shirt',bottom:'Navy performance training trousers',shoes:'White athletic running shoes'};
 const athletic=owned.map(item=>({...item,name:names[item.macroCategory]}));athletic.push({...owned[0],id:'locked-dress-shirt',name:'Formal dress shirt'});
 const body=await request({amount:400,currency:'USD'},athletic).json();body.stylePreferences='sport';body.occasion='Gym workout';body.prompt='Create one gym outfit.';body.anchorItemId='locked-dress-shirt';
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(response.status).toBe(422);expect((await response.json()).code).toBe('no_matching_clothing');expect(fetchMock).not.toHaveBeenCalled();
});

it.each([
 {limit:100000}, {limit:'3'}, {limit:0}, {prompt:'x'.repeat(1001)}, {selectedItemIds:Array(201).fill('item')},
 {preferences:{avoidColors:'red'}}, {wardrobeItems:'not-an-array'},
])('rejects malformed or excessive request fields before provider work: %j',async fields=>{
 const {handler,fetchMock}=load();const body={...await request({amount:300,currency:'USD'}).json(),...fields};
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(response.status).toBe(400);expect((await response.json()).error).toBe('invalid_generation_request');expect(fetchMock).not.toHaveBeenCalled();
});

test('formal candidate ranking cannot show the provider a known-ineligible extra garment',async()=>{
 const {handler,fetchMock}=load();const items=[...owned,{...owned[1],id:'bad-relaxed',name:'100% LINEN RELAXED FIT TROUSERS'}];
 const body=await request({amount:300,currency:'USD'},items).json();body.stylePreferences='semi_classic';body.occasion='formal';body.prompt='Create one complete formal outfit.';
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(response.status).toBe(200);
 expect(JSON.parse(fetchMock.mock.calls[0][1].body).messages[1].content).not.toContain('bad-relaxed');
});
it.each(['anchorItemId','selectedItemIds'])('an incompatible formal %s is rejected before provider work',async field=>{
 const {handler,fetchMock}=load();const items=[...owned,{...owned[1],id:'bad-relaxed',name:'100% LINEN RELAXED FIT TROUSERS'}];
 const body=await request({amount:300,currency:'USD'},items).json();body.stylePreferences='semi_classic';body.occasion='formal';body.prompt='Create one complete formal outfit.';body[field]=field==='anchorItemId'?'bad-relaxed':['bad-relaxed'];
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(response.status).toBe(422);expect((await response.json()).blockedSelectedItemIds).toContain('bad-relaxed');expect(fetchMock).not.toHaveBeenCalled();
});
test('a formal pool lacking an eligible bottom is rejected before provider work',async()=>{
 const {handler,fetchMock}=load();const items=owned.map(item=>item.macroCategory==='bottom'?{...item,name:'100% LINEN RELAXED FIT TROUSERS'}:item);
 const body=await request({amount:300,currency:'USD'},items).json();body.stylePreferences='semi_classic';body.occasion='formal';body.prompt='Create one complete formal outfit.';
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(response.status).toBe(422);expect((await response.json()).missingSlots).toContain('bottom');expect(fetchMock).not.toHaveBeenCalled();
});
test('eligibility rejects dress-and-shoes as an incomplete four-category pool',async()=>{
 const {handler,fetchMock}=load(['wardrobe-dress','wardrobe-shoes']);
 const items=[{...owned[0],id:'wardrobe-dress',name:'Cotton midi dress',macroCategory:'dress'},owned[2]];
 const body=await request({amount:300,currency:'USD'},items).json();body.prompt='Create one complete outfit with a dress.';
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(response.status).toBe(422);expect(fetchMock).not.toHaveBeenCalled();expect((await response.json()).outfits||[]).toEqual([]);
});
test('a coat and dress still cannot replace a shirt and pants',async()=>{
 const {handler}=load(['wardrobe-dress','wardrobe-shoes','wardrobe-coat']);
 const items=[{...owned[0],id:'wardrobe-dress',name:'Wool midi dress',macroCategory:'dress'},owned[2],{...owned[0],id:'wardrobe-coat',name:'Wool coat',macroCategory:'outerwear'}];
 const body=await request({amount:300,currency:'USD'},items).json();body.prompt='Create one complete outfit with a dress.';body.weather={temp:10};
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(response.status).toBe(422);expect((await response.json()).outfits||[]).toEqual([]);
});
test('six unsuitable owned bottoms no longer suppress a valid shop supplement',async()=>{
 const data=[{id:'catalog-regular',name:'Regular fit cotton trousers',category:'bottoms',garment_type:'lower_body',price:50,currency:'USD',image_url:'https://example.invalid/trousers.jpg',source_url:'https://example.invalid/trousers'}];
 const q={select:jest.fn(),eq:jest.fn(),limit:jest.fn(),or:jest.fn(),order:jest.fn(),gt:jest.fn(),abortSignal:jest.fn(),then:resolve=>Promise.resolve({data,error:null}).then(resolve)};
 for(const name of ['select','eq','limit','or','order','gt','abortSignal'])q[name].mockReturnValue(q);
 const {handler,fetchMock,catalogRpc}=load(['wardrobe-shirt-0','shop_catalog-regular','wardrobe-shoes-0'],null,null,null,q);
 const items=owned.flatMap(item=>Array.from({length:6},(_,i)=>({...item,id:`${item.id}-${i}`,name:item.macroCategory==='bottom'?'LINEN RELAXED FIT TROUSERS':item.name})));
 const body=await request({amount:300,currency:'USD'},items).json();body.wardrobeOnly=false;body.stylePreferences='semi_classic';body.occasion='formal';body.prompt='Create one complete formal outfit.';
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(response.status).toBe(200);expect(catalogRpc).toHaveBeenCalledWith('budget_catalog_candidates',expect.objectContaining({p_slot:'bottom'}));
 expect((await response.json()).outfits[0].items.map(item=>item.id)).toContain('shop_catalog-regular');
 expect(JSON.parse(fetchMock.mock.calls[0][1].body).messages[1].content).not.toContain('RELAXED FIT TROUSERS');
});
it.each([1,2])('server response never exceeds the requested %s outfit count',async limit=>{
 const {handler}=load([['wardrobe-shirt','wardrobe-pants','wardrobe-shoes','wardrobe-layer'],['alternative-shirt','alternative-pants','wardrobe-shoes','wardrobe-layer']]);
 const items=[...owned,{...owned[0],id:'alternative-shirt',name:'Cream merino knit polo shirt'},{...owned[1],id:'alternative-pants',name:'Stone regular fit linen chinos'}];
 const body=await request({amount:300,currency:'USD'},items).json();body.limit=limit;
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(response.status).toBe(200);expect((await response.json()).outfits).toHaveLength(limit);
});
test('catalog lookup failure is reported as unavailable before provider work',async()=>{
 const q={select:()=>q,eq:()=>q,limit:()=>q,or:()=>q,order:()=>q,gt:()=>q,abortSignal:()=>q,then:resolve=>Promise.resolve({data:null,error:{message:'fixture unavailable'}}).then(resolve)};
 const {handler,fetchMock}=load(undefined,null,null,null,q);
 const body=await request({amount:300,currency:'USD'},[]).json();body.wardrobeOnly=false;
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(response.status).toBe(503);expect((await response.json()).error).toBe('catalog_lookup_unavailable');expect(fetchMock).not.toHaveBeenCalled();
});

test('ranking and similarity collapse preserve an affordable complete look from the provided pool',async()=>{
 const expensive=owned.map(item=>({...item,id:'shop_expensive_'+item.id,isWardrobe:false,isShopItem:true,price:80}));
 const cheap=owned.map(item=>({...item,id:'shop_cheap_'+item.id,isWardrobe:false,isShopItem:true,price:10}));
 const {handler,fetchMock}=load(cheap.map(item=>item.id));
 const response=await handler(request({amount:100,currency:'USD'},[...expensive,...cheap]));
 expect(response.status).toBe(200);expect(fetchMock).toHaveBeenCalled();
 const body=await response.json();expect(body.outfits).toHaveLength(1);
 expect(body.outfits[0].items.map(item=>item.id).sort()).toEqual(cheap.map(item=>item.id).sort());
});

it.each([
 ['avoidColors',['black'],'Black cotton trousers'],
 ['dislikedPatterns',['striped'],'Striped cotton trousers'],
])('catalog discovery applies %s before page completion and per-slot selection',async(preference,values,name)=>{
 const first=Array.from({length:100},(_,i)=>({id:`blocked-${String(i).padStart(3,'0')}`,name:`${name} variant${i}`,category:'bottoms',garment_type:'lower_body',price:50,currency:'USD',image_url:'https://example.invalid/blocked.jpg'}));
 const valid={id:'catalog-allowed',name:'Regular fit cotton trousers',category:'bottoms',garment_type:'lower_body',price:50,currency:'USD',image_url:'https://example.invalid/allowed.jpg',source_url:'https://example.invalid/allowed'};
 let cursor;const q={select:()=>q,eq:()=>q,limit:()=>q,or:()=>q,order:()=>q,abortSignal:()=>q,gt:jest.fn((_field,value)=>{cursor=value;return q;}),then:resolve=>Promise.resolve({data:cursor?[valid]:first,error:null}).then(resolve)};
 const {handler,fetchMock}=load(['wardrobe-shirt-0','shop_catalog-allowed','wardrobe-shoes-0'],null,null,null,q);
 const items=owned.filter(item=>item.macroCategory!=='bottom').flatMap(item=>Array.from({length:6},(_,i)=>({...item,id:`${item.id}-${i}`})));
 const body=await request({amount:300,currency:'USD'},items).json();body.wardrobeOnly=false;body.preferences[preference]=values;
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(response.status).toBe(200);expect(q.gt).toHaveBeenCalledWith('id','blocked-099');
 const output=await response.json();expect(output.outfits[0].items.map(item=>item.id)).toContain('shop_catalog-allowed');
 expect(JSON.parse(fetchMock.mock.calls[0][1].body).messages[1].content).not.toContain('shop_blocked-');
});


test('a later cheaper catalog bottom is discovered before an affordable budget is rejected',async()=>{
 const expensive=Array.from({length:100},(_,i)=>({id:`a${String(i).padStart(3,'0')}`,name:`Cotton trousers variant${i}`,category:'bottoms',garment_type:'lower_body',price:80,currency:'USD',image_url:'https://example.invalid/pants.jpg'}));
 const cheap={...expensive[0],id:'z-cheap',name:'Regular fit cotton trousers',price:10};
 const q={select:()=>q,eq:()=>q,limit:()=>q,or:()=>q,order:()=>q,gt:()=>q,abortSignal:()=>q,then:resolve=>Promise.resolve({data:expensive,error:null}).then(resolve),rpc:jest.fn((_name,args)=>{const result={abortSignal:()=>result,then:resolve=>Promise.resolve({data:[cheap,...expensive.slice(0,99)].map(row=>({...row,_budget_cost:String(row.price*args.p_rates.USD)})),error:null}).then(resolve)};return result;})};
 const {handler}=load(['wardrobe-shirt-0','shop_z-cheap','wardrobe-shoes-0'],null,null,null,q);
 const items=owned.filter(item=>item.macroCategory!=='bottom').flatMap(item=>Array.from({length:6},(_,i)=>({...item,id:`${item.id}-${i}`})));
 const body=await request({amount:20,currency:'USD'},items).json();body.wardrobeOnly=false;body.prompt='Create one complete casual outfit.';
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(response.status).toBe(200);expect(q.rpc).toHaveBeenCalledWith('budget_catalog_candidates',expect.objectContaining({p_slot:'bottom',p_limit:100}));
 expect((await response.json()).outfits[0].items.map(item=>item.id)).toContain('shop_z-cheap');
});

test('six provided shop choices cannot suppress discovery of an affordable catalog alternative',async()=>{
 const data=[{id:'cheap-alternative',name:'Regular fit cotton trousers',category:'bottoms',garment_type:'lower_body',price:10,currency:'USD',image_url:'https://example.invalid/pants.jpg'}];
 const q={select:()=>q,eq:()=>q,limit:()=>q,or:()=>q,order:()=>q,gt:()=>q,abortSignal:()=>q,then:resolve=>Promise.resolve({data,error:null}).then(resolve)};
 const {handler,catalogRpc}=load(['wardrobe-shirt-0','shop_cheap-alternative','wardrobe-shoes-0'],null,null,null,q);
 const items=owned.flatMap(item=>Array.from({length:6},(_,i)=>({...item,id:`${item.macroCategory==='bottom'?'shop_':''}${item.id}-${i}`,...item.macroCategory==='bottom'?{isShopItem:true,isWardrobe:false,price:80}:{} })));
 const body=await request({amount:20,currency:'USD'},items).json();body.wardrobeOnly=false;body.prompt='Create one complete casual outfit.';
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(response.status).toBe(200);expect(catalogRpc).toHaveBeenCalledWith('budget_catalog_candidates',expect.objectContaining({p_slot:'bottom'}));
 expect((await response.json()).outfits[0].items.map(item=>item.id)).toContain('shop_cheap-alternative');
});

test('a dress-loafer proposal missing a top enters correction before final validation',async()=>{
 const {handler,fetchMock}=load();
 const encode=items=>new Response(JSON.stringify({choices:[{message:{content:JSON.stringify({outfits:[{items}]})}}]}));
 fetchMock.mockImplementationOnce(async()=>encode(['wardrobe-pants','wardrobe-shoes','dress-penny']))
  .mockImplementationOnce(async()=>encode(['wardrobe-shirt','wardrobe-pants','wardrobe-shoes','wardrobe-layer']));
 const items=[...owned,{...owned[2],id:'dress-penny',name:'DRESS PENNY LOAFERS'}];
 const body=await request({amount:300,currency:'USD'},items).json();body.stylePreferences='semi_classic';body.occasion='formal';body.prompt='Create one complete formal outfit.';
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(response.status).toBe(200);expect(fetchMock).toHaveBeenCalledTimes(2);
 const retryMessages=JSON.parse(fetchMock.mock.calls[1][1].body).messages;
 expect(retryMessages[1].content).toContain('duplicate_slot');
 expect((await response.json()).outfits[0].items.map(item=>item.id).sort()).toEqual(owned.map(item=>item.id).sort());
});

test('repeated structurally invalid provider choices never return a malformed outfit',async()=>{
 const {handler,fetchMock}=load(['wardrobe-pants','wardrobe-shoes','dress-penny']);
 const items=[...owned,{...owned[2],id:'dress-penny',name:'DRESS PENNY LOAFERS'}];
 const body=await request({amount:300,currency:'USD'},items).json();body.stylePreferences='semi_classic';body.occasion='formal';body.prompt='Create one complete formal outfit.';
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(fetchMock).toHaveBeenCalledTimes(2);
 const data=await response.json();
 // The provider answer is unusable. The deterministic fallback takes over,
 // and everything it returns must be a VALID, authoritative wardrobe item —
 // never a malformed or invented outfit.
 expect(response.status).toBe(200);
 expect(data.source).toBe('deterministic_fallback');
 const allowed=new Set(items.map(item=>item.id));
 for(const outfit of data.outfits)for(const item of outfit.items)expect(allowed.has(item.id)).toBe(true);
 expect(data.outfits.length).toBeGreaterThan(0);
});

test('an unparseable provider response gets one bounded format correction',async()=>{
 const {handler,fetchMock}=load();
 fetchMock.mockImplementationOnce(async()=>new Response(JSON.stringify({choices:[{message:{content:'This is not outfit JSON.'}}]})));
 const response=await handler(request({amount:300,currency:'USD'}));
 expect(response.status).toBe(200);expect(fetchMock).toHaveBeenCalledTimes(2);
 const retry=JSON.parse(fetchMock.mock.calls[1][1].body).messages;
 expect(retry[1].content).toContain('FORMAT CORRECTION');
 expect((await response.json()).outfits[0].items.map(item=>item.id).sort()).toEqual(owned.map(item=>item.id).sort());
});
test('format and quality correction together cannot exceed two provider requests',async()=>{
 const {handler,fetchMock}=load(['wardrobe-shirt','wardrobe-shoes']);
 fetchMock.mockImplementationOnce(async()=>new Response(JSON.stringify({choices:[{message:{content:'Invalid output'}}]})));
 const response=await handler(request({amount:300,currency:'USD'}));
 expect(fetchMock).toHaveBeenCalledTimes(2);
 const data=await response.json();
 // The provider budget is two logical calls. When both fail, the
 // deterministic fallback still delivers a validated deck.
 expect(response.status).toBe(200);
 expect(data.source).toBe('deterministic_fallback');
 const allowed=new Set(owned.map(item=>item.id));
 for(const outfit of data.outfits)for(const item of outfit.items)expect(allowed.has(item.id)).toBe(true);
});
test('two unparseable responses fall back to a validated deck without inventing an outfit',async()=>{
 const {handler,fetchMock}=load();fetchMock.mockImplementation(async()=>new Response(JSON.stringify({choices:[{message:{content:'Invalid output'}}]})));
 const response=await handler(request({amount:300,currency:'USD'}));const data=await response.json();
 expect(fetchMock).toHaveBeenCalledTimes(2);
 expect(data.providerAttempts.map(attempt=>attempt.status)).toEqual(['unparseable','unparseable']);
 expect(response.status).toBe(200);
 expect(data.source).toBe('deterministic_fallback');
 const allowed=new Set(owned.map(item=>item.id));
 for(const outfit of data.outfits)for(const item of outfit.items)expect(allowed.has(item.id)).toBe(true);
});
test('rate-limited provider requests do not trigger a format correction',async()=>{
 const {handler,fetchMock}=load();fetchMock.mockImplementation(async()=>new Response('{}',{status:429}));
 const response=await handler(request({amount:300,currency:'USD'}));
 // One logical provider call with ONE bounded transient retry — never a
 // format correction (the provider answered, it was just rate limited).
 expect(fetchMock).toHaveBeenCalledTimes(2);
 for(const call of fetchMock.mock.calls)expect(JSON.parse(call[1].body).messages[1].content).not.toContain('FORMAT CORRECTION');
 const data=await response.json();
 expect(response.status).toBe(200);
 expect(data.source).toBe('deterministic_fallback');
});

test('primary and format-correction requests both describe the required climate layer',async()=>{
 const coat={...owned[0],id:'wardrobe-jacket',name:'Navy wool blazer',macroCategory:'outerwear',type:'blazer'};
 const {handler,fetchMock}=load([...owned.map(item=>item.id),coat.id]);
 fetchMock.mockImplementationOnce(async()=>new Response(JSON.stringify({choices:[{message:{content:'Unreadable output'}}]})));
 const body=await request(null,[...owned,coat]).json();body.stylePreferences='old_money';body.occasion='Work';body.prompt='Work';body.weather={temp:20};
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(response.status).toBe(200);expect(fetchMock).toHaveBeenCalledTimes(2);
 for(const call of fetchMock.mock.calls){const system=JSON.parse(call[1].body).messages[0].content;const example=JSON.parse(system.split('OUTPUT CONTRACT: Return ')[1].split('. Every item')[0]);expect(example.outfits[0].items.map(item=>item.id)).toContain('EXACT OUTERWEAR CANDIDATE ID');}
});

test('curated mode returns only AI-selected, fully validated whole looks',async()=>{
 const {handler,fetchMock}=load(undefined,null,null,null,null,'curated');
 fetchMock.mockImplementation(async(_url,init)=>{
  const body=JSON.parse(init.body),options=JSON.parse(body.messages[1].content);
  expect(body.max_tokens).toBe(700);expect(options.looks.length).toBeGreaterThan(0);
  return new Response(JSON.stringify({choices:[{message:{content:JSON.stringify({outfits:[{description:'A coordinated outfit.',items:[{id:options.looks[0].id}]}]})}}]}));
 });
 const response=await handler(request({amount:300,currency:'USD'}));expect(response.status).toBe(200);
 expect((await response.json()).outfits[0].items.map(i=>i.id).sort()).toEqual(owned.map(i=>i.id).sort());expect(fetchMock).toHaveBeenCalledTimes(1);
});
test.each(['unknown','partial','duplicate','timeout'])('curated mode rejects %s provider output and returns independently validated options',async mode=>{
 const {handler,fetchMock}=load(undefined,null,null,null,null,'curated');
 fetchMock.mockImplementation(async(_url,init)=>{
  if(mode==='timeout')throw new TypeError('network unavailable');
  const options=JSON.parse(JSON.parse(init.body).messages[1].content);
  const choice={items:[{id:options.looks[0].id}]};
  const outfits=mode==='unknown'?[{items:[{id:'invented-look'}]}]:mode==='partial'?[{items:[{id:owned[0].id}]}]:[choice,choice];
  return new Response(JSON.stringify({choices:[{message:{content:JSON.stringify({outfits})}}]}));
 });
 const response=await handler(request(null));const data=await response.json();expect(response.status).toBe(200);expect(data.source).toBe('deterministic_fallback');expect(data.outfits.length).toBeGreaterThan(0);expect(data.outfits[0].items.map(item=>item.id).sort()).toEqual(owned.map(item=>item.id).sort());
 // A network failure is retried once inside the provider call; the
 // correction budget is untouched.
 expect(fetchMock).toHaveBeenCalledTimes(mode==='timeout'?2:1);
});
test('curated mode never offers a combination over the selected shopping budget',async()=>{
 const {handler,fetchMock}=load(undefined,null,null,null,null,'curated');
 const shopping=owned.map(i=>({...i,id:`shop_${i.id}`,isWardrobe:false,isShopItem:true,price:50,currency:'USD'}));
 const response=await handler(request({amount:100,currency:'USD'},shopping));expect(response.status).toBe(422);expect(fetchMock).not.toHaveBeenCalled();
});

test('enrichment cannot return a Russian-named sweater in the base shirt slot',async()=>{
 const layer={...owned[0],id:'knitted-polo-layer',name:'Свитер Dc Knitted Zip Polo',type:'polo',macroCategory:'top'};
 const {handler,fetchMock}=load(undefined,null,null,null,null,'curated');
 fetchMock.mockImplementation(async(_url,init)=>{const options=JSON.parse(JSON.parse(init.body).messages[1].content);return new Response(JSON.stringify({choices:[{message:{content:JSON.stringify({outfits:[{items:[{id:options.looks[0].id}]}]})}}]}));});
 const blazer={...owned[0],id:'wool-blazer',name:'Navy wool blazer',type:'blazer',macroCategory:'outerwear'};
 const body=await request(null,[...owned,layer,blazer]).json();body.stylePreferences='old_money';body.occasion='Work';body.prompt='Work';body.weather={temp:20};
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(response.status).toBe(200);const data=await response.json();expect(data.outfits.every(outfit=>outfit.items.every(item=>item.id!==layer.id))).toBe(true);expect(data.outfits[0].items).toHaveLength(4);
});

test('capacity is assessed against the displayed three-card target even when AI returns two looks',async()=>{
 const fixtures=[
  ...['a','b','c'].map((key,i)=>({...owned[3],id:'layer-'+key,name:['Navy linen blazer','Beige cotton overshirt','Grey knit cardigan'][i]})),
  {...owned[0],id:'top-a',name:'White cotton shirt'},
  {...owned[0],id:'top-b',name:'Navy long sleeve oxford shirt'},
  {...owned[0],id:'top-c',name:'Fine gauge knit polo'},
  {...owned[1],id:'bottom-a',name:'Navy cotton trousers'},
  {...owned[1],id:'bottom-b',name:'Beige cotton chinos'},
  {...owned[2],id:'shoes-a',name:'Brown leather loafers'},
  {...owned[2],id:'shoes-b',name:'Tan penny leather loafers'},
 ].map(item=>({...item,imageUrl:`https://example.invalid/${item.id}.jpg`}));
 const {handler}=load([['top-a','bottom-a','shoes-a','layer-a'],['top-b','bottom-b','shoes-b','layer-b']]);
 const body=await request(null,fixtures).json();body.limit=5;body.deckTarget=3;
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(response.status).toBe(200);const data=await response.json();
 // Hosted code may complete the batch; capacity must still use the displayed target.
 expect(data.outfits.length).toBeGreaterThanOrEqual(2);expect(data.outfits.length).toBeLessThanOrEqual(5);
 expect(data.diversity.strictSlots).toContain('top');
 expect(data.diversity.forcedRepeatSlots.sort()).toEqual(['bottom','shoes']);
});

test('multi-card curation jointly selects a compatible deck before bounded AI ranking',async()=>{
 const names=[['White cotton shirt','Navy long sleeve oxford shirt','Fine gauge knit polo','Blue cotton shirt'],['Navy cotton trousers','Beige cotton chinos','Grey pleated trousers','Olive tailored chinos'],['Brown leather loafers','Tan penny leather loafers','Brown derby shoes','Black leather oxford shoes'],['Navy linen blazer','Beige linen blazer','Grey wool blazer','Olive cotton blazer']];
 const fixtures=owned.flatMap((base,slot)=>names[slot].map((name,index)=>({...base,id:`slot${slot}-${index}`,name,imageUrl:`https://example.invalid/slot${slot}-${index}.jpg`})));
 const {handler,fetchMock}=load(undefined,null,null,null,null,'curated');let options,system;
 fetchMock.mockImplementation(async(_url,init)=>{const body=JSON.parse(init.body);options=JSON.parse(body.messages[1].content);system=body.messages[0].content;return new Response(JSON.stringify({choices:[{message:{content:JSON.stringify({rankedLookIds:options.looks.map(look=>look.id)})}}]}));});
 const body=await request(null,fixtures).json();body.limit=5;body.deckTarget=3;
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(response.status).toBe(200);expect(options.looks.length).toBe(3);expect(options.requiredCount).toBe(options.looks.length);expect(system).toContain('Rank all');expect(system).toContain('without descriptions');
 const data=await response.json();expect(data.outfits.length).toBeGreaterThanOrEqual(3);expect(data.diversity.strictSlots.sort()).toEqual(['bottom','outerwear','shoes','top']);expect(data.outfits.every(outfit=>Number.isSafeInteger(outfit.curationRank))).toBe(true);
 // A partial ranking must never masquerade as the complete requested deck.
 fetchMock.mockImplementation(async(_url,init)=>{const sent=JSON.parse(JSON.parse(init.body).messages[1].content);return new Response(JSON.stringify({choices:[{message:{content:JSON.stringify({rankedLookIds:sent.looks.slice(0,2).map(look=>look.id)})}}]}));});
 const partial=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(partial.status).toBe(200);const recovered=await partial.json();expect(recovered.source).toBe('deterministic_fallback');expect(recovered.outfits).toHaveLength(3);expect(new Set(recovered.outfits.flatMap(outfit=>outfit.items.map(item=>item.id))).size).toBe(12);
});

test('an exchange-rate outage still permits a verified native-currency deck',async()=>{
 const {handler,loadRates}=load();loadRates.mockRejectedValue(new fxCore.BudgetFxError());
 const rows=owned.map(item=>({...item,id:'shop_'+item.id,isWardrobe:false,isShopItem:true,price:40,currency:'USD'}));
 rows.push({...rows[0],id:'shop_foreign',currency:'EUR',price:30});
 const body=await request({amount:250,currency:'USD'},rows).json();body.prompt='A complete outfit under $250.';
 const response=await handler(new Request('https://example.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},body:JSON.stringify(body)}));
 expect(response.status).toBe(200);const data=await response.json();expect(data.outfits.length).toBeGreaterThan(0);
 for(const outfit of data.outfits){expect(outfit.items.every(item=>item.currency==='USD')).toBe(true);expect(outfit.items.reduce((sum,item)=>sum+item.price,0)).toBeLessThanOrEqual(250);}
});
