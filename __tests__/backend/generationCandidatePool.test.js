/** @jest-environment node */
const fs=require('fs'),path=require('path'),ts=require('typescript');
const root=process.env.NVIDIA_REQUEST_SOURCE_ROOT||path.resolve(__dirname,'../../backend/supabase/functions');
const judge={...require(path.join(root,'../../../src/domain/occasionRules.ts')),...require(path.join(root,'../../../src/domain/outfitStructure.ts')),...require(path.join(root,'../../../src/domain/outfitDiversity.ts')),...require(path.join(root,'../../../src/domain/outfitRotation.ts'))};
Object.assign(judge,require('../../src/domain/garmentContext'),require('../../src/domain/stylingContext'));
judge.styleInference = require('../../src/domain/styleInference');
const strip=s=>s.replace(/^import .*\n/gm,'').replace(/\bexport\s+/g,'');
const source=['_shared/modelOutfitResponse.ts','generate-outfits/index.ts'].map(file=>strip(fs.readFileSync(path.join(root,file),'utf8'))).join('\n');
const compiled=ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.None,target:ts.ScriptTarget.ES2022}}).outputText;
const api=new Function('serve','createClient','Deno',...Object.keys(judge),compiled+'\nreturn {inferAttrsEdge,scoreItemForStyleEdge,rankItemsForStyleEdge,requiredShopSupplementSlots,fetchShopItemsForAllSlotsEdge,retainBudgetFloorCandidates: typeof retainBudgetFloorCandidates === "function" ? retainBudgetFloorCandidates : undefined};')(()=>{},()=>{},{env:{get:()=>undefined}},...Object.values(judge));
test('ranking floors cannot restore a garment rejected by the authoritative eligibility predicate',()=>{
 const items=[{id:'relaxed',name:'100% LINEN RELAXED FIT TROUSERS',macroCategory:'bottom'},
 {id:'regular',name:'100% WOOL REGULAR FIT SUIT TROUSERS',macroCategory:'bottom'},
 {id:'shirt',name:'Long sleeve cotton shirt',macroCategory:'top'},
 {id:'shoes',name:'Leather penny loafers',macroCategory:'shoes'}];
 const eligible=item=>!judge.rejectForStyleOccasion(item,'semi_classic','formal')&&!judge.rejectForClimate(item,{temp:20});
 expect(eligible(items[0])).toBe(false);expect(eligible(items[1])).toBe(true);
 const result=api.rankItemsForStyleEdge(items,'semi_classic',{minKeep:16,perCategoryFloor:4,eligibility:eligible});
 expect(result.map(i=>i.id)).not.toContain('relaxed');expect(result.map(i=>i.id)).toContain('regular');
 expect(result.every(eligible)).toBe(true);
});
test('six ineligible bottoms cannot make a supplemented wardrobe appear complete',()=>{
 const items=['top','shoes','bottom'].flatMap(slot=>Array.from({length:6},(_,i)=>({id:slot+i,macroCategory:slot,name:slot==='bottom'?'LINEN RELAXED FIT TROUSERS':slot==='top'?'Cotton shirt':'Black leather derby shoes'})));
 const eligible=item=>!judge.rejectForStyleOccasion(item,'semi_classic','formal');
 expect(api.requiredShopSupplementSlots(items,['top','bottom','shoes'],1,[],eligible).slots).toEqual(['bottom']);
});
test('a shop eligibility predicate applies before per-slot truncation',async()=>{
 const data=[{id:'relaxed',name:'LINEN RELAXED FIT TROUSERS',category:'bottoms',garment_type:'lower_body',image_url:'https://example.invalid/pants.jpg'}];
 const q={select:()=>q,eq:()=>q,limit:()=>q,or:()=>q,order:()=>q,gt:()=>q,abortSignal:()=>q,then:resolve=>Promise.resolve({data,error:null}).then(resolve)};
 const eligible=item=>!judge.rejectForStyleOccasion(item,'semi_classic','formal');
 expect(await api.fetchShopItemsForAllSlotsEdge({from:()=>q},['bottom'],'casual',1,undefined,eligible)).toEqual([]);
});

test('an eligible catalog garment after the first100rows remains discoverable',async()=>{
 const invalid=Array.from({length:100},(_,i)=>({id:`bad-${String(i).padStart(3,'0')}`,name:'LINEN RELAXED FIT TROUSERS',category:'bottoms',garment_type:'lower_body',image_url:'https://example.invalid/pants.jpg'}));
 const valid={id:'valid-last',name:'Regular fit tailored cotton trousers',category:'bottoms',garment_type:'lower_body',image_url:'https://example.invalid/valid.jpg'};
 let cursor;const cursors=[],orders=[];
 const q={select:()=>q,eq:()=>q,limit:()=>q,or:()=>q,abortSignal:()=>q,order:(...args)=>{orders.push(args);return q;},gt:(_field,value)=>{cursor=value;cursors.push(value);return q;},then:resolve=>Promise.resolve({data:cursor?[valid]:invalid,error:null}).then(resolve)};
 const eligible=item=>!judge.rejectForStyleOccasion(item,'semi_classic','formal');
 const result=await api.fetchShopItemsForAllSlotsEdge({from:()=>q},['bottom'],'casual',1,undefined,eligible);
 expect(result.map(i=>i.id)).toEqual(['shop_valid-last']);expect(cursors).toEqual(['bad-099']);expect(orders[0]).toEqual(['id',{ascending:true}]);
});
test('excluded first-page products do not suppress a fresh later-page option',async()=>{
 const first=Array.from({length:100},(_,i)=>({id:`seen-${String(i).padStart(3,'0')}`,name:'Cotton trousers',category:'bottoms',garment_type:'lower_body',image_url:'https://example.invalid/a.jpg'}));
 let cursor;const q={select:()=>q,eq:()=>q,limit:()=>q,or:()=>q,order:()=>q,abortSignal:()=>q,gt:(_field,value)=>{cursor=value;return q;},then:resolve=>Promise.resolve({data:cursor?[{...first[0],id:'unseen-final',name:'Linen chinos'}]:first,error:null}).then(resolve)};
 const result=await api.fetchShopItemsForAllSlotsEdge({from:()=>q},['bottom'],'casual',1,new Set(first.map(r=>r.id)));
 expect(result.map(i=>i.id)).toEqual(['shop_unseen-final']);
});
test('a failed later page cannot silently return a partial catalog search',async()=>{
 const first=Array.from({length:100},(_,i)=>({id:`a-${i}`,name:'Cotton trousers',category:'bottoms',garment_type:'lower_body',image_url:'https://example.invalid/a.jpg'}));
 let cursor;const q={select:()=>q,eq:()=>q,limit:()=>q,or:()=>q,order:()=>q,abortSignal:()=>q,gt:(_field,value)=>{cursor=value;return q;},then:resolve=>Promise.resolve(cursor?{data:null,error:{message:'unavailable'}}:{data:first,error:null}).then(resolve)};
 await expect(api.fetchShopItemsForAllSlotsEdge({from:()=>q},['bottom'],'casual',2)).rejects.toThrow('catalog_lookup_unavailable');
});
test('stalled catalog transport settles at the shared search deadline even when abort is ignored',async()=>{
 jest.useFakeTimers();try{
  let signal;const q={select:()=>q,eq:()=>q,limit:()=>q,or:()=>q,order:()=>q,abortSignal:s=>{signal=s;return q;},then:()=>new Promise(()=>{})};
  const result=api.fetchShopItemsForAllSlotsEdge({from:()=>q},['bottom'],'casual',1);
  const rejected=expect(result).rejects.toThrow('catalog_lookup_unavailable');await jest.advanceTimersByTimeAsync(15000);await rejected;
  expect(signal.aborted).toBe(true);expect(jest.getTimerCount()).toBe(0);
 }finally{jest.useRealTimers();}
});

const priced=(id,slot,price,name=id)=>({id,macroCategory:slot,price,name});
test('budget candidates retain the cheapest eligible choice in every slot beyond the rank cap',()=>{
 const expensive=Array.from({length:45},(_,i)=>priced('premium'+i,'top',100+i));
 const cheap=['top','bottom','shoes','outerwear','dress'].map(slot=>priced('cheap-'+slot,slot,1));
 const result=api.retainBudgetFloorCandidates([...expensive,...cheap],expensive,item=>item.price,()=>true);
 expect(result).toHaveLength(40);
 for(const item of cheap)expect(result.map(i=>i.id)).toContain(item.id);
});
test('a costly variant cannot hide its affordable product-family variant',()=>{
 const costly=priced('costly','shoes',100,'Leather penny loafers');
 const cheap=priced('cheap','shoes',10,'Leather penny loafers');
 const result=api.retainBudgetFloorCandidates([costly,cheap],[costly],item=>item.price,()=>true);
 expect(result.map(i=>i.id)).toEqual(['cheap']);
});
test('budget reserves cannot reintroduce ineligible or unpriced garments',()=>{
 const source=[priced('invalid','top',0),priced('unknown','bottom',Infinity),priced('valid','top',20)];
 const result=api.retainBudgetFloorCandidates(source,[],item=>item.price,item=>item.id!=='invalid');
 expect(result.map(i=>i.id)).toEqual(['valid']);
});
test('budget reserves preserve the locked anchor and treat owned garments as zero spend',()=>{
 const anchor=priced('anchor','top',80,'Anchor shirt');
 const owned=priced('owned','top',0,'Owned cotton tee');
 const result=api.retainBudgetFloorCandidates([anchor,owned],[anchor],item=>item.price,()=>true,anchor);
 expect(result.map(i=>i.id)).toEqual(['anchor','owned']);
});

const budgetContext={snapshot:{uzsPerUnit:{UZS:1,USD:12000,EUR:13000,GBP:15000}}};
test('budget discovery carries exact database cost text through a later-page cursor',async()=>{
 const cost='123456.123456789012345678901';
 const first=Array.from({length:100},(_,i)=>({id:`a${i}`,name:'Cotton trousers',category:'bottoms',garment_type:'lower_body',image_url:'https://example.invalid/pants.jpg',_budget_cost:cost,price:20,currency:'USD'}));
 const last={...first[0],id:'last',_budget_cost:'999999'};
 const calls=[];const client={rpc:(name,args)=>{calls.push({name,args});const q={abortSignal:()=>q,then:resolve=>Promise.resolve({data:args.p_after_id?[last]:first,error:null}).then(resolve)};return q;}};
 const result=await api.fetchShopItemsForAllSlotsEdge(client,['bottom'],'casual',1,undefined,item=>item.id==='last',budgetContext);
 expect(result.map(item=>item.id)).toEqual(['shop_last']);
 expect(calls[1]).toEqual({name:'budget_catalog_candidates',args:expect.objectContaining({p_after_cost:cost,p_after_id:'a99',p_rates:budgetContext.snapshot.uzsPerUnit})});
});
test('budget discovery preserves a cheap eligible choice over a higher style score',async()=>{
 const rows=[{id:'cheap',name:'Cotton trousers',brand:'',price:10,currency:'USD',_budget_cost:'120000'},
 {id:'costly',name:'Tailored wool suit trousers',brand:'Massimo Dutti',price:100,currency:'USD',_budget_cost:'1200000'}].map(row=>({...row,category:'bottoms',garment_type:'lower_body',image_url:'https://example.invalid/pants.jpg'}));
 const client={rpc:()=>{const q={abortSignal:()=>q,then:resolve=>Promise.resolve({data:rows,error:null}).then(resolve)};return q;}};
 const result=await api.fetchShopItemsForAllSlotsEdge(client,['bottom'],'semi_classic',1,undefined,()=>true,budgetContext);
 expect(result.map(item=>item.id)).toEqual(['shop_cheap']);
});
test('budget discovery rejects a malformed database cost instead of losing cursor precision',async()=>{
 const client={rpc:()=>{const q={abortSignal:()=>q,then:resolve=>Promise.resolve({data:[{id:'bad',_budget_cost:'NaN'}],error:null}).then(resolve)};return q;}};
 await expect(api.fetchShopItemsForAllSlotsEdge(client,['bottom'],'casual',1,undefined,()=>true,budgetContext)).rejects.toThrow('catalog_lookup_unavailable');
});

test.each(['43Mm Silicone Strap Sport Watch','SUIT CARRIER','SMALL TRAINING TOWEL','LARGE TRAINING TOWEL'])('a catalog %s cannot fill the top slot even if feed metadata says upper_body',async name=>{
 const data=[{id:'catalog-non-garment',name,category:'tops',garment_type:'upper_body',image_url:'https://example.invalid/item.jpg'}];
 const q={select:()=>q,eq:()=>q,limit:()=>q,or:()=>q,order:()=>q,gt:()=>q,abortSignal:()=>q,then:resolve=>Promise.resolve({data,error:null}).then(resolve)};
 expect(await api.fetchShopItemsForAllSlotsEdge({from:()=>q},['top'],'casual',1)).toEqual([]);
});

test.each(['METAL KEY RING WITH KEYS','METAL CHAIN KEY RING','TOTE BAG','Textured Mini Reporter Bag'])('catalog discovery excludes %s despite a top label',async name=>{
 const data=[{id:'accessory',name,category:'tops',garment_type:'upper_body',image_url:'https://example.invalid/item.jpg'}];
 const q={select:()=>q,eq:()=>q,limit:()=>q,or:()=>q,order:()=>q,gt:()=>q,abortSignal:()=>q,then:resolve=>Promise.resolve({data,error:null}).then(resolve)};
 expect(await api.fetchShopItemsForAllSlotsEdge({from:()=>q},['top'],'casual',1)).toEqual([]);
});

test('unbudgeted 40-item trim preserves four representatives of every eligible clothing slot',()=>{
 const source=[...Array.from({length:60},(_,i)=>({id:`top-${i}`,name:`Cotton shirt design${i}`,macroCategory:'top'})),...['bottom','shoes','outerwear'].flatMap(macroCategory=>Array.from({length:4},(_,i)=>({id:`${macroCategory}-${i}`,name:`${macroCategory} candidate design${i}`,macroCategory})))];
 const result=api.retainBudgetFloorCandidates(source,source.slice(0,40),()=>0,()=>true,undefined,4);
 expect(result.length).toBeLessThanOrEqual(40);
 for(const slot of ['top','bottom','shoes','outerwear'])expect(result.filter(item=>item.macroCategory===slot)).toHaveLength(slot==='top'?28:4);
});

test('catalog discovery evaluates later pages even when the first page fills the shortlist',async()=>{
 const first=Array.from({length:100},(_,i)=>({id:`early-${String(i).padStart(3,'0')}`,name:'Cotton trousers',category:'bottoms',garment_type:'lower_body',image_url:`https://example.invalid/${i}.jpg`}));
 const last={...first[0],id:'late-product',name:'Wool tailored trousers',image_url:'https://example.invalid/late.jpg'};
 let cursor;const visited=[];
 const q={select:()=>q,eq:()=>q,limit:()=>q,or:()=>q,order:()=>q,abortSignal:()=>q,gt:(_field,value)=>{cursor=value;return q;},then:resolve=>Promise.resolve({data:cursor?[last]:first,error:null}).then(resolve)};
 const result=await api.fetchShopItemsForAllSlotsEdge({from:()=>q},['bottom'],'casual',1,undefined,item=>{visited.push(item.id);return true;});
 expect(visited).toContain('late-product');
 expect(visited).toHaveLength(101);
 expect(result).toHaveLength(1);
});

test.each([
 {name:'Navy wool tailored blazer',macroCategory:'outerwear'},
 {name:'Brown driving moccasins',macroCategory:'shoes'},
 {name:'Graphic oversized hoodie',macroCategory:'top'},
 {name:'Linen relaxed-fit trousers',macroCategory:'bottom'},
])('Edge style adapters preserve the canonical domain result for $name', item=>{
 const domain=judge.styleInference;
 const attrs=domain.inferItemAttributes(item);
 expect(api.inferAttrsEdge(item)).toEqual({styleTags:attrs.styleTags,materials:attrs.materials.join(', '),patterns:attrs.patterns.join(', '),formality:attrs.formality});
 expect(api.scoreItemForStyleEdge(item,'old_money')).toBe(domain.rawStyleScore(item,'old_money'));
});

test('activated discovery reaches high IDs with bounded pages and preserves hard eligibility',async()=>{
 const rows=Array.from({length:100},(_,i)=>({id:`interior-${5000+i}`,name:`Model ${i} cotton trousers`,category:'bottoms',garment_type:'lower_body',image_url:`https://image.invalid/${i}.jpg`,source:'product-only',currency:'EUR',price:50}));
 const rpc=jest.fn(()=>({abortSignal:async()=>({data:rows,error:null})}));
 const from=jest.fn(()=>{throw new Error('must not full-scan when the ranked pool is sufficient');});
 const result=await api.fetchShopItemsForAllSlotsEdge({rpc,from},['bottom'],'casual',4,undefined,item=>item.name!=='Model 0 cotton trousers',undefined,undefined,'',0,undefined,undefined,'replay-seed');
 expect(result).toHaveLength(4);expect(result.every(item=>item.id.startsWith('shop_interior-'))).toBe(true);
 expect(result.some(item=>item.name==='Model 0 cotton trousers')).toBe(false);
 expect(rpc).toHaveBeenCalledTimes(3);expect(from).not.toHaveBeenCalled();
 expect(rpc).toHaveBeenCalledWith('catalog_discovery_candidates',expect.objectContaining({p_seed:'replay-seed',p_limit:100,p_offset:200}));
});
test('RPC missing is a degraded full path, but permissions failures never bypass the RPC',async()=>{
 const row={id:'late',name:'Cotton trousers',category:'bottoms',garment_type:'lower_body',image_url:'https://image.invalid/late.jpg'};
 const q={select:()=>q,eq:()=>q,limit:()=>q,or:()=>q,order:()=>q,abortSignal:()=>q,then:resolve=>Promise.resolve({data:[row],error:null}).then(resolve)};
 const rpc=jest.fn(()=>({abortSignal:async()=>({data:null,error:{code:'PGRST202'}})}));
 expect(await api.fetchShopItemsForAllSlotsEdge({rpc,from:()=>q},['bottom'],'casual',1)).toHaveLength(1);
 const from=jest.fn();rpc.mockImplementation(()=>({abortSignal:async()=>({data:null,error:{code:'42501'}})}));
 await expect(api.fetchShopItemsForAllSlotsEdge({rpc,from},['bottom'],'casual',1)).rejects.toThrow('catalog_lookup_unavailable');
 expect(from).not.toHaveBeenCalled();
});
test('a known missing RPC is cached for its project, without retrying on the next request',async()=>{
 const row={id:'cached-late',name:'Cotton trousers',category:'bottoms',garment_type:'lower_body',image_url:'https://image.invalid/cached.jpg'};
 const q={select:()=>q,eq:()=>q,limit:()=>q,or:()=>q,order:()=>q,abortSignal:()=>q,then:resolve=>Promise.resolve({data:[row],error:null}).then(resolve)};
 const rpc=jest.fn(()=>({abortSignal:async()=>({data:null,error:{code:'PGRST202'}})}));
 for(let i=0;i<2;i++)expect(await api.fetchShopItemsForAllSlotsEdge({supabaseUrl:'https://capability-fixture.invalid',rpc,from:()=>q},['bottom'],'casual',1)).toHaveLength(1);
 expect(rpc).toHaveBeenCalledTimes(1);
});

test('the 40-item pool reserves ranked practical companions ahead of catalog arrival order',()=>{
 const early=Array.from({length:45},(_,i)=>priced('early-'+i,'outerwear',100,'Wool formal blazer '+i));
 const practical=['top','outerwear','bottom','shoes'].flatMap(slot=>Array.from({length:6},(_,i)=>priced('practical-'+slot+'-'+i,slot,100,slot+' practical '+i)));
 const result=api.retainBudgetFloorCandidates([...early,...practical],[...practical,...early],()=>0,()=>true,undefined,6);
 expect(result.length).toBeLessThanOrEqual(40);
 for(const slot of ['top','outerwear','bottom','shoes'])expect(result.filter(item=>item.macroCategory===slot).length).toBeGreaterThanOrEqual(6);
 for(const item of practical.slice(-6))expect(result.map(row=>row.id)).toContain(item.id);
});
test('catalog retrieval favors walking evidence over dress footwear for Classic Travel within the same cap',async()=>{
 const data=[{id:'dress',name:'Black leather Oxford shoes',category:'shoes',garment_type:'shoes',image_url:'https://example.invalid/dress.jpg'},
 {id:'walking',name:'Flexible cushioned leather loafers',category:'shoes',garment_type:'shoes',image_url:'https://example.invalid/walking.jpg'}];
 const q={};for(const method of ['select','eq','limit','or','order','gt','abortSignal'])q[method]=()=>q;
 q.then=resolve=>Promise.resolve({data,error:null}).then(resolve);
 const rows=await api.fetchShopItemsForAllSlotsEdge({from:()=>q},['shoes'],'classic',1,undefined,()=>true,undefined,undefined,'Travel');
 expect(rows.map(row=>row.id)).toEqual(['shop_walking']);
});
