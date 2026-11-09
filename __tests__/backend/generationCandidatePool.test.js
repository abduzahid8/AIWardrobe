/** @jest-environment node */
const fs=require('fs'),path=require('path'),ts=require('typescript');
const root=process.env.NVIDIA_REQUEST_SOURCE_ROOT||path.resolve(__dirname,'../../backend/supabase/functions');
const judge={...require(path.join(root,'../../../src/domain/occasionRules.ts')),...require(path.join(root,'../../../src/domain/outfitStructure.ts')),...require(path.join(root,'../../../src/domain/outfitDiversity.ts')),...require(path.join(root,'../../../src/domain/outfitRotation.ts'))};
Object.assign(judge,require('../../src/domain/garmentContext'),require('../../src/domain/stylingContext'));
// Candidate retrieval now reads the shared editorial, concept, visual and
// catalog-admission authorities; the stripped-import harness must inject them.
Object.assign(judge,
  require('../../src/domain/editorialIntelligence'),
  require('../../src/domain/catalogIdentity'),
  require('../../src/domain/visualFashion'),
  require('../../src/domain/garmentDescriptor'),
  require('../../src/domain/outfitConcepts'),
  require('../../src/domain/catalogCandidates'));
judge.styleInference = require('../../src/domain/styleInference');
const strip=s=>s.replace(/^import .*\n/gm,'').replace(/\bexport\s+/g,'');
const source=['_shared/modelOutfitResponse.ts','generate-outfits/index.ts'].map(file=>strip(fs.readFileSync(path.join(root,file),'utf8'))).join('\n');
const compiled=ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.None,target:ts.ScriptTarget.ES2022}}).outputText;
const api=new Function('serve','createClient','Deno',...Object.keys(judge),compiled+'\nreturn {inferAttrsEdge,scoreItemForStyleEdge,rankItemsForStyleEdge,requiredShopSupplementSlots,fetchShopItemsForAllSlotsEdge,retainBudgetFloorCandidates: typeof retainBudgetFloorCandidates === "function" ? retainBudgetFloorCandidates : undefined};')(()=>{},()=>{},{env:{get:()=>undefined}},...Object.values(judge));
// RPC-first retrieval contract: rows are served by
// catalog_stylist_candidates_compact (legacy fallback catalog_stylist_candidates)
// and hydration stays empty unless a test provides it.
const thenable=value=>{const q={abortSignal:()=>q,then:resolve=>Promise.resolve(value).then(resolve)};return q;};
const rowsFor=rows=>typeof rows==='function'?rows:()=>rows;
const rpcFor=(rows,{legacy,hydration=[]}={})=>({rpc:(name,args)=>{
 if(name==='catalog_stylist_candidate_hydration')return thenable({data:hydration,error:null});
 if(name==='catalog_stylist_candidates')return thenable({data:(legacy!==undefined?legacy:rowsFor(rows)(args)),error:null});
 return thenable({data:rowsFor(rows)(args),error:null});
}});
const slotRow=(slot,category,garment_type,name,extra={})=>({id:`catalog-${slot}`,name,category,garment_type,fashion_slot:slot,image_url:`https://example.invalid/${slot}.jpg`,...extra});
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
 const row=slotRow('bottom','bottoms','lower_body','LINEN RELAXED FIT TROUSERS');
 const eligible=item=>!judge.rejectForStyleOccasion(item,'semi_classic','formal');
 expect(await api.fetchShopItemsForAllSlotsEdge(rpcFor([row]),['bottom'],'casual',1,undefined,eligible)).toEqual([]);
});

test.each([
 ['top','tops','upper_body','White long sleeve cotton shirt'],
 ['bottom','bottoms','lower_body','Tailored wool trousers'],
 ['outerwear','outerwear','outerwear','Navy wool blazer'],
 ['shoes','shoes','shoes','Black leather Oxford shoes'],
])('catalog-only %s eligibility receives the shop identity before filtering raw rows', async (slot,category,garment_type,name) => {
 const row=slotRow(slot,category,garment_type,name);
 const context=judge.stylingContext({style:'classic',occasion:'work',sourceMode:'catalog'});
 const eligible=jest.fn(item=>!judge.garmentContextReason(item,context)&&judge.isFourSlotCandidate(item)&&!judge.rejectForStyleOccasion(item,'classic','work'));
 expect(judge.garmentContextReason({...row,macroCategory:slot},context)).toBe('source_requires_catalog');
 const result=await api.fetchShopItemsForAllSlotsEdge(rpcFor([row]),[slot],'classic',1,undefined,eligible,undefined,undefined,'work');
 expect(result.map(item=>item.id)).toEqual([`shop_catalog-${slot}`]);
 expect(eligible).toHaveBeenCalledWith(expect.objectContaining({id:`shop_catalog-${slot}`,isShopItem:true,isWardrobe:false,macroCategory:slot}),expect.any(Array));
 expect(judge.garmentContextReason(result[0],judge.stylingContext({sourceMode:'wardrobe'}))).toBe('source_requires_wardrobe');
});

test('an eligible catalog garment after the first100rows remains discoverable',async()=>{
 const invalid=Array.from({length:100},(_,i)=>({id:`bad-${String(i).padStart(3,'0')}`,name:'LINEN RELAXED FIT TROUSERS',category:'bottoms',garment_type:'lower_body',fashion_slot:'bottom',image_url:'https://example.invalid/pants.jpg'}));
 const valid={id:'valid-last',name:'Regular fit tailored cotton trousers',category:'bottoms',garment_type:'lower_body',fashion_slot:'bottom',image_url:'https://example.invalid/valid.jpg'};
 const offsets=[];
 const client=rpcFor(args=>{offsets.push(args.p_offset);return args.p_offset?[valid]:invalid;});
 const eligible=item=>!judge.rejectForStyleOccasion(item,'semi_classic','formal');
 const result=await api.fetchShopItemsForAllSlotsEdge(client,['bottom'],'casual',1,undefined,eligible);
 expect(result.map(i=>i.id)).toEqual(['shop_valid-last']);
 expect(offsets).toEqual([0,100]);
});
test('excluded first-page products do not suppress a fresh later-page option',async()=>{
 const first=Array.from({length:100},(_,i)=>({id:`seen-${String(i).padStart(3,'0')}`,name:'Cotton trousers',category:'bottoms',garment_type:'lower_body',fashion_slot:'bottom',image_url:'https://example.invalid/a.jpg'}));
 const client=rpcFor(args=>args.p_offset?[{...first[0],id:'unseen-final',name:'Linen chinos'}]:first);
 const result=await api.fetchShopItemsForAllSlotsEdge(client,['bottom'],'casual',1,new Set(first.map(r=>r.id)));
 expect(result.map(i=>i.id)).toEqual(['shop_unseen-final']);
});
test('a failed later page cannot silently return a partial catalog search',async()=>{
 const first=Array.from({length:100},(_,i)=>({id:`a-${i}`,name:'Cotton trousers',category:'bottoms',garment_type:'lower_body',fashion_slot:'bottom',image_url:'https://example.invalid/a.jpg'}));
 const client={rpc:(name,args)=>{if(name==='catalog_stylist_candidate_hydration')return thenable({data:[],error:null});
  return thenable(args.p_offset?{data:null,error:{message:'unavailable'}}:{data:first,error:null});}};
 await expect(api.fetchShopItemsForAllSlotsEdge(client,['bottom'],'casual',2)).rejects.toThrow('catalog_lookup_unavailable');
});
test('stalled catalog transport settles at the shared search deadline even when abort is ignored',async()=>{
 jest.useFakeTimers();try{
  let signal;const client={rpc:()=>{const q={abortSignal:s=>{signal=s;return q;},then:()=>new Promise(()=>{})};return q;}};
  const result=api.fetchShopItemsForAllSlotsEdge(client,['bottom'],'casual',1);
  const rejected=expect(result).rejects.toThrow('catalog_lookup_unavailable');await jest.advanceTimersByTimeAsync(15000);await rejected;
  expect(signal.aborted).toBe(true);expect(jest.getTimerCount()).toBe(0);
 }finally{jest.useRealTimers();}
});

const priced=(id,slot,price,name=id)=>({id,macroCategory:slot,price,name});
test('budget candidates retain the cheapest eligible choice in every slot beyond the rank cap',()=>{
 const expensive=Array.from({length:45},(_,i)=>priced('premium'+i,'top',100+i));
 const cheap=['top','bottom','shoes','outerwear','dress'].map(slot=>priced('cheap-'+slot,slot,1));
 const result=api.retainBudgetFloorCandidates([...expensive,...cheap],expensive,item=>item.price,()=>true,undefined,10);
 // Every slot keeps its cheapest option; the top slot is bounded by the
 // per-slot reserve instead of letting 45 variants flood the shortlist.
 for(const item of cheap)expect(result.map(i=>i.id)).toContain(item.id);
 expect(result.filter(i=>i.macroCategory==='top')).toHaveLength(10);
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
 const result=api.retainBudgetFloorCandidates([anchor,owned],[anchor],item=>item.price,()=>true,anchor,2);
 expect(result.map(i=>i.id)).toEqual(['anchor','owned']);
});

test('ten-per-slot reserves keep forty candidates balanced without enlarging the provider prompt',()=>{
 const source=['top','outerwear','bottom','shoes'].flatMap(slot=>Array.from({length:12},(_,i)=>priced(`${slot}-${i}`,slot,10+i)));
 const result=api.retainBudgetFloorCandidates(source,source,item=>item.price,()=>true,undefined,10);
 expect(result).toHaveLength(40);
 for(const slot of ['top','outerwear','bottom','shoes'])expect(result.filter(item=>item.macroCategory===slot)).toHaveLength(10);
});

const budgetContext={snapshot:{uzsPerUnit:{UZS:1,USD:12000,EUR:13000,GBP:15000}}};
test('budget discovery carries the exact cap and reference rates into compact retrieval',async()=>{
 const rows=Array.from({length:100},(_,i)=>({id:`a${i}`,name:'Cotton trousers',category:'bottoms',garment_type:'lower_body',fashion_slot:'bottom',price:20,currency:'USD',image_url:'https://example.invalid/pants.jpg'}));
 const calls=[];const client={rpc:(name,args)=>{calls.push({name,args});return thenable(name==='catalog_stylist_candidate_hydration'?{data:[],error:null}:{data:args.p_offset?[]:rows,error:null});}};
 const result=await api.fetchShopItemsForAllSlotsEdge(client,['bottom'],'casual',1,undefined,item=>item.id==='shop_a0',budgetContext);
 expect(result.map(item=>item.id)).toEqual(['shop_a0']);
 const compact=calls.find(call=>call.name==='catalog_stylist_candidates_compact');
 expect(compact.args).toEqual(expect.objectContaining({p_currency:null,p_max_price:null,p_context:expect.objectContaining({rates:budgetContext.snapshot.uzsPerUnit})}));
});
test('budget discovery preserves a cheap eligible choice over a higher style score',async()=>{
 const rows=[{id:'cheap',name:'Cotton trousers',brand:'',price:10,currency:'USD'},
 {id:'costly',name:'Tailored wool suit trousers',brand:'Massimo Dutti',price:100,currency:'USD'}].map(row=>({...row,category:'bottoms',garment_type:'lower_body',fashion_slot:'bottom',image_url:'https://example.invalid/pants.jpg'}));
 const result=await api.fetchShopItemsForAllSlotsEdge(rpcFor(rows),['bottom'],'semi_classic',1,undefined,()=>true,budgetContext,{amount:50,currency:'USD'});
 // The cheapest eligible bottom is the mandatory budget floor and is picked
 // before any higher-scoring premium variant.
 expect(result[0].id).toBe('shop_cheap');
});
test('a malformed catalog payload is rejected instead of being treated as an empty result',async()=>{
 const client={rpc:(name,args)=>thenable(name==='catalog_stylist_candidate_hydration'?{data:[],error:null}:{data:{rows:'not-an-array'},error:null})};
 await expect(api.fetchShopItemsForAllSlotsEdge(client,['bottom'],'casual',1,undefined,()=>true,budgetContext)).rejects.toThrow('catalog_lookup_unavailable');
});

test.each(['43Mm Silicone Strap Sport Watch','SUIT CARRIER','SMALL TRAINING TOWEL','LARGE TRAINING TOWEL'])('a catalog %s cannot fill the top slot even if feed metadata says upper_body',async name=>{
 const row=slotRow('top','tops','upper_body',name);
 expect(await api.fetchShopItemsForAllSlotsEdge(rpcFor([row]),['top'],'casual',1)).toEqual([]);
});

test.each(['METAL KEY RING WITH KEYS','METAL CHAIN KEY RING','TOTE BAG','Textured Mini Reporter Bag'])('catalog discovery excludes %s despite a top label',async name=>{
 const row=slotRow('top','tops','upper_body',name);
 expect(await api.fetchShopItemsForAllSlotsEdge(rpcFor([row]),['top'],'casual',1)).toEqual([]);
});

test('unbudgeted trim preserves four representatives of every eligible clothing slot',()=>{
 const source=[...Array.from({length:60},(_,i)=>({id:`top-${i}`,name:`Cotton shirt design${i}`,macroCategory:'top'})),...['bottom','shoes','outerwear'].flatMap(macroCategory=>Array.from({length:4},(_,i)=>({id:`${macroCategory}-${i}`,name:`${macroCategory} candidate design${i}`,macroCategory})))];
 const result=api.retainBudgetFloorCandidates(source,source.slice(0,40),()=>0,()=>true,undefined,4);
 // Per-slot reserves: the top slot no longer consumes the whole shortlist.
 for(const slot of ['top','bottom','shoes','outerwear'])expect(result.filter(item=>item.macroCategory===slot)).toHaveLength(4);
});

test('catalog discovery evaluates later pages even when the first page fills the shortlist',async()=>{
 const first=Array.from({length:100},(_,i)=>({id:`early-${String(i).padStart(3,'0')}`,name:'Cotton trousers',category:'bottoms',garment_type:'lower_body',fashion_slot:'bottom',image_url:`https://example.invalid/${i}.jpg`}));
 const last={...first[0],id:'late-product',name:'Wool tailored trousers',image_url:'https://example.invalid/late.jpg'};
 const visited=[];
 const client=rpcFor(args=>args.p_offset?[last]:first);
 const result=await api.fetchShopItemsForAllSlotsEdge(client,['bottom'],'casual',1,undefined,item=>{visited.push(item.id);return true;});
 expect(visited).toContain('shop_late-product');
 // The predicate is evaluated at several stages; unique rows must still be 101.
 expect(new Set(visited).size).toBe(101);
 expect(result.map(item=>item.id)).toContain('shop_late-product');
 expect(result.length).toBeLessThanOrEqual(24);
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
 const rows=Array.from({length:100},(_,i)=>({id:`interior-${5000+i}`,name:`Model ${i} cotton trousers`,category:'bottoms',garment_type:'lower_body',fashion_slot:'bottom',image_url:`https://image.invalid/${i}.jpg`,source:'product-only',currency:'EUR',price:50}));
 const offsets=[];const rpc=jest.fn((name,args)=>{if(name==='catalog_stylist_candidate_hydration')return thenable({data:[],error:null});offsets.push(args.p_offset);return thenable({data:args.p_offset?[]:rows,error:null});});
 const from=jest.fn(()=>{throw new Error('must not full-scan when the ranked pool is sufficient');});
 const result=await api.fetchShopItemsForAllSlotsEdge({rpc,from},['bottom'],'casual',4,undefined,item=>item.name!=='Model 0 cotton trousers',undefined,undefined,'',0,undefined,undefined,'replay-seed');
 expect(result.length).toBeGreaterThan(0);expect(result.every(item=>item.id.startsWith('shop_interior-'))).toBe(true);
 expect(result.some(item=>item.name==='Model 0 cotton trousers')).toBe(false);
 // Page offsets stay bounded to the two-page retrieval window.
 expect(offsets.every(offset=>offset===0||offset===100)).toBe(true);
 expect(from).not.toHaveBeenCalled();
 expect(rpc).toHaveBeenCalledWith('catalog_stylist_candidates_compact',expect.objectContaining({p_seed:'replay-seed',p_limit:100}));
});
test('a full mixed-slot page cannot hide eligible products on the next page',async()=>{
 const slots=['top','outerwear','bottom','shoes'];
 const row=(slot,i,label)=>({id:`${label}-${slot}-${i}`,name:`${label} ${slot} cotton ${i}`,
  category:slot,garment_type:slot==='shoes'?'shoes':slot,fashion_slot:slot,
  image_url:`https://image.invalid/${label}-${slot}-${i}.jpg`,source_url:`https://shop.invalid/${label}-${slot}-${i}`,
  price:100,currency:'USD'});
 const first=slots.flatMap(slot=>Array.from({length:25},(_,i)=>row(slot,i,'blocked')));
 const second=slots.flatMap(slot=>Array.from({length:2},(_,i)=>row(slot,i,'eligible')));
 const offsets=[];
 const rpc=jest.fn((name,args)=>{
  if(name==='catalog_stylist_candidate_hydration')return thenable({data:[],error:null});
  offsets.push(args.p_offset);
  return thenable({data:args.p_offset===0?first:args.p_offset===100?second:[],error:null});
 });
 const result=await api.fetchShopItemsForAllSlotsEdge({rpc},slots,'classic',4,undefined,
  item=>item.name.startsWith('eligible'));
 expect(offsets).toEqual([0,100]);
 expect(new Set(result.map(item=>item.macroCategory))).toEqual(new Set(slots));
 expect(result.every(item=>item.name.startsWith('eligible'))).toBe(true);
});
test('a missing compact RPC degrades to the legacy RPC, but permissions failures never bypass it',async()=>{
 const row={id:'late',name:'Cotton trousers',category:'bottoms',garment_type:'lower_body',fashion_slot:'bottom',image_url:'https://image.invalid/late.jpg'};
 const rpc=jest.fn((name)=>thenable(name==='catalog_stylist_candidates'?{data:[row],error:null}:{data:null,error:{code:'PGRST202'}}));
 const from=jest.fn();
 expect(await api.fetchShopItemsForAllSlotsEdge({rpc,from},['bottom'],'casual',1)).toHaveLength(1);
 expect(from).not.toHaveBeenCalled();
 rpc.mockImplementation(()=>thenable({data:null,error:{code:'42501'}}));
 await expect(api.fetchShopItemsForAllSlotsEdge({rpc,from},['bottom'],'casual',1)).rejects.toThrow('catalog_lookup_unavailable');
 expect(from).not.toHaveBeenCalled();
});
test('a missing compact RPC retries the legacy path on the next request instead of caching failure',async()=>{
 const row={id:'cached-late',name:'Cotton trousers',category:'bottoms',garment_type:'lower_body',fashion_slot:'bottom',image_url:'https://image.invalid/cached.jpg'};
 const names=[];const rpc=jest.fn((name)=>{names.push(name);return thenable(name==='catalog_stylist_candidates'?{data:[row],error:null}:{data:null,error:{code:'PGRST202'}});});
 const from=jest.fn();
 for(let i=0;i<2;i++)expect(await api.fetchShopItemsForAllSlotsEdge({supabaseUrl:'https://capability-fixture.invalid',rpc,from},['bottom'],'casual',1)).toHaveLength(1);
 expect(rpc.mock.calls.filter(call=>call[0]==='catalog_stylist_candidates_compact')).toHaveLength(2);
 expect(rpc.mock.calls.filter(call=>call[0]==='catalog_stylist_candidates')).toHaveLength(2);
 expect(from).not.toHaveBeenCalled();
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
 const rows=[{id:'dress',name:'Black leather Oxford shoes',category:'shoes',garment_type:'shoes',fashion_slot:'shoes',image_url:'https://example.invalid/dress.jpg'},
 {id:'walking',name:'Flexible cushioned leather loafers',category:'shoes',garment_type:'shoes',fashion_slot:'shoes',image_url:'https://example.invalid/walking.jpg'}];
 const result=await api.fetchShopItemsForAllSlotsEdge(rpcFor(rows),['shoes'],'classic',1,undefined,()=>true,undefined,undefined,'Travel');
 // Retrieval keeps a role-diverse representative set; the authoritative
 // Classic Travel preference is the shared context score, which must favor
 // walking evidence over dress oxfords before the planner selects.
 const context=judge.stylingContext({style:'classic',occasion:'Travel'});
 expect(judge.scoreGarmentForContext(rows[1],context)).toBeGreaterThan(judge.scoreGarmentForContext(rows[0],context));
 expect(result.map(row=>row.id)).toContain('shop_walking');
});
test('quality memo returns the cached verdict for identical candidate sets',()=>{
 const items=[{id:'memo-shirt',name:'White cotton dress shirt',macroCategory:'top',color:'white'},{id:'memo-blazer',name:'Navy wool blazer',macroCategory:'outerwear',color:'navy'},{id:'memo-trouser',name:'Charcoal wool trousers',macroCategory:'bottom',color:'charcoal'},{id:'memo-shoe',name:'Black leather Oxford shoes',macroCategory:'shoes',color:'black'}];
 const memo=judge.createQualityMemo();
 const first=memo(items,'old_money','Formal',{temp:18},undefined,{locale:'en'});
 const second=memo(items,'old_money','Formal',{temp:18},undefined,{locale:'en'});
 expect(second).toBe(first);
 const direct=judge.assessOutfitQuality(items,'old_money','Formal',{temp:18},undefined,{locale:'en'});
 expect(first).toEqual(direct);
 const otherStyle=memo(items,'casual','Casual',{temp:18},undefined,{locale:'en'});
 expect(otherStyle).not.toBe(first);
});
