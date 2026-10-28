/** @jest-environment node */
const fs=require('fs'),ts=require('typescript');const core={};
const compile=s=>ts.transpileModule(s,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
Function('exports',compile(fs.readFileSync('backend/supabase/functions/_shared/budgetFx.ts','utf8')))(core);
const now=Date.parse('2026-10-20T12:00:00Z');
const rows=()=>[['USD','12044.72'],['EUR','13525.02'],['GBP','16062.84']].map(([Ccy,Rate])=>({Ccy,Rate,Nominal:'1',Date:'20.10.2026'}));
const snapshot=()=>core.parseCbuBudgetRates(rows(),now);
test('official quote units are normalized per one unit, including Nominal > 1',()=>{
 const data=rows();data[1].Rate='1352502';data[1].Nominal='100';expect(core.parseCbuBudgetRates(data,now).uzsPerUnit.EUR).toBe(13525.02);
});
test('reference conversion goes through UZS and does not reverse EUR/USD',()=>{
 expect(core.referenceAmount(100,'EUR','USD',snapshot(),now)).toBeCloseTo(112.290032479,8);
 expect(core.referenceAmount(100,'USD','UZS',snapshot(),now)).toBe(1204472);
 expect(core.referenceAmount(100,'USD','EUR',snapshot(),now)).toBeCloseTo(89.055099364,8);
});
test('one immutable snapshot pins all request calculations',()=>{
 const data=snapshot();expect(Object.isFrozen(data)).toBe(true);expect(Object.isFrozen(data.uzsPerUnit)).toBe(true);
 expect(core.referenceAmount(100,'USD','USD',data,now)).toBe(100);
});
test.each(['0','-1','NaN','Infinity','1e3',''])('bad rate %s is rejected',Rate=>{const data=rows();data[0].Rate=Rate;expect(()=>core.parseCbuBudgetRates(data,now)).toThrow('budget_rates_unavailable');});
test.each(['0','-1','NaN',''])('bad nominal %s is rejected',Nominal=>{const data=rows();data[0].Nominal=Nominal;expect(()=>core.parseCbuBudgetRates(data,now)).toThrow();});
test('missing and duplicate required currencies are rejected',()=>{
 expect(()=>core.parseCbuBudgetRates(rows().slice(0,2),now)).toThrow();expect(()=>core.parseCbuBudgetRates([...rows(),rows()[0]],now)).toThrow();
});
test.each(['21.10.2026','01.10.2026','30.02.2026','not-a-date'])('future/stale/invalid reference date %s is rejected',Date=>expect(()=>core.parseCbuBudgetRates(rows().map(row=>({...row,Date})),now)).toThrow());
test('mixed publication dates are rejected',()=>{const data=rows();data[0].Date='19.10.2026';expect(()=>core.parseCbuBudgetRates(data,now)).toThrow();});
test('weekend/holiday reference dates are retained and exposed, never relabelled today',()=>{
 const data=rows().map(row=>({...row,Date:'16.10.2026'}));expect(core.parseCbuBudgetRates(data,now).asOf).toBe('2026-10-16');
});
test('expired snapshots and fabricated identities cannot be reused',()=>{
 expect(()=>core.validateBudgetFxSnapshot(snapshot(),now+core.BUDGET_FX_TTL_MS+1)).toThrow();expect(()=>core.validateBudgetFxSnapshot({...snapshot(),id:'made-up'},now)).toThrow();
});
test('unknown units and invalid amounts cannot pass a cap',()=>{
 expect(core.referenceAmount(100,'ZZZ','USD',snapshot(),now)).toBe(Infinity);expect(core.referenceAmount(Infinity,'USD','EUR',snapshot(),now)).toBe(Infinity);
});
function cacheSetup(cached=null){
 const api={};Function('require','exports',compile(fs.readFileSync('backend/supabase/functions/_shared/budgetFxCache.ts','utf8')))(()=>core,api);
 const read=jest.fn().mockResolvedValue({data:cached?{snapshot:cached}:null,error:null});
 const saved=jest.fn().mockResolvedValue({error:null});const upsert=jest.fn(row=>{cached=row.snapshot;return{abortSignal:()=>saved()}});
 let owner=null,cooldown=0;
 const rpc=jest.fn((name,args)=>({abortSignal:async()=>{if(name==='claim_budget_rate_refresh'){if(cooldown>Date.now())return{data:'cooldown'};if(owner)return{data:'busy'};owner=args.p_token;return{data:'acquired'};}if(name==='finish_budget_rate_refresh'){if(owner!==args.p_token)return{data:false};if(args.p_snapshot){await upsert({snapshot:args.p_snapshot}).abortSignal()}else cooldown=Date.now()+10000;owner=null;return{data:true}}throw Error('Unknown test RPC')}}));
 const query={select(){return this;},eq(){return this;},order(){return this;},limit(){return this;},abortSignal(){return this;},maybeSingle:read,upsert};
 return{api,read,upsert,rpc,admin:{from:()=>query,rpc}};
}
afterEach(()=>jest.useRealTimers());
test('fresh database cache avoids any CBU request',async()=>{
 const app=cacheSetup(snapshot()),fetch=jest.fn();expect(await app.api.currentBudgetRates(app.admin,now,fetch)).toEqual(snapshot());expect(fetch).not.toHaveBeenCalled();
});
test('simultaneous cache misses share one upstream refresh and persist before publishing',async()=>{
 const app=cacheSetup();let finish;const fetch=jest.fn(()=>new Promise(resolve=>{finish=resolve;}));
 const a=app.api.currentBudgetRates(app.admin,now,fetch),b=app.api.currentBudgetRates(app.admin,now,fetch);
 await new Promise(setImmediate);expect(fetch).toHaveBeenCalledTimes(1);finish(new Response(JSON.stringify(rows())));
 expect(await a).toEqual(snapshot());expect(await b).toEqual(snapshot());expect(app.upsert).toHaveBeenCalledTimes(1);
});
test('a stalled initial database read has a hard deadline',async()=>{
 jest.useFakeTimers();const app=cacheSetup();app.read.mockReturnValue(new Promise(()=>{}));const result=expect(app.api.currentBudgetRates(app.admin,now,jest.fn())).rejects.toThrow();await jest.advanceTimersByTimeAsync(5000);await result;
});
test('a stalled upstream settles and later cannot publish a snapshot',async()=>{
 jest.useFakeTimers();const app=cacheSetup();let finish;const fetch=jest.fn(()=>new Promise(resolve=>{finish=resolve;}));
 const result=expect(app.api.currentBudgetRates(app.admin,now,fetch)).rejects.toThrow();await jest.advanceTimersByTimeAsync(8000);await result;
 finish(new Response(JSON.stringify(rows())));await jest.advanceTimersByTimeAsync(1);expect(app.upsert).not.toHaveBeenCalled();
});
test('oversized upstream body is rejected without writing rates',async()=>{
 const app=cacheSetup();await expect(app.api.currentBudgetRates(app.admin,now,async()=>new Response('x'.repeat(262145)))).rejects.toThrow();expect(app.upsert).not.toHaveBeenCalled();
});

test('a request retains its initially validated snapshot while an hourly cache expires',()=>{
 const context=core.budgetFxContext(snapshot(),now);
 expect(core.referenceAmount(100,'EUR','USD',context.snapshot,context.checkedAt)).toBeCloseTo(112.290032479,8);
 expect(()=>core.validateBudgetFxSnapshot(context.snapshot,now+3600001)).toThrow();
});
test('budget comparison allows floating-point noise but never a fractional-cent overrun',()=>{
 expect(core.fitsBudgetAmount(0.1+0.2,0.3)).toBe(true);expect(core.fitsBudgetAmount(100.008,100)).toBe(false);expect(core.fitsBudgetAmount(Infinity,100)).toBe(false);
});


test('a named snapshot is validated and must match the requested identifier',async()=>{
 const app=cacheSetup(snapshot());expect(await app.api.budgetRatesById(app.admin,snapshot().id,now)).toEqual(snapshot());
 await expect(app.api.budgetRatesById(app.admin,'CBU:forged',now)).rejects.toThrow();
});
test('an expired named snapshot cannot be admitted',async()=>{
 const app=cacheSetup(snapshot());await expect(app.api.budgetRatesById(app.admin,snapshot().id,now+3600001)).rejects.toThrow();
});

test('an unavailable shared coordinator fails closed before fetching rates',async()=>{
 const app=cacheSetup();app.rpc.mockReturnValue({abortSignal:async()=>({error:{code:'PGRST202'}})});const fetch=jest.fn();
 await expect(app.api.currentBudgetRates(app.admin,now,fetch)).rejects.toThrow('budget_rates_unavailable');expect(fetch).not.toHaveBeenCalled();
});
test('a snapshot published between initial read and lease claim avoids a duplicate upstream fetch',async()=>{
 const app=cacheSetup(snapshot());app.read.mockResolvedValueOnce({data:null,error:null});const fetch=jest.fn();
 expect(await app.api.currentBudgetRates(app.admin,now,fetch)).toEqual(snapshot());expect(fetch).not.toHaveBeenCalled();expect(app.upsert).toHaveBeenCalledTimes(1);
});
test('a failed refresh activates shared cooldown instead of immediate upstream retries',async()=>{
 const app=cacheSetup();const fetch=jest.fn().mockRejectedValue(Error('upstream unreachable'));
 await expect(app.api.currentBudgetRates(app.admin,now,fetch)).rejects.toThrow();
 await expect(app.api.currentBudgetRates(app.admin,now,fetch)).rejects.toThrow();
 expect(fetch).toHaveBeenCalledTimes(1);expect(app.upsert).not.toHaveBeenCalled();
});
test('a worker rejected by publication fencing cannot return its unpublished snapshot',async()=>{
 const app=cacheSetup();const original=app.rpc.getMockImplementation();app.rpc.mockImplementation((name,args)=>name==='finish_budget_rate_refresh'?{abortSignal:async()=>({data:false})}:original(name,args));
 await expect(app.api.currentBudgetRates(app.admin,now,async()=>new Response(JSON.stringify(rows())))).rejects.toThrow();expect(app.upsert).not.toHaveBeenCalled();
});
test('waiting for an owner is bounded by the complete helper deadline',async()=>{
 jest.useFakeTimers();const app=cacheSetup();app.rpc.mockReturnValue({abortSignal:async()=>({data:'busy'})});const fetch=jest.fn();
 const result=expect(app.api.currentBudgetRates(app.admin,now,fetch)).rejects.toThrow();await jest.advanceTimersByTimeAsync(12000);await result;expect(fetch).not.toHaveBeenCalled();
});
