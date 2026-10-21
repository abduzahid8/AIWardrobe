const mockInvoke = jest.fn();
jest.mock('../../lib/supabase', () => ({supabase: {functions: {invoke: (...args: any[]) => mockInvoke(...args)}}}));
const now = Date.parse('2026-10-20T12:00:00Z');
const snapshot = (fetchedAt = now) => ({id:'CBU:2026-10-20:12044.72:13525.02:16062.84',source:'CBU',asOf:'2026-10-20',fetchedAt:new Date(fetchedAt).toISOString(),uzsPerUnit:{USD:12044.72,EUR:13525.02,GBP:16062.84,UZS:1}});
let getBudgetRates: typeof import('../../src/services/budgetRates').getBudgetRates;
beforeEach(() => {jest.resetModules();jest.clearAllMocks();jest.useFakeTimers();jest.setSystemTime(now);getBudgetRates=require('../../src/services/budgetRates').getBudgetRates;mockInvoke.mockResolvedValue({data:{success:true,snapshot:snapshot()},error:null});});
afterEach(() => jest.useRealTimers());
it('loads a server snapshot and returns immutable request-pinned values',async()=>{
 const result=await getBudgetRates();expect(result).toEqual(snapshot());expect(Object.isFrozen(result)).toBe(true);expect(Object.isFrozen(result.uzsPerUnit)).toBe(true);expect(mockInvoke).toHaveBeenCalledWith('budget-rates',expect.objectContaining({signal:expect.any(AbortSignal)}));
});
it('coalesces overlapping loads and reuses only a fresh cache',async()=>{
 const [a,b]=await Promise.all([getBudgetRates(),getBudgetRates()]);expect(a.id).toBe(b.id);await getBudgetRates();expect(mockInvoke).toHaveBeenCalledTimes(1);
});
it('refreshes an expired cache without changing a snapshot already held by a caller',async()=>{
 const first=await getBudgetRates();await jest.advanceTimersByTimeAsync(3600001);mockInvoke.mockResolvedValueOnce({data:{success:true,snapshot:snapshot(Date.now())},error:null});const second=await getBudgetRates();expect(mockInvoke).toHaveBeenCalledTimes(2);expect(first.fetchedAt).toBe(snapshot().fetchedAt);expect(second.fetchedAt).not.toBe(first.fetchedAt);
});
it('does not use expired rates when refresh fails',async()=>{
 await getBudgetRates();await jest.advanceTimersByTimeAsync(3600001);mockInvoke.mockResolvedValueOnce({data:null,error:{message:'private detail'}});await expect(getBudgetRates()).rejects.toThrow('budget_rates_unavailable');
});
it('rejects a fabricated rate snapshot rather than caching it',async()=>{
 mockInvoke.mockResolvedValueOnce({data:{success:true,snapshot:{...snapshot(),id:'fabricated'}},error:null});await expect(getBudgetRates()).rejects.toThrow('budget_rates_unavailable');expect(await getBudgetRates()).toEqual(snapshot());expect(mockInvoke).toHaveBeenCalledTimes(2);
});
it('a timed-out response cannot later overwrite a successful refresh',async()=>{
 let finish: (value: any)=>void=()=>{};mockInvoke.mockReturnValueOnce(new Promise(resolve=>{finish=resolve;}));const first=expect(getBudgetRates()).rejects.toThrow('timed out');await jest.advanceTimersByTimeAsync(15000);await first;
 mockInvoke.mockResolvedValueOnce({data:{success:true,snapshot:snapshot(Date.now())},error:null});const newer=await getBudgetRates();finish({data:{success:true,snapshot:snapshot()},error:null});await jest.advanceTimersByTimeAsync(1);expect((await getBudgetRates()).fetchedAt).toBe(newer.fetchedAt);expect(mockInvoke).toHaveBeenCalledTimes(2);
});
