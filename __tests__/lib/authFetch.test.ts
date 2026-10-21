import {createClient} from '@supabase/supabase-js';
import {createAuthDeadlineFetch} from '../../src/lib/authFetch';
import {RequestTimeoutError} from '../../src/lib/requestDeadline';
type FetchInit = NonNullable<Parameters<typeof fetch>[1]>;
const base='https://project.test', url=base+'/auth/v1/token?grant_type=password';
beforeEach(()=>jest.useFakeTimers());afterEach(()=>jest.useRealTimers());
const response=()=>({clone:()=>({arrayBuffer:async()=>new ArrayBuffer(0)}),json:async()=>({ok:true})}) as unknown as Response;
test('hung auth fetch rejects on time, cancels transport and ignores late success',async()=>{
 let finish!:(response:Response)=>void;let signal!:AbortSignal;
 const raw=jest.fn((_:unknown,init:FetchInit)=>{signal=init.signal!;return new Promise<Response>(resolve=>{finish=resolve;});});
 const pending=createAuthDeadlineFetch(base,raw as typeof fetch,25)(url);
 const rejected=expect(pending).rejects.toBeInstanceOf(RequestTimeoutError);
 await jest.advanceTimersByTimeAsync(25);await rejected;expect(signal.aborted).toBe(true);
 finish(response());await Promise.resolve();await expect(pending).rejects.toBeInstanceOf(RequestTimeoutError);expect(raw).toHaveBeenCalledTimes(1);
});
test('a stalled auth body is also bounded',async()=>{
 const raw=jest.fn().mockResolvedValue({clone:()=>({arrayBuffer:()=>new Promise(()=>undefined)})});
 const pending=createAuthDeadlineFetch(base,raw,25)(url);const rejected=expect(pending).rejects.toBeInstanceOf(RequestTimeoutError);
 await jest.advanceTimersByTimeAsync(25);await rejected;expect(raw.mock.calls[0][1].signal.aborted).toBe(true);
});
test('successful responses stay readable, preserve request fields and clear timers',async()=>{
 const value=response(),raw=jest.fn().mockResolvedValue(value);const init={method:'POST',headers:{apikey:'test'},body:'test-body'};
 const result=await createAuthDeadlineFetch(base,raw,25)(url,init);
 expect(result).toBe(value);expect(await result.json()).toEqual({ok:true});expect(raw).toHaveBeenCalledWith(url,expect.objectContaining(init));expect(jest.getTimerCount()).toBe(0);
});
it.each([base+'/rest/v1/items',base+'/storage/v1/object/photo','https://other.test/auth/v1/token'] )('non-auth or other-origin transport stays unchanged: %s',async(other)=>{
 const raw=jest.fn().mockResolvedValue(response()),init={method:'POST'};await createAuthDeadlineFetch(base,raw,25)(other,init);
 expect(raw).toHaveBeenCalledWith(other,init);expect(jest.getTimerCount()).toBe(0);
});
test('caller cancellation settles even an uncooperative auth transport',async()=>{
 const controller=new AbortController(),raw=jest.fn(()=>new Promise<Response>(()=>undefined));
 const pending=createAuthDeadlineFetch(base,raw,25)(url,{signal:controller.signal});const rejected=expect(pending).rejects.toMatchObject({name:'AbortError'});
 controller.abort();await rejected;expect((raw.mock.calls[0] as unknown as [unknown,FetchInit])[1].signal?.aborted).toBe(true);expect(jest.getTimerCount()).toBe(0);
});
test('already-cancelled auth does not reach transport',async()=>{
 const controller=new AbortController();controller.abort();const raw=jest.fn();await expect(createAuthDeadlineFetch(base,raw,25)(url,{signal:controller.signal})).rejects.toMatchObject({name:'AbortError'});expect(raw).not.toHaveBeenCalled();
});


test('installed Supabase auth returns the deadline failure without creating a session or retrying sign-in',async()=>{
 const raw=jest.fn(()=>new Promise<Response>(()=>undefined));
 const client=createClient(base,'test-anon',{auth:{persistSession:false,autoRefreshToken:false,detectSessionInUrl:false},global:{fetch:createAuthDeadlineFetch(base,raw,25)}});
 const pending=client.auth.signInWithPassword({email:'fixture@example.invalid',password:'fixture-password'});
 await jest.advanceTimersByTimeAsync(25);
 const result=await pending;
 expect(result.error?.message).toContain('timed out');expect(result.data.session).toBeNull();expect(raw).toHaveBeenCalledTimes(1);
 expect((await client.auth.getSession()).data.session).toBeNull();
});
