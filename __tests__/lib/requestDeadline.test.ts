import { createClient } from '@supabase/supabase-js';
import { withRequestDeadline, RequestTimeoutError } from '../../src/lib/requestDeadline';
beforeEach(() => { jest.useFakeTimers(); });
afterEach(() => { jest.useRealTimers(); });

test('a hung operation settles at the deadline and receives cancellation', async () => {
  let signal!: AbortSignal;
  const pending = withRequestDeadline(s => { signal = s; return new Promise(() => undefined); }, 25);
  const failure = expect(pending).rejects.toBeInstanceOf(RequestTimeoutError);
  await jest.advanceTimersByTimeAsync(25); await failure;
  expect(signal.aborted).toBe(true);
});

test('completed requests clear their deadline timer', async () => {
  let signal!: AbortSignal;
  expect(await withRequestDeadline(s => { signal = s; return Promise.resolve('saved'); }, 25)).toBe('saved');
  await jest.advanceTimersByTimeAsync(25);
  expect(signal.aborted).toBe(false);
  expect(jest.getTimerCount()).toBe(0);
});

test('a late success cannot replace a timeout outcome', async () => {
  let finish!: (value: string) => void;
  const pending = withRequestDeadline(() => new Promise<string>(resolve => { finish = resolve; }), 25);
  const failure = expect(pending).rejects.toBeInstanceOf(RequestTimeoutError);
  await jest.advanceTimersByTimeAsync(25); await failure;
  finish('late'); await Promise.resolve();
  await expect(pending).rejects.toBeInstanceOf(RequestTimeoutError);
});

test('synchronous errors clean up the timer and are preserved', async () => {
  await expect(withRequestDeadline(() => { throw new Error('bad request'); }, 25)).rejects.toThrow('bad request');
  expect(jest.getTimerCount()).toBe(0);
});

test('the installed Supabase query builder passes cancellation to its transport', async () => {
  let transportSignal: AbortSignal | undefined;
  const transport = jest.fn((_url: unknown, init?: RequestInit) => new Promise<Response>((_, reject) => {
    transportSignal = init?.signal as AbortSignal;
    const abort = () => { const error = new Error('transport aborted'); error.name = 'AbortError'; reject(error); };
    if (transportSignal.aborted) abort();
    else transportSignal.addEventListener('abort', abort, { once: true });
  }));
  const client = createClient('https://deadline-test.invalid', 'test-anon', { auth: { persistSession: false, autoRefreshToken: false, detectSessionInUrl: false }, global: { fetch: transport } });
  const pending = withRequestDeadline(signal => client.from('profiles').select('id').abortSignal(signal), 25);
  const failure = expect(pending).rejects.toBeInstanceOf(RequestTimeoutError);
  await jest.advanceTimersByTimeAsync(25); await failure;
  expect(transport).toHaveBeenCalledTimes(1);
  expect(transportSignal?.aborted).toBe(true);
});

test('the installed Supabase Edge client cancels a hung outfit transport at the deadline', async () => {
  let transportSignal: AbortSignal | undefined;
  const transport = jest.fn((_url: unknown, init?: RequestInit) => new Promise<Response>((_, reject) => {
    transportSignal = init?.signal as AbortSignal;
    const abort = () => { const error = new Error('transport aborted'); error.name = 'AbortError'; reject(error); };
    if (transportSignal.aborted) abort();
    else transportSignal.addEventListener('abort', abort, { once: true });
  }));
  const client = createClient('https://deadline-test.invalid', 'test-anon', { auth: { persistSession: false, autoRefreshToken: false, detectSessionInUrl: false }, global: { fetch: transport } });
  const pending = withRequestDeadline(signal => client.functions.invoke('generate-outfits', { signal, body: { occasion: 'Casual' } }), 25);
  const failure = expect(pending).rejects.toBeInstanceOf(RequestTimeoutError);
  await jest.advanceTimersByTimeAsync(25); await failure;
  expect(transport).toHaveBeenCalledTimes(1);
  expect(transportSignal?.aborted).toBe(true);
  expect(jest.getTimerCount()).toBe(0);
});

test('explicit cancellation settles even when the transport ignores abort',async()=>{
 const cancel=new AbortController();let transport!:AbortSignal;
 const pending=withRequestDeadline(signal=>{transport=signal;return new Promise(()=>{});},1000,cancel.signal);
 const failure=expect(pending).rejects.toMatchObject({name:'AbortError'});
 cancel.abort();await failure;expect(transport.aborted).toBe(true);expect(jest.getTimerCount()).toBe(0);
});
test('an already cancelled request never starts transport or a timer',async()=>{
 const cancel=new AbortController();cancel.abort();const start=jest.fn();
 await expect(withRequestDeadline(start,1000,cancel.signal)).rejects.toMatchObject({name:'AbortError'});
 expect(start).not.toHaveBeenCalled();expect(jest.getTimerCount()).toBe(0);
});
test('successful work removes the external cancellation listener',async()=>{
 const cancel=new AbortController();const remove=jest.spyOn(cancel.signal,'removeEventListener');let transport!:AbortSignal;
 expect(await withRequestDeadline(signal=>{transport=signal;return Promise.resolve('done');},1000,cancel.signal)).toBe('done');
 expect(remove).toHaveBeenCalledWith('abort',expect.any(Function));cancel.abort();expect(transport.aborted).toBe(false);
});
