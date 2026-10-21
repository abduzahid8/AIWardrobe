import { createClient } from '@supabase/supabase-js';
import { createStorageUploadDeadlineFetch } from '../../src/lib/authFetch';
const base = 'https://project.test';
beforeEach(() => jest.useFakeTimers());
afterEach(() => jest.useRealTimers());

it.each(['headers', 'body'])('installed SDK bounds stalled upload %s and does not retry', async phase => {
  const raw = jest.fn().mockImplementation(() => phase === 'headers' ? new Promise(() => undefined) : Promise.resolve({
    clone: () => ({ arrayBuffer: () => new Promise(() => undefined) }),
  }));
  const client = createClient(base, 'test-anon', { auth: { persistSession: false, autoRefreshToken: false, detectSessionInUrl: false },
    global: { fetch: createStorageUploadDeadlineFetch(base, raw, 25) } });
  const result = client.storage.from('wardrobe-images').upload('owner/item/original.jpg', new Uint8Array([255,216,255,217]).buffer, { upsert: true, contentType: 'image/jpeg' });
  await jest.advanceTimersByTimeAsync(25);
  expect((await result).error?.message).toContain('timed out');
  expect(raw).toHaveBeenCalledTimes(1);
  expect(raw.mock.calls[0][1].signal.aborted).toBe(true);
  expect(jest.getTimerCount()).toBe(0);
});

it.each([
  [base + '/storage/v1/object/wardrobe-images/owner/item.jpg', 'GET'],
  [base + '/storage/v1/object/other-bucket/item.jpg', 'POST'],
  ['https://other.test/storage/v1/object/wardrobe-images/item.jpg', 'POST'],
  [base + '/rest/v1/items', 'POST'],
])('leaves unrelated transport unchanged: %s %s', async (url, method) => {
  const response = {} as Response, raw = jest.fn().mockResolvedValue(response), init = { method };
  expect(await createStorageUploadDeadlineFetch(base, raw, 25)(url, init)).toBe(response);
  expect(raw).toHaveBeenCalledWith(url, init);
  expect(jest.getTimerCount()).toBe(0);
});

test('lost acknowledgement may commit once; late success stays failed and explicit retry uses the same object', async () => {
  const objects = new Map<string, number[]>();
  let release!: (response: Response) => void;
  const acknowledgement = () => new Response(JSON.stringify({ Id: 'object-id', Key: 'wardrobe-images/owner/item/original.jpg' }), { status: 200, headers: { 'content-type': 'application/json' } });
  const raw = jest.fn().mockImplementation((url: string, init: RequestInit) => {
    expect(new Headers(init.headers).get('x-upsert')).toBe('true');
    objects.set(url, [...new Uint8Array(init.body as ArrayBuffer)]);
    return raw.mock.calls.length === 1 ? new Promise<Response>(resolve => { release = resolve; }) : Promise.resolve(acknowledgement());
  });
  const client = createClient(base, 'test-anon', { auth: { persistSession: false, autoRefreshToken: false, detectSessionInUrl: false },
    global: { fetch: createStorageUploadDeadlineFetch(base, raw, 25) } });
  const upload = () => client.storage.from('wardrobe-images').upload('owner/item/original.jpg', new Uint8Array([255,216,255,217]).buffer, { upsert: true, contentType: 'image/jpeg' });
  const pending = upload();
  await jest.advanceTimersByTimeAsync(25);
  expect((await pending).error?.message).toContain('timed out');
  expect(objects.size).toBe(1);
  release(acknowledgement());
  await jest.advanceTimersByTimeAsync(0);
  expect((await pending).error?.message).toContain('timed out');
  expect(raw).toHaveBeenCalledTimes(1);
  const retry = await upload();
  expect(retry.error).toBeNull();
  expect(retry.data?.path).toBe('owner/item/original.jpg');
  expect(objects.size).toBe(1);
  expect([...objects.values()]).toEqual([[255,216,255,217]]);
  expect(jest.getTimerCount()).toBe(0);
});
