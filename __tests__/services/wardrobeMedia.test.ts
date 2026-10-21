import { uploadWardrobeImage, wardrobeMediaHeaders, wardrobeMediaReference, createWardrobeImageAccessUrl, WARDROBE_MEDIA_MAX_BYTES } from '../../src/services/wardrobeMedia';
import { authEvents } from '../../store/authEvents';
const mockUpload = jest.fn(), mockSigned = jest.fn();
jest.mock('../../lib/supabase', () => ({ supabase: { storage: { from: () => ({ upload: mockUpload, createSignedUrl: mockSigned }) } } }));
jest.mock('../../src/config/env', () => ({ Config: { supabase: { url: 'https://storage.test' } } }));
const owner = '11111111-1111-4111-8111-111111111111', other = '22222222-2222-4222-8222-222222222222';
const id = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const jpeg = 'data:image/jpeg;base64,/9j/2Q==';
const canonical = `https://storage.test/storage/v1/object/authenticated/wardrobe-images/${owner}/${id}/original.jpg`;
const originalFetch = global.fetch;
beforeEach(() => {
  jest.resetAllMocks(); authEvents.emitLogin(owner);
  mockUpload.mockResolvedValue({ error: null });
  mockSigned.mockResolvedValue({ data: { signedUrl: 'https://storage.test/temporary-test-link' }, error: null });
  global.fetch = jest.fn().mockResolvedValue({ headers: { get: () => 'application/octet-stream' }, arrayBuffer: async () => new Uint8Array([255,216,255,217]).buffer });
});
afterAll(() => { global.fetch = originalFetch; });

test('uploads unchanged bytes with stable owner path and persists no expiring token', async () => {
  expect(await uploadWardrobeImage(owner,id,jpeg)).toBe(canonical);
  expect(await uploadWardrobeImage(owner,id,jpeg)).toBe(canonical);
  for (const call of mockUpload.mock.calls) {
    expect(call[0]).toBe(`${owner}/${id}/original.jpg`);
    expect([...new Uint8Array(call[1])]).toEqual([255,216,255,217]);
    expect(call[2]).toEqual({ contentType: 'image/jpeg', upsert: true });
  }
  expect(global.fetch).not.toHaveBeenCalled();
});

test('reads native local bytes as ArrayBuffer despite generic file MIME', async () => {
  expect(await uploadWardrobeImage(owner,id,'file:///cache/photo.jpg')).toBe(canonical);
  expect(mockUpload.mock.calls[0][1]).toBeInstanceOf(ArrayBuffer);
});

test('existing private and signed references normalize without reuploading', async () => {
  expect(await uploadWardrobeImage(owner,id,canonical)).toBe(canonical);
  expect(await uploadWardrobeImage(owner,id,canonical.replace('/authenticated/','/sign/')+'?token=temporary')).toBe(canonical);
  await expect(uploadWardrobeImage(owner,id,canonical.replace(owner,other))).rejects.toMatchObject({ code: 'access_denied' });
  expect(mockUpload).not.toHaveBeenCalled();
});

test('existing remote catalog images are not downloaded or copied', async () => {
  const uri = 'https://catalog.example.invalid/photo.jpg';
  expect(await uploadWardrobeImage(owner,id,uri)).toBe(uri);
  expect(global.fetch).not.toHaveBeenCalled(); expect(mockUpload).not.toHaveBeenCalled();
});

it.each([
  ['data:image/svg+xml;base64,PHN2Zz4=', 'unsupported_format'],
  ['data:image/png;base64,/9j/2Q==', 'invalid_source'],
  ['data:image/png;base64,data:image/png;base64,AAAA', 'invalid_source'],
  ['javascript:alert(1)', 'invalid_source'],
])('rejects invalid/unsupported media before upload: %s', async (uri, code) => {
  await expect(uploadWardrobeImage(owner,id,uri)).rejects.toMatchObject({ code });
  expect(mockUpload).not.toHaveBeenCalled();
});

it.each([0, WARDROBE_MEDIA_MAX_BYTES + 1])('rejects invalid native size %s', async size => {
  global.fetch = jest.fn().mockResolvedValue({ arrayBuffer: async () => new ArrayBuffer(size) });
  await expect(uploadWardrobeImage(owner,id,'file:///cache/photo.jpg')).rejects.toMatchObject({ code: 'invalid_size' });
  expect(mockUpload).not.toHaveBeenCalled();
});

test('storage errors reject rather than returning a saved URL', async () => {
  mockUpload.mockResolvedValueOnce({ error: new Error('Storage unavailable') });
  await expect(uploadWardrobeImage(owner,id,jpeg)).rejects.toThrow('Storage unavailable');
});

test('account changes while reading bytes prevent upload', async () => {
  global.fetch = jest.fn().mockImplementation(async () => {
    authEvents.emitLogout(); authEvents.emitLogin(other);
    return { arrayBuffer: async () => new Uint8Array([255,216,255,217]).buffer };
  });
  await expect(uploadWardrobeImage(owner,id,'file:///cache/photo.jpg')).rejects.toMatchObject({ code: 'account_changed' });
  expect(mockUpload).not.toHaveBeenCalled();
});

test('a late completed upload cannot publish into a new account', async () => {
  mockUpload.mockImplementationOnce(async () => { authEvents.emitLogout(); authEvents.emitLogin(other); return { error: null }; });
  await expect(uploadWardrobeImage(owner,id,jpeg)).rejects.toMatchObject({ code: 'account_changed' });
});

test('export token creation validates its lifetime and owner', async () => {
  expect(await createWardrobeImageAccessUrl(owner,canonical)).toBe('https://storage.test/temporary-test-link');
  expect(mockSigned).toHaveBeenCalledWith(`${owner}/${id}/original.jpg`,300);
  await expect(createWardrobeImageAccessUrl(owner,canonical.replace(owner,other))).rejects.toMatchObject({ code: 'access_denied' });
  await expect(createWardrobeImageAccessUrl(owner,canonical,86400)).rejects.toMatchObject({ code: 'invalid_source' });
  mockSigned.mockResolvedValueOnce({ data: null, error: null });
  await expect(createWardrobeImageAccessUrl(owner,canonical)).rejects.toThrow('not confirmed');
});

test('image credentials never attach to other owners, hosts or signed endpoints', () => {
  expect(wardrobeMediaHeaders(canonical,owner,'session')).toEqual({ Authorization: 'Bearer session' });
  for (const uri of [canonical.replace(owner,other),canonical.replace('storage.test','storage.test.evil.invalid'),
    canonical.replace('/authenticated/','/sign/'),canonical.replace('https://','https://untrusted@'), 'https://external.invalid/photo.jpg']) {
    expect(wardrobeMediaHeaders(uri,owner,'session')).toBeUndefined();
  }
  expect(wardrobeMediaReference(canonical.replace(owner,owner+'%2Fforeign'))).toBeNull();
});

it.each([301,400,401,403,404,413,429,500,503])('rejects a local fetch response with status %s before reading or uploading its body',async status=>{
 const body=jest.fn().mockResolvedValue(new Uint8Array([255,216,255,217]).buffer);
 global.fetch=jest.fn().mockResolvedValue({status,ok:false,headers:{get:()=> 'image/jpeg'},arrayBuffer:body});
 await expect(uploadWardrobeImage(owner,id,'file:///cache/photo.jpg')).rejects.toMatchObject({code:'invalid_source'});
 expect(body).not.toHaveBeenCalled();expect(mockUpload).not.toHaveBeenCalled();
});
it.each([0,200,206])('allows byte-validated local reader status %s',async status=>{
 global.fetch=jest.fn().mockResolvedValue({status,ok:status>=200,headers:{get:()=> 'image/jpeg'},arrayBuffer:async()=>new Uint8Array([255,216,255,217]).buffer});
 expect(await uploadWardrobeImage(owner,id,'file:///cache/photo.jpg')).toBe(canonical);
});

it.each(['headers', 'body'])('bounds a stalled local media %s read and never uploads late bytes', async phase => {
  jest.useFakeTimers();
  let release!: (value: any) => void;
  const stalled = new Promise<any>(resolve => { release = resolve; });
  const bytes = new Uint8Array([255,216,255,217]).buffer;
  global.fetch = jest.fn().mockImplementation(() => phase === 'headers' ? stalled : Promise.resolve({ arrayBuffer: () => stalled }));
  try {
    const result = uploadWardrobeImage(owner, id, 'file:///cache/photo.jpg');
    const rejected = expect(result).rejects.toMatchObject({ name: 'RequestTimeoutError' });
    await jest.advanceTimersByTimeAsync(15_000);
    await rejected;
    expect((global.fetch as jest.Mock).mock.calls[0][1].signal.aborted).toBe(true);
    release(phase === 'headers' ? { arrayBuffer: async () => bytes } : bytes);
    await jest.advanceTimersByTimeAsync(0);
    expect(mockUpload).not.toHaveBeenCalled();
    expect(jest.getTimerCount()).toBe(0);
  } finally { jest.useRealTimers(); }
});
