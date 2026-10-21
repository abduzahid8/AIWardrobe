import { createStore } from 'zustand/vanilla';
import { createItemsSlice } from '../../store/slices/wardrobeItemsSlice';
import type { WardrobeState } from '../../store/wardrobeStore';
import { stageNativeWardrobeImage, resolvePendingWardrobeImage, removePendingWardrobeImage, clearPendingWardrobeImages, clearAllPendingWardrobeImages, flushPendingWardrobeImageWork } from '../../src/services/pendingWardrobeMedia';
import * as FileSystem from 'expo-file-system/legacy';
import { authEvents } from '../../store/authEvents';
const mockFiles = new Map<string,string>();
let mockRoot: string | null = 'file:///first/Documents/';
jest.mock('../../lib/supabase', () => ({ supabase: {} }));
jest.mock('../../src/services/wardrobeThumbnail', () => ({ createNativeWardrobeThumbnail: (owner: string, id: string, source: string) => jest.requireActual('../../src/services/pendingWardrobeMedia').stageNativeWardrobeImage(owner, id, source) }));
jest.mock('../../store/auth', () => ({ __esModule: true, default: { getState: () => ({ user: { id: '11111111-1111-4111-8111-111111111111' } }) } }));
jest.mock('../../store/subscriptionStore', () => ({ __esModule: true, default: { getState: () => ({ tier: 'premium', effectiveTier: 'premium', refreshTimeState: () => {} }) }, TIER_FEATURES: { premium: { wardrobeItems: -1 } } }));
jest.mock('../../src/config/env', () => ({ Config: { supabase: { url: 'https://storage.test' } } }));
jest.mock('expo-file-system/legacy', () => ({
  get documentDirectory() { return mockRoot; }, EncodingType: { Base64: 'base64' },
  makeDirectoryAsync: jest.fn(), getInfoAsync: jest.fn(), readAsStringAsync: jest.fn(),
  writeAsStringAsync: jest.fn(), moveAsync: jest.fn(), deleteAsync: jest.fn(),
}));
const owner='11111111-1111-4111-8111-111111111111', other='22222222-2222-4222-8222-222222222222';
const id='aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa', next='bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';
const input='data:image/jpeg;base64,/9j/2Q==';
const fetchBefore=global.fetch;
beforeEach(async () => {
  await flushPendingWardrobeImageWork(); jest.resetAllMocks(); mockFiles.clear(); mockRoot='file:///first/Documents/'; authEvents.emitLogin(owner);
  (FileSystem.makeDirectoryAsync as jest.Mock).mockResolvedValue(undefined);
  (FileSystem.getInfoAsync as jest.Mock).mockImplementation(async (path:string) => mockFiles.has(path)
    ? { exists:true,isDirectory:false,size:Buffer.from(mockFiles.get(path)!,'base64').length }
    : path.startsWith('file:///cache/') ? {exists:true,isDirectory:false,size:4} : {exists:false});
  (FileSystem.readAsStringAsync as jest.Mock).mockImplementation(async (path:string) => {
    if(!mockFiles.has(path))throw new Error('missing');return mockFiles.get(path);
  });
  (FileSystem.writeAsStringAsync as jest.Mock).mockImplementation(async (path:string,value:string) => {mockFiles.set(path,value);});
  (FileSystem.moveAsync as jest.Mock).mockImplementation(async ({from,to}:{from:string;to:string}) => {
    if(!mockFiles.has(from))throw new Error('missing temporary');mockFiles.set(to,mockFiles.get(from)!);mockFiles.delete(from);
  });
  (FileSystem.deleteAsync as jest.Mock).mockImplementation(async (path:string) => {
    for(const file of mockFiles.keys())if(file===path||file.startsWith(path.endsWith('/')?path:path+'/'))mockFiles.delete(file);
  });
  global.fetch=jest.fn().mockResolvedValue({arrayBuffer:async()=>new Uint8Array([255,216,255,217]).buffer});
});
afterEach(async()=>{await flushPendingWardrobeImageWork();});
afterAll(()=>{global.fetch=fetchBefore;});

test('stages exact bytes in Documents via temporary write and move',async()=>{
  const uri=await stageNativeWardrobeImage(owner,id,input);
  expect(uri).toBe(`${mockRoot}aiwardrobe-pending-media/${owner}/${id}.jpg`);
  expect([...mockFiles]).toEqual([[uri,'/9j/2Q==']]);
  expect(FileSystem.writeAsStringAsync).toHaveBeenCalledWith(expect.stringMatching(/\.pending$/),'/9j/2Q==',{encoding:'base64'});
  expect(FileSystem.moveAsync).toHaveBeenCalledWith({from:expect.stringMatching(/\.pending$/),to:uri});
});
test('cache-file input is copied into durable Documents storage',async()=>{
  const uri=await stageNativeWardrobeImage(owner,id,'file:///cache/temporary.jpg');
  expect(global.fetch).toHaveBeenCalledWith('file:///cache/temporary.jpg', { signal: expect.any(AbortSignal) });
  expect(uri).not.toContain('/cache/');expect(mockFiles.get(uri)).toBe('/9j/2Q==');
});
it.each([0,-1,10*1024*1024+1,Number.NaN,1.5])('native file size %s rejects before buffering or writing',async size=>{
  (FileSystem.getInfoAsync as jest.Mock).mockResolvedValueOnce({exists:true,isDirectory:false,size});
  await expect(stageNativeWardrobeImage(owner,id,'file:///cache/oversized.jpg')).rejects.toMatchObject({code:'invalid_size'});
  expect(global.fetch).not.toHaveBeenCalled();expect(FileSystem.writeAsStringAsync).not.toHaveBeenCalled();
});
it.each([{exists:false},{exists:true,isDirectory:true,size:4}])('missing files and directories never reach native fetch: %j',async info=>{
  (FileSystem.getInfoAsync as jest.Mock).mockResolvedValueOnce(info);
  await expect(stageNativeWardrobeImage(owner,id,'file:///cache/invalid.jpg')).rejects.toMatchObject({code:'invalid_source'});
  expect(global.fetch).not.toHaveBeenCalled();expect(FileSystem.writeAsStringAsync).not.toHaveBeenCalled();
});
test('account switch during metadata lookup prevents reading private source bytes',async()=>{
  (FileSystem.getInfoAsync as jest.Mock).mockImplementationOnce(async()=>{
    authEvents.emitLogout();authEvents.emitLogin(other);return {exists:true,isDirectory:false,size:4};
  });
  await expect(stageNativeWardrobeImage(owner,id,'file:///cache/old.jpg')).rejects.toMatchObject({code:'account_changed'});
  expect(global.fetch).not.toHaveBeenCalled();
});
test('same-content retry is idempotent; another photo cannot replace its ID',async()=>{
  const uri=await stageNativeWardrobeImage(owner,id,input);
  expect(await stageNativeWardrobeImage(owner,id,input)).toBe(uri);
  expect(FileSystem.writeAsStringAsync).toHaveBeenCalledTimes(1);
  await expect(stageNativeWardrobeImage(owner,id,'data:image/jpeg;base64,/9j/AA==')).rejects.toMatchObject({code:'invalid_source'});
  expect(mockFiles.get(uri)).toBe('/9j/2Q==');
});
test('restored documents rebase after application container changes',async()=>{
  const old=await stageNativeWardrobeImage(owner,id,input);
  mockRoot='file:///restored/Documents/';
  const current=resolvePendingWardrobeImage(old,owner)!;
  mockFiles.set(current,mockFiles.get(old)!);mockFiles.delete(old);
  expect(current).toContain('/restored/Documents/');
  expect(await stageNativeWardrobeImage(owner,id,input)).toBe(current);
  expect(resolvePendingWardrobeImage(old,other)).toBeUndefined();
});
test('rename failure leaves no published partial photo and later work can succeed',async()=>{
  (FileSystem.moveAsync as jest.Mock).mockRejectedValueOnce(new Error('disk failure'));
  await expect(stageNativeWardrobeImage(owner,id,input)).rejects.toThrow('disk failure');
  expect(mockFiles.size).toBe(0);
  expect(await stageNativeWardrobeImage(owner,next,input)).toContain(next);
});
test('committed-size mismatch removes temporary and final files',async()=>{
  (FileSystem.getInfoAsync as jest.Mock).mockResolvedValueOnce({exists:false})
    .mockResolvedValueOnce({exists:true,isDirectory:false,size:999});
  await expect(stageNativeWardrobeImage(owner,id,input)).rejects.toMatchObject({code:'invalid_source'});
  expect(mockFiles.size).toBe(0);
});
test('account switch during source read prevents writing',async()=>{
  global.fetch=jest.fn().mockImplementation(async()=>{
    authEvents.emitLogout();authEvents.emitLogin(other);
    return {arrayBuffer:async()=>new Uint8Array([255,216,255,217]).buffer};
  });
  await expect(stageNativeWardrobeImage(owner,id,'file:///cache/photo.jpg')).rejects.toMatchObject({code:'account_changed'});
  expect(FileSystem.writeAsStringAsync).not.toHaveBeenCalled();
});
test('late old-account writes and cleanup cannot erase the next account photo',async()=>{
  let release!:()=>void;
  (FileSystem.writeAsStringAsync as jest.Mock).mockImplementationOnce(async(path:string,value:string)=>{
    await new Promise<void>(resolve=>{release=resolve;});mockFiles.set(path,value);
  });
  const old=stageNativeWardrobeImage(owner,id,input);
  const rejected=expect(old).rejects.toMatchObject({code:'account_changed'});
  while(!release)await new Promise(resolve=>setTimeout(resolve,0));
  authEvents.emitLogout();authEvents.emitLogin(other);
  const clear=clearPendingWardrobeImages(owner);
  const fresh=stageNativeWardrobeImage(other,next,input);
  release();await rejected;await clear;
  const current=await fresh;
  expect([...mockFiles.keys()]).toEqual([current]);expect(current).toContain(other);
});
test('removal is limited to explicitly owned managed files',async()=>{
  const uri=await stageNativeWardrobeImage(owner,id,input);
  await removePendingWardrobeImage(other,uri);expect(mockFiles.has(uri)).toBe(true);
  await removePendingWardrobeImage(owner,'file:///cache/unrelated.jpg');expect(mockFiles.has(uri)).toBe(true);
  await removePendingWardrobeImage(owner,uri);expect(mockFiles.size).toBe(0);
});
test('account cleanup removes all managed owners and orphaned temporary files, retaining unrelated Documents files', async () => {
  mockFiles.set(`${mockRoot}aiwardrobe-pending-media/${owner}/${id}.jpg`, 'private');
  mockFiles.set(`${mockRoot}aiwardrobe-pending-media/${other}/.${next}.jpg.pending`, 'orphan');
  const unrelated = `${mockRoot}unrelated-document.txt`;
  mockFiles.set(unrelated, 'keep');
  await clearAllPendingWardrobeImages(authEvents.getRevision());
  expect([...mockFiles]).toEqual([[unrelated, 'keep']]);
  expect(FileSystem.deleteAsync).toHaveBeenCalledWith(`${mockRoot}aiwardrobe-pending-media/`, { idempotent: true });
});
test('superseded root cleanup queued behind a write cannot erase the next account files', async () => {
  let release!: () => void;
  (FileSystem.writeAsStringAsync as jest.Mock).mockImplementationOnce(async (path: string, value: string) => {
    await new Promise<void>(resolve => { release = resolve; }); mockFiles.set(path, value);
  });
  const old = stageNativeWardrobeImage(owner, id, input);
  const rejected = expect(old).rejects.toMatchObject({ code: 'account_changed' });
  while (!release) await new Promise(resolve => setTimeout(resolve, 0));
  const cleanup = clearAllPendingWardrobeImages(authEvents.getRevision());
  const stale = expect(cleanup).rejects.toMatchObject({ code: 'account_changed' });
  authEvents.emitLogout(); authEvents.emitLogin(other);
  const fresh = stageNativeWardrobeImage(other, next, input);
  release(); await rejected; await stale;
  const current = await fresh;
  expect([...mockFiles.keys()]).toEqual([current]);
  expect(FileSystem.deleteAsync).not.toHaveBeenCalledWith(`${mockRoot}aiwardrobe-pending-media/`, expect.anything());
});
test('root deletion errors propagate, and a subsequent cleanup can retry', async () => {
  (FileSystem.deleteAsync as jest.Mock).mockRejectedValueOnce(new Error('synthetic filesystem failure'));
  await expect(clearAllPendingWardrobeImages(authEvents.getRevision())).rejects.toThrow('synthetic filesystem failure');
  await clearAllPendingWardrobeImages(authEvents.getRevision());
  expect(FileSystem.deleteAsync).toHaveBeenCalledTimes(2);
});
test('invalid paths and unavailable native Documents storage fail before persistence',async()=>{
  await expect(stageNativeWardrobeImage(owner,'../escape',input)).rejects.toMatchObject({code:'invalid_source'});
  mockRoot=null;
  await expect(stageNativeWardrobeImage(owner,id,input)).rejects.toMatchObject({code:'invalid_source'});
  expect(FileSystem.writeAsStringAsync).not.toHaveBeenCalled();
});


test('actual native garment save queues staged original bytes rather than picker or inline references', async () => {
  const sync = jest.fn().mockImplementation(async () => {
    const saved = state.getState().items[0];
    expect(saved.imageUrl).toContain(`/aiwardrobe-pending-media/${owner}/`);
    expect(mockFiles.get(saved.imageUrl)).toBe('/9j/2Q==');
    expect(state.getState().pendingActions[0].payload.imageUrl).toBe(saved.imageUrl);
    expect(saved.thumbnailUrl).not.toBe(saved.imageUrl);
    expect(mockFiles.get(saved.thumbnailUrl!)).toBe('/9j/2Q==');
  });
  const state = createStore<WardrobeState>()((set,get,store) => ({ ...createItemsSlice(set,get,store),
    pendingActions: [], outfits: [], syncToServer: sync,
  } as unknown as WardrobeState));
  await state.getState().addItem({ imageUrl: input, thumbnailUrl: input, category: 'top', userId: owner } as never);
  expect(sync).toHaveBeenCalledTimes(1);
  expect(FileSystem.moveAsync).toHaveBeenCalledTimes(2);
  expect([...mockFiles.keys()]).toEqual([state.getState().items[0].imageUrl, state.getState().items[0].thumbnailUrl]);
});

test('concurrent local staging waits for the first complete read and commit before buffering the next image',async()=>{
 let release!:(bytes:ArrayBuffer)=>void;
 (global.fetch as jest.Mock).mockResolvedValueOnce({arrayBuffer:()=>new Promise<ArrayBuffer>(resolve=>{release=resolve;})});
 const first=stageNativeWardrobeImage(owner,id,'file:///cache/first.jpg');
 while(!release)await new Promise(resolve=>setTimeout(resolve,0));
 const second=stageNativeWardrobeImage(owner,next,'file:///cache/second.jpg');
 try{
  for(let i=0;i<20;i++)await Promise.resolve();
  expect(global.fetch).toHaveBeenCalledTimes(1);
  expect(FileSystem.moveAsync).not.toHaveBeenCalled();
 }finally{release(new Uint8Array([255,216,255,217]).buffer);await Promise.allSettled([first,second]);}
 expect((FileSystem.moveAsync as jest.Mock).mock.calls[0][0].to).toContain(id);
 expect(global.fetch).toHaveBeenCalledTimes(2);
});
test('a stale queued local image is rejected before its metadata or body is read',async()=>{
 let release!:(bytes:ArrayBuffer)=>void;
 (global.fetch as jest.Mock).mockResolvedValueOnce({arrayBuffer:()=>new Promise<ArrayBuffer>(resolve=>{release=resolve;})});
 const first=stageNativeWardrobeImage(owner,id,'file:///cache/first.jpg');
 while(!release)await new Promise(resolve=>setTimeout(resolve,0));
 const second=stageNativeWardrobeImage(owner,next,'file:///cache/second.jpg');
 const completed=Promise.allSettled([first,second]);authEvents.emitLogout();authEvents.emitLogin(other);
 release(new Uint8Array([255,216,255,217]).buffer);
 const results=await completed;expect(results.every(result=>result.status==='rejected')).toBe(true);
 expect(FileSystem.getInfoAsync).not.toHaveBeenCalledWith('file:///cache/second.jpg');
 expect(global.fetch).toHaveBeenCalledTimes(1);
});
test('a body-read rejection does not prevent the next staged image from committing',async()=>{
 (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('synthetic read failure'));
 const results=await Promise.allSettled([stageNativeWardrobeImage(owner,id,'file:///cache/first.jpg'),stageNativeWardrobeImage(owner,next,'file:///cache/second.jpg')]);
 expect(results[0]).toMatchObject({status:'rejected'});expect(results[1]).toMatchObject({status:'fulfilled',value:expect.stringContaining(next)});
});
