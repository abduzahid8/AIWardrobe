import { authEvents } from '../../store/authEvents';
import AsyncStorage from '@react-native-async-storage/async-storage';
import useSavedShopStore, {resetSavedShopState,flushSavedShopPersistence} from '../../store/savedShopStore';
import type {ShopCatalogItem} from '../../src/types/shop';
const item=(id:string)=>({id} as ShopCatalogItem);
beforeEach(async()=>{await flushSavedShopPersistence();jest.restoreAllMocks();jest.clearAllMocks();await AsyncStorage.clear();authEvents.emitLogin('owner-a');resetSavedShopState();await flushSavedShopPersistence();});

test('old in-flight write, account cleanup and new write commit in order',async()=>{
 const set=AsyncStorage.setItem as jest.Mock, original=set.getMockImplementation()!;
 let finish!:()=>void;let started!:()=>void;const active=new Promise<void>(resolve=>{started=resolve;});
 set.mockImplementationOnce((key:string,value:string)=>new Promise<void>(resolve=>{finish=()=>{Promise.resolve(original(key,value)).then(()=>resolve());};started();}));
 const old=useSavedShopStore.getState().toggleSave(item('old-owner'));
 await active;resetSavedShopState();
 const fresh=useSavedShopStore.getState().toggleSave(item('new-owner'));
 const hydration=useSavedShopStore.getState().hydrate();
 finish();await Promise.all([old,fresh,hydration]);
 expect(useSavedShopStore.getState().items.map(x=>x.id)).toEqual(['new-owner']);
 expect(JSON.parse((await AsyncStorage.getItem('savedInspo'))!).items).toEqual([{id:'new-owner'}]);
});

test('a failed account cleanup blocks reading old disk data until a fresh write replaces it',async()=>{
 await AsyncStorage.setItem('savedInspo',JSON.stringify({version:1,ownerId:'owner-a',items:[item('private-old-owner')]}));
 (AsyncStorage.removeItem as jest.Mock).mockRejectedValueOnce(new Error('disk error'));
 resetSavedShopState();await flushSavedShopPersistence();
 await useSavedShopStore.getState().hydrate();
 expect(useSavedShopStore.getState().items).toEqual([]);
 expect(await AsyncStorage.getItem('savedInspo')).toContain('private-old-owner');
 await useSavedShopStore.getState().toggleSave(item('new-owner'));
 await useSavedShopStore.getState().hydrate();
 expect(useSavedShopStore.getState().items.map(x=>x.id)).toEqual(['new-owner']);
});

test('a save during initial hydration preserves earlier same-account likes',async()=>{
 let finish!:(value:string)=>void;
 (AsyncStorage.getItem as jest.Mock).mockImplementationOnce(()=>new Promise<string>(resolve=>{finish=resolve;}));
 const hydration=useSavedShopStore.getState().hydrate();await Promise.resolve();
 const save=useSavedShopStore.getState().toggleSave(item('new-like'));
 finish(JSON.stringify({version:1,ownerId:'owner-a',items:[item('older-like')]}));
 await Promise.all([hydration,save]);
 expect(useSavedShopStore.getState().items.map(x=>x.id)).toEqual(['older-like','new-like']);
});

test('a failed save rejects and retains the last confirmed list',async()=>{
 await useSavedShopStore.getState().toggleSave(item('existing-like'));
 (AsyncStorage.setItem as jest.Mock).mockRejectedValueOnce(new Error('disk full'));
 await expect(useSavedShopStore.getState().toggleSave(item('new-like'))).rejects.toThrow('disk full');
 await useSavedShopStore.getState().hydrate();
 expect(useSavedShopStore.getState().items.map(x=>x.id)).toEqual(['existing-like']);
 expect(JSON.parse((await AsyncStorage.getItem('savedInspo'))!).items).toEqual([{id:'existing-like'}]);
});

test('a synchronous account reset during store notification cannot persist the old action as new-owner work',async()=>{
 const unsubscribe=useSavedShopStore.subscribe(state=>{if(state.items.some(x=>x.id==='old-like'))resetSavedShopState();});
 try{await useSavedShopStore.getState().toggleSave(item('old-like'));await flushSavedShopPersistence();
 expect(useSavedShopStore.getState().items).toEqual([]);expect(await AsyncStorage.getItem('savedInspo')).toBeNull();
 }finally{unsubscribe();}
});

test('a new-account save triggered by reset notification is ordered after cleanup',async()=>{
 await useSavedShopStore.getState().toggleSave(item('old-like'));
 let fresh:Promise<boolean>|undefined;
 const unsubscribe=useSavedShopStore.subscribe(state=>{if(!state.hydrated&&!state.items.length)fresh=state.toggleSave(item('new-like'));});
 try{resetSavedShopState();await fresh;await flushSavedShopPersistence();
 expect(JSON.parse((await AsyncStorage.getItem('savedInspo'))!).items).toEqual([{id:'new-like'}]);
 }finally{unsubscribe();}
});

test('hydration triggered by reset notification cannot read the pre-cleanup cache',async()=>{
 await useSavedShopStore.getState().toggleSave(item('old-like'));
 let hydration:Promise<void>|undefined;
 const unsubscribe=useSavedShopStore.subscribe(state=>{if(!state.hydrated&&!state.items.length)hydration=state.hydrate();});
 try{resetSavedShopState();await hydration;
 expect(useSavedShopStore.getState().items).toEqual([]);
 expect(await AsyncStorage.getItem('savedInspo')).toBeNull();
 }finally{unsubscribe();}
});

test('persisted records identify their owner and survive a same-account auth refresh', async () => {
 await useSavedShopStore.getState().toggleSave(item('mine'));
 expect(JSON.parse((await AsyncStorage.getItem('savedInspo'))!)).toEqual({version:1,ownerId:'owner-a',items:[{id:'mine'}]});
 authEvents.emitLogin('owner-a');
 expect(useSavedShopStore.getState().items.map(x=>x.id)).toEqual(['mine']);
});

test('a direct SDK account switch clears visible likes and rejects the previous disk owner', async () => {
 await useSavedShopStore.getState().toggleSave(item('a-private'));
 authEvents.emitLogin('owner-b');
 expect(useSavedShopStore.getState().items).toEqual([]);
 await useSavedShopStore.getState().hydrate();
 expect(useSavedShopStore.getState().items).toEqual([]);
 await useSavedShopStore.getState().toggleSave(item('b-private'));
 expect(JSON.parse((await AsyncStorage.getItem('savedInspo'))!).ownerId).toBe('owner-b');
});

test('a foreign owner record left by interrupted cleanup is never hydrated', async () => {
 // Disk contains a valid old record; there is no in-process failure flag to rely on.
 await AsyncStorage.setItem('savedInspo',JSON.stringify({version:1,ownerId:'previous-owner',items:[item('private')]}));
 await useSavedShopStore.getState().hydrate();
 expect(useSavedShopStore.getState().items).toEqual([]);
});

test('unowned legacy arrays are not attributed to the current login', async () => {
 await AsyncStorage.setItem('savedInspo',JSON.stringify([item('unknown-owner')]));
 await useSavedShopStore.getState().hydrate();
 expect(useSavedShopStore.getState().items).toEqual([]);
});

test('a tagged same-owner record loads from disk without an earlier in-memory save', async () => {
 await AsyncStorage.setItem('savedInspo',JSON.stringify({version:1,ownerId:'owner-a',items:[item('mine')]}));
 await useSavedShopStore.getState().hydrate();
 expect(useSavedShopStore.getState().items.map(x=>x.id)).toEqual(['mine']);
});

test('logout immediately clears visible likes and anonymous actions cannot write them back', async () => {
 await useSavedShopStore.getState().toggleSave(item('mine'));
 authEvents.emitLogout();
 expect(useSavedShopStore.getState().items).toEqual([]);
 const before=await AsyncStorage.getItem('savedInspo');
 expect(await useSavedShopStore.getState().toggleSave(item('anonymous'))).toBe(false);
 await useSavedShopStore.getState().setItems([item('anonymous')]);
 await useSavedShopStore.getState().hydrate();
 expect(useSavedShopStore.getState().items).toEqual([]);
 expect(await AsyncStorage.getItem('savedInspo')).toBe(before);
});

test('an old write finishing after direct account switch cannot be read by the new owner', async () => {
 const write=AsyncStorage.setItem as jest.Mock, original=write.getMockImplementation()!;
 let finish!:()=>void, started!:()=>void;
 const active=new Promise<void>(resolve=>{started=resolve;});
 write.mockImplementationOnce((key:string,value:string)=>new Promise<void>(resolve=>{finish=()=>{Promise.resolve(original(key,value)).then(()=>resolve());};started();}));
 const old=useSavedShopStore.getState().toggleSave(item('private-a'));await active;
 authEvents.emitLogin('owner-b');
 const hydration=useSavedShopStore.getState().hydrate();finish();await Promise.all([old,hydration]);
 expect(JSON.parse((await AsyncStorage.getItem('savedInspo'))!).ownerId).toBe('owner-a');
 expect(useSavedShopStore.getState().items).toEqual([]);
});

test('a fresh runtime rejects leftover foreign disk data before and after establishing its identity', async () => {
 let fresh!: typeof useSavedShopStore;
 let events!: typeof authEvents;
 let disk!: typeof AsyncStorage;
 jest.isolateModules(()=>{
   fresh=require('../../store/savedShopStore').default;
   events=require('../../store/authEvents').authEvents;
   const storage=require('@react-native-async-storage/async-storage');
   disk=storage.default ?? storage;
 });
 await disk.setItem('savedInspo',JSON.stringify({version:1,ownerId:'old-runtime-owner',items:[item('private')]}));
 await fresh.getState().hydrate();
 expect(fresh.getState().items).toEqual([]);
 expect(events.getUserId()).toBeNull();
 events.emitLogin('fresh-runtime-owner');
 await fresh.getState().hydrate();
 expect(fresh.getState().items).toEqual([]);
});

test('concurrent first saves merge with persisted likes and each other', async () => {
 await AsyncStorage.setItem('savedInspo',JSON.stringify({version:1,ownerId:'owner-a',items:[item('earlier')]}));
 await Promise.all([useSavedShopStore.getState().toggleSave(item('first')),useSavedShopStore.getState().toggleSave(item('second'))]);
 expect(useSavedShopStore.getState().items.map(x=>x.id)).toEqual(['earlier','first','second']);
 expect(JSON.parse((await AsyncStorage.getItem('savedInspo'))!).items.map((x:{id:string})=>x.id)).toEqual(['earlier','first','second']);
});

test('a first toggle removes an already persisted like rather than adding it again', async () => {
 await AsyncStorage.setItem('savedInspo',JSON.stringify({version:1,ownerId:'owner-a',items:[item('existing'),item('keep')]}));
 expect(await useSavedShopStore.getState().toggleSave(item('existing'))).toBe(false);
 expect(useSavedShopStore.getState().items.map(x=>x.id)).toEqual(['keep']);
});

test('a first removal preserves other persisted likes', async () => {
 await AsyncStorage.setItem('savedInspo',JSON.stringify({version:1,ownerId:'owner-a',items:[item('remove'),item('keep')]}));
 await useSavedShopStore.getState().removeSave('remove');
 expect(useSavedShopStore.getState().items.map(x=>x.id)).toEqual(['keep']);
});

test('failed initial storage reads cannot overwrite an unread cache and can be retried', async () => {
 const record=JSON.stringify({version:1,ownerId:'owner-a',items:[item('existing')]});
 await AsyncStorage.setItem('savedInspo',record);
 (AsyncStorage.getItem as jest.Mock).mockRejectedValueOnce(new Error('read unavailable'));
 await expect(useSavedShopStore.getState().toggleSave(item('new'))).rejects.toThrow('could not be loaded');
 expect(await AsyncStorage.getItem('savedInspo')).toBe(record);
 expect(useSavedShopStore.getState().hydrated).toBe(false);
 await useSavedShopStore.getState().toggleSave(item('new'));
 expect(useSavedShopStore.getState().items.map(x=>x.id)).toEqual(['existing','new']);
});

test('an account switch during the initial read cancels the waiting old-account mutation', async () => {
 let finish!:(value:string)=>void;
 (AsyncStorage.getItem as jest.Mock).mockImplementationOnce(()=>new Promise<string>(resolve=>{finish=resolve;}));
 const save=useSavedShopStore.getState().toggleSave(item('old-action'));await Promise.resolve();
 authEvents.emitLogin('owner-b');
 finish(JSON.stringify({version:1,ownerId:'owner-a',items:[item('old-cache')]}));
 expect(await save).toBe(false);
 expect(useSavedShopStore.getState().items).toEqual([]);
 expect(await AsyncStorage.getItem('savedInspo')).toBeNull();
});

test('malformed persisted JSON is retained when the first removal cannot safely read it', async () => {
 await AsyncStorage.setItem('savedInspo','{broken');
 await expect(useSavedShopStore.getState().removeSave('anything')).rejects.toThrow('could not be loaded');
 expect(await AsyncStorage.getItem('savedInspo')).toBe('{broken');
});

test('an explicit replacement wins over an earlier pending initial hydration', async () => {
 let finish!:(value:string)=>void;
 (AsyncStorage.getItem as jest.Mock).mockImplementationOnce(()=>new Promise<string>(resolve=>{finish=resolve;}));
 const hydration=useSavedShopStore.getState().hydrate();await Promise.resolve();
 await useSavedShopStore.getState().setItems([item('replacement')]);
 finish(JSON.stringify({version:1,ownerId:'owner-a',items:[item('older')]}));await hydration;
 expect(useSavedShopStore.getState().items.map(x=>x.id)).toEqual(['replacement']);
});

test('visible likes remain confirmed until a pending disk write succeeds', async () => {
 await useSavedShopStore.getState().toggleSave(item('existing'));
 const write=AsyncStorage.setItem as jest.Mock, original=write.getMockImplementation()!;
 let finish!:()=>void, started!:()=>void;
 const active=new Promise<void>(resolve=>{started=resolve;});
 write.mockImplementationOnce((key:string,value:string)=>new Promise<void>(resolve=>{finish=()=>{Promise.resolve(original(key,value)).then(()=>resolve());};started();}));
 const save=useSavedShopStore.getState().toggleSave(item('pending'));await active;
 expect(useSavedShopStore.getState().items.map(x=>x.id)).toEqual(['existing']);
 finish();expect(await save).toBe(true);
 expect(useSavedShopStore.getState().items.map(x=>x.id)).toEqual(['existing','pending']);
});

test('a failed removal does not hide the persisted saved item', async () => {
 await useSavedShopStore.getState().setItems([item('keep'),item('remove')]);
 (AsyncStorage.setItem as jest.Mock).mockRejectedValueOnce(new Error('disk full'));
 await expect(useSavedShopStore.getState().removeSave('remove')).rejects.toThrow('disk full');
 expect(useSavedShopStore.getState().items.map(x=>x.id)).toEqual(['keep','remove']);
 expect(JSON.parse((await AsyncStorage.getItem('savedInspo'))!).items.map((x:{id:string})=>x.id)).toEqual(['keep','remove']);
});

test('a mutation after a failed concurrent save derives from the last confirmed list', async () => {
 await useSavedShopStore.getState().toggleSave(item('existing'));
 (AsyncStorage.setItem as jest.Mock).mockRejectedValueOnce(new Error('disk full'));
 const results=await Promise.allSettled([useSavedShopStore.getState().toggleSave(item('failed')),useSavedShopStore.getState().toggleSave(item('succeeded'))]);
 expect(results.map(x=>x.status)).toEqual(['rejected','fulfilled']);
 expect(useSavedShopStore.getState().items.map(x=>x.id)).toEqual(['existing','succeeded']);
 expect(JSON.parse((await AsyncStorage.getItem('savedInspo'))!).items.map((x:{id:string})=>x.id)).toEqual(['existing','succeeded']);
});

test('overlapping toggles of the same item commit in order without losing unrelated likes', async () => {
 await useSavedShopStore.getState().toggleSave(item('keep'));
 expect(await Promise.all([useSavedShopStore.getState().toggleSave(item('twice')),useSavedShopStore.getState().toggleSave(item('twice'))])).toEqual([true,false]);
 expect(useSavedShopStore.getState().items.map(x=>x.id)).toEqual(['keep']);
});

test('failed replacement and clear both reject without discarding confirmed likes', async () => {
 await useSavedShopStore.getState().toggleSave(item('keep'));
 (AsyncStorage.setItem as jest.Mock).mockRejectedValueOnce(new Error('disk full'));
 await expect(useSavedShopStore.getState().setItems([item('replacement')])).rejects.toThrow('disk full');
 (AsyncStorage.setItem as jest.Mock).mockRejectedValueOnce(new Error('disk full'));
 await expect(useSavedShopStore.getState().clearAll()).rejects.toThrow('disk full');
 expect(useSavedShopStore.getState().items.map(x=>x.id)).toEqual(['keep']);
});
