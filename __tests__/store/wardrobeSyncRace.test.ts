const mockReviewAck = jest.fn(), mockFindConfirmedReviews = jest.fn();
jest.mock('../../src/services/scanReviews',()=>({acknowledgeSavedScanReviewItems:(...args:unknown[])=>mockReviewAck(...args),findConfirmedScanReviewItems:(...args:unknown[])=>mockFindConfirmedReviews(...args)}));
import { createStore } from 'zustand/vanilla';
import { createItemsSlice } from '../../store/slices/wardrobeItemsSlice';
jest.mock('../../store/subscriptionStore', () => ({ __esModule: true, default: { getState: () => ({ tier: 'premium', effectiveTier: 'premium', refreshTimeState: () => {} }) }, TIER_FEATURES: { premium: { wardrobeItems: -1 } } }));
import { createSyncSlice, invalidateRehydrateState } from '../../store/slices/wardrobeSyncSlice';
import { authEvents } from '../../store/authEvents';
import { processPendingActions, fetchItemsFromServer, fetchWearLogsFromServer } from '../../store/wardrobeSyncService';
import type { WardrobeState } from '../../store/wardrobeStore';
import type { ClothingItem } from '../../src/types/domain';

let mockUserId = 'user-a';
jest.mock('../../store/auth', () => ({ __esModule: true, default: { getState: () => ({ user: { id: mockUserId } }) } }));
jest.mock('../../lib/supabase', () => ({ supabase: {} }));
jest.mock('../../src/lib/api', () => ({ wardrobeApi: { list: jest.fn() }, wearLogApi: {} }));
jest.mock('../../store/wardrobeSyncService', () => ({ processPendingActions: jest.fn(), fetchItemsFromServer: jest.fn(), fetchWearLogsFromServer: jest.fn(), reconcileCloudItems: jest.requireActual('../../store/wardrobeSyncService').reconcileCloudItems }));

function fixture() {
  const original = { id: 'edited', name: 'before' } as ClothingItem;
  const removed = { id: 'removed' } as ClothingItem;
  const untouched = { id: 'untouched', name: 'before' } as ClothingItem;
  const state = createStore<WardrobeState>()((set, get, store) => ({
    ...createItemsSlice(set, get, store), ...createSyncSlice(set, get, store), wearLogs: [], items: [original, removed, untouched],
    pendingActions: [{ id: 'old-action', type: 'add_item', payload: {}, createdAt: '' }],
  } as unknown as WardrobeState));
  const persistSnapshot = jest.fn().mockResolvedValue(undefined);
  Object.assign(state, { persist: { getOptions: () => ({ name: 'test-wardrobe', version: 1,
    storage: { setItem: persistSnapshot }, partialize: (value: WardrobeState) => ({ pendingActions: value.pendingActions, items: value.items, wardrobeCapacityIssue: value.wardrobeCapacityIssue, pendingReviewAcknowledgements: value.pendingReviewAcknowledgements }) }) } });
  return { state, original, untouched, persistSnapshot };
}

beforeEach(() => {
  jest.clearAllMocks();
  mockReviewAck.mockReset().mockResolvedValue(undefined);
  mockFindConfirmedReviews.mockReset().mockResolvedValue([]);
  invalidateRehydrateState();
  (processPendingActions as jest.Mock).mockResolvedValue({ processedIds: [], updatedItems: [] });
  mockUserId = 'user-a';
  authEvents.emitLogin(mockUserId);
});

test('cloud writes wait for confirmed persistence of the queue snapshot', async () => {
  const { state, persistSnapshot } = fixture();
  let persisted!: () => void;
  persistSnapshot.mockImplementationOnce(() => new Promise<void>(resolve => { persisted = resolve; }));
  const pending = state.getState().syncToServer();
  expect(persistSnapshot).toHaveBeenCalledWith('test-wardrobe', expect.objectContaining({ state: expect.objectContaining({
    pendingActions: expect.arrayContaining([expect.objectContaining({ id: 'old-action' })]),
  }), version: 1 }));
  expect(processPendingActions).not.toHaveBeenCalled();
  persisted(); await pending;
  expect(processPendingActions).toHaveBeenCalledTimes(1);
});

test('failed local queue persistence prevents cloud mutation and retains work', async () => {
  const { state, persistSnapshot } = fixture();
  persistSnapshot.mockRejectedValueOnce(new Error('storage full'));
  const error = jest.spyOn(console, 'error').mockImplementation(() => undefined);
  try {
    await state.getState().syncToServer();
    expect(processPendingActions).not.toHaveBeenCalled();
    expect(state.getState().pendingActions.map(action => action.id)).toEqual(['old-action']);
    expect(state.getState().isSyncing).toBe(false);
  } finally { error.mockRestore(); }
});

test('sync preserves edits, added/deleted items and new queued actions while applying unchanged-item server conflicts', async () => {
  const { state, untouched } = fixture();
  let finish!: (value: unknown) => void;
  let started!: () => void;
  const cloudStarted = new Promise<void>(resolve => { started = resolve; });
  (processPendingActions as jest.Mock).mockImplementationOnce(() => new Promise(resolve => { finish = resolve; started(); }));
  const pending = state.getState().syncToServer();
  await cloudStarted;
  state.setState({ items: [{ id: 'new' } as ClothingItem, { id: 'edited', name: 'local edit' } as ClothingItem, untouched],
    pendingActions: [...state.getState().pendingActions, { id: 'new-action', type: 'remove_item', payload: {}, createdAt: '' }] });
  finish({ processedIds: ['old-action'], updatedItems: [{ id: 'edited', name: 'old server edit' }, { id: 'removed' }, { id: 'untouched', name: 'server conflict' }] });
  await pending;
  expect(state.getState().items.map(item => [item.id, item.name])).toEqual([['new', undefined], ['edited', 'local edit'], ['untouched', 'server conflict']]);
  expect(state.getState().pendingActions.map(action => action.id)).toEqual(['new-action']);
  expect(processPendingActions).toHaveBeenCalledTimes(2); // One follow-up, no failed-action retry spin.
});

test('overlapping sync calls share the same network operation', async () => {
  const { state } = fixture();
  let finish!: (value: unknown) => void;
  let started!: () => void;
  const cloudStarted = new Promise<void>(resolve => { started = resolve; });
  (processPendingActions as jest.Mock).mockImplementationOnce(() => new Promise(resolve => { finish = resolve; started(); }));
  const first = state.getState().syncToServer(), second = state.getState().syncToServer();
  await cloudStarted;
  expect(processPendingActions).toHaveBeenCalledTimes(1);
  finish({ processedIds: [], updatedItems: state.getState().items });
  await Promise.all([first, second]);
  expect(state.getState().isSyncing).toBe(false);
});

test('old account sync result cannot replace a newly signed-in account state', async () => {
  const { state } = fixture();
  let finish!: (value: unknown) => void;
  let started!: () => void;
  const cloudStarted = new Promise<void>(resolve => { started = resolve; });
  (processPendingActions as jest.Mock).mockImplementationOnce(() => new Promise(resolve => { finish = resolve; started(); }));
  const pending = state.getState().syncToServer();
  await cloudStarted;
  authEvents.emitLogout(); mockUserId = 'user-b'; authEvents.emitLogin(mockUserId);
  state.setState({ items: [{ id: 'user-b-item' } as ClothingItem], pendingActions: [], isSyncing: false });
  finish({ processedIds: ['old-action'], updatedItems: [{ id: 'private-user-a-item' }] });
  await pending;
  expect(state.getState().items.map(item => item.id)).toEqual(['user-b-item']);
  expect(state.getState().lastSyncedAt).toBeNull();
});


test('a late wardrobe fetch cannot publish the previous account data', async () => {
  const { state } = fixture();
  let finish!: (value: unknown) => void;
  (fetchItemsFromServer as jest.Mock).mockImplementationOnce(() => new Promise(resolve => { finish = resolve; }));
  const pending = state.getState().fetchItems();
  authEvents.emitLogout(); mockUserId = 'user-b'; authEvents.emitLogin(mockUserId);
  state.setState({ items: [{ id: 'user-b-item' } as ClothingItem], isLoading: false });
  finish([{ id: 'private-user-a-item' }]);
  await pending;
  expect(state.getState().items.map(item => item.id)).toEqual(['user-b-item']);
  expect(state.getState().isLoading).toBe(false);
});

test('rehydration refreshes remote wear history even when local history exists', async () => {
  const { state } = fixture();
  const oldLog = { id: 'old', date: '2026-10-14', itemIds: [] };
  const cloudLog = { id: 'cloud', date: '2026-10-15', itemIds: [] };
  state.setState({ pendingActions: [], wearLogs: [oldLog as never] });
  (fetchItemsFromServer as jest.Mock).mockResolvedValueOnce([{ id: 'cloud-item' }]);
  (fetchWearLogsFromServer as jest.Mock).mockResolvedValueOnce([oldLog, cloudLog]);
  await state.getState().rehydrateFromCloud(true);
  expect(fetchWearLogsFromServer).toHaveBeenCalledTimes(1);
  expect(state.getState().wearLogs.map(log => log.id)).toEqual(['cloud', 'old']);
  expect(state.getState().items.map(item => item.id)).toEqual(['cloud-item']);
  expect(state.getState().isLoading).toBe(false);
});

test('account change during storage hydration cannot start an old-account refresh', async () => {
  const { state } = fixture();
  let finishHydration!: () => void;
  Object.assign(state, { persist: { hasHydrated: () => false, onFinishHydration: (callback: () => void) => {
    finishHydration = callback; return () => undefined;
  } } });
  const pending = state.getState().rehydrateFromCloud(true);
  authEvents.emitLogout(); mockUserId = 'user-b'; authEvents.emitLogin(mockUserId);
  state.setState({ items: [{ id: 'user-b-item' } as ClothingItem], isLoading: false, pendingActions: [] });
  finishHydration();
  await pending;
  expect(fetchItemsFromServer).not.toHaveBeenCalled();
  expect(state.getState().isLoading).toBe(false);
  expect(state.getState().items.map(item => item.id)).toEqual(['user-b-item']);
});

test('recent cloud refresh cannot suppress retrying pending offline changes', async () => {
  const { state } = fixture();
  state.setState({ pendingActions: [] });
  (fetchItemsFromServer as jest.Mock).mockResolvedValue([{ id: 'cloud-item' }]);
  (fetchWearLogsFromServer as jest.Mock).mockResolvedValue([]);
  await state.getState().rehydrateFromCloud(true);
  jest.clearAllMocks();
  state.setState({ pendingActions: [{ id: 'offline-wear', type: 'add_wear_log', payload: {}, createdAt: '' }] });
  (processPendingActions as jest.Mock).mockResolvedValueOnce({ processedIds: ['offline-wear'], updatedItems: state.getState().items });
  await state.getState().rehydrateFromCloud();
  expect(processPendingActions).toHaveBeenCalledTimes(1);
  expect(state.getState().pendingActions).toEqual([]);
});

test('recent cloud refresh still skips redundant reads when no work is pending', async () => {
  const { state } = fixture();
  state.setState({ pendingActions: [] });
  (fetchItemsFromServer as jest.Mock).mockResolvedValue([{ id: 'cloud-item' }]);
  (fetchWearLogsFromServer as jest.Mock).mockResolvedValue([]);
  await state.getState().rehydrateFromCloud(true);
  jest.clearAllMocks();
  await state.getState().rehydrateFromCloud();
  expect(processPendingActions).not.toHaveBeenCalled();
  expect(fetchItemsFromServer).not.toHaveBeenCalled();
});


test('account change while persisting prevents the old-account cloud write', async () => {
  const { state, persistSnapshot } = fixture();
  let persisted!: () => void;
  persistSnapshot.mockImplementationOnce(() => new Promise<void>(resolve => { persisted = resolve; }));
  const pending = state.getState().syncToServer();
  authEvents.emitLogout(); mockUserId = 'user-b'; authEvents.emitLogin(mockUserId);
  state.setState({ pendingActions: [], items: [{ id: 'new-owner' } as ClothingItem], isSyncing: false });
  persisted(); await pending;
  expect(processPendingActions).not.toHaveBeenCalled();
  expect(state.getState().items[0].id).toBe('new-owner');
});

test('missing persistence fails closed and keeps the pending action', async () => {
  const { state } = fixture();
  Object.assign(state, { persist: undefined });
  const error = jest.spyOn(console, 'error').mockImplementation(() => undefined);
  try {
    await state.getState().syncToServer();
    expect(processPendingActions).not.toHaveBeenCalled();
    expect(state.getState().pendingActions).toHaveLength(1);
    expect(state.getState().isSyncing).toBe(false);
  } finally { error.mockRestore(); }
});


test('initial garment creation reaches cloud only after its queued snapshot is durable', async () => {
  const { state, persistSnapshot } = fixture();
  state.setState({ pendingActions: [] });
  let persisted!: () => void;
  persistSnapshot.mockImplementationOnce(() => new Promise<void>(resolve => { persisted = resolve; }));
  const pending = state.getState().addItem({ category: 'top', imageUrl: 'https://example.com/shirt.jpg' } as never);
  const item = state.getState().items[0];
  expect(persistSnapshot).toHaveBeenCalledWith('test-wardrobe', expect.objectContaining({ state: expect.objectContaining({
    items: expect.arrayContaining([expect.objectContaining({ id: item.id })]),
    pendingActions: expect.arrayContaining([expect.objectContaining({ type: 'add_item', payload: expect.objectContaining({ id: item.id }) })]),
  }) }));
  expect(processPendingActions).not.toHaveBeenCalled();
  persisted(); await pending;
  expect(processPendingActions).toHaveBeenCalledWith(expect.arrayContaining([expect.objectContaining({ type: 'add_item' })]), expect.any(Array), 'user-a', expect.any(Function));
});


test('sync completion waits for durable cloud acknowledgement', async () => {
  const { state, persistSnapshot } = fixture();
  let acknowledged!: () => void;
  (processPendingActions as jest.Mock).mockResolvedValueOnce({ processedIds: ['old-action'], updatedItems: state.getState().items });
  let secondWrite!: () => void;
  const ackStarted = new Promise<void>(resolve => { secondWrite = resolve; });
  persistSnapshot.mockImplementationOnce(async () => undefined).mockImplementationOnce(() => { secondWrite(); return new Promise<void>(resolve => { acknowledged = resolve; }); });
  let settled = false;
  const pending = state.getState().syncToServer().then(() => { settled = true; });
  await ackStarted;
  expect(state.getState().pendingActions).toEqual([]);
  expect(settled).toBe(false);
  acknowledged(); await pending;
  expect(settled).toBe(true);
});

test('failed acknowledgement persistence restores stable actions for a later retry', async () => {
  const { state, persistSnapshot } = fixture();
  persistSnapshot.mockResolvedValueOnce(undefined).mockRejectedValueOnce(new Error('disk full'));
  (processPendingActions as jest.Mock).mockResolvedValueOnce({ processedIds: ['old-action'], updatedItems: state.getState().items });
  const error = jest.spyOn(console, 'error').mockImplementation(() => undefined);
  try {
    await state.getState().syncToServer();
    expect(state.getState().pendingActions.map(action => action.id)).toEqual(['old-action']);
    expect(processPendingActions).toHaveBeenCalledTimes(1);
    (processPendingActions as jest.Mock).mockResolvedValueOnce({ processedIds: ['old-action'], updatedItems: state.getState().items });
    await state.getState().syncToServer();
    expect(state.getState().pendingActions).toEqual([]);
    expect(processPendingActions).toHaveBeenCalledTimes(2);
  } finally { error.mockRestore(); }
});


test('failed acknowledgement keeps restored creation before newer queued edits', async () => {
  const { state, persistSnapshot } = fixture();
  persistSnapshot.mockResolvedValueOnce(undefined).mockImplementationOnce(async () => {
    state.setState({ pendingActions: [{ id: 'new-edit', type: 'update_item', payload: { itemId: 'edited', isFavorite: true }, createdAt: '' }] });
    throw new Error('disk full');
  });
  (processPendingActions as jest.Mock).mockResolvedValueOnce({ processedIds: ['old-action'], updatedItems: state.getState().items });
  const error = jest.spyOn(console, 'error').mockImplementation(() => undefined);
  try {
    await state.getState().syncToServer();
    expect(state.getState().pendingActions.map(action => action.id)).toEqual(['old-action', 'new-edit']);
    expect((processPendingActions as jest.Mock).mock.calls[1][0].map((action: { id: string }) => action.id)).toEqual(['old-action', 'new-edit']);
  } finally { error.mockRestore(); }
});

test('failed old-account acknowledgement cannot restore work into the next account', async () => {
  const { state, persistSnapshot } = fixture();
  persistSnapshot.mockResolvedValueOnce(undefined).mockImplementationOnce(async () => {
    authEvents.emitLogout(); mockUserId = 'user-b'; authEvents.emitLogin(mockUserId);
    state.setState({ pendingActions: [], items: [{ id: 'new-owner' } as ClothingItem], isSyncing: false });
    throw new Error('disk full');
  });
  (processPendingActions as jest.Mock).mockResolvedValueOnce({ processedIds: ['old-action'], updatedItems: state.getState().items });
  const error = jest.spyOn(console, 'error').mockImplementation(() => undefined);
  try {
    await state.getState().syncToServer();
    expect(state.getState().pendingActions).toEqual([]);
    expect(state.getState().items[0].id).toBe('new-owner');
  } finally { error.mockRestore(); }
});


test('uploaded references replace unchanged photo fields and outfit snapshots while preserving later edits', async () => {
  const { state } = fixture();
  const old = { id: 'edited', imageUrl: 'file:///cache/photo.jpg', thumbnailUrl: 'file:///cache/thumb.jpg', name: 'before', isFavorite: false } as ClothingItem;
  state.setState({ items: [old], outfits: [{ id: 'look', previewImageUrl: old.imageUrl, snapshotItems: [{ id: old.id, image: old.imageUrl }, { id: 'shop', image: 'https://shop.test/photo.jpg' }] }] as never });
  (processPendingActions as jest.Mock).mockImplementationOnce(async () => {
    state.setState({ items: [{ ...old, isFavorite: true, name: 'new name' }] });
    return { processedIds: ['old-action'], updatedItems: [{ ...old, imageUrl: 'https://storage.test/original.jpg', thumbnailUrl: 'https://storage.test/thumbnail.jpg' }] };
  });
  await state.getState().syncToServer();
  expect(state.getState().items[0]).toMatchObject({ name: 'new name', isFavorite: true, imageUrl: 'https://storage.test/original.jpg', thumbnailUrl: 'https://storage.test/thumbnail.jpg' });
  expect(state.getState().outfits[0]).toMatchObject({ previewImageUrl: 'https://storage.test/original.jpg', snapshotItems: [{ id: 'edited', image: 'https://storage.test/original.jpg' }, { id: 'shop', image: 'https://shop.test/photo.jpg' }] });
});

test('a newly selected photo is not replaced by an older upload response', async () => {
  const { state } = fixture();
  const old = { id: 'edited', imageUrl: 'file:///cache/old.jpg' } as ClothingItem;
  state.setState({ items: [old] });
  (processPendingActions as jest.Mock).mockImplementationOnce(async () => {
    state.setState({ items: [{ ...old, imageUrl: 'file:///cache/new.jpg' }] });
    return { processedIds: ['old-action'], updatedItems: [{ ...old, imageUrl: 'https://storage.test/old.jpg' }] };
  });
  await state.getState().syncToServer();
  expect(state.getState().items[0].imageUrl).toBe('file:///cache/new.jpg');
});

it.each([false,true])('favorite reconciliation preserves unrelated edits and newer favorite taps (new tap=%s)',async newerFavorite=>{
 const {state}=fixture();const original={id:'edited',name:'before',isFavorite:true} as ClothingItem;
 state.setState({items:[original],pendingActions:[{id:'old-favorite',type:'update_item',payload:{itemId:'edited',isFavorite:true},createdAt:''}]});
 let finish!:(value:unknown)=>void;let started!:()=>void;const ready=new Promise<void>(resolve=>{started=resolve;});
 (processPendingActions as jest.Mock).mockImplementationOnce(()=>new Promise(resolve=>{finish=resolve;started();}));
 const pending=state.getState().syncToServer();await ready;
 state.setState({items:[{...original,name:'local rename'}],pendingActions:[...state.getState().pendingActions,
  newerFavorite?{id:'new-favorite',type:'update_item',payload:{itemId:'edited',isFavorite:true},createdAt:''}:{id:'new-name',type:'add_item',payload:{...original,name:'local rename'},createdAt:''}]});
 finish({processedIds:['old-favorite'],updatedItems:[{...original,isFavorite:false}]});await pending;
 expect(state.getState().items[0].name).toBe('local rename');
 expect(state.getState().items[0].isFavorite).toBe(newerFavorite);
 expect(state.getState().pendingActions.map(action=>action.id)).toEqual([newerFavorite?'new-favorite':'new-name']);
 if(!newerFavorite)expect(state.getState().pendingActions[0].payload.isFavorite).toBe(false);
});

test('an unconfirmed favorite intent survives a confirmed older item upload',async()=>{
 const {state}=fixture();const original={id:'edited',isFavorite:true} as ClothingItem;
 state.setState({items:[original],pendingActions:[
  {id:'upload',type:'add_item',payload:{...original,isFavorite:false},createdAt:''},
  {id:'favorite',type:'update_item',payload:{itemId:'edited',isFavorite:true},createdAt:''},
 ]});
 (processPendingActions as jest.Mock).mockResolvedValueOnce({processedIds:['upload'],updatedItems:[{...original,isFavorite:false}]});
 await state.getState().syncToServer();
 expect(state.getState().items[0].isFavorite).toBe(true);
 expect(state.getState().pendingActions.map(action=>action.id)).toEqual(['favorite']);
});

it.each(['success','checkpoint failure','acknowledgement failure'])('favorite version persistence ordering: %s',async mode=>{
 const {state,persistSnapshot}=fixture();state.setState({pendingActions:[{id:'favorite',type:'update_item',payload:{itemId:'edited',isFavorite:true},createdAt:''}]});
 const mutation=jest.fn();let release!:()=>void;let reached!:()=>void;const ready=new Promise<void>(resolve=>{reached=resolve;});
 persistSnapshot.mockResolvedValueOnce(undefined);
 if(mode==='success')persistSnapshot.mockImplementationOnce(()=>new Promise<void>(resolve=>{release=resolve;}));
 else if(mode==='checkpoint failure')persistSnapshot.mockRejectedValueOnce(Error('checkpoint disk full'));
 else persistSnapshot.mockResolvedValueOnce(undefined).mockRejectedValueOnce(Error('ack disk full'));
 (processPendingActions as jest.Mock).mockImplementationOnce(async(actions,items,_owner,checkpoint)=>{
  const work=checkpoint(actions.map((action:any)=>({...action,payload:{...action.payload,favoriteVersion:'2026-10-21T00:00:00Z'}})));reached();await work;mutation();return{processedIds:['favorite'],updatedItems:items};
 });
 const error=jest.spyOn(console,'error').mockImplementation(()=>undefined);
 try{
  const pending=state.getState().syncToServer();await ready;
  if(mode==='success'){expect(mutation).not.toHaveBeenCalled();release();}
  await pending;expect(state.getState().isSyncing).toBe(false);
  if(mode==='success'){expect(mutation).toHaveBeenCalledTimes(1);expect(state.getState().pendingActions).toEqual([]);}
  else{expect(mutation).toHaveBeenCalledTimes(mode==='checkpoint failure'?0:1);expect(state.getState().pendingActions[0].payload.favoriteVersion).toBe('2026-10-21T00:00:00Z');}
 }finally{error.mockRestore();}
});

test('capacity rejections persist a retryable notice with the untouched queued item', async () => {
 const {state,persistSnapshot}=fixture();
 (processPendingActions as jest.Mock).mockResolvedValueOnce({processedIds:[],updatedItems:state.getState().items,capacityIssue:{limit:20,actionIds:['old-action']}});
 await state.getState().syncToServer();
 expect(state.getState().wardrobeCapacityIssue).toEqual({limit:20,actionIds:['old-action']});
 expect(state.getState().pendingActions.map(a=>a.id)).toContain('old-action');
 expect(persistSnapshot).toHaveBeenCalledTimes(2);
 expect(persistSnapshot).toHaveBeenLastCalledWith('test-wardrobe',expect.objectContaining({state:expect.objectContaining({wardrobeCapacityIssue:{limit:20,actionIds:['old-action']}})}));
});
test('confirming the blocked actions clears the capacity notice',async()=>{
 const {state}=fixture();state.setState({wardrobeCapacityIssue:{limit:20,actionIds:['old-action']}});
 (processPendingActions as jest.Mock).mockResolvedValueOnce({processedIds:['old-action'],updatedItems:state.getState().items});
 await state.getState().syncToServer();expect(state.getState().wardrobeCapacityIssue).toBeNull();
});
const reviewItem='88888888-8888-4888-8888-888888888888';
function confirmedReviewFixture(){
 const app=fixture();app.state.setState({pendingActions:[{id:'review-add',type:'add_item',payload:{id:reviewItem},createdAt:''}]});
 (processPendingActions as jest.Mock).mockResolvedValueOnce({processedIds:['review-add'],updatedItems:app.state.getState().items});
 return app;
}
test('review cleanup waits for durable cloud acknowledgement and a persisted cleanup journal',async()=>{
 const {state,persistSnapshot}=confirmedReviewFixture();let release!:()=>void,started!:()=>void;
 const writing=new Promise<void>(resolve=>{started=resolve;});
 persistSnapshot.mockResolvedValueOnce(undefined).mockImplementationOnce(()=>{started();return new Promise<void>(resolve=>{release=resolve;});});
 const pending=state.getState().syncToServer();await writing;
 expect(mockReviewAck).not.toHaveBeenCalled();
 expect(state.getState().pendingReviewAcknowledgements).toEqual([reviewItem]);
 release();await pending;
 expect(mockReviewAck).toHaveBeenCalledWith('user-a',[reviewItem]);
 expect(state.getState().pendingReviewAcknowledgements).toEqual([]);
 expect(persistSnapshot).toHaveBeenCalledTimes(3);
});
test('failed review cleanup leaves a durable journal that replays without another cloud write',async()=>{
 const {state,persistSnapshot}=confirmedReviewFixture();mockReviewAck.mockRejectedValueOnce(Error('disk full'));
 const log=jest.spyOn(console,'error').mockImplementation(()=>undefined);
 try{
  await state.getState().syncToServer();expect(state.getState().pendingActions).toEqual([]);
  expect(state.getState().pendingReviewAcknowledgements).toEqual([reviewItem]);
  expect(persistSnapshot).toHaveBeenLastCalledWith('test-wardrobe',expect.objectContaining({state:expect.objectContaining({pendingReviewAcknowledgements:[reviewItem]})}));
  await state.getState().syncToServer();
  expect(processPendingActions).toHaveBeenCalledTimes(1);expect(mockReviewAck).toHaveBeenCalledTimes(2);
  expect(state.getState().pendingReviewAcknowledgements).toEqual([]);
 }finally{log.mockRestore();}
});
test('failed journal-clear persistence restores the cleanup entry for safe replay',async()=>{
 const {state,persistSnapshot}=confirmedReviewFixture();
 persistSnapshot.mockResolvedValueOnce(undefined).mockResolvedValueOnce(undefined).mockRejectedValueOnce(Error('disk unavailable'));
 const log=jest.spyOn(console,'error').mockImplementation(()=>undefined);
 try{await state.getState().syncToServer();expect(state.getState().pendingActions).toEqual([]);expect(state.getState().pendingReviewAcknowledgements).toEqual([reviewItem]);}
 finally{log.mockRestore();}
});
test('an authoritative wardrobe fetch recovers pre-journal confirmed scan candidates',async()=>{
 const {state}=fixture();state.setState({pendingActions:[],items:[]});
 (fetchItemsFromServer as jest.Mock).mockResolvedValueOnce([{id:reviewItem,userId:'user-a'}]);
 (fetchWearLogsFromServer as jest.Mock).mockResolvedValueOnce([]);
 mockFindConfirmedReviews.mockResolvedValueOnce([reviewItem]);
 await state.getState().rehydrateFromCloud(true);
 expect(mockFindConfirmedReviews).toHaveBeenCalledWith('user-a',[reviewItem]);
 expect(mockReviewAck).toHaveBeenCalledWith('user-a',[reviewItem]);
 expect(state.getState().pendingReviewAcknowledgements).toEqual([]);
});
test('a failed cloud fetch cannot acknowledge a review using only cached garment IDs',async()=>{
 const {state}=fixture();state.setState({pendingActions:[],items:[{id:reviewItem} as ClothingItem]});
 (fetchItemsFromServer as jest.Mock).mockResolvedValueOnce(null);
 (fetchWearLogsFromServer as jest.Mock).mockResolvedValueOnce([]);
 await state.getState().rehydrateFromCloud(true);
 expect(mockFindConfirmedReviews).not.toHaveBeenCalled();expect(mockReviewAck).not.toHaveBeenCalled();
});
