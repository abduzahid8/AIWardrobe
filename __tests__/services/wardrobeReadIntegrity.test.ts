import { createClient } from '@supabase/supabase-js';
import { fetchWearLogsFromServer, fetchItemsFromServer, reconcileCloudItems, type PendingAction } from '../../store/wardrobeSyncService';
import type { ClothingItem } from '../../src/types/domain';
import { authEvents } from '../../store/authEvents';
const mockRange = jest.fn();
const mockEq = jest.fn();
const mockOrder = jest.fn();
const mockLt = jest.fn();
const mockAbort = jest.fn();
const mockGetSession = jest.fn();
const mockFrom = jest.fn();
jest.mock('../../lib/supabase', () => ({ supabase: { auth: { getSession: () => mockGetSession() },
  from: (...args: unknown[]) => mockFrom(...args),
} }));
beforeEach(() => {
  jest.resetAllMocks(); authEvents.emitLogin('owner');
  mockFrom.mockReturnValue({ select: () => ({ eq: mockEq }) });
  mockGetSession.mockResolvedValue({ data: { session: { user: { id: 'owner' } } } });
  const chain = { order: mockOrder, range: mockRange, limit: mockRange, lt: mockLt, abortSignal: mockAbort };
  mockAbort.mockReturnValue(chain);
  mockEq.mockReturnValue(chain); mockOrder.mockReturnValue(chain); mockLt.mockReturnValue(chain);
});

test('loads more than 500 owner items using stable ordered pages', async () => {
  const rows = Array.from({ length: 501 }, (_, i) => ({ id: `item-${String(501-i).padStart(4,'0')}`, user_id: 'owner', category: 'top' }));
  mockRange.mockResolvedValueOnce({ data: rows.slice(0, 250), error: null }).mockResolvedValueOnce({ data: rows.slice(250, 500), error: null }).mockResolvedValueOnce({ data: rows.slice(500), error: null });
  const result = await fetchItemsFromServer();
  expect(result?.map(item => item.id)).toEqual(rows.map(row => row.id));
  expect(mockRange.mock.calls).toEqual([[250], [250], [250]]);
  expect(mockLt.mock.calls).toEqual([['id', rows[249].id], ['id', rows[499].id]]);
  expect(mockEq).toHaveBeenCalledWith('user_id', 'owner');
  expect(mockOrder).toHaveBeenCalledWith('id', { ascending: false });
});

test('later page failures never return a misleading partial wardrobe', async () => {
  mockRange.mockResolvedValueOnce({ data: Array.from({ length: 250 }, (_, i) => ({ id: `${i}` })), error: null }).mockResolvedValueOnce({ data: null, error: { message: 'offline' } });
  expect(await fetchItemsFromServer()).toBeNull();
});

test('a pending refresh preserves unsynced additions, removals and favorite values', () => {
  const removed = { id: 'removed' } as ClothingItem;
  const changed = { id: 'changed', isFavorite: false } as ClothingItem;
  const added = { id: 'added', isFavorite: false } as ClothingItem;
  const pending: PendingAction[] = [
    { id: '1', type: 'remove_item', payload: { itemId: 'removed' }, createdAt: '' },
    { id: '2', type: 'add_item', payload: added as unknown as Record<string, unknown>, createdAt: '' },
    { id: '3', type: 'update_item', payload: { itemId: 'changed', isFavorite: true }, createdAt: '' },
  ];
  const result = reconcileCloudItems([removed, changed], [removed, changed], [changed, added], pending);
  expect(result.map(item => item.id).sort()).toEqual(['added', 'changed']);
  expect(result.find(item => item.id === 'changed')?.isFavorite).toBe(true);
});

test('refresh preserves newer local edits without resurrecting a completed deletion', () => {
  const original = { id: 'edited', name: 'old' } as ClothingItem;
  const removed = { id: 'removed' } as ClothingItem;
  const result = reconcileCloudItems([original, removed], [original, removed], [{ ...original, name: 'new' }], []);
  expect(result).toEqual([{ ...original, name: 'new' }]);
});

test('cloud refresh preserves optimistic wear statistics while its log is still queued', () => {
  const local = { id: 'owned', name: 'local name', wearCount: 3, lastWornAt: '2026-10-16T12:00:00Z' } as ClothingItem;
  const remote = { ...local, name: 'remote name', wearCount: 2, lastWornAt: '2026-10-15T12:00:00Z' };
  const pending: PendingAction[] = [{ id: 'pending-log', type: 'add_wear_log', createdAt: '', payload: { id: 'log', itemIds: ['owned'] } }];
  const result = reconcileCloudItems([remote], [local], [local], pending);
  expect(result).toEqual([{ ...remote, wearCount: 3, lastWornAt: local.lastWornAt }]);
  expect(reconcileCloudItems([remote], [local], [local], [])).toEqual([remote]);
  // A successful upload with a lost acknowledgement must not add a second increment.
  expect(reconcileCloudItems([local], [local], [local], pending)).toEqual([local]);
  const newerRemote = { ...remote, wearCount: 5, lastWornAt: '2026-10-17T12:00:00Z' };
  expect(reconcileCloudItems([newerRemote], [local], [local], pending)).toEqual([newerRemote]);
  expect(reconcileCloudItems([], [local], [], pending)).toEqual([]);
});


test('loads a complete 1001-entry wear history with stable date/id ordering', async () => {
  const rows = Array.from({ length: 1001 }, (_, i) => ({ id: `log-${String(1001-i).padStart(4,'0')}`, user_id: 'owner', date: '2026-10-16', item_ids: ['garment'] }));
  for (let start = 0; start < rows.length; start += 250) mockRange.mockResolvedValueOnce({ data: rows.slice(start, start + 250), error: null });
  const result = await fetchWearLogsFromServer();
  expect(result?.map(log => log.id)).toEqual(rows.map(row => row.id));
  expect(result?.[1000].itemIds).toEqual(['garment']);
  expect(mockRange.mock.calls).toEqual(Array.from({length:5}, () => [250]));
  expect(mockEq).toHaveBeenCalledWith('user_id', 'owner');
  expect(mockOrder.mock.calls).toEqual(Array.from({length:5}, () => ['id', {ascending:false}]));
});

test('wear history page failure preserves the cache instead of returning partial history', async () => {
  mockRange.mockResolvedValueOnce({ data: Array.from({ length: 250 }, (_, i) => ({ id: `${i}` })), error: null })
    .mockResolvedValueOnce({ data: null, error: { message: 'network unavailable' } });
  expect(await fetchWearLogsFromServer()).toBeNull();
});

test('a wear history page arriving after an account change is discarded', async () => {
  let finish!: (value: unknown) => void;
  mockRange.mockReturnValueOnce(new Promise(resolve => { finish = resolve; }));
  const pending = fetchWearLogsFromServer();
  await Promise.resolve();
  authEvents.emitLogout(); authEvents.emitLogin('other');
  finish({ data: [{ id: 'private-owner-log' }], error: null });
  expect(await pending).toBeNull();
});

it.each([fetchItemsFromServer, fetchWearLogsFromServer])('does not skip existing rows when another device deletes an earlier page', async fetchRows => {
  const original = Array.from({ length: 501 }, (_, index) => ({
    id: `row-${String(501-index).padStart(4,'0')}`, user_id: 'owner', date: '2026-10-16',
  }));
  let live = [...original];
  let cursor: string | undefined;
  let requests = 0;
  mockLt.mockImplementation((_field, value) => { cursor = value; return { limit: mockRange }; });
  mockRange.mockImplementation(async (start: number, end?: number) => {
    const eligible = cursor ? live.filter(row => row.id < cursor!) : live;
    const page = end === undefined ? eligible.slice(0, start) : eligible.slice(start, end + 1);
    if (requests++ === 0) live = live.slice(1);
    return { data: page, error: null };
  });
  const rows = await fetchRows();
  expect(rows?.map(row => row.id)).toEqual(original.map(row => row.id));
});

it.each([fetchItemsFromServer, fetchWearLogsFromServer])('rejects a non-advancing cursor rather than looping or returning partial results', async fetchRows => {
  const page = Array.from({ length: 250 }, (_, index) => ({ id: `row-${String(250-index).padStart(4,'0')}`, date: '2026-10-16' }));
  mockRange.mockResolvedValue({ data: page, error: null });
  expect(await fetchRows()).toBeNull();
  expect(mockRange).toHaveBeenCalledTimes(2);
});

test('ID cursor traversal preserves chronological display ordering', async () => {
  const rows = [
    { id: 'c', date: '2026-10-14', created_at: '2026-10-14T12:00:00Z' },
    { id: 'b', date: '2026-10-16', created_at: '2026-10-16T12:00:00Z' },
    { id: 'a', date: '2026-10-15', created_at: '2026-10-15T12:00:00Z' },
  ];
  mockRange.mockResolvedValue({ data: rows, error: null });
  expect((await fetchItemsFromServer())?.map(row => row.id)).toEqual(['b', 'a', 'c']);
  expect((await fetchWearLogsFromServer())?.map(row => row.id)).toEqual(['b', 'a', 'c']);
});

it.each([fetchItemsFromServer,fetchWearLogsFromServer])('a stalled read settles without publishing a late page or requesting another page',async fetchRows=>{
 jest.useFakeTimers();let finish!:(value:unknown)=>void;
 mockRange.mockReturnValueOnce(new Promise(resolve=>{finish=resolve;}));
 let settled=false;const pending=fetchRows().then(value=>{settled=true;return value;});
 try{
  await jest.advanceTimersByTimeAsync(30000);
  expect(settled).toBe(true);expect(await pending).toBeNull();
  expect(mockAbort.mock.calls[0][0].aborted).toBe(true);
  finish({data:Array.from({length:250},(_,i)=>({id:String(1000-i),date:'2026-10-21'})),error:null});
  await jest.advanceTimersByTimeAsync(0);
  expect(mockRange).toHaveBeenCalledTimes(1);expect(jest.getTimerCount()).toBe(0);
 }finally{finish?.({data:[],error:null});await pending.catch(()=>undefined);jest.useRealTimers();}
});
it.each([fetchItemsFromServer,fetchWearLogsFromServer])('a stalled session read settles before any database query',async fetchRows=>{
 jest.useFakeTimers();let finish!:(value:unknown)=>void;
 mockGetSession.mockReturnValueOnce(new Promise(resolve=>{finish=resolve;}));mockRange.mockResolvedValue({data:[],error:null});
 let settled=false;const pending=fetchRows().then(value=>{settled=true;return value;});
 try{
  await jest.advanceTimersByTimeAsync(30000);expect(settled).toBe(true);expect(await pending).toBeNull();
  finish({data:{session:{user:{id:'owner'}}}});await jest.advanceTimersByTimeAsync(0);
  expect(mockEq).not.toHaveBeenCalled();expect(jest.getTimerCount()).toBe(0);
 }finally{finish?.({data:{session:null}});await pending.catch(()=>undefined);jest.useRealTimers();}
});

it.each([fetchItemsFromServer,fetchWearLogsFromServer])('the read deadline is shared across all pages rather than reset per page',async fetchRows=>{
 jest.useFakeTimers();
 const page=Array.from({length:250},(_,i)=>({id:String(1000-i),date:'2026-10-21'}));
 mockRange.mockImplementationOnce(()=>new Promise(resolve=>setTimeout(()=>resolve({data:page,error:null}),8000)))
  .mockImplementationOnce(()=>new Promise(()=>{}));
 let settled=false;const pending=fetchRows().then(value=>{settled=true;return value;});
 try{
  await jest.advanceTimersByTimeAsync(29999);expect(settled).toBe(false);
  await jest.advanceTimersByTimeAsync(1);expect(await pending).toBeNull();
  expect(mockRange).toHaveBeenCalledTimes(2);
  expect(mockAbort.mock.calls[0][0]).toBe(mockAbort.mock.calls[1][0]);
  expect(jest.getTimerCount()).toBe(0);
 }finally{jest.useRealTimers();}
});
it.each([
 [fetchItemsFromServer,'headers'],[fetchItemsFromServer,'body'],[fetchWearLogsFromServer,'headers'],[fetchWearLogsFromServer,'body'],
] as const)('installed SDK read cancellation covers uncooperative transport/body: %s %s',async(fetchRows,phase)=>{
 jest.useFakeTimers();
 let signal:AbortSignal|undefined;
 const transport=jest.fn(async(_url:unknown,init?:RequestInit)=>{
  signal=init?.signal??undefined;
  if(phase==='headers')return await new Promise<Response>(()=>{});
  return {ok:true,status:200,statusText:'OK',headers:new Headers(),text:()=>new Promise(()=>{})} as unknown as Response;
 });
 const client=createClient('https://query.test','fixture-anon',{auth:{persistSession:false,autoRefreshToken:false,detectSessionInUrl:false},global:{fetch:transport}});
 mockFrom.mockImplementation(table=>client.from(table));
 try{
  const pending=fetchRows();await jest.advanceTimersByTimeAsync(30000);
  expect(await pending).toBeNull();expect(signal?.aborted).toBe(true);expect(transport).toHaveBeenCalledTimes(1);expect(jest.getTimerCount()).toBe(0);
 }finally{jest.useRealTimers();}
});
