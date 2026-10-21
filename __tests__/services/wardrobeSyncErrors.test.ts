import type { ClothingItem } from '../../src/types/domain';
import { processPendingActions, type PendingAction } from '../../store/wardrobeSyncService';
import { authEvents } from '../../store/authEvents';
const mockFrom = jest.fn();
jest.mock('../../lib/supabase', () => ({ supabase: {
  auth: { getSession: jest.fn(async () => ({ data: { session: { user: { id: 'owner' } } } })) },
  from: (...args: unknown[]) => mockFrom(...args),
} }));
const favoriteVersion = (updatedAt: string | null = '2026-10-21T00:00:00Z') => ({
  eq: jest.fn().mockReturnThis(), abortSignal: jest.fn().mockReturnThis(),
  single: jest.fn().mockResolvedValue({ data: { id: 'item', is_favorite: false, updated_at: updatedAt }, error: null }),
});
const actions: PendingAction[] = [
  { id: 'delete', type: 'remove_item', payload: { itemId: 'item' }, createdAt: '' },
  { id: 'wear', type: 'add_wear_log', payload: { id: 'log' }, createdAt: '' },
];
beforeEach(() => { jest.clearAllMocks(); authEvents.emitLogin('owner'); });

it.each([true, false])('only acknowledges successful Supabase responses (error=%s)', async (hasError) => {
  const response = { error: hasError ? { message: 'request rejected' } : null };
  const eq = jest.fn().mockReturnValue({ eq: jest.fn().mockResolvedValue(response) });
  mockFrom.mockReturnValue({ delete: () => ({ eq }), upsert: jest.fn().mockResolvedValue(response) });
  const error = jest.spyOn(console, 'error').mockImplementation(() => undefined);
  const result = await processPendingActions(actions, [], 'owner');
  expect(result.processedIds).toEqual(hasError ? [] : ['delete', 'wear']);
  error.mockRestore();
});

test('refuses a queue from a different account before issuing mutations', async () => {
  const result = await processPendingActions(actions, [], 'other-owner');
  expect(result.processedIds).toEqual([]);
  expect(mockFrom).not.toHaveBeenCalled();
});


test('favorite writes are explicit owner-scoped values and require a matching returned row', async () => {
  const single = jest.fn().mockResolvedValueOnce({ data: { id: 'item', is_favorite: true, updated_at: '2026-10-21T00:00:01Z' }, error: null })
    .mockResolvedValueOnce({ data: null, error: { message: 'denied' } });
  const eq = jest.fn(); eq.mockReturnValue({ eq, select: () => ({ single }), single: async () => ({ data: { id: 'item', updated_at: '2026-10-21T00:00:00Z' }, error: null }) });
  const update = jest.fn().mockReturnValue({ eq });
  mockFrom.mockReturnValue({ update, select: () => favoriteVersion() });
  const action: PendingAction = { id: 'favorite', type: 'update_item', payload: { itemId: 'item', isFavorite: true }, createdAt: '2026-10-16T00:00:00Z' };
  expect((await processPendingActions([action], [], 'owner')).processedIds).toEqual(['favorite']);
  expect(update).toHaveBeenCalledWith({ is_favorite: true, updated_at: action.createdAt });
  expect(eq).toHaveBeenCalledWith('user_id', 'owner');
  const error = jest.spyOn(console, 'error').mockImplementation(() => undefined);
  expect((await processPendingActions([action], [], 'owner')).processedIds).toEqual([]);
  error.mockRestore();
});

test('favorite retries cannot replay an older tap after a newer value succeeds', async () => {
  const single = jest.fn().mockResolvedValue({ data: { id: 'item', is_favorite: false, updated_at: '2026-10-21T00:00:01Z' }, error: null });
  const eq = jest.fn(); eq.mockReturnValue({ eq, select: () => ({ single }), single: async () => ({ data: { id: 'item', updated_at: '2026-10-21T00:00:00Z' }, error: null }) });
  const update = jest.fn().mockReturnValue({ eq });
  mockFrom.mockReturnValue({ update, select: () => favoriteVersion() });
  const pending: PendingAction[] = [
    { id: 'old', type: 'update_item', payload: { itemId: 'item', isFavorite: true }, createdAt: '2026-10-16T00:00:00Z' },
    { id: 'new', type: 'update_item', payload: { itemId: 'item', isFavorite: false }, createdAt: '2026-10-16T00:00:01Z' },
  ];
  const result = await processPendingActions(pending, [], 'owner');
  expect(update).toHaveBeenCalledTimes(1);
  expect(update).toHaveBeenCalledWith({ is_favorite: false, updated_at: pending[1].createdAt });
  expect(result.processedIds.sort()).toEqual(['new', 'old']);
});

test('failed final favorite confirmation keeps all superseded entries for retry', async () => {
  const single = jest.fn().mockResolvedValueOnce({ data: null, error: { message: 'offline' } })
    .mockResolvedValueOnce({ data: { id: 'item', is_favorite: false, updated_at: '2026-10-21T00:00:01Z' }, error: null });
  const eq = jest.fn(); eq.mockReturnValue({ eq, select: () => ({ single }), single: async () => ({ data: { id: 'item', updated_at: '2026-10-21T00:00:00Z' }, error: null }) });
  const update = jest.fn().mockReturnValue({ eq });
  mockFrom.mockReturnValue({ update, select: () => favoriteVersion() });
  const pending: PendingAction[] = [true, false].map((isFavorite, index) => ({
    id: String(index), type: 'update_item', payload: { itemId: 'item', isFavorite }, createdAt: '',
  }));
  const error = jest.spyOn(console, 'error').mockImplementation(() => undefined);
  try {
    expect((await processPendingActions(pending, [], 'owner')).processedIds).toEqual([]);
    expect((await processPendingActions(pending, [], 'owner')).processedIds).toEqual(['0', '1']);
    expect(update.mock.calls.map(([value]) => value.is_favorite)).toEqual([false, false]);
  } finally { error.mockRestore(); }
});

test('a queued deletion supersedes old uploads even if deletion must retry', async () => {
  const pending: PendingAction[] = [
    { id: 'upload', type: 'add_item', payload: { id: 'item' }, createdAt: '' },
    { id: 'favorite', type: 'update_item', payload: { itemId: 'item', isFavorite: true }, createdAt: '' },
    { id: 'delete', type: 'remove_item', payload: { itemId: 'item' }, createdAt: '' },
  ];
  const response = { error: { message: 'offline' } as { message: string } | null };
  const eq = jest.fn().mockReturnValue({ eq: jest.fn().mockImplementation(async () => response) });
  const upsert = jest.fn();
  mockFrom.mockReturnValue({ delete: () => ({ eq }), upsert });
  const error = jest.spyOn(console, 'error').mockImplementation(() => undefined);
  try {
    expect((await processPendingActions(pending, [], 'owner')).processedIds).toEqual([]);
    response.error = null;
    expect((await processPendingActions(pending, [], 'owner')).processedIds.sort()).toEqual(['delete', 'favorite', 'upload']);
    expect(upsert).not.toHaveBeenCalled();
    expect(eq).toHaveBeenCalledTimes(2);
  } finally { error.mockRestore(); }
});

test.each(['remove_item', 'update_item', 'add_wear_log'] as const)(
  'does not acknowledge %s when its response arrives after an account change', async (type) => {
    const response = jest.fn(async () => {
      authEvents.emitLogin('different-owner');
      return { data: { id: 'item', is_favorite: true, updated_at: '2026-10-21T00:00:01Z' }, error: null };
    });
    const eq = jest.fn();
    eq.mockReturnValue({ eq, select: () => ({ single: response }), then: (resolve: (value: unknown) => unknown) => response().then(resolve) });
    mockFrom.mockReturnValue({ delete: () => ({ eq }), update: () => ({ eq }), upsert: response, select: () => favoriteVersion() });
    const action: PendingAction = {
      id: 'late', type, createdAt: '',
      payload: type === 'add_wear_log' ? { id: 'log', itemIds: [] } : { itemId: 'item', isFavorite: true },
    };
    const error = jest.spyOn(console, 'error').mockImplementation(() => undefined);
    try {
      expect((await processPendingActions([action], [], 'owner')).processedIds).toEqual([]);
      expect(response).toHaveBeenCalledTimes(1);
    } finally { error.mockRestore(); }
  }
);

test('a delayed older favorite write cannot overwrite a newer confirmed choice', async () => {
  const { createClient } = require('@supabase/supabase-js');
  let row = { id: 'item', user_id: 'owner', is_favorite: false, updated_at: '2026-10-21T00:00:00Z' };
  let release!: () => void;
  let reached!: () => void;
  const held = new Promise<void>(resolve => { release = resolve; });
  const firstReached = new Promise<void>(resolve => { reached = resolve; });
  let writes = 0;
  const reply = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
  const transport = jest.fn(async (input: string, init: RequestInit) => {
    const url = new URL(input);
    expect(url.searchParams.get('user_id')).toBe('eq.owner');
    if (init.method === 'GET') return reply({ id: row.id, is_favorite: row.is_favorite, updated_at: row.updated_at });
    const body = JSON.parse(String(init.body));
    if (++writes === 1) { reached(); await held; }
    const version = url.searchParams.get('updated_at');
    if (version && version !== `eq.${row.updated_at}`) return reply({ code: 'PGRST116', message: 'No row matched' }, 406);
    row = { ...row, is_favorite: body.is_favorite, updated_at: `2026-10-21T00:00:0${writes}Z` };
    return reply({ id: row.id, is_favorite: row.is_favorite, updated_at: row.updated_at });
  });
  const client = createClient('https://favorite.test', 'fixture-anon', { auth: { persistSession: false, autoRefreshToken: false, detectSessionInUrl: false }, global: { fetch: transport } });
  mockFrom.mockImplementation(table => client.from(table));
  const action = (id: string, isFavorite: boolean): PendingAction => ({ id, type: 'update_item', payload: { itemId: 'item', isFavorite }, createdAt: '2026-10-21T00:00:00Z' });
  const error = jest.spyOn(console, 'error').mockImplementation(() => undefined);
  let older: ReturnType<typeof processPendingActions> | undefined;
  try {
    older = processPendingActions([action('older', true)], [{ id: 'item', isFavorite: true } as ClothingItem], 'owner');
    await firstReached;
    expect((await processPendingActions([action('newer', false)], [], 'owner')).processedIds).toEqual(['newer']);
    release();
    const retired = await older;
    expect(retired.processedIds).toEqual(['older']);
    expect(retired.updatedItems[0].isFavorite).toBe(false);
    expect(row.is_favorite).toBe(false);
  } finally { release(); await older; error.mockRestore(); }
});

test('favorite mutation waits for the version checkpoint and leaves the original action immutable',async()=>{
 const read=favoriteVersion();const chain={eq:jest.fn().mockReturnThis(),select:jest.fn().mockReturnThis(),single:jest.fn().mockResolvedValue({data:{id:'item',is_favorite:true,updated_at:'2026-10-21T00:00:01Z'},error:null})};
 const update=jest.fn().mockReturnValue(chain);mockFrom.mockReturnValue({select:()=>read,update});
 let release!:()=>void;let reached!:()=>void;const ready=new Promise<void>(resolve=>{reached=resolve;});
 const checkpoint=jest.fn(async(_actions:PendingAction[])=>{reached();await new Promise<void>(resolve=>{release=resolve;});});
 const action:PendingAction={id:'favorite',type:'update_item',payload:{itemId:'item',isFavorite:true},createdAt:'2026-10-21T00:00:00Z'};
 const pending=processPendingActions([action],[],'owner',checkpoint);await ready;expect(update).not.toHaveBeenCalled();release();
 expect((await pending).processedIds).toEqual(['favorite']);expect(action.payload).not.toHaveProperty('favoriteVersion');
 expect(checkpoint.mock.calls[0][0][0].payload.favoriteVersion).toBe('2026-10-21T00:00:00Z');
});
test('failed checkpoint prevents a favorite mutation',async()=>{
 const update=jest.fn();mockFrom.mockReturnValue({select:()=>favoriteVersion(),update});
 const action:PendingAction={id:'favorite',type:'update_item',payload:{itemId:'item',isFavorite:true},createdAt:''};
 const error=jest.spyOn(console,'error').mockImplementation(()=>undefined);
 try{const result=await processPendingActions([action],[],'owner',async()=>{throw Error('disk full');});expect(result.processedIds).toEqual([]);expect(update).not.toHaveBeenCalled();}finally{error.mockRestore();}
});
test('a failed conflict readback retains the pinned version; serialized retry cannot overwrite the newer value',async()=>{
 const read=favoriteVersion();read.single.mockResolvedValueOnce({data:{id:'item',is_favorite:false,updated_at:'2026-10-21T00:00:00Z'},error:null})
  .mockResolvedValueOnce({data:null,error:{message:'read unavailable'}})
  .mockResolvedValueOnce({data:{id:'item',is_favorite:false,updated_at:'2026-10-21T00:00:01Z'},error:null});
 const chain={eq:jest.fn().mockReturnThis(),select:jest.fn().mockReturnThis(),single:jest.fn().mockResolvedValue({data:null,error:{code:'PGRST116'}})};
 const update=jest.fn().mockReturnValue(chain);mockFrom.mockReturnValue({select:()=>read,update});
 let saved:PendingAction[]=[];const checkpoint=jest.fn(async(actions:PendingAction[])=>{saved=JSON.parse(JSON.stringify(actions));});
 const action:PendingAction={id:'favorite',type:'update_item',payload:{itemId:'item',isFavorite:true},createdAt:''};
 const error=jest.spyOn(console,'error').mockImplementation(()=>undefined);
 try{
  const first=await processPendingActions([action],[{id:'item',isFavorite:true} as ClothingItem],'owner',checkpoint);expect(first.processedIds).toEqual([]);expect(saved[0].payload.favoriteVersion).toBe('2026-10-21T00:00:00Z');
  const retry=await processPendingActions(saved,[{id:'item',isFavorite:true} as ClothingItem],'owner',checkpoint);
  expect(retry.processedIds).toEqual(['favorite']);expect(retry.updatedItems[0].isFavorite).toBe(false);expect(update).toHaveBeenCalledTimes(1);expect(checkpoint).toHaveBeenCalledTimes(1);
 }finally{error.mockRestore();}
});
test('a null legacy version uses an atomic IS NULL condition',async()=>{
 const chain={eq:jest.fn().mockReturnThis(),is:jest.fn().mockReturnThis(),select:jest.fn().mockReturnThis(),single:jest.fn().mockResolvedValue({data:{id:'item',is_favorite:true,updated_at:'2026-10-21T00:00:01Z'},error:null})};
 mockFrom.mockReturnValue({select:()=>favoriteVersion(null),update:()=>chain});
 const action:PendingAction={id:'favorite',type:'update_item',payload:{itemId:'item',isFavorite:true},createdAt:''};
 expect((await processPendingActions([action],[],'owner',async()=>{})).processedIds).toEqual(['favorite']);expect(chain.is).toHaveBeenCalledWith('updated_at',null);
});
test('a stalled favorite version read settles without writing',async()=>{
 jest.useFakeTimers();const read=favoriteVersion();read.single.mockImplementation(()=>new Promise(()=>{}));const update=jest.fn();mockFrom.mockReturnValue({select:()=>read,update});
 const error=jest.spyOn(console,'error').mockImplementation(()=>undefined);
 try{const pending=processPendingActions([{id:'favorite',type:'update_item',payload:{itemId:'item',isFavorite:true},createdAt:''}],[],'owner');await jest.advanceTimersByTimeAsync(30000);expect((await pending).processedIds).toEqual([]);expect(update).not.toHaveBeenCalled();expect(read.abortSignal.mock.calls[0][0].aborted).toBe(true);}finally{error.mockRestore();jest.useRealTimers();}
});
