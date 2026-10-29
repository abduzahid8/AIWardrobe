jest.mock('../../store/wardrobeStore', () => ({ __esModule: true, default: { getState: jest.fn(() => ({ items: [] })) } }));
import { supabase } from '../../lib/supabase';
import { generateOutfitsFromDB } from '../../src/services/outfitGenerationService';
jest.mock('../../lib/supabase', () => ({ supabase: { auth: { onAuthStateChange: jest.fn().mockReturnValue({data:{subscription:{unsubscribe:jest.fn()}}}), getSession: jest.fn().mockResolvedValue({ data: { session: { user: { id: 'account' }, access_token: 'captured-token' } } }) }, functions: { invoke: jest.fn() }, from: jest.fn() } }));
jest.mock('../../src/services/recentOutfitIds', () => ({ getRecentShownIds: jest.fn().mockResolvedValue(['recent']), getRecentShownProductKeys: jest.fn().mockResolvedValue([]), recordShownOutfits: jest.fn() }));
const invoke = supabase.functions.invoke as jest.Mock;
describe('canonical stylist failure handling', () => {
  beforeEach(() => jest.clearAllMocks());
  it('returns failure without querying a local replacement when edge fails', async () => {
    invoke.mockResolvedValue({ data: null, error: new Error('unavailable') });
    expect((await generateOutfitsFromDB({ prompt: 'workout' })).success).toBe(false);
    expect(invoke).toHaveBeenCalledTimes(1);
    expect(supabase.from).not.toHaveBeenCalled();
  });
  it('rejects a successful response with an incomplete outfit', async () => {
    invoke.mockResolvedValue({ data: { success: true, outfits: [{ items: [{ id: 'shirt', name: 'Cotton shirt', macroCategory: 'top' }] }] }, error: null });
    expect((await generateOutfitsFromDB({ prompt: 'office' })).success).toBe(false);
  });
  it('sends preferences, actual weather, context and exclusions to one endpoint', async () => {
    invoke.mockResolvedValue({ data: { success: false, outfits: [] }, error: null });
    await generateOutfitsFromDB({ prompt: 'meeting', conversationContext: 'Prefer navy', weather: { temp: 0, condition: 'snow' }, preferences: { avoidColors: ['red'], maxShopBudget: {amount:250,currency:'USD'} } });
    expect(invoke).toHaveBeenCalledWith('generate-outfits', expect.objectContaining({ body: expect.objectContaining({ prompt: 'meeting', conversationContext: 'Prefer navy', excludeItemIds: ['recent'], weather: { temp: 0, condition: 'snow' }, preferences: { avoidColors: ['red'], maxShopBudget: {amount:250,currency:'USD'} } }) }));
  });
});

const complete = { items: [
  { id: 'layer', name: 'Linen blazer', macroCategory: 'outerwear' },
  { id: 'shirt', name: 'Cotton shirt', macroCategory: 'top' },
  { id: 'pants', name: 'Navy trousers', macroCategory: 'bottom' },
  { id: 'shoes', name: 'Brown loafers', macroCategory: 'shoes' },
] };
it('does not accept a legacy server local fallback', async () => {
  invoke.mockResolvedValue({ data: { success: true, source: 'local', outfits: [complete] }, error: null });
  expect((await generateOutfitsFromDB({})).success).toBe(false);
});
it('keeps a valid sibling when another returned outfit is malformed', async () => {
  invoke.mockResolvedValue({ data: { success: true, source: 'ai', outfits: [null, { items: {} }, complete] }, error: null });
  const result = await generateOutfitsFromDB({});
  expect(result.success).toBe(true);
  expect(result.outfits).toHaveLength(1);
  expect(result.outfits[0].items.map(item => item.id)).toEqual(['layer', 'shirt', 'pants', 'shoes']);
});
it('aborts the network request on timeout instead of leaving it running', async () => {
  jest.useFakeTimers();
  try {
    invoke.mockImplementation(() => new Promise(() => {}));
    const pending = generateOutfitsFromDB({});
    await jest.advanceTimersByTimeAsync(120_001);
    expect((await pending).error).toBe('outfit_generation_timeout');
    expect(invoke.mock.calls[invoke.mock.calls.length - 1][1].signal.aborted).toBe(true);
  } finally { jest.useRealTimers(); }
});

it('preserves authoritative item currency while mapping returned catalog prices',async()=>{
 const priced={items:complete.items.map(item=>({...item,isShopItem:true,price:800000,currency:'UZS',source_url:`https://retailer.invalid/products/${item.id}`}))};
 invoke.mockResolvedValue({data:{success:true,source:'ai',outfits:[priced]},error:null});
 const result=await generateOutfitsFromDB({weather:{temp:20,condition:'Clear'}});
 expect(result.success).toBe(true);
 expect(result.outfits[0].items.map(item=>item.currency)).toEqual(['UZS','UZS','UZS','UZS']);
 expect(result.outfits[0].items.map(item=>item.price)).toEqual([800000,800000,800000,800000]);
 expect(result.outfits[0].items.map(item=>item.shopUrl)).toEqual(complete.items.map(item=>`https://retailer.invalid/products/${item.id}`));
});

it('preserves a known budget failure for localized downstream presentation',async()=>{
 invoke.mockResolvedValue({data:null,error:{context:new Response(JSON.stringify({code:'no_outfit_within_budget'}),{status:422})}});
 const result=await generateOutfitsFromDB({});
 expect(result.success).toBe(false);expect(result.outfits).toEqual([]);expect(result.errorTranslationKey).toBe('outfitMaker.budgetTooLow');
});


describe('generation account ownership', () => {
  let listener: (event: string, session: any) => void;
  const unsubscribe = jest.fn();
  beforeEach(() => {
    jest.clearAllMocks();
    (supabase.auth.getSession as jest.Mock).mockResolvedValue({data:{session:{user:{id:'account'},access_token:'captured-token'}}});
    (supabase.auth.onAuthStateChange as jest.Mock).mockImplementation(cb => {
      listener = cb;
      return {data:{subscription:{unsubscribe}}};
    });
  });
  it('pins authorization to the requesting account', async () => {
    invoke.mockResolvedValue({data:{success:true,source:'ai',outfits:[complete]},error:null});
    expect((await generateOutfitsFromDB({})).success).toBe(true);
    expect(invoke.mock.calls[0][1].headers).toEqual({Authorization:'Bearer captured-token'});
    expect(unsubscribe).toHaveBeenCalledTimes(1);
  });
  it('rejects a result after switching accounts and does not record it', async () => {
    let finish!: (value: any) => void;
    invoke.mockImplementation(() => new Promise(resolve => {finish=resolve;}));
    const pending = generateOutfitsFromDB({});
    while (!finish) await Promise.resolve();
    listener?.('SIGNED_IN',{user:{id:'other'},access_token:'other-token'});
    finish({data:{success:true,source:'ai',outfits:[complete]},error:null});
    expect((await pending).success).toBe(false);
    expect(require('../../src/services/recentOutfitIds').recordShownOutfits).not.toHaveBeenCalled();
    expect(invoke.mock.calls[0][1].signal.aborted).toBe(true);
    expect(unsubscribe).toHaveBeenCalledTimes(1);
  });
  it('does not dispatch after an account switch while recency is loading', async () => {
    const recent = require('../../src/services/recentOutfitIds').getRecentShownIds;
    let finish!: (ids:string[]) => void;
    recent.mockImplementationOnce(() => new Promise((resolve:any) => {finish=resolve;}));
    const pending=generateOutfitsFromDB({});
    while (!finish) await Promise.resolve();
    listener?.('SIGNED_OUT',null);
    finish(['private-old-account-id']);
    expect((await pending).success).toBe(false);
    expect(invoke).not.toHaveBeenCalled();
  });
  it('rejects missing authentication without calling the provider', async () => {
    (supabase.auth.getSession as jest.Mock).mockResolvedValueOnce({data:{session:null}});
    expect((await generateOutfitsFromDB({})).success).toBe(false);
    expect(invoke).not.toHaveBeenCalled();
  });
  it.each(['session', 'recency'])('bounds stalled %s loading before provider dispatch', async stage => {
    jest.useFakeTimers();
    try {
      if (stage === 'session') (supabase.auth.getSession as jest.Mock).mockImplementationOnce(() => new Promise(() => {}));
      else require('../../src/services/recentOutfitIds').getRecentShownIds.mockImplementationOnce(() => new Promise(() => {}));
      const pending = generateOutfitsFromDB({});
      await jest.advanceTimersByTimeAsync(120_001);
      expect((await pending).error).toBe('outfit_generation_timeout');
      expect(invoke).not.toHaveBeenCalled();
      expect(jest.getTimerCount()).toBe(0);
    } finally { jest.useRealTimers(); }
  });
  it('permits same-account token refresh without changing the captured credential', async () => {
    let finish!: (value:any) => void;
    invoke.mockImplementation(() => new Promise(resolve => {finish=resolve;}));
    const pending=generateOutfitsFromDB({});
    while (!finish) await Promise.resolve();
    listener('TOKEN_REFRESHED',{user:{id:'account'},access_token:'refreshed-token'});
    finish({data:{success:true,source:'ai',outfits:[complete]},error:null});
    expect((await pending).success).toBe(true);
    expect(invoke.mock.calls[0][1].headers.Authorization).toBe('Bearer captured-token');
  });

  it('rejects an account change between the initial read and listener registration', async () => {
    (supabase.auth.getSession as jest.Mock)
      .mockResolvedValueOnce({data:{session:{user:{id:'account'},access_token:'captured-token'}}})
      .mockResolvedValueOnce({data:{session:{user:{id:'other'},access_token:'other-token'}}});
    expect((await generateOutfitsFromDB({})).error).toBe('generation_account_changed');
    expect(invoke).not.toHaveBeenCalled();
    expect(unsubscribe).toHaveBeenCalledTimes(1);
  });

});

describe('client request admission',()=>{
 beforeEach(()=>{
  jest.clearAllMocks();
  (supabase.auth.getSession as jest.Mock).mockResolvedValue({data:{session:{user:{id:'account'},access_token:'captured-token'}}});
  (supabase.auth.onAuthStateChange as jest.Mock).mockReturnValue({data:{subscription:{unsubscribe:jest.fn()}}});
  require('../../src/services/recentOutfitIds').getRecentShownIds.mockResolvedValue(['recent']);
  invoke.mockResolvedValue({data:{success:false,outfits:[]},error:null});
 });
 it.each([{limit:16},{prompt:'x'.repeat(1001)},{selectedItemIds:Array(201).fill('item')}])('invalid client request does not call the metered endpoint: %j',async params=>{
  const result=await generateOutfitsFromDB(params);
  expect(result.success).toBe(false);expect(result.errorTranslationKey).toBe('outfitMaker.invalidRequest');expect(invoke).not.toHaveBeenCalled();
 });
 it('retains the complete supported prompt and selected pool at the boundary',async()=>{
  const params={prompt:'x'.repeat(1000),selectedItemIds:Array.from({length:200},(_,i)=>'item-'+i),limit:15};
  await generateOutfitsFromDB(params);
  expect(invoke).toHaveBeenCalledTimes(1);expect(invoke.mock.calls[0][1].body).toEqual(expect.objectContaining(params));
 });
});
