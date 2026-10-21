let AsyncStorage: typeof import('@react-native-async-storage/async-storage').default;

beforeEach(() => {
  jest.resetModules();
  jest.resetAllMocks();
  const storageModule = require('@react-native-async-storage/async-storage');
  AsyncStorage = storageModule.default ?? storageModule;
  (AsyncStorage.getItem as jest.Mock).mockResolvedValue(null);
  (AsyncStorage.setItem as jest.Mock).mockResolvedValue(undefined);
  (AsyncStorage.removeItem as jest.Mock).mockResolvedValue(undefined);
});

function loadStores() {
  require('../../store/authEvents').authEvents.emitLogin('default-owner');
  const useSubscriptionStore = require('../../store/subscriptionStore').default;
  const useDailyUsageStore = require('../../store/dailyUsageStore').default;
  useSubscriptionStore.setState({
    tier: 'free',
    hasActiveSubscription: false,
    isPremium: false,
    isVIP: false,
    expiryDate: null,
  });
  useDailyUsageStore.setState({
    snapshot: { date: '2099-01-01', counts: {} },
    hydrated: false,
  });
  return { useSubscriptionStore, useDailyUsageStore };
}

describe('dailyUsageStore', () => {
  it('free tier (trial expired): aiOutfits locked at 0', async () => {
    const { useDailyUsageStore } = loadStores();
    await useDailyUsageStore.getState().hydrate();
    expect(useDailyUsageStore.getState().getRemaining('aiOutfits')).toBe(0);
    expect(useDailyUsageStore.getState().canUse('aiOutfits')).toBe(false);
  });

  it('free tier (trial expired): starts at 3 remaining for homeSwipes', async () => {
    const { useDailyUsageStore } = loadStores();
    await useDailyUsageStore.getState().hydrate();
    expect(useDailyUsageStore.getState().getRemaining('homeSwipes')).toBe(3);
    expect(useDailyUsageStore.getState().canUse('homeSwipes')).toBe(true);
  });

  it('consume decrements remaining (homeSwipes)', async () => {
    const { useDailyUsageStore } = loadStores();
    await useDailyUsageStore.getState().hydrate();
    await useDailyUsageStore.getState().consume('homeSwipes');
    await useDailyUsageStore.getState().consume('homeSwipes');
    expect(useDailyUsageStore.getState().getUsed('homeSwipes')).toBe(2);
    expect(useDailyUsageStore.getState().getRemaining('homeSwipes')).toBe(1);
  });

  it('free tier: blocks after 3 home swipes', async () => {
    const { useDailyUsageStore } = loadStores();
    await useDailyUsageStore.getState().hydrate();
    for (let i = 0; i < 3; i++) {
      await useDailyUsageStore.getState().consume('homeSwipes');
    }
    expect(useDailyUsageStore.getState().getRemaining('homeSwipes')).toBe(0);
    expect(useDailyUsageStore.getState().canUse('homeSwipes')).toBe(false);
  });

  it('Pro tier: unlimited (-1)', async () => {
    const { useSubscriptionStore, useDailyUsageStore } = loadStores();
    await useSubscriptionStore.getState().setSubscription('premium');
    await useDailyUsageStore.getState().hydrate();
    expect(useDailyUsageStore.getState().getRemaining('aiOutfits')).toBe(-1);
  });

  it('Max tier: unlimited (-1)', async () => {
    const { useSubscriptionStore, useDailyUsageStore } = loadStores();
    await useSubscriptionStore.getState().setSubscription('vip');
    await useDailyUsageStore.getState().hydrate();
    expect(useDailyUsageStore.getState().getRemaining('aiOutfits')).toBe(-1);
    expect(useDailyUsageStore.getState().canUse('aiOutfits')).toBe(true);
  });

  it('tryOns on free tier is blocked (limit = 0)', async () => {
    const { useDailyUsageStore } = loadStores();
    await useDailyUsageStore.getState().hydrate();
    expect(useDailyUsageStore.getState().getRemaining('tryOns')).toBe(0);
    expect(useDailyUsageStore.getState().canUse('tryOns')).toBe(false);
  });

  it('hydrate rolls over stored snapshot from a previous day', async () => {
    const yesterdaySnapshot = JSON.stringify({
      date: '1999-01-01',
      counts: { homeSwipes: 2 },
    });
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(yesterdaySnapshot);
    const { useDailyUsageStore } = loadStores();
    await useDailyUsageStore.getState().hydrate();
    expect(useDailyUsageStore.getState().getUsed('homeSwipes')).toBe(0);
    expect(useDailyUsageStore.getState().getRemaining('homeSwipes')).toBe(3);
  });

  it('resetToday clears counters', async () => {
    const { useDailyUsageStore } = loadStores();
    await useDailyUsageStore.getState().hydrate();
    await useDailyUsageStore.getState().consume('homeSwipes', 2);
    await useDailyUsageStore.getState().resetToday();
    expect(useDailyUsageStore.getState().getUsed('homeSwipes')).toBe(0);
  });
});


describe('quota boundary consistency', () => {
  it('uses the server UTC date at local midnight', async () => {
    jest.useFakeTimers().setSystemTime(new Date('2026-10-15T00:30:00+05:00'));
    try {
      const { useDailyUsageStore } = loadStores();
      await useDailyUsageStore.getState().hydrate();
      expect(useDailyUsageStore.getState().snapshot.date).toBe('2026-10-14');
    } finally { jest.useRealTimers(); }
  });

  it('never decrements cached usage for invalid consumption amounts', async () => {
    const { useDailyUsageStore } = loadStores();
    await useDailyUsageStore.getState().hydrate();
    for (const amount of [-5, 0, NaN, 0.5, 101]) {
      const result = await useDailyUsageStore.getState().consume('homeSwipes', amount);
      expect(result.allowed).toBe(false);
      expect(useDailyUsageStore.getState().getUsed('homeSwipes')).toBe(0);
    }
  });
});

describe('delayed usage and cache integrity', () => {
  it('rejects a previous-account hydration result', async () => {
    const { useDailyUsageStore } = loadStores();
    const events = require('../../store/authEvents').authEvents;
    events.emitLogin('a');
    let finish!: (value: string) => void;
    (AsyncStorage.getItem as jest.Mock).mockImplementationOnce(() => new Promise(resolve => { finish = resolve; }));
    const pending = useDailyUsageStore.getState().hydrate();
    events.emitLogout(); events.emitLogin('b');
    finish(JSON.stringify({ date: new Date().toISOString().slice(0, 10), counts: { homeSwipes: 77 } }));
    await pending;
    expect(useDailyUsageStore.getState().getUsed('homeSwipes')).toBe(0);
    expect(useDailyUsageStore.getState().hydrated).toBe(false);
  });

  it('does not apply a previous-account quota response or authorize its caller', async () => {
    const { useDailyUsageStore } = loadStores();
    const events = require('../../store/authEvents').authEvents;
    events.emitLogin('a');
    const client = require('../../lib/supabase').supabase;
    let finish!: (value: unknown) => void;
    client.rpc = jest.fn(() => new Promise(resolve => { finish = resolve; }));
    const pending = useDailyUsageStore.getState().consume('homeSwipes');
    events.emitLogout(); events.emitLogin('b');
    finish({ data: [{ allowed: true, used: 2, remaining: 1 }], error: null });
    expect((await pending).allowed).toBe(false);
    expect(useDailyUsageStore.getState().getUsed('homeSwipes')).toBe(0);
    expect(AsyncStorage.setItem).not.toHaveBeenCalled();
  });

  it('keeps cached usage monotonic when responses arrive out of order', async () => {
    const { useDailyUsageStore } = loadStores();
    const client = require('../../lib/supabase').supabase;
    const finishes: Array<(value: unknown) => void> = [];
    client.rpc = jest.fn(() => new Promise(resolve => finishes.push(resolve)));
    const first = useDailyUsageStore.getState().consume('homeSwipes');
    const second = useDailyUsageStore.getState().consume('homeSwipes');
    finishes[1]({ data: [{ allowed: true, used: 2, remaining: 1 }], error: null }); await second;
    finishes[0]({ data: [{ allowed: true, used: 1, remaining: 2 }], error: null }); await first;
    expect(useDailyUsageStore.getState().getUsed('homeSwipes')).toBe(2);
    expect(useDailyUsageStore.getState().getRemaining('homeSwipes')).toBe(1);
  });

  it.each([null, { date: 'today', counts: null }, { date: new Date().toISOString().slice(0, 10), counts: { homeSwipes: -5, tryOns: 'broken' } }])('ignores invalid cached counters: %j', async (snapshot) => {
    const { useDailyUsageStore } = loadStores();
    (AsyncStorage.getItem as jest.Mock).mockResolvedValueOnce(JSON.stringify(snapshot));
    await useDailyUsageStore.getState().hydrate();
    expect(useDailyUsageStore.getState().getUsed('homeSwipes')).toBe(0);
    expect(useDailyUsageStore.getState().getRemaining('homeSwipes')).toBe(3);
  });
});

test('a response from yesterday cannot populate today and fast RPCs leave no deadline timer', async () => {
  jest.useFakeTimers().setSystemTime(new Date('2026-10-15T23:59:59Z'));
  try {
    const { useDailyUsageStore } = loadStores();
    const client = require('../../lib/supabase').supabase;
    let finish!: (value: unknown) => void;
    client.rpc = jest.fn(() => new Promise(resolve => { finish = resolve; }));
    const timersBefore = jest.getTimerCount();
    const pending = useDailyUsageStore.getState().consume('homeSwipes');
    jest.setSystemTime(new Date('2026-10-16T00:00:01Z'));
    finish({ data: [{ allowed: true, used: 3, remaining: 0 }], error: null });
    expect((await pending).allowed).toBe(false);
    expect(useDailyUsageStore.getState().getUsed('homeSwipes')).toBe(0);
    expect(jest.getTimerCount()).toBe(timersBefore);
  } finally { jest.useRealTimers(); }
});

describe('quota publication after persistence', () => {
  it.each(['server', 'offline'])('rejects an account switch during %s counter persistence', async mode => {
    const { useDailyUsageStore } = loadStores();
    const events = require('../../store/authEvents').authEvents;
    events.emitLogin('a');
    const client = require('../../lib/supabase').supabase;
    client.rpc = jest.fn().mockResolvedValue(mode === 'server'
      ? { data: [{ allowed: true, used: 1, remaining: 2 }], error: null }
      : { data: null, error: new Error('offline') });
    let finish!: () => void;
    (AsyncStorage.setItem as jest.Mock).mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve; }));
    const pending = useDailyUsageStore.getState().consume('homeSwipes');
    while (!finish) await Promise.resolve();
    events.emitLogout(); events.emitLogin('b');
    finish();
    expect((await pending).allowed).toBe(false);
  });
  it('does not authorize yesterday’s reservation after a delayed cache write crosses midnight', async () => {
    jest.useFakeTimers().setSystemTime(new Date('2026-10-20T23:59:59Z'));
    try {
      const { useDailyUsageStore } = loadStores();
      const client = require('../../lib/supabase').supabase;
      client.rpc = jest.fn().mockResolvedValue({ data: [{ allowed: true, used: 3, remaining: 0 }], error: null });
      let finish!: () => void;
      (AsyncStorage.setItem as jest.Mock).mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve; }));
      const pending = useDailyUsageStore.getState().consume('homeSwipes');
      while (!finish) await Promise.resolve();
      jest.setSystemTime(new Date('2026-10-21T00:00:01Z'));
      finish();
      expect((await pending).allowed).toBe(false);
    } finally { jest.useRealTimers(); }
  });
});


test('late account A persistence cannot overwrite account B counters on disk', async () => {
  const {useDailyUsageStore}=loadStores();
  const events=require('../../store/authEvents').authEvents;
  const client=require('../../lib/supabase').supabase;
  const disk=new Map<string,string>();
  let finish!:()=>void;
  (AsyncStorage.setItem as jest.Mock)
    .mockImplementationOnce((key:string,value:string)=>new Promise<void>(resolve=>{finish=()=>{disk.set(key,value);resolve();};}))
    .mockImplementation((key:string,value:string)=>{disk.set(key,value);return Promise.resolve();});
  (AsyncStorage.getItem as jest.Mock).mockImplementation((key:string)=>Promise.resolve(disk.get(key)||null));
  events.emitLogin('a');
  client.rpc=jest.fn().mockResolvedValueOnce({data:[{allowed:true,used:3,remaining:0}],error:null})
    .mockResolvedValueOnce({data:[{allowed:true,used:1,remaining:2}],error:null});
  const first=useDailyUsageStore.getState().consume('homeSwipes');
  while(!finish)await Promise.resolve();
  events.emitLogout();events.emitLogin('b');
  useDailyUsageStore.setState({snapshot:{date:new Date().toISOString().slice(0,10),counts:{}},hydrated:false});
  const second=useDailyUsageStore.getState().consume('homeSwipes');
  for(let i=0;i<10;i++)await Promise.resolve();
  finish();await Promise.all([first,second]);
  useDailyUsageStore.setState({snapshot:{date:new Date().toISOString().slice(0,10),counts:{}},hydrated:false});
  await useDailyUsageStore.getState().hydrate();
  expect(useDailyUsageStore.getState().getUsed('homeSwipes')).toBe(1);
});

test('does not attribute an unowned legacy cache to the current account', async()=>{
  const {useDailyUsageStore}=loadStores();
  (AsyncStorage.getItem as jest.Mock).mockImplementation((key:string)=>Promise.resolve(key==='daily_usage_v1'?JSON.stringify({date:new Date().toISOString().slice(0,10),counts:{homeSwipes:3}}):null));
  await useDailyUsageStore.getState().hydrate();
  expect(useDailyUsageStore.getState().getUsed('homeSwipes')).toBe(0);
});
