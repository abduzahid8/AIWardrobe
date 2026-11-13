/** @jest-environment node */
import { OutfitGenerationFailure } from '../../src/services/outfitGenerationFailure';

function loadDevModule(): typeof import('../../src/services/stylist6Dev') {
  let mod: typeof import('../../src/services/stylist6Dev') | undefined;
  jest.isolateModules(() => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    mod = require('../../src/services/stylist6Dev') as typeof import('../../src/services/stylist6Dev');
  });
  if (!mod) throw new Error('module failed to load');
  return mod;
}

const previousDev = (globalThis as any).__DEV__;
const previousFlag = process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE;
afterEach(() => {
  (globalThis as any).__DEV__ = previousDev;
  if (previousFlag === undefined) delete process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE;
  else process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE = previousFlag;
  jest.restoreAllMocks();
});

describe('stylist6 cloud completion safety', () => {
  test('three server-owned looks retain their existing complete contract', () => {
    const { stylist6CloudCompletion } = loadDevModule();
    expect(stylist6CloudCompletion({
      success: true, generationVersion: 'stylist-6-cloud-staging-1',
      status: 'complete', outfits: [{}, {}, {}],
    })).toBe('complete');
  });
  test('one or two model-approved displayed looks may render honestly, never three invented looks', () => {
    const { stylist6CloudCompletion } = loadDevModule();
    const partial = {
      success: true, generationVersion: 'stylist-6-cloud-staging-1',
      status: 'partial', visualStatus: 'NVIDIA_APPLIED',
      outfits: [{ id: 'look1', fashionDiagnostics: { visualVerdict: 'JUDGED' } },
        { id: 'look2', fashionDiagnostics: { visualVerdict: 'JUDGED' } }],
      visualReceipt: { displayedJudged: 2, displayedLooks: 2, visualApproval: 'verified_partial' },
    };
    expect(stylist6CloudCompletion(partial)).toBe('verified_partial');
    expect(stylist6CloudCompletion({ ...partial, visualReceipt: {
      displayedJudged: 1, displayedLooks: 2, visualApproval: 'verified_partial',
    } })).toBeNull();
    expect(stylist6CloudCompletion({ ...partial, outfits: [{ id: 'unreviewed' },
      partial.outfits[1]] })).toBeNull();
    expect(stylist6CloudCompletion({ ...partial, visualReceipt: {
      displayedJudged: 2, displayedLooks: 2, visualApproval: 'unverified_fallback',
    } })).toBeNull();
    expect(stylist6CloudCompletion({ ...partial, visualStatus: 'NVIDIA_ABSTAINED' })).toBeNull();
    expect(stylist6CloudCompletion({ ...partial, outfits: [{}, {}, {}],
      visualReceipt: { displayedJudged: 3, displayedLooks: 3 } })).toBeNull();
  });
  test('wrong engine, unsuccessful response and missing receipt cannot masquerade as cloud results', () => {
    const { stylist6CloudCompletion } = loadDevModule();
    expect(stylist6CloudCompletion({ success: true, generationVersion: 'stylist-5-v2',
      status: 'complete', outfits: [{}, {}, {}] })).toBeNull();
    expect(stylist6CloudCompletion({ success: false, generationVersion: 'stylist-6-cloud-staging-1',
      status: 'complete', outfits: [{}, {}, {}] })).toBeNull();
    expect(stylist6CloudCompletion({ success: true, generationVersion: 'stylist-6-cloud-staging-1',
      status: 'partial', visualStatus: 'NVIDIA_APPLIED', outfits: [{}] })).toBeNull();
  });
});

describe('stylist6 dev bridge client', () => {
  test('gate is off without the explicit dev opt-in, even in dev bundles', () => {
    (globalThis as any).__DEV__ = true;
    delete process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE;
    expect(loadDevModule().STYLIST6_DEV_ENABLED).toBe(false);
  });

  test('gate is off in release builds even with the flag set', () => {
    (globalThis as any).__DEV__ = false;
    process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE = 'true';
    expect(loadDevModule().STYLIST6_DEV_ENABLED).toBe(false);
  });

  test('gate opens only for dev bundles with the explicit opt-in', () => {
    (globalThis as any).__DEV__ = true;
    process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE = 'true';
    expect(loadDevModule().STYLIST6_DEV_ENABLED).toBe(true);
  });

  test('only stylist-6-dev versions count as Stylist 6 decks', () => {
    (globalThis as any).__DEV__ = true;
    process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE = 'true';
    const { isStylist6DevVersion } = loadDevModule();
    expect(isStylist6DevVersion('stylist-6-dev-local-20261111')).toBe(true);
    expect(isStylist6DevVersion('stylist-5-editorial-v2-rc-20261109-r2')).toBe(false);
    expect(isStylist6DevVersion(undefined)).toBe(false);
    expect(isStylist6DevVersion('')).toBe(false);
  });

  test('request body preserves SKU identity, prices, photos and engine inputs', () => {
    (globalThis as any).__DEV__ = true;
    process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE = 'true';
    const { buildStylist6BridgeBody } = loadDevModule();
    const row = {
      id: 'shop_123', name: 'Wool Blazer', brand: 'Burton', price: 79, currency: 'GBP',
      image_url: 'https://example.com/blazer.jpg', source_url: 'https://example.com/buy',
      description: 'blazer', is_active: true,
    };
    const body = buildStylist6BridgeBody({
      request: { style: 'classic', occasion: 'Work', brief: 'Work', weather: { temp: 20, condition: 'clear' } },
      catalogRows: [row],
    });
    expect(body.catalogRows).toEqual([row]);
    expect(body.request.style).toBe('classic');
    expect(body.request.occasion).toBe('Work');
  });

  test('disabled gate throws without touching the network', async () => {
    (globalThis as any).__DEV__ = false;
    process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE = 'true';
    const { invokeStylist6Bridge } = loadDevModule();
    const fetchSpy = jest.spyOn(globalThis, 'fetch' as any);
    await expect(invokeStylist6Bridge({
      request: { style: 'classic', occasion: 'Work', brief: '' },
      catalogRows: [{ id: 'x' }],
    })).rejects.toMatchObject({ code: 'stylist6_dev_disabled' });
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  test('a Stylist 6 ranking failure surfaces its real code, never a Stylist 5 label', async () => {
    (globalThis as any).__DEV__ = true;
    process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE = 'true';
    const { invokeStylist6Bridge } = loadDevModule();
    jest.spyOn(globalThis, 'fetch' as any).mockResolvedValue({
      ok: false,
      status: 502,
      json: async () => ({ success: false, code: 'stylist6_no_feasible_looks', stage: 'stylist6_ranking' }),
    } as any);
    const failure: any = await invokeStylist6Bridge({
      request: { style: 'classic', occasion: 'Work', brief: '' },
      catalogRows: [{ id: 'x' }],
    }).catch((error) => error);
    // Identity check by contract, not instanceof: the module under test is
    // loaded in an isolated registry with its own class copy.
    expect(failure?.name).toBe('OutfitGenerationFailure');
    expect(failure?.code).toBe('stylist6_no_feasible_looks');
    expect(String(failure?.code)).not.toMatch(/stylist-?5/i);
  });

  test('exact-deck acceptance keeps knit-top Stylist 6 looks with identity intact', () => {
    (globalThis as any).__DEV__ = true;
    process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE = 'true';
    const { acceptStylist6DevDeck } = loadDevModule();
    const knitTop = {
      id: 'top-knit-1', macroCategory: 'top', color: 'cream', name: 'Milk-colored extra-fine merino wool crew neck',
      brand: 'B', imageUrl: 'https://example.com/knit.jpg', image: 'https://example.com/knit.jpg',
      isShopItem: true, price: 120, currency: 'USD', shopUrl: 'https://example.com/buy-knit',
    };
    const look = {
      id: 's6_1', description: 'Stylist 6 look',
      items: [
        knitTop,
        { id: 'outer-1', macroCategory: 'outerwear', name: 'Wool Overcoat', imageUrl: 'https://example.com/o.jpg', isShopItem: true, price: 300, currency: 'USD' },
        { id: 'bottom-1', macroCategory: 'bottom', name: 'Wool pants', imageUrl: 'https://example.com/b.jpg', isShopItem: true, price: 150, currency: 'USD' },
        { id: 'shoes-1', macroCategory: 'shoes', name: 'Loafer', imageUrl: 'https://example.com/s.jpg', isShopItem: true, price: 200, currency: 'USD' },
      ],
    };
    const { accepted, rejections } = acceptStylist6DevDeck([look]);
    expect(accepted).toHaveLength(1);
    expect(accepted[0].items[0]).toMatchObject({ id: 'top-knit-1', price: 120, currency: 'USD' });
    expect(rejections).toEqual({});
  });

  test('exact-deck acceptance counts unrenderable looks without repairing them', () => {
    (globalThis as any).__DEV__ = true;
    process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE = 'true';
    const { acceptStylist6DevDeck } = loadDevModule();
    const item = (id: string, macroCategory: string, extra: Record<string, unknown> = {}) => ({
      id, macroCategory, name: `${macroCategory} ${id}`, imageUrl: 'https://example.com/x.jpg', isShopItem: true, ...extra,
    });
    const deck = (items: Record<string, unknown>[]) => ({ id: `look-${Math.random()}`, items });
    const { accepted, rejections } = acceptStylist6DevDeck([
      // Duplicate ids with otherwise valid slots.
      deck([item('same', 'top'), item('same', 'outerwear'), item('b', 'bottom'), item('s', 'shoes')]),
      // One cleartext photo with otherwise valid slots.
      deck([{ ...item('t', 'top'), imageUrl: 'http://plain.example.com/a.jpg', image: undefined }, item('o', 'outerwear'), item('b', 'bottom'), item('s', 'shoes')]),
      // Wrong slot multiplicity.
      deck([item('a', 'top'), item('b', 'top'), item('c', 'bottom')]),
      // Owned piece in a shop-pipeline deck.
      deck([{ ...item('t', 'top'), isShopItem: false }, item('o', 'outerwear'), item('b', 'bottom'), item('s', 'shoes')]),
      'not-a-look',
    ]);
    expect(accepted).toHaveLength(0);
    expect(rejections).toMatchObject({
      invalid_product_identity: 1, unrenderable_photo: 1, structure: 2, source_or_anchor: 1,
    });
  });

  test('a Stylist 6 deck resolves with its own engine version', async () => {
    (globalThis as any).__DEV__ = true;
    process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE = 'true';
    const { invokeStylist6Bridge, isStylist6DevVersion } = loadDevModule();
    const deck = {
      success: true,
      generationVersion: 'stylist-6-dev-local-20261111',
      source: 'stylist-6-dev-local',
      outfits: [
        {
          id: 's6_1', description: 'Stylist 6 look', concept: { id: 'c', title: 'Concept' },
          stylingTips: [], confidence: 0.8, matchScore: 0.8, engineVersion: 'stylist-6-dev-local-20261111',
          items: [{
            id: 'shop_1', macroCategory: 'top', color: 'navy', name: 'Shirt', brand: 'B',
            imageUrl: 'https://example.com/s.jpg', image: 'https://example.com/s.jpg',
            isShopItem: true, price: 50, currency: 'USD', shopUrl: 'https://example.com/buy',
          }],
        },
      ],
    };
    jest.spyOn(globalThis, 'fetch' as any).mockResolvedValue({
      ok: true, status: 200, json: async () => deck,
    } as any);
    const data = await invokeStylist6Bridge({
      request: { style: 'classic', occasion: 'Work', brief: '' },
      catalogRows: [{ id: 'shop_1' }],
    });
    expect(isStylist6DevVersion(data.generationVersion)).toBe(true);
    expect(data.outfits).toHaveLength(1);
    expect(data.outfits[0].items[0]).toMatchObject({ id: 'shop_1', price: 50, currency: 'USD' });
  });
});

describe('safe visual receipt reporting for owner iOS screenshots',()=>{
  test('NVIDIA_ABSTAINED exposes bounded cause and counts without leaking provider data',()=>{
    (globalThis as any).__DEV__ = true;
    process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE = 'true';
    const {stylist6VisualDiagnostic}=loadDevModule();
    const received={
      visualStatus:'NVIDIA_ABSTAINED',
      visualReceipt:{
        providerCalls:0,validVerdicts:0,abstentions:4,
        events:[{
          reason:'reference_unresolved',outcome:'abstain',
          candidateId:'PRIVATE-SKU', imageUrl:'https://seller.example/private.jpg',
          jwt:'PRIVATE-JWT',apiKey:'PRIVATE-KEY',
        }],
      },
    };
    expect(stylist6VisualDiagnostic(received)).toBe('NVIDIA_ABSTAINED c0/v0/a4 · reference_unresolved');
  });

  test('partial NVIDIA_APPLIED names how many displayed looks actually have evidence',()=>{
    (globalThis as any).__DEV__=true;
    process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE='true';
    const {stylist6VisualDiagnostic}=loadDevModule();
    const response={
      visualStatus:'NVIDIA_APPLIED',
      visualReceipt:{
        providerCalls:3, validVerdicts:2, abstentions:2,
        displayedLooks:3, displayedJudged:1,
        events:[{reason:'product_image_unresolved',mediaSlot:'outerwear',
          mediaFailure:'http_403',source:'Bearer PRIVATE_TOKEN',url:'https://private.example'}],
      },
    };
    expect(stylist6VisualDiagnostic(response)).toBe(
      'NVIDIA_APPLIED c3/v2/a2 · shown1/3 · product_image_unresolved (outerwear/http_403)');
  });

  test('unknown server strings are never echoed or treated as a verdict',()=>{
    (globalThis as any).__DEV__ = true;
    process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE = 'true';
    const {stylist6VisualDiagnostic}=loadDevModule();
    expect(stylist6VisualDiagnostic({visualStatus:'NVIDIA_ABSTAINED',visualReceipt:{
      events:[{reason:'Bearer secret-key'}],providerCalls:-100,validVerdicts:0,abstentions:20,
    }})).toBe('NVIDIA_ABSTAINED');
    expect(stylist6VisualDiagnostic({visualStatus:'arbitrary_server_echo'})).toBeNull();
    expect(stylist6VisualDiagnostic({visualStatus:'NVIDIA_NOT_RUN'})).toBe('NVIDIA_NOT_RUN');
  });
});

describe('stylist6 cloud staging client', () => {
  const mockInvoke = jest.fn();
  beforeEach(() => {
    mockInvoke.mockReset();
    // The cloud client resolves supabase lazily; mock the whole module so no
    // network or real client is involved.
    jest.mock('../../lib/supabase', () => ({
      supabase: {
        auth: { getSession: async () => ({ data: { session: { access_token: 'test-token' } } }) },
        functions: { invoke: mockInvoke },
      },
    }));
  });

  test('cloud versions are recognized only with the stylist-6-cloud prefix', () => {
    (globalThis as any).__DEV__ = true;
    process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE = 'true';
    const mod = loadDevModule();
    expect(mod.isStylist6CloudVersion('stylist-6-cloud-staging-1')).toBe(true);
    expect(mod.isStylist6CloudVersion('stylist-6-dev-local-20261111-r2')).toBe(false);
    expect(mod.isStylist6CloudVersion('stylist-5-editorial-v2-rc-20261109-r2')).toBe(false);
    expect(mod.isStylist6CloudVersion(null)).toBe(false);
  });

  test('cloud invoke refuses without the dev gate, never touching the network', async () => {
    (globalThis as any).__DEV__ = false;
    process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE = 'true';
    const mod = loadDevModule();
    const spy = jest.spyOn(globalThis, 'fetch' as any);
    await expect(mod.invokeStylist6Cloud({
      request: { style: 'classic', occasion: 'Work', brief: '' },
    })).rejects.toMatchObject({ code: 'stylist6_dev_disabled' });
    expect(spy).not.toHaveBeenCalled();
  });

  test('a non-2xx FunctionsHttpError surfaces the server code and stage, not a generic label', async () => {
    (globalThis as any).__DEV__ = true;
    process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE = 'true';
    const { invokeStylist6Cloud } = loadDevModule();
    mockInvoke.mockResolvedValue({
      data: null,
      error: Object.assign(new Error('FunctionsHttpError'), {
        name: 'FunctionsHttpError',
        context: new Response(
          JSON.stringify({ success: false, code: 'anchor_not_found', stage: 'stylist6_admission' }),
          { status: 422 },
        ),
      }),
    });
    const failure: any = await invokeStylist6Cloud({
      request: { style: 'classic', occasion: 'Work', brief: '' },
    }).catch((error) => error);
    expect(failure?.name).toBe('OutfitGenerationFailure');
    expect(failure?.code).toBe('anchor_not_found');
    expect(failure?.stage).toBe('stylist6_admission');
  });

  test('an error response without a JSON body still fails with a transport-safe code', async () => {
    (globalThis as any).__DEV__ = true;
    process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE = 'true';
    const { invokeStylist6Cloud } = loadDevModule();
    mockInvoke.mockResolvedValue({
      data: null,
      error: Object.assign(new Error('FunctionsHttpError'), {
        name: 'FunctionsHttpError',
        context: new Response('Bad Gateway', { status: 502 }),
      }),
    });
    const failure: any = await invokeStylist6Cloud({
      request: { style: 'classic', occasion: 'Work', brief: '' },
    }).catch((error) => error);
    expect(failure?.name).toBe('OutfitGenerationFailure');
    // No application body to read: the transport message is kept verbatim
    // instead of inventing a server verdict.
    expect(failure?.code).toBe('FunctionsHttpError');
  });

  test('a wrong-engine deck is rejected even when the transport succeeds', async () => {
    (globalThis as any).__DEV__ = true;
    process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE = 'true';
    const { invokeStylist6Cloud } = loadDevModule();
    mockInvoke.mockResolvedValue({
      data: {
        success: true, generationVersion: 'stylist-5-editorial-v2-rc-20261109-r2', outfits: [],
      },
      error: null,
    });
    await expect(invokeStylist6Cloud({
      request: { style: 'classic', occasion: 'Work', brief: '' },
    })).rejects.toMatchObject({ code: 'stylist6_wrong_engine' });
  });

  test('an abort signal propagates to the transport invocation', async () => {
    (globalThis as any).__DEV__ = true;
    process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE = 'true';
    const { invokeStylist6Cloud } = loadDevModule();
    const controller = new AbortController();
    mockInvoke.mockResolvedValue({ data: { success: true, generationVersion: 'stylist-6-cloud-staging-1', outfits: [] }, error: null });
    await invokeStylist6Cloud(
      { request: { style: 'classic', occasion: 'Work', brief: '' } },
      { signal: controller.signal },
    );
    expect(mockInvoke).toHaveBeenCalledTimes(1);
    const options = mockInvoke.mock.calls[0][1];
    expect(options?.signal).toBe(controller.signal);
  });

  test('stylist6Language normalizes the active i18n locale to a supported copy language', () => {
    (globalThis as any).__DEV__ = true;
    process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE = 'true';
    const { stylist6Language } = loadDevModule();
    expect(stylist6Language('en')).toBe('en');
    expect(stylist6Language('ru')).toBe('ru');
    expect(stylist6Language('uz')).toBe('uz');
    expect(stylist6Language('ru-RU')).toBe('ru');
    expect(stylist6Language('en-US')).toBe('en');
    expect(stylist6Language(undefined)).toBe('en');
    expect(stylist6Language('de')).toBe('en');
    expect(stylist6Language(42)).toBe('en');
  });

  test('the cloud request forwards the active language for localized deck copy', async () => {
    (globalThis as any).__DEV__ = true;
    process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE = 'true';
    const { invokeStylist6Cloud } = loadDevModule();
    mockInvoke.mockResolvedValue({ data: { success: true, generationVersion: 'stylist-6-cloud-staging-1', outfits: [] }, error: null });
    await invokeStylist6Cloud({
      request: { style: 'classic', occasion: 'Work', brief: '', language: 'ru' },
    });
    expect(mockInvoke).toHaveBeenCalledTimes(1);
    const [, invokeOptions] = mockInvoke.mock.calls[0];
    expect(invokeOptions?.body?.request?.language).toBe('ru');
  });

  test('cloud deck acceptance is source-aware and never repairs', () => {
    (globalThis as any).__DEV__ = true;
    process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE = 'true';
    const { acceptStylist6CloudDeck } = loadDevModule();
    const shop = (id: string, macroCategory: string, extra: Record<string, unknown> = {}) => ({
      id, macroCategory, name: `${macroCategory} ${id}`, imageUrl: `https://example.com/${id}.jpg`,
      isShopItem: true, isWardrobe: false, price: 100, currency: 'USD', shopUrl: `https://example.com/buy/${id}`, ...extra,
    });
    const owned = (id: string, macroCategory: string) => ({
      id, macroCategory, name: `${macroCategory} ${id}`, imageUrl: `https://example.com/${id}.jpg`,
      isShopItem: false, isWardrobe: true, price: null, currency: null, shopUrl: null,
    });
    const look = (...items: Record<string, unknown>[]) => ({ id: 'look', items });
    // catalog: a shop deck passes; one owned piece rejects the look.
    expect(acceptStylist6CloudDeck([look(shop('t', 'top'), shop('o', 'outerwear'), shop('b', 'bottom'), shop('s', 'shoes'))], 'catalog').accepted).toHaveLength(1);
    expect(acceptStylist6CloudDeck([look(owned('t', 'top'), shop('o', 'outerwear'), shop('b', 'bottom'), shop('s', 'shoes'))], 'catalog').rejections).toMatchObject({ source_or_anchor: 1 });
    // wardrobe: owned pieces pass; shop pieces reject.
    expect(acceptStylist6CloudDeck([look(owned('t', 'top'), owned('b', 'bottom'), owned('s', 'shoes'))], 'wardrobe').accepted).toHaveLength(1);
    expect(acceptStylist6CloudDeck([look(owned('t', 'top'), owned('b', 'bottom'), shop('s', 'shoes'))], 'wardrobe').rejections).toMatchObject({ source_or_anchor: 1 });
    // mixed: both provenances coexist, shop pieces keep the merchant contract.
    expect(acceptStylist6CloudDeck([look(owned('t', 'top'), shop('o', 'outerwear'), owned('b', 'bottom'), shop('s', 'shoes'))], 'mixed').accepted).toHaveLength(1);
    expect(acceptStylist6CloudDeck(
      [look(owned('t', 'top'), shop('o', 'outerwear', { price: null, currency: null }), owned('b', 'bottom'), shop('s', 'shoes'))],
      'mixed',
    ).rejections).toMatchObject({ source_or_anchor: 1 });
    // Structure is still enforced exactly like the local bridge deck.
    expect(acceptStylist6CloudDeck([look(shop('a', 'top'), shop('b', 'top'), shop('c', 'bottom'))], 'catalog').rejections).toMatchObject({ structure: 1 });
  });
  test('owner reproduction: complete three-look cloud catalog deck survives unchanged', () => {
    (globalThis as any).__DEV__ = true;
    process.env.EXPO_PUBLIC_STYLIST6_DEV_BRIDGE = 'true';
    const { acceptStylist6CloudDeck } = loadDevModule();
    const roles = ['top', 'outerwear', 'bottom', 'shoes'];
    const outfits = ['a', 'b', 'c'].map((lookId) => ({
      id: `stylist6-${lookId}`,
      items: roles.map(role => ({
        id: `sku-${lookId}-${role}`, macroCategory: role,
        name: `${lookId} ${role}`, imageUrl: `https://merchant.example/${lookId}/${role}.jpg`,
        isShopItem: true, isWardrobe: false,
        shopUrl: `https://merchant.example/product/${lookId}-${role}`,
        price: 100, currency: 'UZS',
      })),
    }));
    const result = acceptStylist6CloudDeck(outfits, 'catalog');
    expect(result.rejections).toEqual({});
    expect(result.accepted).toHaveLength(3);
    expect(result.accepted).toEqual(outfits);
    expect(result.accepted.map((look: any) => look.items.map((item: any) => item.id))).toEqual(
      outfits.map(look => look.items.map(item => item.id)),
    );
  });

});
