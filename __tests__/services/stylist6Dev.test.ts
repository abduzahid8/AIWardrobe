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

describe('stylist6 cloud staging client', () => {
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
});
