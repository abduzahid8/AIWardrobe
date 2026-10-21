import {referenceFx} from '../fixtures/budgetFx';
import { createSlotNormalizer, getBackendMacroCategory } from '../../features/outfit-generator/utils/normalizeSlots';
import { parseBudgetFromText, outfitShopSpend } from '../../src/utils/outfitBudget';
// Explicit feed fixtures: the production bundled catalog was intentionally removed.
const SHOP_CATALOG_ITEMS = [
  ['expensive-top', 'White cotton Oxford shirt', 'upper_body', 110],
  ['expensive-layer', 'Navy wool blazer', 'outerwear', 180],
  ['expensive-bottom', 'Grey tailored trousers', 'lower_body', 130],
  ['expensive-shoes', 'Brown leather penny loafers', 'shoes', 140],
  ['affordable-top', 'Blue cotton shirt', 'upper_body', 35],
  ['affordable-layer', 'Brown wool blazer', 'outerwear', 65],
  ['affordable-bottom', 'Navy tailored trousers', 'lower_body', 45],
  ['affordable-shoes', 'Black leather loafers', 'shoes', 60],
].map(([id, name, garmentType, price]) => ({
  id: String(id), name: String(name), garmentType: String(garmentType), price: Number(price),
  imageUrl: `https://example.com/${id}.jpg`, currency: 'USD', brand: 'Test catalog',
}));

function mapPoolRow(item: any, keepPrice: boolean) {
  const macroCategory = getBackendMacroCategory(item.garmentType || '', item.garmentType || '', item.name || '');
  return {
    id: item.id,
    name: item.name,
    image: typeof item.imageUrl === 'string' ? item.imageUrl : 'https://example.com/x.jpg',
    imageUrl: typeof item.imageUrl === 'string' ? item.imageUrl : 'https://example.com/x.jpg',
    type: macroCategory,
    macroCategory,
    color: 'neutral',
    brand: item.brand || '',
    isShopItem: true,
    ...(keepPrice ? { price: item.price, currency: item.currency } : {}),
  };
}

describe('live shop pool budget (250$)', () => {
  it('parses 250$ from the Generate details field', () => {
    expect(parseBudgetFromText('250$')).toEqual({ amount: 250, currency: 'USD' });
    expect(parseBudgetFromText('Work: 250$')).toEqual({ amount: 250, currency: 'USD' });
  });

  it('fits a 250$ cap when pool rows keep listed prices', () => {
    const pool = SHOP_CATALOG_ITEMS
      .filter((i) => typeof i.price === 'number' && i.price > 0)
      .slice(0, 80)
      .map((i) => mapPoolRow(i, true));
    const normalize = createSlotNormalizer({
      weather: undefined,
      baseItem: null,
      baseItemId: null,
      currentSource: 'shop',
      selectedOccasion: 'work',
      generationNonceRef: { current: 1 },
      poolFallback: pool,
      budgetText: '250$',
    });
    const expensiveIncoming = pool
      .filter((i) => (i.price || 0) > 80)
      .slice(0, 4);
    expect(expensiveIncoming.length).toBeGreaterThan(0);
    const out = normalize(expensiveIncoming, 'old_money', 0, 7, pool);
    expect(out.length).toBe(4);
    expect(outfitShopSpend(out)).toBeGreaterThan(0);
    expect(outfitShopSpend(out)).toBeLessThanOrEqual(250);
  });

  it('cannot enforce a cap when candidatePool strips prices (the live bug)', () => {
    const stripped = SHOP_CATALOG_ITEMS
      .filter((i) => typeof i.price === 'number' && i.price > 0)
      .slice(0, 80)
      .map((i) => mapPoolRow(i, false));
    // Incoming from the edge still has prices — without pool prices the
    // normalizer used to keep those expensive pieces.
    const incoming = SHOP_CATALOG_ITEMS
      .filter((i) => typeof i.price === 'number' && i.price > 90)
      .slice(0, 4)
      .map((i) => mapPoolRow(i, true));
    const incomingSpend = outfitShopSpend(incoming);
    expect(incomingSpend).toBeGreaterThan(250);
    const normalize = createSlotNormalizer({
      weather: undefined,
      baseItem: null,
      baseItemId: null,
      currentSource: 'shop',
      selectedOccasion: 'work',
      generationNonceRef: { current: 1 },
      poolFallback: stripped,
      budgetText: '250$',
    });
    const out = normalize(incoming, 'old_money', 0, 7, stripped);
    // With stripped pool, cheaper swaps are invisible. This documents the
    // failure mode; the keep-price test above is the required production path.
    expect(stripped.every((i) => i.price == null)).toBe(true);
    expect(out.length).toBeGreaterThan(0);
  });
});

describe('budget transport after prompt sanitization', () => {
  const { withOutfitBudget, stripBudgetFromText } = require('../../src/utils/outfitBudget');
  it('retains the exact converted cap in the JSON body when prompt budget wording is removed', () => {
    const stored = { avoidColors: ['red'], maxShopBudget: { amount: 1000, currency: 'USD' } };
    const budget = { amount: 3150000, currency: 'UZS' };
    const body = JSON.parse(JSON.stringify({ prompt: stripBudgetFromText('office outfit under $250'), preferences: withOutfitBudget(stored, budget) }));
    expect(body.prompt).not.toContain('250');
    expect(body.preferences.maxShopBudget).toEqual(budget);
    expect(body.preferences.avoidColors).toEqual(['red']);
    expect(stored.maxShopBudget).toEqual({ amount: 1000, currency: 'USD' });
  });
  it('preserves an explicit zero cap and does not mutate caller budget data', () => {
    const budget = { amount: 0, currency: 'USD' };
    const outgoing = withOutfitBudget({}, budget);
    expect(outgoing.maxShopBudget).toEqual(budget);
    outgoing.maxShopBudget.amount = 100;
    expect(budget.amount).toBe(0);
  });
});

it('recognizes an explicit zero-dollar limit rather than dropping it',()=>{
  expect(parseBudgetFromText('Budget under $0')).toEqual({amount:0,currency:'USD'});
});

it('selects the stricter converted limit when text contains different currencies', () => {
  expect(parseBudgetFromText('Use $100 or £90 as the maximum', referenceFx)).toMatchObject({amount:100,currency:'USD'});
  expect(parseBudgetFromText('Use $100 or €105 as the maximum', referenceFx)).toMatchObject({amount:100,currency:'USD'});
});
