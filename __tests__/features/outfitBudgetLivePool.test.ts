import { createSlotNormalizer, getBackendMacroCategory } from '../../features/outfit-generator/utils/normalizeSlots';
import { parseBudgetFromText, outfitShopSpend } from '../../src/utils/outfitBudget';
import { SHOP_CATALOG_ITEMS } from '../../data/shopCatalogItems';

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
