import { resolveOutfitSelectionSource, offerCaveatLabel } from '../../src/services/stylingCopy';

describe('Stylist 6 per-look visual & merchant offer truth helpers', () => {
  test('only grants ai badge to individually JUDGED Stylist 6 looks when deck is NVIDIA_APPLIED', () => {
    const cloudEnvelope = { source: 'stylist-6-cloud-staging', visualStatus: 'NVIDIA_APPLIED' };
    expect(resolveOutfitSelectionSource(
      { selectionSource: 'ai', fashionDiagnostics: { visualVerdict: 'JUDGED' } },
      cloudEnvelope,
    )).toBe('ai');

    // A partial or fallback look inside an NVIDIA_APPLIED batch must NEVER wear 'ai'.
    expect(resolveOutfitSelectionSource(
      { selectionSource: 'deterministic_fallback', fashionDiagnostics: { visualVerdict: 'NOT_JUDGED' } },
      cloudEnvelope,
    )).toBe('deterministic_fallback');

    // Global NVIDIA_EVALUATED_NO_CHANGE never grants 'ai' even if a field says 'ai'.
    expect(resolveOutfitSelectionSource(
      { selectionSource: 'ai', fashionDiagnostics: { visualVerdict: 'JUDGED' } },
      { source: 'stylist-6-cloud-staging', visualStatus: 'NVIDIA_EVALUATED_NO_CHANGE' },
    )).toBe('deterministic_fallback');

    // Legacy Stylist 5 responses keep source-level provenance rules.
    expect(resolveOutfitSelectionSource({}, { source: 'ai' })).toBe('ai');
    expect(resolveOutfitSelectionSource({}, { source: 'hybrid' })).toBe('hybrid');
    expect(resolveOutfitSelectionSource({}, { source: 'legacy' })).toBe('deterministic_fallback');
  });

  test('renders honest offer caveats for shop items with unverified size/stock or stale pricing', () => {
    expect(offerCaveatLabel({
      isWardrobe: true,
      priceFreshness: 'stale',
      offerVerification: { sizeStockStatus: 'unverified' },
    })).toBeNull();

    expect(offerCaveatLabel({
      isWardrobe: false,
      priceFreshness: 'stale',
      offerVerification: { sizeStockStatus: 'unverified' },
    })).toBe('Price outdated · verify at store');

    expect(offerCaveatLabel({
      isWardrobe: false,
      priceFreshness: 'recent',
      offerVerification: { sizeStockStatus: 'unverified', limitations: ['size_stock_unverified'] },
    })).toBe('Check size & stock at store');

    expect(offerCaveatLabel({
      isWardrobe: false,
      priceFreshness: 'fresh',
      offerVerification: { sizeStockStatus: 'in_stock', limitations: [] },
    })).toBeNull();
  });
});
