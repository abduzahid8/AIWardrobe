import { hasProductOnlyCatalogProvenance } from '../../src/utils/productOnlyCatalog';

describe('hasProductOnlyCatalogProvenance', () => {
  it.each([
    { source: 'brand-white-bg-no-model' },
    { source: 'official-menswear-product-only' },
    { description: 'Official product-only photo on a plain background.' },
    { source: 'elegant-menswear-2026' },
  ])('accepts explicit product-only provenance: %#', (row) => {
    expect(hasProductOnlyCatalogProvenance(row)).toBe(true);
  });

  it.each([
    undefined,
    {},
    { source: 'generic-retailer-import' },
    { description: 'Seasonal menswear image' },
    { source: 'brand-no-model', description: 'The product photo shows a male model.' },
    { source: 'brand-no-model', name: 'Model styled in a wool jacket' },
  ])('rejects rows without explicit image provenance: %#', (row) => {
    expect(hasProductOnlyCatalogProvenance(row)).toBe(false);
  });
});
