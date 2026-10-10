import { inferShopRowMacro } from '../../src/utils/shopCatalogOrder';
import { detectSlotMismatch } from '../../features/outfit-generator/utils/occasionRules';
import { getBackendMacroCategory } from '../../features/outfit-generator/utils/normalizeSlots';

// Zone-vs-name conflict rows: shop_catalog md-uz-62899803 is
// garment_type=lower_body with a trousers photo but a stolen blazer name
// ("100% cool wool suit blazer", live screenshot: grey pants in the LAYER
// tile). The row must stay dropped everywhere — upstream classifier,
// keyword fallback, and the pool resolver must never resurrect it by name.
describe('conflict row classification (md-uz-62899803)', () => {
  const conflictName = '100% cool wool suit blazer';

  it('inferShopRowMacro drops lower_body zone + outerwear-only name', () => {
    expect(inferShopRowMacro('lower_body', conflictName, '')).toBe('other');
    expect(inferShopRowMacro('shoes', conflictName, '')).toBe('other');
  });

  it('getBackendMacroCategory never re-derives an explicit non-clothing alias by name', () => {
    // mapShopCatalogRow writes type=category='accessories' for macro 'other'.
    expect(getBackendMacroCategory('accessories', 'accessories', conflictName)).toBe('other');
    expect(getBackendMacroCategory('accessory', 'accessory', conflictName)).toBe('other');
  });

  it('structured zone still beats a conflicting name when present', () => {
    expect(getBackendMacroCategory('lower_body', '', conflictName)).toBe('bottom');
    expect(getBackendMacroCategory('shoes', '', conflictName)).toBe('shoes');
  });

  it('absent slots still classify from the name (edge items)', () => {
    expect(getBackendMacroCategory('', '', conflictName)).toBe('outerwear');
    expect(getBackendMacroCategory('', '', 'Cotton T-Shirt')).toBe('top');
  });

  it('reverse conflicts resolve by name: pants mistyped as upper_body stay bottoms', () => {
    expect(inferShopRowMacro('upper_body', 'Cargo Tech Pant', '')).toBe('bottom');
    expect(inferShopRowMacro('upper_body', 'Th Performance Straight Fit Pant', '')).toBe('bottom');
  });

  it('legit rows keep their slots', () => {
    expect(inferShopRowMacro('lower_body', '100% cool wool suit trousers', '')).toBe('bottom');
    expect(inferShopRowMacro('upper_body', conflictName, '')).toBe('outerwear');
    expect(inferShopRowMacro('lower_body', 'Classic Cotton T-Shirt', '')).toBe('top');
    expect(inferShopRowMacro('upper_body', '43Mm Silicone Strap Sport Watch', '')).toBe('other');
  });

  it('detectSlotMismatch does not resurrect a dropped (non-canonical) macro', () => {
    expect(detectSlotMismatch('other', conflictName, 'accessories')).toBeNull();
    expect(detectSlotMismatch('accessories', conflictName, 'accessories')).toBeNull();
  });
});
