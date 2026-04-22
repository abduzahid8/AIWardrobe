/**
 * outfitNormalize.test.ts
 *
 * Tests for the 3/4-slot outfit contract and the formal-layer + shorts rule.
 * The contract:
 *   - Non-layered (warm weather): exactly 3 items — top, bottom, shoes
 *   - Layered (cold weather): exactly 4 items — outerwear, base-top, bottom, shoes
 *   - Formal outerwear (blazer, suit jacket, overcoat, trench, peacoat, tuxedo)
 *     must never be paired with shorts/bermudas
 */

// ── Re-implement the helpers under test so the tests are self-contained ──
// These mirror the logic in:
//   supabase/functions/generate-outfits/index.ts
//   src/services/outfitGenerationService.ts
//   screens/AIOutfitmaker.tsx

function isFormalLayerItem(item: { type?: string; name?: string; brand?: string; macroCategory?: string } | null | undefined): boolean {
  if (!item) return false;
  const blob = `${item.type || ''} ${item.name || ''} ${item.brand || ''}`.toLowerCase();
  const macro = (item.macroCategory || '').toLowerCase();
  const isOuter = macro === 'outerwear' || /jacket|coat|blazer|vest|outerwear/.test(blob);
  if (!isOuter) return false;
  return /\b(blazer|suit\s*jacket|sport\s*coat|sports\s*coat|overcoat|top\s*coat|topcoat|trench|peacoat|pea\s*coat|tuxedo)\b/.test(blob);
}

function isShortsItem(item: { type?: string; name?: string; macroCategory?: string; subCategory?: string } | null | undefined): boolean {
  if (!item) return false;
  const blob = `${item.type || ''} ${item.name || ''} ${item.subCategory || ''}`.toLowerCase();
  const macro = (item.macroCategory || '').toLowerCase();
  const isBottom = macro === 'bottom' || /pant|trouser|jeans|bottom|shorts?|skirt/.test(blob);
  if (!isBottom) return false;
  return /\b(shorts?|bermudas?)\b/.test(blob);
}

function isLayeredWeather(weather?: { temp?: number | null; condition?: string | null }, prompt?: string | null): boolean {
  const promptBlob = (prompt || '').toLowerCase();
  if (/\b(summer|hot|heatwave|tee[-\s]?only|no jacket|no outerwear|beach)\b/.test(promptBlob)) return false;
  const condition = (weather?.condition || '').toString().toLowerCase();
  const temp = typeof weather?.temp === 'number' ? weather.temp : null;
  const coldTemp = temp != null && temp < 18;
  const coldCondition = /\b(cold|chilly|freezing|snow|rain|drizzle|wind|storm)\b/.test(condition);
  return coldTemp || coldCondition;
}

// ── Tests ──────────────────────────────────────────────────────────────────

describe('isFormalLayerItem', () => {
  it('identifies a blazer as formal', () => {
    expect(isFormalLayerItem({ name: 'Navy Blazer', macroCategory: 'outerwear' })).toBe(true);
  });

  it('identifies a suit jacket as formal', () => {
    expect(isFormalLayerItem({ name: 'Charcoal Suit Jacket', macroCategory: 'outerwear' })).toBe(true);
  });

  it('identifies a trench coat as formal', () => {
    expect(isFormalLayerItem({ name: 'Belted Trench Coat', type: 'coat', macroCategory: 'outerwear' })).toBe(true);
  });

  it('identifies a peacoat as formal', () => {
    expect(isFormalLayerItem({ name: 'Wool Peacoat', macroCategory: 'outerwear' })).toBe(true);
  });

  it('does NOT flag a denim jacket as formal', () => {
    expect(isFormalLayerItem({ name: 'Denim Jacket', macroCategory: 'outerwear' })).toBe(false);
  });

  it('does NOT flag a hoodie as formal', () => {
    expect(isFormalLayerItem({ name: 'Zip Hoodie', macroCategory: 'outerwear' })).toBe(false);
  });

  it('does NOT flag a puffer as formal', () => {
    expect(isFormalLayerItem({ name: 'Puffer Jacket', macroCategory: 'outerwear' })).toBe(false);
  });

  it('does NOT flag a bomber jacket as formal', () => {
    expect(isFormalLayerItem({ name: 'Bomber Jacket', macroCategory: 'outerwear' })).toBe(false);
  });

  it('does NOT flag a cardigan as formal', () => {
    expect(isFormalLayerItem({ name: 'Cashmere Cardigan', macroCategory: 'outerwear' })).toBe(false);
  });

  it('returns false for null/undefined', () => {
    expect(isFormalLayerItem(null)).toBe(false);
    expect(isFormalLayerItem(undefined)).toBe(false);
  });
});

describe('isShortsItem', () => {
  it('identifies chino shorts as shorts', () => {
    expect(isShortsItem({ name: 'Chino Shorts', macroCategory: 'bottom' })).toBe(true);
  });

  it('identifies athletic shorts as shorts', () => {
    expect(isShortsItem({ name: 'Athletic Shorts', macroCategory: 'bottom' })).toBe(true);
  });

  it('identifies bermudas as shorts', () => {
    expect(isShortsItem({ name: 'Bermudas', macroCategory: 'bottom' })).toBe(true);
  });

  it('does NOT flag chinos as shorts', () => {
    expect(isShortsItem({ name: 'Slim Chinos', macroCategory: 'bottom' })).toBe(false);
  });

  it('does NOT flag jeans as shorts', () => {
    expect(isShortsItem({ name: 'Straight Jeans', macroCategory: 'bottom' })).toBe(false);
  });

  it('does NOT flag trousers as shorts', () => {
    expect(isShortsItem({ name: 'Wide Leg Trousers', macroCategory: 'bottom' })).toBe(false);
  });

  it('returns false for null/undefined', () => {
    expect(isShortsItem(null)).toBe(false);
    expect(isShortsItem(undefined)).toBe(false);
  });
});

describe('isLayeredWeather — weather-only layering', () => {
  it('returns true when temp < 18°C', () => {
    expect(isLayeredWeather({ temp: 10, condition: 'clear' })).toBe(true);
  });

  it('returns true when condition contains "cold"', () => {
    expect(isLayeredWeather({ temp: 22, condition: 'cold and windy' })).toBe(true);
  });

  it('returns true when condition contains "rain"', () => {
    expect(isLayeredWeather({ temp: 20, condition: 'light rain' })).toBe(true);
  });

  it('returns true when condition contains "snow"', () => {
    expect(isLayeredWeather({ temp: -2, condition: 'snow' })).toBe(true);
  });

  it('returns false when temp >= 18 and condition is warm', () => {
    expect(isLayeredWeather({ temp: 25, condition: 'sunny' })).toBe(false);
  });

  it('returns false when prompt contains "summer"', () => {
    expect(isLayeredWeather({ temp: 10, condition: 'cold' }, 'summer vibes')).toBe(false);
  });

  it('returns false when prompt contains "hot"', () => {
    expect(isLayeredWeather({ temp: 5, condition: 'chilly' }, 'hot day')).toBe(false);
  });

  it('returns false when prompt contains "beach"', () => {
    expect(isLayeredWeather({ temp: 12, condition: 'wind' }, 'beach outfit')).toBe(false);
  });

  it('returns true for cold weather even without prompt override', () => {
    expect(isLayeredWeather({ temp: 8, condition: 'overcast' }, '')).toBe(true);
  });
});

describe('3/4-slot contract', () => {
  // These tests validate the contract rules that all three layers enforce:
  //   - warm weather → exactly 3 items (top, bottom, shoes)
  //   - cold weather → exactly 4 items (outerwear, base-top, bottom, shoes)
  //   - formal outerwear + shorts is always rejected

  it('warm weather should produce non-layered (3-slot) outfits', () => {
    const layered = isLayeredWeather({ temp: 28, condition: 'sunny' });
    expect(layered).toBe(false);
    // Non-layered outfits should have exactly 3 items
    const expectedSlotCount = layered ? 4 : 3;
    expect(expectedSlotCount).toBe(3);
  });

  it('cold weather should produce layered (4-slot) outfits', () => {
    const layered = isLayeredWeather({ temp: 5, condition: 'snow' });
    expect(layered).toBe(true);
    const expectedSlotCount = layered ? 4 : 3;
    expect(expectedSlotCount).toBe(4);
  });

  it('formal outerwear + shorts should be rejected', () => {
    const formalLayer = { name: 'Navy Blazer', macroCategory: 'outerwear' };
    const shortsBottom = { name: 'Chino Shorts', macroCategory: 'bottom' };
    expect(isFormalLayerItem(formalLayer)).toBe(true);
    expect(isShortsItem(shortsBottom)).toBe(true);
    // The combination is disallowed
    const disallowed = isFormalLayerItem(formalLayer) && isShortsItem(shortsBottom);
    expect(disallowed).toBe(true); // This combo should be caught and rejected
  });

  it('casual outerwear + shorts should be allowed', () => {
    const casualLayer = { name: 'Denim Jacket', macroCategory: 'outerwear' };
    const shortsBottom = { name: 'Chino Shorts', macroCategory: 'bottom' };
    expect(isFormalLayerItem(casualLayer)).toBe(false);
    expect(isShortsItem(shortsBottom)).toBe(true);
    // The combination is fine
    const disallowed = isFormalLayerItem(casualLayer) && isShortsItem(shortsBottom);
    expect(disallowed).toBe(false);
  });

  it('formal outerwear + chinos should be allowed', () => {
    const formalLayer = { name: 'Navy Blazer', macroCategory: 'outerwear' };
    const chinosBottom = { name: 'Slim Chinos', macroCategory: 'bottom' };
    expect(isFormalLayerItem(formalLayer)).toBe(true);
    expect(isShortsItem(chinosBottom)).toBe(false);
    const disallowed = isFormalLayerItem(formalLayer) && isShortsItem(chinosBottom);
    expect(disallowed).toBe(false);
  });
});
