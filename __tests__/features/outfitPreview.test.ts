import { getOutfitItemMacroCategory, getOutfitPreviewSlots } from '../../features/outfit-generator/utils/outfitPreview';

describe('outfitPreview', () => {
  it('infers bottoms using the canonical bottom category', () => {
    expect(getOutfitItemMacroCategory({ type: 'pants' })).toBe('bottom');
    expect(getOutfitItemMacroCategory({ type: 'clothing', name: 'Wide Leg Trousers' })).toBe('bottom');
    expect(getOutfitItemMacroCategory({ type: 'Slim Fit Jeans', macroCategory: 'pants' })).toBe('bottom');
  });

  it('keeps all outfit items in their original order instead of re-sorting them', () => {
    const slots = getOutfitPreviewSlots([
      { id: 'shoes', type: 'Loafers' },
      { id: 'outerwear', type: 'Blazer' },
      { id: 'bottom', type: 'Wide Leg Trousers' },
      { id: 'top', type: 'Oxford Shirt' },
    ]);

    expect(slots.map(slot => slot.label)).toEqual(['Shoes', 'Outerwear', 'Bottom', 'Top']);
    expect(slots.map(slot => slot.item.id)).toEqual(['shoes', 'outerwear', 'bottom', 'top']);
  });
});
