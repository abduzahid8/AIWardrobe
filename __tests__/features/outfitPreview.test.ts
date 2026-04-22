import { getOutfitItemMacroCategory, getOutfitPreviewSlots, getOutfitPreviewTitle } from '../../features/outfit-generator/utils/outfitPreview';

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

  describe('getOutfitPreviewTitle — label fallback', () => {
    it('prefers name over type', () => {
      expect(getOutfitPreviewTitle({ name: 'Cashmere V-Neck', type: 'upper_body' })).toBe('Cashmere V-Neck');
    });

    it('surfaces type when name is empty and type is not a raw garment code', () => {
      expect(getOutfitPreviewTitle({ type: 'Polo Shirt' })).toBe('Polo Shirt');
    });

    it('falls back to macroCategory label when type is a raw garment code like upper_body', () => {
      // upper_body is a machine tag — user should see "Top" instead
      expect(getOutfitPreviewTitle({ type: 'upper_body', macroCategory: 'top' })).toBe('Top');
    });

    it('falls back to macroCategory label when type is lower_body', () => {
      expect(getOutfitPreviewTitle({ type: 'lower_body', macroCategory: 'bottom' })).toBe('Bottom');
    });

    it('falls back to macroCategory label when type is dresses', () => {
      expect(getOutfitPreviewTitle({ type: 'dresses', macroCategory: 'top' })).toBe('Top');
    });
  });
});
