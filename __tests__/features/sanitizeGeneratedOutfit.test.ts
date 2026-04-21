import {
  sanitizeGeneratedOutfitItems,
  sanitizeGeneratedOutfitItemsDetailed,
} from '../../features/outfit-generator/utils/sanitizeGeneratedOutfit';

describe('sanitizeGeneratedOutfitItems', () => {
  const availableItems = [
    {
      id: 'top_1',
      image: 'top.png',
      type: 'Oxford Shirt',
      name: 'Oxford Shirt',
      macroCategory: 'top',
    },
    {
      id: 'bottom_1',
      image: 'bottom.png',
      type: 'Tailored Trousers',
      name: 'Tailored Trousers',
      macroCategory: 'bottom',
    },
    {
      id: 'shoes_1',
      image: 'shoes.png',
      type: 'Leather Loafers',
      name: 'Leather Loafers',
      macroCategory: 'shoes',
    },
  ];

  it('replaces duplicate tops with valid bottom and shoes slots', () => {
    const sanitizedItems = sanitizeGeneratedOutfitItems(
      [
        { id: 'top_1', image: 'top.png', type: 'Oxford Shirt', name: 'Oxford Shirt', macroCategory: 'top' },
        { id: 'top_1', image: 'top.png', type: 'Tailored Trousers', name: 'Tailored Trousers', macroCategory: 'bottom' },
        { id: 'top_1', image: 'top.png', type: 'Leather Loafers', name: 'Leather Loafers', macroCategory: 'shoes' },
      ],
      availableItems as any
    );

    expect(sanitizedItems.map((item) => item.id)).toEqual(['top_1', 'bottom_1', 'shoes_1']);
    expect(sanitizedItems.map((item) => item.macroCategory)).toEqual(['top', 'bottom', 'shoes']);
  });

  it('keeps an outerwear layer when the AI picked a full outfit correctly', () => {
    const sanitizedItems = sanitizeGeneratedOutfitItems(
      [
        { id: 'outer_1', image: 'outer.png', type: 'Wool Blazer', name: 'Wool Blazer', macroCategory: 'outerwear' },
        { id: 'top_1', image: 'top.png', type: 'Oxford Shirt', name: 'Oxford Shirt', macroCategory: 'top' },
        { id: 'bottom_1', image: 'bottom.png', type: 'Tailored Trousers', name: 'Tailored Trousers', macroCategory: 'bottom' },
        { id: 'shoes_1', image: 'shoes.png', type: 'Leather Loafers', name: 'Leather Loafers', macroCategory: 'shoes' },
      ],
      [
        ...availableItems,
        {
          id: 'outer_1',
          image: 'outer.png',
          type: 'Wool Blazer',
          name: 'Wool Blazer',
          macroCategory: 'outerwear',
        },
      ] as any
    );

    // Main-top (outerwear) comes first, followed by the second-top (base
    // layer), bottom, shoes — this matches the layered slot ordering used
    // across the UI.
    expect(sanitizedItems.map((item) => item.id)).toEqual(['outer_1', 'top_1', 'bottom_1', 'shoes_1']);
    expect(sanitizedItems.map((item) => item.macroCategory)).toEqual([
      'outerwear',
      'top',
      'bottom',
      'shoes',
    ]);
  });

  it('layered mode pulls outerwear from the wardrobe when the AI omits it', () => {
    const result = sanitizeGeneratedOutfitItemsDetailed(
      [
        { id: 'top_1', image: 'top.png', type: 'Oxford Shirt', name: 'Oxford Shirt', macroCategory: 'top' },
        { id: 'bottom_1', image: 'bottom.png', type: 'Tailored Trousers', name: 'Tailored Trousers', macroCategory: 'bottom' },
        { id: 'shoes_1', image: 'shoes.png', type: 'Leather Loafers', name: 'Leather Loafers', macroCategory: 'shoes' },
      ],
      [
        ...availableItems,
        {
          id: 'outer_1',
          image: 'outer.png',
          type: 'Wool Blazer',
          name: 'Wool Blazer',
          macroCategory: 'outerwear',
        },
      ] as any,
      { layered: true, style: 'old_money', maxItems: 5 },
    );

    expect(result.layered).toBe(true);
    expect(result.missingSlots).toEqual([]);
    expect(result.items.map((item) => item.macroCategory)).toEqual([
      'outerwear',
      'top',
      'bottom',
      'shoes',
    ]);
    expect(result.items.map((item) => item.id)).toEqual([
      'outer_1',
      'top_1',
      'bottom_1',
      'shoes_1',
    ]);
  });

  it('layered mode reports missingSlots when the wardrobe has no outerwear', () => {
    const result = sanitizeGeneratedOutfitItemsDetailed(
      [
        { id: 'top_1', image: 'top.png', type: 'Oxford Shirt', name: 'Oxford Shirt', macroCategory: 'top' },
        { id: 'bottom_1', image: 'bottom.png', type: 'Tailored Trousers', name: 'Tailored Trousers', macroCategory: 'bottom' },
        { id: 'shoes_1', image: 'shoes.png', type: 'Leather Loafers', name: 'Leather Loafers', macroCategory: 'shoes' },
      ],
      availableItems as any,
      { layered: true, style: 'old_money', maxItems: 5 },
    );

    expect(result.layered).toBe(true);
    expect(result.missingSlots).toContain('outerwear');
    expect(result.items.map((item) => item.macroCategory)).toEqual(['top', 'bottom', 'shoes']);
  });

  it('still returns AI items when no source lookup is available', () => {
    const sanitizedItems = sanitizeGeneratedOutfitItems(
      [
        { id: 'generated_top', image: 'top.png', type: 'Cropped Tee', name: 'Cropped Tee', macroCategory: 'top' },
        { id: 'generated_bottom', image: 'bottom.png', type: 'Wide Leg Pants', name: 'Wide Leg Pants', macroCategory: 'bottom' },
        { id: 'generated_shoes', image: 'shoes.png', type: 'Platform Sneakers', name: 'Platform Sneakers', macroCategory: 'shoes' },
      ],
      []
    );

    expect(sanitizedItems.map((item) => item.id)).toEqual([
      'generated_top',
      'generated_bottom',
      'generated_shoes',
    ]);
    expect(sanitizedItems.map((item) => item.macroCategory)).toEqual(['top', 'bottom', 'shoes']);
  });
});
