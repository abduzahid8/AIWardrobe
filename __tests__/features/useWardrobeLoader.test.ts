import {
  mapShopCatalogGarmentTypeToCategory,
  mapShopCatalogItemToWardrobeDisplayItem,
} from '../../features/outfit-generator/hooks/useWardrobeLoader';

describe('useWardrobeLoader shop mapping', () => {
  it('maps live Zara upper-body items into tops or outerwear', () => {
    expect(mapShopCatalogGarmentTypeToCategory('upper_body', 'RELAXED FIT INTERLOCK T-SHIRT /04')).toBe('tops');
    expect(mapShopCatalogGarmentTypeToCategory('upper_body', 'CROPPED BOMBER JACKET')).toBe('outerwear');
  });

  it('keeps live Zara lower-body and shoes categories intact', () => {
    expect(mapShopCatalogGarmentTypeToCategory('lower_body', 'STRAIGHT FIT JEANS')).toBe('bottoms');
    expect(mapShopCatalogGarmentTypeToCategory('shoes', 'DRESS PENNY LOAFERS')).toBe('shoes');
  });

  it('converts live Zara catalog rows into AI outfit shop items', () => {
    const mapped = mapShopCatalogItemToWardrobeDisplayItem({
      id: 'zara-apify-basic-heavyweight-tee',
      brand: 'ZARA',
      name: 'BASIC HEAVYWEIGHT T-SHIRT /03',
      price: 29.9,
      currency: 'USD',
      imageUrl: 'https://static.zara.net/photos/basic-heavyweight-tee.jpg',
      garmentType: 'upper_body',
      description: 'Heavyweight basic tee',
    });

    expect(mapped).toEqual({
      id: 'zara-apify-basic-heavyweight-tee',
      image: 'https://static.zara.net/photos/basic-heavyweight-tee.jpg',
      type: 'BASIC HEAVYWEIGHT T-SHIRT /03',
      name: 'BASIC HEAVYWEIGHT T-SHIRT /03',
      brand: 'ZARA',
      price: 29.9,
      macroCategory: 'top',
      category: 'tops',
      isShopItem: true,
    });
  });
});
