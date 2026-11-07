import { resolveOutfitCollageItems, resolveOutfitCollageItemsWithMaps } from '../../src/utils/outfitCollage';
import { toClosetItem } from '../../src/utils/closetItem';
import type { Outfit, ClothingItem } from '../../src/types/domain';

const outfit = { itemIds: ['top', 'layer', 'pants', 'shoes'], snapshotItems: [
  { id: 'top', category: 'upper_body', image: 'https://test/top.png' },
  { id: 'layer', category: 'outerwear', image: '' },
  { id: 'pants', category: 'lower_body', image: 'https://test/pants.png' },
  { id: 'shoes', category: 'footwear', image: '' },
] } as Outfit;

test.each([resolveOutfitCollageItems, (o: Outfit, w: any[]) => resolveOutfitCollageItemsWithMaps(o, new Map(w.map(i => [i.id, i])), new Map())])('saved outfit retains every slot and recovers missing snapshot media', resolve => {
  const pieces = resolve(outfit, [{ id: 'layer', imageUrl: 'https://test/layer.png' }]);
  expect(pieces).toHaveLength(4);
  expect(pieces.map(p => p.category)).toEqual(['top', 'outerwear', 'bottom', 'shoes']);
  expect(pieces[1].image).toBe('https://test/layer.png');
  expect(pieces[3]).toMatchObject({ id: 'shoes', image: '', category: 'shoes' });
});

test('closet conversion retains the subtype used to identify a selected garment', () => {
  const converted = toClosetItem({ id: 'pants', category: 'bottom', subCategory: 'trousers' } as ClothingItem);
  expect(converted).toMatchObject({ type: 'trousers', itemType: 'trousers', category: 'bottom' });
});
