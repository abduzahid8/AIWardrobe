import { buildLocalOutfits, type LocalClothingItem } from '../src/services/outfitGenerationService';
let idCounter = 200;
function makeItem(slot: any, name: string): LocalClothingItem {
  idCounter += 1;
  return { id: `w${idCounter}`, userId: 'u', imageUrl: 'https://e.com/x.jpg', category: slot, subCategory: slot === 'top' ? 't-shirt' : slot === 'bottom' ? 'jeans' : slot === 'shoes' ? 'sneakers' : 'jacket', primaryColor: 'blue', colorHex: '#000', pattern: 'solid', material: 'cotton', seasons: [], occasions: [], wearCount: 0, lastWornAt: null, isFavorite: false, createdAt: '2026-01-01T00:00:00Z', updatedAt: '2026-01-01T00:00:00Z', macroCategory: slot } as LocalClothingItem;
}
it('probe thin', () => {
  const thin = [
    makeItem('top', 'tee one'), makeItem('top', 'tee two'), makeItem('top', 'polo three'),
    makeItem('top', 'shirt four'), makeItem('top', 'henley five'), makeItem('top', 'knit six'),
    makeItem('bottom', 'jeans one'), makeItem('bottom', 'chinos two'),
    makeItem('shoes', 'only sneakers'), makeItem('outerwear', 'only jacket'),
  ];
  for (let r = 0; r < 6; r++) {
    const outfits = buildLocalOutfits(thin, { stylePreferences: 'casual' as any, weather: { temp: 10, condition: 'cool' }, limit: 3 } as any);
    console.log('run', r, 'count', outfits.length, outfits.map((o) => o.items.map((i: any) => i.id).sort().join('|')));
  }
});
