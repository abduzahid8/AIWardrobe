import { closestWardrobeOutfits, closestAvailableOutfits } from '../../features/outfit-generator/utils/closestWardrobeOutfits';
import { outfitStructureReason } from '../../backend/supabase/functions/_shared/outfitStructure';
import { outfitVisualSignature } from '../../src/domain/outfitVisualSignature';

const item = (id: string, macroCategory: string, name: string) => ({ id, macroCategory, name, imageUrl: `https://test/${id}.png`, isWardrobe: true });
const pool = [item('jacket', 'outerwear', 'Denim jacket'), item('shirt', 'top', 'White T-shirt'), item('pants', 'bottom', 'Blue jeans'), item('shoes', 'shoes', 'White sneakers')];

test.each(pool)('thin wardrobe retains the selected $macroCategory despite formal preferences', anchor => {
  const looks = closestWardrobeOutfits(pool, { anchorId: anchor.id, style: 'old_money', occasion: 'formal' });
  expect(looks).toHaveLength(1);
  expect(looks[0].items.map(item => item.id)).toContain(anchor.id);
  expect(outfitStructureReason(looks[0].items, true)).toBeNull();
});

test('closer style matches rank ahead of alternatives', () => {
  const looks = closestWardrobeOutfits([...pool, item('trousers', 'bottom', 'Navy tailored trousers')], { style: 'old_money', occasion: 'work' });
  expect(looks[0].items.some(item => item.id === 'trousers')).toBe(true);
});

test('missing inventory never fabricates a garment or uses shop clothing', () => {
  expect(closestWardrobeOutfits([...pool.slice(0, 3), { ...pool[3], isWardrobe: false }], { style: 'casual', occasion: 'everyday' })).toEqual([]);
  expect(closestWardrobeOutfits(pool, { anchorId: 'missing', style: 'casual', occasion: 'everyday' })).toEqual([]);
});

test('alternatives cannot relax the four-item contract', () => {
  const looks = closestWardrobeOutfits([pool[1], item('shorts', 'bottom', 'Cotton shorts'), pool[3]], { anchorId: 'shorts', style: 'classic', occasion: 'everyday' });
  expect(looks).toEqual([]);
});

const shopPool = pool.map(item => ({ ...item, isWardrobe: false, isShopItem: true, price: 100, currency: 'USD' }));

test.each(pool)('shop alternatives preserve an owned $macroCategory anchor without other wardrobe pieces', anchor => {
  const looks = closestAvailableOutfits([anchor, ...shopPool, item('unrelated-owned', 'shoes', 'Black oxford shoes')], {
    source: 'shop', anchorId: anchor.id, style: 'minimalist', occasion: 'work',
  });
  expect(looks.length).toBeGreaterThan(0);
  for (const look of looks) {
    expect(look.items.filter(item => item.isWardrobe).map(item => item.id)).toEqual([anchor.id]);
    expect(look.items.filter(item => item.id !== anchor.id).every(item => item.isShopItem)).toBe(true);
    expect(outfitStructureReason(look.items, true)).toBeNull();
  }
});

test('shop fallback rejects over-budget proposals before choosing alternatives', () => {
  const cheapShoes = { ...shopPool[3], id: 'cheap-shoes', price: 10 };
  const looks = closestAvailableOutfits([...shopPool, cheapShoes], {
    source: 'shop', style: 'minimalist', occasion: 'work',
    accepts: parts => parts.reduce((total, item) => total + (item.price || 0), 0) <= 310,
  });
  expect(looks).toHaveLength(1);
  expect(looks[0].items.find(item => item.macroCategory === 'shoes')?.id).toBe('cheap-shoes');
  expect(looks[0].items.every(item => item.currency === 'USD')).toBe(true);
});

test('shop mode without an anchor uses only catalog items', () => {
  const looks = closestAvailableOutfits([...pool, ...shopPool], { source: 'shop', style: 'casual', occasion: 'everyday' });
  expect(looks.length).toBeGreaterThan(0);
  expect(looks.flatMap(look => look.items).every(item => item.isShopItem && !item.isWardrobe)).toBe(true);
});

test('a recently shown semantic look scores below its fresh quality without breaking the contract', () => {
  const base = closestWardrobeOutfits(pool, { style: 'casual', occasion: 'everyday' });
  expect(base.length).toBeGreaterThan(0);
  const keyOf = (look: { items: { id: string }[] }) => look.items.map(item => item.id).sort().join('|');
  const reranked = closestWardrobeOutfits(pool, {
    style: 'casual', occasion: 'everyday',
    recentLookSignatures: [outfitVisualSignature(base[0].items)],
  });
  expect(reranked.map(keyOf)).toEqual(expect.arrayContaining(base.map(keyOf)));
  const same = reranked.find(look => keyOf(look) === keyOf(base[0]));
  expect(same).toBeDefined();
  expect(same!.quality).toBeLessThan(base[0].quality);
  for (const look of reranked) expect(outfitStructureReason(look.items, true)).toBeNull();
});
