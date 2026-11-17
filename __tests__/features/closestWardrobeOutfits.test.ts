import { closestWardrobeOutfits, closestAvailableOutfits } from '../../features/outfit-generator/utils/closestWardrobeOutfits';
import { productIdentityKeys } from '../../src/domain/outfitDiversity';
import { outfitStructureReason } from '../../backend/supabase/functions/_shared/outfitStructure';
import { outfitVisualSignature } from '../../src/domain/outfitVisualSignature';

const item = (id: string, macroCategory: string, name: string) => ({ id, macroCategory, name, imageUrl: `https://test/${id}.png`, isWardrobe: true });
const pool = [item('jacket', 'outerwear', 'Denim jacket'), item('shirt', 'top', 'White T-shirt'), item('pants', 'bottom', 'Blue jeans'), item('shoes', 'shoes', 'White sneakers')];

test.each(pool)('a selected $macroCategory never waives an incompatible Formal request', anchor => {
  const looks = closestWardrobeOutfits(pool, { anchorId: anchor.id, style: 'old_money', occasion: 'formal' });
  expect(looks).toEqual([]);
});

test('a tailored bottom cannot make a denim jacket and tee suitable for Work', () => {
  const looks = closestWardrobeOutfits([...pool, item('trousers', 'bottom', 'Navy tailored trousers')], { style: 'old_money', occasion: 'work' });
  expect(looks).toEqual([]);
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
    source: 'shop', anchorId: anchor.id, style: 'minimalist', occasion: 'everyday',
  });
  expect(looks.length).toBeGreaterThan(0);
  for (const look of looks) {
    expect(look.items.filter(item => item.isWardrobe).map(item => item.id)).toEqual([anchor.id]);
    expect(look.items.filter(item => item.id !== anchor.id).every(item => item.isShopItem)).toBe(true);
    expect(outfitStructureReason(look.items, true)).toBeNull();
  }
});

test('shop fallback rejects over-budget proposals before choosing alternatives', () => {
  // Redesign: catalog rows with the same name and photo are ONE physical
  // product ("alias listings"), so the cheaper alternative must be a genuinely
  // different product — a different name and image — to be selectable.
  const cheapShoes = { ...shopPool[3], id: 'cheap-shoes', name: 'Canvas budget sneakers', imageUrl: 'https://test/cheap-shoes.png', price: 10 };
  const looks = closestAvailableOutfits([...shopPool, cheapShoes], {
    source: 'shop', style: 'minimalist', occasion: 'everyday',
    accepts: parts => parts.reduce((total, item) => total + (item.price || 0), 0) <= 310,
  });
  expect(looks).toHaveLength(1);
  expect(looks[0].items.find(item => item.macroCategory === 'shoes')?.id).toBe('cheap-shoes');
  expect(looks[0].items.every(item => item.currency === 'USD')).toBe(true);
});

test('an alias duplicate row is one product: its cheaper copy cannot satisfy the budget', () => {
  const aliasShoes = { ...shopPool[3], id: 'cheap-shoes', price: 10 };
  // Same name + same photo = one physical product. The budget gate still runs
  // during proposal selection: with only the full-price listing selectable,
  // the capped request honestly gets no deck instead of a repriced product.
  expect(productIdentityKeys(aliasShoes).filter(key => key !== 'id:cheap-shoes'))
    .toEqual(expect.arrayContaining(productIdentityKeys(shopPool[3]).filter(key => key !== 'id:shoes')));
  const looks = closestAvailableOutfits([...shopPool, aliasShoes], {
    source: 'shop', style: 'minimalist', occasion: 'everyday',
    accepts: parts => parts.reduce((total, item) => total + (item.price || 0), 0) <= 310,
  });
  expect(looks).toEqual([]);
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

test('fallback forms three disjoint looks and excludes the displayed deck on the next request', () => {
  const colors = ['White','Blue','Black','Beige','Green','Grey'];
  const inventory = colors.flatMap((color,index) => [
    item(`top-${index}`, 'top', `${color} cotton T-shirt`),
    item(`layer-${index}`, 'outerwear', `${color} cotton overshirt`),
    item(`bottom-${index}`, 'bottom', `${color} cotton chinos`),
    item(`shoe-${index}`, 'shoes', `${color} minimal leather sneakers`),
  ]);
  const first = closestWardrobeOutfits(inventory, {style:'casual', occasion:'casual'});
  expect(first).toHaveLength(3);
  const ids = first.flatMap(look => look.items.map(item => item.id));
  expect(new Set(ids).size).toBe(12);
  const productKeys = ids.map(id => `id:${id}`);
  const second = closestWardrobeOutfits(inventory, {style:'casual', occasion:'casual', history:{
    previousDeckProductKeys:productKeys, olderRecentProductKeys:[], rotationDecks:[{productKeys}],
  }});
  expect(second).toHaveLength(3);
  expect(second.flatMap(look => look.items).some(item => ids.includes(item.id))).toBe(false);
});
