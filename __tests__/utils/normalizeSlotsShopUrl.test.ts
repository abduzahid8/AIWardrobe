import { createSlotNormalizer } from '../../features/outfit-generator/utils/normalizeSlots';

// Edge items never carry sourceUrl; the normalizer already merges price and
// image from the pool row but used to drop the shop URL — tiles then rendered
// without a Buy button. Regression: every output item keeps/gets shopUrl.
describe('normalizeTo4Slots reattaches pool shopUrl (Buy button)', () => {
  const poolRow = (id: string, name: string, macroCategory: string) => ({
    id,
    name,
    macroCategory,
    type: macroCategory,
    image: `https://cdn.example.com/${id}.jpg`,
    imageUrl: `https://cdn.example.com/${id}.jpg`,
    price: 500000,
    currency: 'UZS',
    shopUrl: `https://shop.example.com/${id}`,
    isShopItem: true,
    color: 'neutral',
    brand: 'TestBrand',
  });

  const pool = [
    poolRow('p_layer', '100% Wool Suit Blazer', 'outerwear'),
    poolRow('p_top', 'Cotton Striped Shirt', 'top'),
    poolRow('p_pants', 'Wool Blend Flannel Pants', 'bottom'),
    poolRow('p_shoes', 'Leather Derby Shoes', 'shoes'),
  ];

  it('edge-shaped incoming items (no shopUrl, shop_ id prefix) regain shopUrl', () => {
    const incoming = pool.map((r) => {
      const { shopUrl: _drop, ...edgeShape } = r;
      return { ...edgeShape, id: `shop_${r.id}`, fromEdge: true };
    });

    const normalize = createSlotNormalizer({
      weather: { temp: 20, condition: 'clear' },
      currentSource: 'shop',
      selectedOccasion: 'work',
    });
    const out = normalize(incoming, 'old_money', 0, 1, pool, new Set());

    expect(out.length).toBe(4);
    const edgeKept = out.filter((it: any) => it.fromEdge);
    expect(edgeKept.length).toBeGreaterThan(0);
    for (const it of out) {
      expect(typeof it.shopUrl).toBe('string');
      expect(it.shopUrl).toMatch(/^https:\/\/shop\.example\.com\//);
    }
  });
});
