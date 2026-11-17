import { aiProvider } from '../../src/services/aiProviderService';
const mockInvoke = jest.fn();
jest.mock('../../lib/supabase', () => ({
  supabase: { functions: { invoke: (...args: unknown[]) => mockInvoke(...args) } },
}));
const pieces = ['White cotton shirt', 'Fine cardigan', 'Chino trousers', 'Leather loafers'].map(
  (name, i) => ({
    id: `piece-${i}`,
    name,
    macroCategory: ['top', 'outerwear', 'bottom', 'shoes'][i],
  })
);
beforeEach(() => mockInvoke.mockReset());
it.each([pieces.slice(1), [pieces[0], pieces[0], ...pieces.slice(2)]])(
  'invalid provider output never becomes a random subset fallback',
  async (items) => {
    mockInvoke.mockResolvedValue({ data: { outfits: [{ items }] }, error: null });
    const result = await aiProvider.generateOutfit(
      pieces.map((p) => p.id),
      'Weekend'
    );
    expect(result).toMatchObject({ success: false, error: 'insufficient_candidates', items: [] });
    expect(mockInvoke).toHaveBeenCalledTimes(1);
  }
);
it('offline or quota failure does not trigger a second provider call', async () => {
  mockInvoke.mockResolvedValue({ data: null, error: { status: 429 } });
  expect(
    await aiProvider.generateOutfit(
      pieces.map((p) => p.id),
      'Weekend'
    )
  ).toMatchObject({ success: false, items: [] });
  expect(mockInvoke).toHaveBeenCalledTimes(1);
});
it('retains the IDs of a valid canonical quartet', async () => {
  mockInvoke.mockResolvedValue({
    data: { outfits: [{ items: pieces, style: 'casual' }] },
    error: null,
  });
  expect(
    await aiProvider.generateOutfit(
      pieces.map((p) => p.id),
      'Weekend'
    )
  ).toMatchObject({ success: true, items: pieces.map((p) => p.id) });
});
