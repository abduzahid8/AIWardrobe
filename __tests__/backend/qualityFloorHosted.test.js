/** @jest-environment node */
// Real handler/provider parsing, with offline transport/admission fixtures only.
const fs = require('fs'),
  path = require('path');
const harness = fs
  .readFileSync(path.join(__dirname, 'outfitBudgetWire.test.js'), 'utf8')
  .split('const owned=')[0];
const load = new Function('require', '__dirname', 'jest', harness + '\nreturn load;')(
  require,
  __dirname,
  jest
);
const { styleJudgeReason } = require('../../src/domain/occasionRules');
const slots = ['top', 'outerwear', 'bottom', 'shoes'];
function scenario({ spare = true, anchor = false, count = 4, allWeak = false } = {}) {
  const names = [
    'White cotton poplin shirt',
    'Navy structured wool blazer',
    'Navy tailored wool trousers',
    'Black leather Oxford shoes',
  ];
  const rows = slots.flatMap((slot, s) =>
    Array.from({ length: spare ? count : 3 }, (_, i) => ({
      id: `shop_q-${s}-${i}`,
      name: `${names[s]} model ${i}`,
      macroCategory: slot,
      type: slot,
      color: ['white', 'navy', 'navy', 'black'][s],
      imageUrl: `https://example.invalid/q-${s}-${i}.jpg`,
      isShopItem: true,
      isWardrobe: false,
      price: 50,
      currency: 'USD',
    }))
  );
  const find = (s, i) => rows.find((r) => r.id === `shop_q-${s}-${i}`);
  const anchorRow = anchor ? find(1, 0) : undefined;
  if (anchorRow) {
    anchorRow.id = 'wardrobe-quality-anchor';
    anchorRow.isShopItem = false;
    anchorRow.isWardrobe = true;
  }
  const anchorId = anchorRow?.id;
  for (const i of allWeak ? [0, 1, 2] : [2]) {
    if (anchor) {
      find(3, i).name = `Flextride Plus Leather Loafers - Black model ${i}`;
    } else {
      find(0, i).name = `Bold rugby striped cotton shirt model ${i}`;
      find(1, i).name = `Technical leisure zip blazer model ${i}`;
    }
  }
  const provider = Array.from({ length: 3 }, (_, i) =>
    slots.map((_, s) => (anchor && s === 1 ? anchorId : find(s, i).id))
  );
  const body = {
    stylePreferences: 'old_money',
    occasion: 'Work',
    prompt: 'Work',
    limit: 3,
    deckTarget: 3,
    wardrobeItems: rows,
    useProvidedWardrobeOnly: true,
    wardrobeOnly: false,
    anchorItemId: anchorId,
    preferences: { maxShopBudget: { amount: 220, currency: 'USD' } },
  };
  return { rows, provider, body, anchorId };
}
const send = (handler, body) =>
  handler(
    new Request('https://example.invalid', {
      method: 'POST',
      headers: { Authorization: 'Bearer fixture-user' },
      body: JSON.stringify(body),
    })
  );
const signature = (items) =>
  items
    .map((p) => (typeof p === 'string' ? p : p.id))
    .sort()
    .join('|');
function preserved(data, fixture) {
  const signatures = data.outfits.map((o) => signature(o.items));
  for (const strong of fixture.provider.slice(0, 2))
    expect(signatures).toContain(signature(strong));
  expect(signatures).not.toContain(signature(fixture.provider[2]));
  for (const look of data.outfits) {
    expect(look.items).toHaveLength(4);
    expect(look.qualityAssessment.acceptance.passed).toBe(true);
    expect(look.items.reduce((sum, i) => sum + Number(i.price || 0), 0)).toBeLessThanOrEqual(220);
  }
}
test.each([false, true])(
  'replaces only the weak hard-valid provider look; preserves good cards including shoes (anchor=%s)',
  async (anchor) => {
    const f = scenario({ anchor });
    const byId = new Map(f.rows.map((i) => [i.id, i]));
    for (const look of f.provider)
      expect(
        styleJudgeReason(
          look.map((id) => byId.get(id)),
          'old_money',
          'Work',
          null,
          f.anchorId
        )
      ).toBeNull();
    const { handler } = load(f.provider);
    const response = await send(handler, f.body),
      data = await response.json();
    expect({ status: response.status, data }).toMatchObject({
      status: 200,
      data: { success: true },
    });
    expect(data.outfits).toHaveLength(3);
    preserved(data, f);
    if (anchor)
      for (const look of data.outfits)
        expect(look.items.some((i) => i.id === f.anchorId)).toBe(true);
  }
);
test.each(['missing', 'budget', 'recency', 'physical-alias'])(
  'fails closed when reserve alternatives are unavailable by %s',
  async (mode) => {
    const f = scenario({ spare: mode !== 'missing' });
    if (mode === 'budget')
      f.rows.filter((i) => i.id.endsWith('-3')).forEach((i) => (i.price = 1000));
    if (mode === 'recency')
      f.body.excludeItemIds = f.rows.filter((i) => i.id.endsWith('-3')).map((i) => i.id);
    if (mode === 'physical-alias')
      for (let s = 0; s < 4; s++) {
        const original = f.rows.find((i) => i.id === `shop_q-${s}-0`);
        f.rows.find((i) => i.id === `shop_q-${s}-3`).imageUrl = original.imageUrl;
      }
    const { handler } = load(f.provider);
    const response = await send(handler, f.body),
      data = await response.json();
    expect(response.status).toBe(422);
    expect(data.code).toBe('quality_replacement_unavailable');
    expect(data.outfits).toEqual([]);
  }
);
test('all weak provider looks can enter the same bounded reserve without keeping a weak card', async () => {
  const f = scenario({ count: 6, allWeak: true });
  const { handler } = load(f.provider);
  const response = await send(handler, f.body),
    data = await response.json();
  expect({ status: response.status, data }).toMatchObject({ status: 200, data: { success: true } });
  expect(data.outfits).toHaveLength(3);
  for (const look of data.outfits) {
    expect(look.qualityAssessment.acceptance.passed).toBe(true);
    expect(f.provider.map(signature)).not.toContain(signature(look.items));
  }
});
