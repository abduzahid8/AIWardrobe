/** @jest-environment node */
const fs = require('fs'), path = require('path');
const harness = fs.readFileSync(path.join(__dirname, 'outfitBudgetWire.test.js'), 'utf8').split('const owned=')[0];
const load = new Function('require', '__dirname', 'jest', harness + '\nreturn load;')(require, __dirname, jest);
const {assessOutfitQuality} = require('../../src/domain/occasionRules');
const {stylingV193Fixtures, fixtureLook} = require('../fixtures/stylingV193');
const slots = ['top', 'outerwear', 'bottom', 'shoes'];
const names = ['White poplin shirt', 'Navy wool blazer', 'Grey tailored cotton trousers', 'Brown leather penny loafers'];
const rows = slots.flatMap((macroCategory, s) => Array.from({length: 6}, (_, i) => ({
  id: `shop_v193-${s}-${i}`, name: `${names[s]} model ${i}`, macroCategory, type: macroCategory,
  imageUrl: `https://example.invalid/v193-${s}-${i}.jpg`, isShopItem: true, isWardrobe: false,
  price: 300000, currency: 'UZS'
})));
const request = (extra = {}) => new Request('https://example.invalid', {method: 'POST',
  headers: {Authorization: 'Bearer fixture-user'}, body: JSON.stringify({stylePreferences: 'semi_classic',
    occasion: 'Date night', prompt: 'Date night', limit: 3, deckTarget: 3, wardrobeItems: rows,
    useProvidedWardrobeOnly: true, ...extra})});
const assertDeck = data => {
  expect(data.outfits).toHaveLength(3);
  expect(new Set(data.outfits.flatMap(o => o.items.map(i => i.id))).size).toBe(12);
  for (const outfit of data.outfits) {
    expect(outfit.items.map(i => i.macroCategory).sort()).toEqual([...slots].sort());
    expect(outfit.items.every(i => i.isShopItem && String(i.id).startsWith('shop_'))).toBe(true);
    expect(assessOutfitQuality(outfit.items, 'semi_classic', 'Date night').acceptance.passed).toBe(true);
  }
};

test('curated Edge path offers nine safe alternatives and preserves final quality after ranking', async () => {
  // One rejected family per invariant class (pattern clash, washed denim,
  // fabric-weight mismatch) keeps the exclusion proof while leaving the
  // deterministic planner ample wall-clock headroom to complete evaluation
  // even when the suite runs alongside other CPU-heavy workers.
  const bad = stylingV193Fixtures.filter(f =>
    ['busy-plaid-paisley', 'washed-double-denim-oxford', 'airy-heavy-winter'].includes(f.id)).flatMap(f =>
    fixtureLook(f.names, `shop_${f.id}`).map(i => ({...i, type: i.macroCategory,
      isShopItem: true, price: 300000, currency: 'UZS', imageUrl: `https://example.invalid/${i.id}.jpg`})));
  const pool = [...rows, ...bad];
  const {handler, fetchMock} = load([], null, null, null, null, 'curated');
  fetchMock.mockImplementation(async (_url, init) => {
    const options = JSON.parse(JSON.parse(init.body).messages.at(-1).content);
    expect(options.looks).toHaveLength(9);
    for (const look of options.looks) {
      const parts = look.items.map(id => pool.find(i => i.id === id));
      expect(assessOutfitQuality(parts, 'semi_classic', 'Date night').acceptance.passed).toBe(true);
    }
    return new Response(JSON.stringify({choices: [{message: {content: JSON.stringify({
      rankedLookIds: options.looks.map(l => l.id).reverse()
    })}}]}));
  });
  const response = await handler(request({wardrobeItems: pool}));
  const data = await response.json();
  expect({status: response.status, data}).toMatchObject({status: 200, data: {success: true}});
  assertDeck(data);
  expect(fetchMock).toHaveBeenCalledTimes(1);
});

test.each([
  {preferences: {maxShopBudget: {amount: 1500000, currency: 'UZS'}}},
  {prompt: 'Date night. Budget under 1 500 000 UZS.'}
])('submitted UZS cap covers all four slots: %j', async extra => {
  const {handler} = load([], null, null, null, null, 'curated');
  const response = await handler(request(extra)), data = await response.json();
  expect(response.status).toBe(200);
  assertDeck(data);
  for (const outfit of data.outfits)
    expect(outfit.items.reduce((sum, i) => sum + i.price, 0)).toBeLessThanOrEqual(1500000);
});

test('four individually affordable products exceeding the full UZS cap fail before inference', async () => {
  const {handler, fetchMock} = load([], null, null, null, null, 'curated');
  const response = await handler(request({wardrobeItems: rows.map(i => ({...i, price: 450000})),
    preferences: {maxShopBudget: {amount: 1500000, currency: 'UZS'}}}));
  expect(response.status).toBe(422);
  expect((await response.json()).outfits).toEqual([]);
  expect(fetchMock).not.toHaveBeenCalled();
});

test('omitted budget sends no cap; a displayed example is not submitted request data', async () => {
  const {handler, fetchMock} = load([], null, null, null, null, 'curated');
  const response = await handler(request({wardrobeItems: rows.map(i => ({...i, price: 12700000}))}));
  expect(response.status).toBe(200);
  assertDeck(await response.json());
  expect(JSON.stringify(fetchMock.mock.calls)).not.toContain('HARD BUDGET');
});

test('provider cannot inject an unapproved look or absent product through ranked IDs', async () => {
  const {handler, fetchMock} = load([], null, null, null, null, 'curated');
  fetchMock.mockImplementation(async () => new Response(JSON.stringify({choices: [{message: {
    content: JSON.stringify({rankedLookIds: ['shop_absent-product', 'look_unapproved']})
  }}]})));
  const response = await handler(request()), data = await response.json();
  expect(response.status).toBe(200);
  assertDeck(data);
  expect(data.outfits.flatMap(o => o.items).every(i => rows.some(row => row.id === i.id))).toBe(true);
});
