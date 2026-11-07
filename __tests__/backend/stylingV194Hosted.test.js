/** @jest-environment node */
const fs = require('fs'), path = require('path');
const harness = fs.readFileSync(path.join(__dirname, 'outfitBudgetWire.test.js'), 'utf8').split('const owned=')[0];
const load = new Function('require', '__dirname', 'jest', harness + '\nreturn load;')(require, __dirname, jest);
const {assessOutfitQuality} = require('../../src/domain/occasionRules');
const slots = ['top', 'outerwear', 'bottom', 'shoes'];

test.each([
  ['old_money', 'Work', ['White poplin shirt', 'Navy wool blazer', 'Grey tailored trousers', 'Black leather penny loafers'],
    ['Dark casual Tencel shirt', 'Short wool zip jacket', 'Beige drawstring flannel trousers', 'Black leather dress loafers']],
  ['semi_classic', 'Date night', ['White cotton shirt', 'Navy cotton blazer', 'Grey straight chinos', 'Brown suede loafers'],
    ['Light Tencel shirt', 'Dark Blue Denim Leisure Blazer', 'Washed skinny jeans', 'Formal black Derby shoes']],
  ['casual', 'Travel', ['Plain regular cotton T-shirt', 'Olive cotton jacket', 'Black straight jeans', 'White lifestyle sneakers'],
    ['Oversized graphic T-shirt', 'Structured navy blazer', 'Brown tailored chinos', 'Technical cross-training sneakers']],
])('40-row curated handler preserves nine options, one call and three unique looks: %s / %s', async (style, occasion, good, bad) => {
  const rows = slots.flatMap((macroCategory, slot) => Array.from({length: 10}, (_, i) => ({
    id: `shop_v194-${slot}-${i}`, name: `${i === 9 ? bad[slot] : good[slot]} model ${i}`,
    macroCategory, type: macroCategory, isShopItem: true, isWardrobe: false,
    price: 50, currency: 'USD', imageUrl: `https://example.invalid/v194-${slot}-${i}.jpg`,
  })));
  const {handler, fetchMock} = load([], null, null, null, null, 'curated');
  fetchMock.mockImplementation(async (_url, init) => {
    const payload = JSON.parse(JSON.parse(init.body).messages.at(-1).content);
    expect(payload.looks).toHaveLength(9);
    for (const look of payload.looks)
      expect(assessOutfitQuality(look.items.map(id => rows.find(row => row.id === id)), style, occasion).acceptance.passed).toBe(true);
    return new Response(JSON.stringify({choices: [{message: {content: JSON.stringify({rankedLookIds: payload.looks.map(look => look.id).reverse()})}}]}));
  });
  const response = await handler(new Request('https://example.invalid', {method: 'POST',
    headers: {Authorization: 'Bearer fixture-user'}, body: JSON.stringify({
      stylePreferences: style, occasion, prompt: occasion, limit: 3, deckTarget: 3,
      wardrobeItems: rows, useProvidedWardrobeOnly: true,
    })}));
  const data = await response.json();
  expect({status: response.status, code: data.code}).toEqual({status: 200, code: undefined});
  expect(fetchMock).toHaveBeenCalledTimes(1);
  expect(data.outfits).toHaveLength(3);
  expect(new Set(data.outfits.flatMap(look => look.items.map(item => item.id))).size).toBe(12);
  for (const look of data.outfits) {
    expect(look.items.map(item => item.macroCategory).sort()).toEqual([...slots].sort());
    expect(look.qualityAssessment.acceptance.passed).toBe(true);
    expect(assessOutfitQuality(look.items, style, occasion).acceptance.passed).toBe(true);
  }
});
