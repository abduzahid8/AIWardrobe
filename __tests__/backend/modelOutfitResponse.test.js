/** @jest-environment node */
const fs = require('fs');
const path = require('path');
const ts = require('typescript');
const source = fs.readFileSync(path.resolve(__dirname, '../../backend/supabase/functions/_shared/modelOutfitResponse.ts'), 'utf8');
const api = {};
new Function('exports', ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText)(api);
const { normalizeModelOutfits } = api;
test('equivalent provider references preserve exact identities and metadata', () => {
  expect(normalizeModelOutfits([{ description: 'Look', items: [' item-a ', { itemId: 'item-b' }, { item_id: 'item-c', recommendation: 'Match' }] }])).toEqual([
    { description: 'Look', items: [{ id: 'item-a' }, { itemId: 'item-b', id: 'item-b' }, { item_id: 'item-c', id: 'item-c', recommendation: 'Match' }] },
  ]);
});
test('malformed references never receive fabricated or substituted identities', () => {
  const result = normalizeModelOutfits([null, { items: [null, 123, {}, { id: 'unknown', itemId: 'known' }] }]);
  expect(result[0].items.map(item => item.id)).toEqual(['', '', '', 'unknown']);
});
test('duplicates remain visible for strict downstream validation', () => {
  const result = normalizeModelOutfits([{ items: ['same', { itemId: 'same' }] }]);
  expect(result[0].items.map(item => item.id)).toEqual(['same', 'same']);
  expect(normalizeModelOutfits({ outfits: [] })).toEqual([]);
});
