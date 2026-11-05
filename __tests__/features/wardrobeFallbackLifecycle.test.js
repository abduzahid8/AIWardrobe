/** @jest-environment node */
const fs = require('fs'), ts = require('typescript');
const { closestAvailableOutfits } = require('../../features/outfit-generator/utils/closestWardrobeOutfits');
const { OutfitGenerationFailure } = require('../../src/services/outfitGenerationFailure');
const source = fs.readFileSync('screens/AIOutfitmaker.tsx', 'utf8');
const ast = ts.createSourceFile('screen.tsx', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
let catchBody;
function visit(node) {
  if (ts.isCatchClause(node) && node.block.getText(ast).includes('const noMatch =')) catchBody = node.block.getText(ast);
  ts.forEachChild(node, visit);
}
visit(ast);
const code = ts.transpileModule(`async function run(err: any) ${catchBody}`, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText;
function setup(extra = {}) {
  const deps = {
    console: { warn: jest.fn() }, fallbackTimer: { current: null }, fallbackHistory: undefined, isCurrentGeneration: () => true,
    isMounted: { current: true }, effectiveSource: 'wardrobe', closestAvailableOutfits, OutfitGenerationFailure,
    alternativeItems: [
      { id: 'layer', macroCategory: 'outerwear', name: 'Denim jacket' },
      { id: 'top', macroCategory: 'top', name: 'Cotton T-shirt' },
      { id: 'bottom', macroCategory: 'bottom', name: 'Blue jeans' },
      { id: 'shoes', macroCategory: 'shoes', name: 'White sneakers' },
    ], baseItemId: 'bottom', styleToUse: 'old_money', occasionToUse: 'formal', promptText: '',
    fitsCap: () => true, resolvedWeather: undefined, tapNonce: 1, setOutfits: jest.fn(), setError: jest.fn(), setLoading: jest.fn(), t: key => key,
    ...extra,
  };
  return { ...deps, run: Function(...Object.keys(deps), code + '; return run;')(...Object.values(deps)) };
}
test.each(['outfitMaker.noMatchingClothing', 'outfitMaker.aiUnavailable'])('real screen catch presents an anchored alternative for %s', async key => {
  const s = setup(); await s.run(new OutfitGenerationFailure(key));
  const looks = s.setOutfits.mock.calls[0][0];
  expect(looks).toHaveLength(1);
  expect(looks[0].items.map(i => i.id)).toContain('bottom');
  expect(s.setError).toHaveBeenLastCalledWith('');
  expect(s.setLoading).toHaveBeenLastCalledWith(false);
});
test('cancelled or superseded requests cannot publish fallback results', async () => {
  const s = setup({ isCurrentGeneration: () => false }); await s.run(new OutfitGenerationFailure('outfitMaker.noMatchingClothing'));
  expect(s.setOutfits).not.toHaveBeenCalled();
});
test.each(['outfitMaker.budgetTooLow', 'outfitMaker.invalidRequest'])('operational and budget errors do not become wardrobe alternatives: %s', async key => {
  const s = setup(); await s.run(new OutfitGenerationFailure(key));
  expect(s.setOutfits).toHaveBeenCalledWith([]);
  expect(s.setError).toHaveBeenCalledWith(key);
});
test('shop mode cannot fall back to owned wardrobe items', async () => {
  const s = setup({ effectiveSource: 'shop' }); await s.run(new OutfitGenerationFailure('outfitMaker.noMatchingClothing'));
  expect(s.setOutfits).toHaveBeenCalledWith([]);
});

test('the screenshot flow keeps the owned polo and uses shop companions despite work/minimalist mismatch', async () => {
  const anchor = { id: 'polo', macroCategory: 'top', name: 'Polo Piquet 1', type: 'polo', isWardrobe: true, isShopItem: false };
  const companions = [
    { id: 'shop-layer', macroCategory: 'outerwear', name: 'Denim jacket' },
    { id: 'shop-pants', macroCategory: 'bottom', name: 'Blue jeans' },
    { id: 'shop-shoes', macroCategory: 'shoes', name: 'White sneakers' },
  ].map(item => ({ ...item, isShopItem: true, isWardrobe: false, price: 50, currency: 'USD' }));
  const s = setup({ effectiveSource: 'shop', baseItemId: 'polo', styleToUse: 'minimalist', occasionToUse: 'work', alternativeItems: [anchor, ...companions] });
  await s.run(new OutfitGenerationFailure('outfitMaker.noMatchingClothing'));
  const looks = s.setOutfits.mock.calls[0][0];
  expect(looks).toHaveLength(1);
  expect(looks[0].items.map(i => i.id)).toEqual(expect.arrayContaining(['polo', ...companions.map(item => item.id)]));
  expect(looks[0].items.filter(i => i.isWardrobe).map(i => i.id)).toEqual(['polo']);
  expect(looks[0].description).toBe('outfitMaker.closestShopAlternative');
});
test('shop alternatives cannot bypass the budget', async () => {
  const s = setup({ effectiveSource: 'shop', baseItemId: undefined, fitsCap: () => false,
    alternativeItems: setup().alternativeItems.map(item => ({ ...item, isShopItem: true, isWardrobe: false })) });
  await s.run(new OutfitGenerationFailure('outfitMaker.noMatchingClothing'));
  expect(s.setOutfits).toHaveBeenCalledWith([]);
});
