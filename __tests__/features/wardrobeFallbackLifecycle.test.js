/** @jest-environment node */
// Redesign v209 removed the catch-block closest-match wardrobe/shop fallback entirely
// (the closestAvailableOutfits import is gone and the catch now carries a
// "Server failure stays visible" policy): a failed generation publishes an empty deck
// with a localized error instead of silently re-serving local or previous suggestions.
const fs = require('fs'), ts = require('typescript');
const { OutfitGenerationFailure } = require('../../src/services/outfitGenerationFailure');
const { RequestTimeoutError } = require('../../src/lib/requestDeadline');
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
    console: { warn: jest.fn() }, fallbackTimer: { current: null }, isCurrentGeneration: () => true,
    isMounted: { current: true }, effectiveSource: 'wardrobe',
    alternativeItems: [
      { id: 'layer', macroCategory: 'outerwear', name: 'Denim jacket' },
      { id: 'top', macroCategory: 'top', name: 'Cotton T-shirt' },
      { id: 'bottom', macroCategory: 'bottom', name: 'Blue jeans' },
      { id: 'shoes', macroCategory: 'shoes', name: 'White sneakers' },
    ], baseItemId: 'bottom', styleToUse: 'old_money', occasionToUse: 'work', promptText: '',
    resolvedWeather: undefined, tapNonce: 1, setOutfits: jest.fn(), setError: jest.fn(), setLoading: jest.fn(), setLastFailureCode: jest.fn(), t: key => key,
    OutfitGenerationFailure, RequestTimeoutError,
    ...extra,
  };
  return { ...deps, run: Function(...Object.keys(deps), code + '; return run;')(...Object.values(deps)) };
}
test.each(['outfitMaker.noMatchingClothing', 'outfitMaker.aiUnavailable'])('real screen catch surfaces %s with an empty deck and a visible error', async key => {
  const s = setup(); await s.run(new OutfitGenerationFailure(key));
  // v209: no anchored alternative deck — the failure stays visible.
  expect(s.setOutfits).toHaveBeenCalledTimes(1);
  expect(s.setOutfits).toHaveBeenCalledWith([]);
  expect(s.setError).toHaveBeenLastCalledWith(key);
  expect(s.setLoading).toHaveBeenLastCalledWith(false);
  expect(s.setLastFailureCode).toHaveBeenCalledWith(key);
});
test('formal-occasion shortfalls surface the dedicated formal inventory message', async () => {
  const s = setup({ occasionToUse: 'formal' }); await s.run(new OutfitGenerationFailure('outfitMaker.noMatchingClothing'));
  expect(s.setError).toHaveBeenLastCalledWith('outfitMaker.formalInventoryShortfall');
  expect(s.setOutfits).toHaveBeenCalledWith([]);
});
test('cancelled or superseded requests cannot publish fallback results', async () => {
  const s = setup({ isCurrentGeneration: () => false }); await s.run(new OutfitGenerationFailure('outfitMaker.noMatchingClothing'));
  expect(s.setOutfits).not.toHaveBeenCalled();
  expect(s.setError).not.toHaveBeenCalled();
  expect(s.setLoading).not.toHaveBeenCalled();
});
test.each(['outfitMaker.budgetTooLow', 'outfitMaker.invalidRequest'])('operational and budget errors do not become wardrobe alternatives: %s', async key => {
  const s = setup(); await s.run(new OutfitGenerationFailure(key));
  expect(s.setOutfits).toHaveBeenCalledWith([]);
  expect(s.setError).toHaveBeenCalledWith(key);
});
test('shop mode cannot fall back to owned wardrobe items', async () => {
  // v209 removed the local fallback entirely — no owned items are republished on failure.
  const s = setup({ effectiveSource: 'shop' }); await s.run(new OutfitGenerationFailure('outfitMaker.noMatchingClothing'));
  expect(s.setOutfits).toHaveBeenCalledWith([]);
});
test('the screenshot flow with an owned polo never republishes shop companions on failure', async () => {
  // v209 removed the anchored shop recomposition: even with an owned anchor and in-budget shop
  // companions available, a noMatch failure keeps the deck empty and the error visible.
  const anchor = { id: 'polo', macroCategory: 'top', name: 'Polo Piquet 1', type: 'polo', isWardrobe: true, isShopItem: false };
  const companions = [
    { id: 'shop-layer', macroCategory: 'outerwear', name: 'Denim jacket' },
    { id: 'shop-pants', macroCategory: 'bottom', name: 'Blue jeans' },
    { id: 'shop-shoes', macroCategory: 'shoes', name: 'White sneakers' },
  ].map(item => ({ ...item, isShopItem: true, isWardrobe: false, price: 50, currency: 'USD' }));
  const s = setup({ effectiveSource: 'shop', baseItemId: 'polo', styleToUse: 'minimalist', occasionToUse: 'work', alternativeItems: [anchor, ...companions] });
  await s.run(new OutfitGenerationFailure('outfitMaker.noMatchingClothing'));
  expect(s.setOutfits).toHaveBeenCalledTimes(1);
  expect(s.setOutfits).toHaveBeenCalledWith([]); // the owned polo is not re-served as a "closest alternative" look
  expect(s.setError).toHaveBeenLastCalledWith('outfitMaker.noMatchingClothing');
  expect(s.setLoading).toHaveBeenLastCalledWith(false);
});
test('shop alternatives cannot bypass the budget', async () => {
  // v209 composes no local deck at all, so no client-side cap filtering happens either;
  // a failed shop generation publishes nothing.
  const s = setup({ effectiveSource: 'shop', baseItemId: undefined, fitsCap: () => false,
    alternativeItems: setup().alternativeItems.map(item => ({ ...item, isShopItem: true, isWardrobe: false })) });
  await s.run(new OutfitGenerationFailure('outfitMaker.noMatchingClothing'));
  expect(s.setOutfits).toHaveBeenCalledWith([]);
  expect(s.setError).toHaveBeenLastCalledWith('outfitMaker.noMatchingClothing');
});
