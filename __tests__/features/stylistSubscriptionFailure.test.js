/** @jest-environment node */
const fs = require('fs'), ts = require('typescript');
const { OutfitGenerationFailure } = require('../../src/services/outfitGenerationFailure');
const source = fs.readFileSync('screens/AIOutfitmaker.tsx', 'utf8');
const ast = ts.createSourceFile('screen.tsx', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
let body;
function visit(node) {
  if (ts.isCatchClause(node) && node.block.getText(ast).includes('const noMatch =')) body = node.block.getText(ast);
  ts.forEachChild(node, visit);
}
visit(ast);
const code = ts.transpileModule(`async function run(err: any) ${body}`, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText;
function setup(current = true) {
  const deps = {
    fallbackTimer: {current: null}, isCurrentGeneration: () => current, OutfitGenerationFailure,
    setOutfits: jest.fn(), setLoading: jest.fn(), setError: jest.fn(), t: key => key,
    navigation: {navigate: jest.fn()}, closestAvailableOutfits: jest.fn(),
  };
  return {...deps, run: Function(...Object.keys(deps), code + '; return run;')(...Object.values(deps))};
}
test('server subscription denial opens the paywall with localized guidance and no fallback', async () => {
  const s = setup(); await s.run(new OutfitGenerationFailure('outfitMaker.subscriptionRequired'));
  expect(s.navigation.navigate).toHaveBeenCalledTimes(1);
  expect(s.navigation.navigate).toHaveBeenCalledWith('Paywall');
  expect(s.setOutfits).toHaveBeenCalledWith([]);
  expect(s.setLoading).toHaveBeenCalledWith(false);
  expect(s.setError).toHaveBeenCalledWith('outfitMaker.subscriptionRequired');
  expect(s.closestAvailableOutfits).not.toHaveBeenCalled();
});
test('a stale account response cannot open a paywall for the current account', async () => {
  const s = setup(false); await s.run(new OutfitGenerationFailure('outfitMaker.subscriptionRequired'));
  expect(s.navigation.navigate).not.toHaveBeenCalled();
  expect(s.setError).not.toHaveBeenCalled();
});

test('the outfit chat keeps the localized denial and opens the same paywall', () => {
  const chat = ts.createSourceFile('chat.tsx', fs.readFileSync('screens/OutfitAIScreen.tsx', 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  let catchBody;
  function walk(node) {
    if (ts.isCatchClause(node) && node.block.getText(chat).includes('const aiMessage:')) catchBody = node.block.getText(chat);
    ts.forEachChild(node, walk);
  }
  walk(chat);
  const chatCode = ts.transpileModule(`function run(error: unknown) ${catchBody}`, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText;
  const navigation = {navigate: jest.fn()}, setMessages = jest.fn(), deps = {OutfitGenerationFailure, navigation, setMessages, t: key => key, console: {error: jest.fn()}};
  const run = Function(...Object.keys(deps), chatCode + ';return run;')(...Object.values(deps));
  run(new OutfitGenerationFailure('outfitMaker.subscriptionRequired'));
  expect(setMessages.mock.calls[0][0]([])[0].text).toBe('outfitMaker.subscriptionRequired');
  expect(navigation.navigate).toHaveBeenCalledWith('Paywall');
});
