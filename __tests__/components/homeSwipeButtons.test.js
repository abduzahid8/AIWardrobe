// Execute the shipped Home callbacks; native gesture/authorization behavior is
// separately covered by outfitSwipeAuthorization and live receipt tests.
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const source = ts.createSourceFile('HomeScreen.tsx', fs.readFileSync(path.resolve(__dirname, '../../screens/HomeScreen.tsx'), 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
function find(predicate) {
  let found;
  function visit(node) { if (predicate(node)) found = node; else ts.forEachChild(node, visit); }
  visit(source); if (!found) throw Error('Home callback missing'); return found;
}
function callback(node, bindings) {
  const js = ts.transpileModule(`const __handler = (${node.getText(source)});`, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText;
  return new Function(...Object.keys(bindings), `${js}\nreturn __handler;`)(...Object.values(bindings));
}
test.each([['home.saveOutfit', 'swipeRight'], ['home.dislikeOutfit', 'swipeLeft']])('%s can recover a server receipt despite a cached zero', (label, method) => {
  const element = find(node => ts.isJsxOpeningElement(node) && node.attributes.properties.some(attr => ts.isJsxAttribute(attr) && attr.name.getText(source) === 'accessibilityLabel' && attr.getText(source).includes(label)));
  const attr = element.attributes.properties.find(attr => ts.isJsxAttribute(attr) && attr.name.getText(source) === 'onPress');
  const stackRef = { current: { swipeRight: jest.fn(), swipeLeft: jest.fn() } }, requireFeature = jest.fn();
  const press = callback(attr.initializer.expression, { stackRef, canUseNow: () => false, requireFeature });
  press(); expect(stackRef.current[method]).toHaveBeenCalledTimes(1); expect(requireFeature).not.toHaveBeenCalled();
});
test('Home supplies the actual displayed card identity and cancellation signal to verified authorization', async () => {
  const declaration = find(node => ts.isVariableDeclaration(node) && node.name.getText(source) === 'authorizeHomeSwipe');
  const consumeOrPaywall = jest.fn().mockResolvedValue(true), signal = new AbortController().signal;
  const authorize = callback(declaration.initializer, { consumeOrPaywall, authEvents: { getRevision: () => 1 }, Alert: { alert: jest.fn() }, t: key => key });
  expect(await authorize({ key: 'displayed-outfit-42' }, 'left', signal)).toBe(true);
  expect(consumeOrPaywall).toHaveBeenCalledWith('homeSwipes', { requireServer: true, cancelSignal: signal, homeCardKey: 'displayed-outfit-42' });
});
