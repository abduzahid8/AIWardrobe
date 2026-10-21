/** @jest-environment node */
const fs = require('fs');
const ts = require('typescript');
const { mapCategoryToType, normalizeCategory } = require('../../src/utils/mappingUtils');
function extract(path, name) {
  const text = fs.readFileSync(require.resolve(path), 'utf8');
  const file = ts.createSourceFile(path, text, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  let code;
  function visit(node) {
    if (ts.isVariableDeclaration(node) && node.name.getText(file) === name) code = `const handler = ${node.initializer.getText(file)};`;
    if (ts.isFunctionDeclaration(node) && node.name?.text === name) code = `${node.getText(file)}; const handler = ${name};`;
    ts.forEachChild(node, visit);
  }
  visit(file);
  if (!code) throw new Error(`Missing production function ${name}`);
  return ts.transpile(code, { target: ts.ScriptTarget.ES2020 });
}
const saveCode = extract('../../components/ClothingDetailEditor.tsx', 'saveChanges');
const readCode = extract('../../store/wardrobeSyncService.ts', 'mapRowToItem');
function editor(overrides = {}) {
  const addItem = jest.fn(async () => undefined);
  const context = {
    authEvents: { getRevision: () => 1 }, selectedType: 'outerwear', selectedColor: 'beige', selectedSeason: 'winter', selectedWeather: [],
    route: { params: {} }, onSave: undefined, COLORS: [{ id: 'beige', hex: '#C7B299', label: 'Beige' }],
    isEditingExisting: false, existingItem: undefined, retainedCreate: { current: null },
    useAuthStore: { getState: () => ({ user: { id: 'owner' } }) }, useWardrobeStore: { getState: () => ({ addItem }) },
    itemImageUri: 'file:///jacket.jpg', detectedSpecificType: 'jacket', detectedPattern: 'plaid', detectedMaterial: 'wool',
    detectedDescription: 'Wool plaid jacket', detectedStyle: 'casual', aiConfidence: 0.93,
    editorSeasonToDomain: value => value, styleToOccasions: () => ['casual'], mapCategoryToType, normalizeCategory,
    Haptics: { notificationAsync: jest.fn(), NotificationFeedbackType: { Success: 1 } },
    Alert: { alert: jest.fn() }, t: key => key, navigation: { goBack: jest.fn() },
    ...overrides,
  };
  const handler = Function(...Object.keys(context), `${saveCode} return handler;`)(...Object.values(context));
  return { ...context, handler, addItem };
}
it('new save preserves specific garment identity, pattern, material and confidence under authenticated owner', async () => {
  const app = editor(); await app.handler();
  expect(app.addItem).toHaveBeenCalledTimes(1);
  expect(app.addItem).toHaveBeenCalledWith(expect.objectContaining({ userId: 'owner', category: 'outerwear', subCategory: 'jacket',
    pattern: 'plaid', material: 'wool', detectionConfidence: 0.93, primaryColor: 'beige' }));
});
it('user category correction supersedes conflicting detected subtype', async () => {
  const app = editor({ selectedType: 'bottoms' }); await app.handler();
  expect(app.addItem).toHaveBeenCalledWith(expect.objectContaining({ category: 'bottoms', subCategory: 'bottoms' }));
});
it.each([undefined, null, ''])('unknown pattern %j is not saved as solid', async pattern => {
  const app = editor({ detectedPattern: pattern }); await app.handler();
  expect(app.addItem.mock.calls[0][0].pattern).toBe('');
});
it('zero confidence is retained', async () => {
  const app = editor({ aiConfidence: 0 }); await app.handler();
  expect(app.addItem.mock.calls[0][0].detectionConfidence).toBe(0);
});
it.each([undefined, -1, 2, NaN])('invalid confidence %j is omitted', async confidence => {
  const app = editor({ aiConfidence: confidence }); await app.handler();
  expect(app.addItem.mock.calls[0][0]).not.toHaveProperty('detectionConfidence');
});
it.each([null, undefined, '', 'plaid'])('reload preserves known or unknown pattern %j without fabricating solid', pattern => {
  const read = Function(`${readCode} return handler;`)();
  expect(read({ pattern }).pattern).toBe(typeof pattern === 'string' ? pattern : '');
});
it('capacity-rejected creates show retained-work guidance and reuse the same ID on retry', async () => {
  const app=editor(),id='77777777-7777-4777-8777-777777777777';
  app.addItem.mockRejectedValue(Object.assign(new Error('full'),{code:'WARDROBE_LIMIT_REACHED',limit:20,savedLocally:true,itemId:id}));
  await app.handler();await app.handler();
  expect(app.addItem.mock.calls[0]).toHaveLength(1);
  expect(app.addItem.mock.calls[1][1]).toEqual({idempotencyId:id});
  expect(app.Alert.alert).toHaveBeenLastCalledWith('clothingEditor.wardrobeFull','wardrobeSync.capacityMessage',expect.any(Array));
  expect(app.Haptics.notificationAsync).not.toHaveBeenCalled();
});
