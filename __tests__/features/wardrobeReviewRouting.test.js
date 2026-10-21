/** @jest-environment node */
const fs = require('fs');
const ts = require('typescript');
const { mapCategoryToType } = require('../../src/utils/mappingUtils');
const text = fs.readFileSync(require.resolve('../../screens/WardrobeVideoScreen.tsx'), 'utf8');
const file = ts.createSourceFile('WardrobeVideoScreen.tsx', text, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
let expression;
function visit(node) {
  if (ts.isVariableDeclaration(node) && node.name.getText(file) === 'handleReviewAndSave') expression = node.initializer.getText(file);
  ts.forEachChild(node, visit);
}
visit(file);
if (!expression) throw new Error('Missing production review handler');
const compiled = ts.transpile(`const handler = ${expression};`, { target: ts.ScriptTarget.ES2020 });
function review(results, index) {
  const navigation = { navigate: jest.fn() };
  Function('results', 'navigation', 'mapCategoryToType', `${compiled} handler(${index === undefined ? '' : JSON.stringify(index)});`)(results, navigation, mapCategoryToType);
  return navigation.navigate;
}
it('review preserves recognized jacket metadata instead of defaulting to tops and dropping attributes', () => {
  const item = { itemType: 'jacket', color: 'Beige', material: 'cotton', style: 'casual', confidence: 0.93, description: 'Beige cotton jacket', frameImage: 'file:///frame.jpg' };
  expect(review({ detectedItems: [item] })).toHaveBeenCalledWith('ClothingDetailEditor', {
    imageUri: item.frameImage, detectedType: 'outerwear', detectedColor: 'beige', detectedMaterial: 'cotton',
    detectedStyle: 'casual', aiConfidence: 0.93, detectedDescription: item.description, detectedItem: item,
  });
});
it.each([null, { detectedItems: [] }])('absent detections never open a fabricated editor item: %j', results => {
  expect(review(results)).not.toHaveBeenCalled();
});

it('opens a later selected garment without reusing the first image or attributes', () => {
  const first = { itemType: 'shirt', color: 'White', frameImage: 'file:///shirt.jpg' };
  const second = { itemType: 'jeans', color: 'Blue', frameImage: 'file:///jeans.jpg', confidence: 0 };
  expect(review({ detectedItems: [first, second] }, 1)).toHaveBeenCalledWith('ClothingDetailEditor', expect.objectContaining({
    imageUri: second.frameImage, detectedType: 'bottoms', detectedColor: 'blue', aiConfidence: 0, detectedItem: second,
  }));
});
it.each([-1, 1, 0.5])('rejects invalid detection index %s', index => {
  expect(review({ detectedItems: [{ itemType: 'jacket' }] }, index)).not.toHaveBeenCalled();
});
it('the production result-card callback sends its own item to the review handler', () => {
  let callback;
  function find(node) {
    if (ts.isJsxAttribute(node) && node.name.getText(file) === 'onPress' && node.initializer
        && node.initializer.getText(file).includes('results.detectedItems.indexOf(item)')) {
      callback = node.initializer.expression.getText(file);
    }
    ts.forEachChild(node, find);
  }
  find(file);
  expect(callback).toBeDefined();
  const first = { itemType: 'shirt', frameImage: 'file:///shirt.jpg' };
  const second = { itemType: 'jeans', frameImage: 'file:///jeans.jpg' };
  const results = { detectedItems: [first, second] };
  const navigate = jest.fn();
  const invokeCard = Function('results', 'navigation', 'mapCategoryToType', 'item',
    `${compiled} const handleReviewAndSave = handler; (${callback})();`);
  invokeCard(results, { navigate }, mapCategoryToType, second);
  expect(navigate).toHaveBeenLastCalledWith('ClothingDetailEditor', expect.objectContaining({ detectedItem: second }));
  invokeCard(results, { navigate }, mapCategoryToType, first);
  expect(navigate).toHaveBeenLastCalledWith('ClothingDetailEditor', expect.objectContaining({ detectedItem: first }));
});
