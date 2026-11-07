import fs from 'fs';
import path from 'path';
import ts from 'typescript';

const source = fs.readFileSync(path.resolve(__dirname, '../../screens/HomeScreen.tsx'), 'utf8');
const tree = ts.createSourceFile('HomeScreen.tsx', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
let predicate = '';
let shopOnly = '';
function visit(node: ts.Node) {
  if (ts.isVariableDeclaration(node) && node.name.getText(tree) === 'isShopOnlyDeck') shopOnly = node.initializer!.getText(tree);
  if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression) && node.expression.name.text === 'filter' && node.arguments[0]?.getText(tree).includes('isShopOnlyDeck(item.style')) predicate = node.arguments[0].getText(tree);
  ts.forEachChild(node, visit);
}
visit(tree);
if (!predicate || !shopOnly) throw new Error('Home occasion visibility logic not found');
const compiled = ts.transpileModule(`function filter(items: any[], occasions: any[]) { const isShopOnlyDeck = ${shopOnly}; return occasions.filter(${predicate}); }`, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText;
const filter = new Function(`${compiled}; return filter;`)();
const team = { occasion: 'Team Collaboration', style: 'business_casual' };
const dinner = { occasion: 'Night-Time Dinner', style: 'old_money' };
it('shows the shop-only Team Collaboration deck with an empty wardrobe', () => {
  expect(filter([], [team, dinner])).toEqual([team]);
});
it('continues to show all configured decks with a full wardrobe', () => {
  expect(filter([1, 2, 3, 4, 5], [team, dinner])).toEqual([team, dinner]);
});

let deckEnabled = '';
function findEnabled(node: ts.Node) {
  if (ts.isJsxAttribute(node) && node.name.getText(tree) === 'enabled' && node.parent.getText(tree).includes('occasion={item.occasion}')) {
    deckEnabled = (node.initializer as ts.JsxExpression).expression!.getText(tree);
  }
  ts.forEachChild(node, findEnabled);
}
findEnabled(tree);
if (!deckEnabled) throw new Error('Home deck enablement not found');
const enabledCode = ts.transpileModule(`function enabled(item:any,decksFetchEnabled:boolean,isWardrobeLoading:boolean){ const isShopOnlyDeck = ${shopOnly}; return ${deckEnabled}; }`, {compilerOptions: {target: ts.ScriptTarget.ES2022}}).outputText;
const enabled = new Function(`${enabledCode}; return enabled;`)();
it('starts Team Collaboration while wardrobe hydration is still running', () => {
  expect(enabled(team, true, true)).toBe(true);
  expect(enabled(dinner, true, true)).toBe(false);
  expect(enabled(team, false, false)).toBe(false);
});
