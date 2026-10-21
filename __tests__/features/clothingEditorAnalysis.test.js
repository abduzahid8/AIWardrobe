/** @jest-environment node */
const fs = require('fs');
const ts = require('typescript');
const { mapCategoryToType, mapColorToId } = require('../../src/utils/mappingUtils');
const text = fs.readFileSync(require.resolve('../../components/ClothingDetailEditor.tsx'), 'utf8');
const ast = ts.createSourceFile('editor.tsx', text, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
function expression(name) {
  let value;
  function visit(node) {
    if (ts.isVariableDeclaration(node) && node.name.getText(ast) === name) value = node.initializer.getText(ast);
    ts.forEachChild(node, visit);
  }
  visit(ast);
  if (!value) throw new Error(`Missing production expression ${name}`);
  return ts.transpile(`const value = ${value};`, { target: ts.ScriptTarget.ES2020 });
}
const providerResult = { success: true, description: 'Beige cotton jacket', classification: { category: 'jacket', section: 'outerwear', confidence: 0.92, attributes: { color: 'beige', material: 'cotton', style: 'casual' } } };
function harness(mode = 'success') {
  let active = true;
  const context = { itemImageUri: 'file:///jacket.jpg',
    FileSystem: { readAsStringAsync: jest.fn(async () => { if (mode === 'file-account-change') active = false; return 'photo-base64'; }) },
    ExternalAIService: { classifyOnly: jest.fn(async () => {
      if (mode === 'provider-account-change') active = false;
      if (mode === 'exception') throw new Error('provider unavailable');
      return mode === 'unavailable' ? { success: false, classification: null } : providerResult;
    }) }, canApply: () => active,
    setAnalysisMetadata: jest.fn(), setSelectedType: jest.fn(), setSelectedColor: jest.fn(),
    setSelectedSeason: jest.fn(), setSelectedWeather: jest.fn(), setIsAnalyzing: jest.fn(),
    mapCategoryToType, mapColorToId, inferSeasonFromType: () => 'spring', inferWeatherFromType: () => ['sun'],
    Alert: { alert: jest.fn() }, t: x => x,
  };
  const handler = Function(...Object.keys(context), `${expression('analyzeEditorImage')} return value;`)(...Object.values(context));
  return { handler, ...context };
}
it('direct photo classification retains full result and updates visible garment attributes', async () => {
  const app = harness(); await app.handler();
  expect(app.FileSystem.readAsStringAsync).toHaveBeenCalledWith('file:///jacket.jpg', { encoding: 'base64' });
  expect(app.setAnalysisMetadata).toHaveBeenCalledWith(providerResult);
  expect(app.setSelectedType).toHaveBeenCalledWith('outerwear');
  expect(app.setSelectedColor).toHaveBeenCalledWith('beige');
  expect(app.setIsAnalyzing).toHaveBeenLastCalledWith(false);
  expect(app.Alert.alert).not.toHaveBeenCalled();
});
it.each(['unavailable', 'exception'])('%s reports a visible error and publishes no classification', async mode => {
  const app = harness(mode); await app.handler();
  expect(app.setAnalysisMetadata).not.toHaveBeenCalled();
  expect(app.setSelectedType).not.toHaveBeenCalled();
  expect(app.Alert.alert).toHaveBeenCalledWith('videoAnalysis.analysisFailed', 'videoAnalysis.somethingWrong');
  expect(app.setIsAnalyzing).toHaveBeenLastCalledWith(false);
});
it.each(['file-account-change', 'provider-account-change'])('%s prevents late results or alerts from crossing accounts', async mode => {
  const app = harness(mode); await app.handler();
  expect(app.setAnalysisMetadata).not.toHaveBeenCalled();
  expect(app.setSelectedType).not.toHaveBeenCalled();
  expect(app.Alert.alert).not.toHaveBeenCalled();
  expect(app.setIsAnalyzing).not.toHaveBeenCalled();
  if (mode === 'file-account-change') expect(app.ExternalAIService.classifyOnly).not.toHaveBeenCalled();
});
it('save metadata resolves to the retained direct-analysis result', () => {
  const context = { existingItem: undefined, route: { params: {} }, analysisMetadata: providerResult };
  const expected = { detectedStyle: 'casual', detectedMaterial: 'cotton', aiConfidence: 0.92, detectedDescription: 'Beige cotton jacket', detectedSpecificType: 'jacket' };
  for (const [name, expectedValue] of Object.entries(expected)) {
    const value = Function(...Object.keys(context), `${expression(name)} return value;`)(...Object.values(context));
    expect(value).toBe(expectedValue);
  }
});
it('an existing zero confidence takes precedence over later analysis metadata', () => {
  const context = { existingItem: { detectionConfidence: 0 }, route: { params: { aiConfidence: 0.8 } }, analysisMetadata: providerResult };
  expect(Function(...Object.keys(context), `${expression('aiConfidence')} return value;`)(...Object.values(context))).toBe(0);
});
