/** @jest-environment node */
/**
 * Shop-catalog color provenance in the Edge function: the merchant color slug
 * is trusted evidence and must be read before free-text descriptions, which
 * often name companion garments ("pair with white trousers").
 */
const fs = require('fs');
const path = require('path');
const ts = require('typescript');
const source = fs.readFileSync(
  path.resolve(__dirname, '../../backend/supabase/functions/generate-outfits/index.ts'),
  'utf8'
);
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;
const api = new Function('require', 'exports', 'Deno', compiled + '\nreturn {inferShopColor};')(
  (name) => {
    if (name.includes('server.ts')) return { serve: () => {} };
    if (name.includes('budgetFx.ts')) return require('../../src/domain/budgetFx');
    const domain = [
      'occasionRules', 'outfitStructure', 'outfitDiversity', 'outfitRotation', 'garmentContext',
      'stylingContext', 'editorialIntelligence', 'outfitConcepts', 'visualFashion',
      'stylist5VisualContract', 'garmentDescriptor', 'catalogCandidates', 'lookExplanation',
      'validatedQuartets', 'legacyClientCompatibility', 'outfitDeckSelection', 'styleInference',
      'catalogIdentity',
    ];
    for (const module of domain) {
      if (name.includes(module)) return require(path.resolve(__dirname, '../../src/domain/', module + '.ts'));
    }
    return {};
  },
  {},
  { env: { get: () => undefined } }
);
const { catalogImageAdmission } = require('../../src/domain/catalogIdentity');

test('the merchant color slug beats companion colors named in the description', () => {
  expect(
    api.inferShopColor({
      name: 'Slim fit blazer',
      description: 'Pair with white trousers and black shoes',
      source_url: 'https://www.zara.com/uz/en/blazer-p01234567.html?color_name=navy-blue',
    })
  ).toBe('navy');
});

test('the product title still beats the URL slug, and the slug beats description-only evidence', () => {
  expect(
    api.inferShopColor({
      name: 'Dark green cardigan',
      description: 'Pair with white trousers',
      source_url: 'https://www.zara.com/uz/en/cardigan-p01234567.html?color_name=red',
    })
  ).toBe('dark green');
  expect(
    api.inferShopColor({
      name: 'Slim fit overshirt',
      description: 'Wear it over a white tee',
      source_url: 'https://www.zara.com/uz/en/overshirt-p01234567.html?color_name=chocolate-brown',
    })
  ).toBe('chocolate');
  expect(api.inferShopColor({ name: 'Slim fit overshirt', description: 'Wear it over a white tee' })).toBe('white');
  expect(api.inferShopColor({ name: 'Slim fit overshirt' })).toBe('neutral');
});

test('the Edge catalog path repairs the known Zara image construction before admission', () => {
  const malformed = 'https://static.zara.net/x/04395705_251_1.jpg&w=1024';
  expect(catalogImageAdmission(malformed)).toEqual({
    ok: true,
    url: 'https://static.zara.net/x/04395705_251_1.jpg?w=1024',
    repaired: true,
  });
});
