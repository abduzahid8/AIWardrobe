/** @jest-environment node */
/**
 * Regression for the owner's 2026-11-12 cloud-generation incident:
 * the Stylist 6 handler completed authenticated requests, but the client
 * discarded their exact ranked deck and returned noMatchingClothing.
 *
 * Exercise expressions from the REAL screen AST, not a parallel replica of
 * the implementation. These tests perform no network or quota operations.
 */
const fs = require('fs');
const ts = require('typescript');

const source = fs.readFileSync('screens/AIOutfitmaker.tsx', 'utf8');
const ast = ts.createSourceFile('AIOutfitmaker.tsx', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);

let deckSelection;
let rotationCondition;
let quotaCondition;
let defaultEngine;

function visit(node) {
  if (ts.isVariableDeclaration(node)) {
    if (ts.isObjectBindingPattern(node.name) &&
        node.name.elements.some(e => e.name.getText(ast) === 'assembledOutfits')) {
      deckSelection = node.initializer?.getText(ast);
    }
    if (ts.isArrayBindingPattern(node.name) && node.name.elements.some(e => e.name.getText(ast) === 'devEngine')) defaultEngine = node.initializer?.getText(ast);
  }
  if (ts.isIfStatement(node)) {
    if (node.thenStatement.getText(ast).includes("throw new Error('client_rotation_overlap')")) {
      rotationCondition = node.expression.getText(ast);
    }
    if (node.expression.getText(ast).includes("consumeOrPaywall('aiOutfits'")) {
      quotaCondition = node.expression.getText(ast);
    }
  }
  ts.forEachChild(node, visit);
}
visit(ast);

function compiled(expression, args, asynchronous = false) {
  if (!expression) throw new Error('Expected screen expression is missing');
  const js = ts.transpileModule(asynchronous ? 'return (async () => (' + expression + '))();' : 'return (' + expression + ');', {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
  }).outputText;
  return Function(...Object.keys(args), js)(...Object.values(args));
}

test('explicit TestFlight release feature chooses Stylist 6 Cloud; regular App Store stays Stylist 5', () => {
  const choose = enabled => compiled(defaultEngine, {
    STYLIST6_CLOUD_ENABLED: enabled, useState: value => [value, () => {}],
  })[0];
  expect(choose(true)).toBe('stylist6cloud');
  expect(choose(false)).toBe('stylist5');
  expect(source).toContain('const serverMeteredCloudQA = STYLIST6_CLOUD_ENABLED');
  expect(source).toContain("let s6CloudActive = STYLIST6_CLOUD_ENABLED");
});

test('three complete server-ranked Stylist 6 Cloud looks bypass legacy deck reconstruction', () => {
  const looks = [
    { id: 'cloud-look-a', items: ['top-a', 'bottom-a', 'shoes-a'] },
    { id: 'cloud-look-b', items: ['top-b', 'bottom-b', 'shoes-b'] },
    { id: 'cloud-look-c', items: ['top-c', 'bottom-c', 'shoes-c'] },
  ];
  const legacyAssembler = jest.fn(() => { throw new Error('legacy assembler must not touch cloud results'); });
  const selected = compiled(deckSelection, {
    s6Active: false, s6CloudActive: true,
    data: { generationVersion: 'stylist-6-cloud-staging-1' },
    budgetedOutfits: looks,
    assembleDiverseDeck: legacyAssembler,
  });
  expect(selected.deck).toBe(looks);
  expect(selected.deck.map(x => x.id)).toEqual(['cloud-look-a', 'cloud-look-b', 'cloud-look-c']);
  expect(legacyAssembler).not.toHaveBeenCalled();
});

test('client rotation-overlap veto does not reject the server-owned cloud plan', () => {
  const evaluate = (cloud, local) => compiled(rotationCondition, {
    s6Active: local, s6CloudActive: cloud, count: 1, slot: 'shoes',
    data: {
      generationVersion: 'stylist-6-cloud-staging-1',
      previousDeckOverlapBySlot: { shoes: 0 },
    },
  });
  expect(evaluate(true, false)).toBe(false);
  expect(evaluate(false, true)).toBe(false);
  // Proves the test reaches the conditional rather than a constant false.
  expect(evaluate(false, false)).toBe(true);
});

test('server-admitted cloud success does not trigger a second client quota consumption', async () => {
  const quota = jest.fn(async () => true);
  const evaluate = (cloud, isCurrent) => compiled(quotaCondition, {
    s6CloudActive: cloud, cancelSignal: undefined,
    consumeOrPaywall: quota, isCurrentGeneration: () => isCurrent,
  }, true);
  expect(await evaluate(true, true)).toBe(false);
  expect(quota).not.toHaveBeenCalled();
  expect(await evaluate(true, false)).toBe(true); // still fail-closed after an account/session switch
  expect(quota).not.toHaveBeenCalled();
  expect(await evaluate(false, true)).toBe(false);
  expect(quota).toHaveBeenCalledTimes(1);
  expect(quota).toHaveBeenCalledWith('aiOutfits', { cancelSignal: undefined });
});

test('Stylist 6 cloud beta defaults to independent full-photo fashion review; production Stylist 5 stays unchanged', () => {
  // This is a source-contract test: the real screen must default to visual
  // judgement ONLY inside the explicit Stylist 6 cloud beta (never S5).
  expect(source).toContain('const [nvidiaVisual, setNvidiaVisual] = useState(STYLIST6_CLOUD_ENABLED)');
  expect(source).toContain('enableVisual: nvidiaVisual === true');
  expect(source).toContain("const [devEngine, setDevEngine] = useState<DevEngine>(STYLIST6_CLOUD_ENABLED ? 'stylist6cloud' : 'stylist5')");
  expect(source).toContain('NVIDIA visual review (optional)');
});

test('actual displayed look count stays visible in both simulator diagnostic configurations', () => {
  expect(source).toContain('testID="stylistEngineBanner"');
  expect(source).toContain('${outfits.length} looks');
  expect(source).not.toContain("STYLIST6_DEV_ENABLED && lastEngineVersion && process.env.EXPO_PUBLIC_STYLIST_DIAGNOSTICS !== 'true'");
});

test('every returned look has a directly selectable tab and retains its exact card identity', () => {
  expect(source).toContain('testID={`stylistLookTab-${index}`}');
  expect(source).toContain('accessibilityState={{ selected: visibleLookIndex === index }}');
  expect(source).toContain('onPress={() => setVisibleLookIndex(index)}');
  expect(source).toContain('key={`stylist-look-${visibleLookIndex}`}');
  expect(source).toContain('index !== visibleLookIndex ? null : (');
  expect(source).toContain('testID={`stylistResult-${index}`}');
});

test('cloud rendering failures retain a classified code rather than hiding rejection reasons', () => {
  expect(source).toContain("'stylist6_client_deck_rejected'");
  expect(source).toContain("'stylist6_incomplete_cloud_deck'");
  expect(source).toContain("'stylist6_render'");
});


test('S6 cloud repeats whole new looks without exhausting finite catalog SKUs; Stylist 5 keeps its own rotation',()=>{
  expect(source).toContain('excludeItemIds: stylist6CloudRecentExclusions(history, baseItemId)');
  expect(source).toContain('const excludeProductKeys = [...new Set([...history.olderRecentProductKeys,...history.previousDeckProductKeys])]');
  expect(source).toContain('const excludeItemIds = excludeProductKeys.filter');
});


test('default Stylist 6 iOS staging gateway reaches the real AI SKU composer runtime, not the 50s Vercel fallback', () => {
  const fs = require('fs');
  const defaultGateway = fs.readFileSync('backend/supabase/functions/aiw-stylist6-node-staging/index.ts', 'utf8');
  const renderGateway = fs.readFileSync('backend/stylist-proxy/stylist6-render-staging.ts', 'utf8');
  const nodeRoute = fs.readFileSync('backend/stylist-node/api/stylist6-generate-outfits.ts', 'utf8');
  const cloudHandler = fs.readFileSync('backend/stylist6/cloud_handler.mjs', 'utf8');

  // A 50-second fallback silently disables both real-SKU composition and final
  // portfolio editing. Keep the normal QA route on the already-authenticated
  // 72-second Render canary, without touching production Stylist 5.
  expect(defaultGateway).toContain('stylist6-render-staging.ts');
  expect(defaultGateway).not.toMatch(/^import .*stylist6-staging\.ts/m);
  expect(renderGateway).toContain("STYLIST6_RENDER_NODE_SHARED_SECRET");
  expect(renderGateway).toContain('AbortSignal.timeout(85000)');
  expect(renderGateway).toContain('signStylistTransport');
  expect(nodeRoute).toContain('STYLIST6_RENDER_CANARY_BUDGET_MS = 72_000');
  expect(cloudHandler).toContain('budgetMs>=70000');
  expect(cloudHandler).toContain('composeCatalogSkuLooks(');
  expect(cloudHandler).toContain('selectPhotoEditorialDeck(');
  expect(renderGateway).toContain("RENDER_PATH='/api/stylist6-generate-outfits'");
});
