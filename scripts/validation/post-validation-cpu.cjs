/* Offline replay of the actual Edge handler after provider validation. No network/auth stubs are used to authorize a request: this starts AFTER admission. */
const fs = require('node:fs'),
  path = require('node:path'),
  ts = require('typescript'),
  assert = require('node:assert/strict');
const root = path.resolve(process.argv[2] || process.cwd());
const cache = new Map();
let metrics, logs;
const cpuMs = (start) => {
  const n = process.cpuUsage(start);
  return (n.user + n.system) / 1000;
};
function measure(name, fn) {
  const start = process.cpuUsage();
  try {
    return fn();
  } finally {
    const m = (metrics[name] ||= { calls: 0, cpuMs: 0 });
    m.calls++;
    m.cpuMs += cpuMs(start);
  }
}
function load(file) {
  if (cache.has(file)) return cache.get(file).exports;
  const module = { exports: {} };
  cache.set(file, module);
  let source = fs.readFileSync(file, 'utf8');
  if (file.endsWith('/generate-outfits/index.ts')) {
    const begin = source.indexOf('        const itemMap = new Map<string, any>');
    const end = source.indexOf('        async function generateCuratedChoices()', begin);
    const post = source.indexOf('        // Resolve immutable product identity');
    const finish = source.indexOf('\n    } catch (error: any)', post);
    assert(begin > 0 && end > begin && post > end && finish > post);
    const signals = source.includes('        const lookSignals =')
      ? source.slice(
          source.indexOf('        const lookSignals ='),
          source.indexOf('        const eligibleCandidate =')
        )
      : '';
    const eligible = source.slice(
      source.indexOf('        const eligibleCandidate ='),
      source.indexOf('        const requiredIds =')
    );
    source += `\nexport {validateOutfitAgainstStyle,validateOutfitCompatibility};\nexport async function replay(fixture:any) {\n const {wardrobeItems,aiOutfits,styleKey,occasion,anchorItemId,preferences,body,excludeSet,previousSet,previousShoeSet,rotation,wornItemIds,selectedItemIds,catalogOnly,wardrobeOnly}=fixture;\n const stylePreferences=styleKey,selectionStyle=styleKey,prompt='',weather=null,layered=true,deckTarget=3,limit=3,requestFx=undefined,aiProviderUsed='offline-provider-replay',providerTrace=[],fetchedShoeIds=[]; let generationSource='ai';\n const qualityContext=stylingContext({style:styleKey,occasion,weather,anchorId:anchorItemId,preferences,sourceMode:catalogOnly?'catalog':wardrobeOnly?'wardrobe':'mixed'});const memoQuality=createQualityMemo();\n const deckPriority=createShoeRotationPriority(excludeSet,previousSet,body.rotationDecks||[],anchorItemId,{previous:previousShoeSet,decks:body.shoeRotationDecks||[]});\n ${signals}\n ${eligible}\n ${source.slice(begin, end)}\n ${source.slice(post, finish)}\n }`;
  }
  source = source.replaceAll(
    'finalOutfits.push(...replacements)',
    'finalOutfits.push(...replacements); __probe({choices:completeChoices().length,rejections:Object.fromEntries(choiceRejections)})'
  );
  const code = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const req = (name) =>
    name.startsWith('https:')
      ? name.includes('server.ts')
        ? { serve: () => {} }
        : {
            createClient: () => {
              throw Error('network forbidden');
            },
          }
      : name.startsWith('.')
        ? load(path.resolve(path.dirname(file), name.endsWith('.ts') ? name : name + '.ts'))
        : require(name);
  const quiet = { log: () => logs++, info: () => logs++, warn: () => logs++, error: () => logs++ };
  new Function('require', 'module', 'exports', 'Deno', 'console', '__probe', code)(
    req,
    module,
    module.exports,
    { env: { get: () => undefined } },
    quiet,
    (data) => {
      metrics.expansion = data;
    }
  );
  for (const name of [
    'validatedQuartets',
    'assembleDiverseDeck',
    'assignRotatingShoes',
    'assessOutfitQuality',
    'styleJudgeReason',
    'outfitCoherence',
  ])
    if (typeof module.exports[name] === 'function') {
      const fn = module.exports[name];
      module.exports[name] = (...args) => measure(name, () => fn(...args));
    }
  return module.exports;
}
function fixture(occasion, anchored = false, mode = 'fresh') {
  const names = [
    'White cotton poplin dress shirt',
    'Navy structured wool blazer',
    'Navy tailored wool trousers',
    'Black leather Oxford shoes',
  ];
  const slots = ['top', 'outerwear', 'bottom', 'shoes'];
  const wardrobeItems = slots.flatMap((macroCategory, s) =>
    Array.from({ length: 24 }, (_, i) => ({
      id: `shop_${s}-${i}`,
      name: `${names[s]} ${i}`,
      description: `${names[s]}. ${s === 3 ? 'Polished calfskin leather outsole.' : 'Regular fit, smooth woven cloth.'}`,
      type: s === 3 ? 'shoes' : macroCategory,
      macroCategory,
      color: ['white', 'navy', 'navy', 'black'][s],
      material: s === 1 || s === 2 ? 'wool' : s === 3 ? 'leather' : 'cotton',
      pattern: 'solid',
      isShopItem: true,
      isWardrobe: false,
      price: 50,
      currency: 'USD',
      imageUrl: `https://offline.invalid/${s}-${i}.jpg`,
    }))
  );
  const anchorItemId = anchored ? 'shop_1-0' : undefined;
  const aiOutfits = Array.from({ length: 3 }, (_, i) => ({
    items: slots.map((slot, s) => ({
      id: anchorItemId && s === 1 ? anchorItemId : `shop_${s}-${i}`,
    })),
  }));
  if (mode === 'repair') aiOutfits[1].items[0] = { ...aiOutfits[0].items[0] };
  if (mode === 'budget-pressure') {
    wardrobeItems.forEach((item, i) => {
      item.price = i % 24 < 3 ? 50 : 180;
      item.description +=
        ' Woven garment construction with a smooth finish and documented regular proportions.'.repeat(
          4
        );
    });
    aiOutfits[1].items[0] = { ...aiOutfits[0].items[0] };
  }
  if (mode === 'rejected-reserve') {
    wardrobeItems.forEach((item, i) => {
      if (i % 24 >= 3 && item.macroCategory === 'bottom')
        item.name = item.name.replace('Navy', 'Charcoal');
    });
    aiOutfits[1].items[0] = { ...aiOutfits[0].items[0] };
  }
  if (mode === 'quality-floor' || mode === 'minimalist-floor') {
    const row = (slot) =>
      wardrobeItems.find((item) => item.macroCategory === slot && item.id.endsWith('-2'));
    if (mode === 'quality-floor') {
      if (anchored) row('shoes').name = 'Flextride Plus Leather Loafers - Black';
      else {
        row('top').name = 'Bold rugby striped cotton shirt';
        row('outerwear').name = 'Technical leisure zip blazer';
      }
    } else {
      wardrobeItems.forEach((item) => {
        if (item.macroCategory === 'top')
          item.name = item.name.replace('poplin dress shirt', 'T-shirt');
        if (item.macroCategory === 'bottom')
          item.name = item.name.replace('tailored wool trousers', 'straight denim jeans');
        if (item.macroCategory === 'shoes')
          item.name = item.name.replace('Oxford shoes', 'minimal leather sneakers');
        item.description = item.name + '. Regular fit.';
        if (item.macroCategory === 'bottom') item.material = 'cotton denim';
      });
      row('bottom').description = 'Shredded denim with large holes';
    }
  }
  const keys = wardrobeItems.filter((_, i) => i % 24 >= 20).map((i) => 'id:' + i.id);
  return {
    wardrobeItems,
    aiOutfits,
    styleKey: mode === 'minimalist-floor' ? 'minimalist' : 'old_money',
    occasion,
    anchorItemId,
    preferences: { maxShopBudget: { amount: 250, currency: 'USD' } },
    body: {
      generationRequestId: 'offline-cpu',
      recentLookSignatures: [],
      rotationDecks: [{ productKeys: keys }],
      shoeRotationDecks: [{ productKeys: keys }],
    },
    excludeSet: new Set(keys),
    previousSet: new Set(keys),
    previousShoeSet: new Set(keys),
    rotation: { recycledSlots: [] },
    wornItemIds: new Set(),
    selectedItemIds: [],
    catalogOnly: true,
    wardrobeOnly: false,
  };
}
(async () => {
  metrics = {};
  logs = 0;
  const api = load(path.join(root, 'backend/supabase/functions/generate-outfits/index.ts'));
  const reports = [];
  const cases = [
    ['Work', false, 'fresh'],
    ['Formal', false, 'fresh'],
    ['Work', true, 'fresh'],
    ['Formal', true, 'fresh'],
    ['Work', false, 'repair'],
    ['Formal', true, 'repair'],
    ['Work', false, 'budget-pressure'],
    ['Formal', true, 'budget-pressure'],
    ['Work', false, 'rejected-reserve'],
    ['Work', false, 'quality-floor'],
    ['Work', true, 'quality-floor'],
    ['Casual', false, 'minimalist-floor'],
  ];
  for (const [occasion, anchored, mode] of cases) {
    metrics = {};
    logs = 0;
    const f = fixture(occasion, anchored, mode);
    const itemMap = new Map(f.wardrobeItems.map((item) => [item.id, item]));
    for (const outfit of f.aiOutfits) {
      const verdict = api.validateOutfitAgainstStyle(
        outfit,
        f.styleKey,
        itemMap,
        true,
        f.preferences
      );
      assert(verdict.ok, verdict.reason);
      assert(api.validateOutfitCompatibility(outfit, itemMap, f.styleKey).ok);
    }
    const start = process.cpuUsage(),
      wall = performance.now();
    const response = await api.replay(f),
      cpu = cpuMs(start),
      data = await response.json();
    const report = {
      occasion,
      anchored,
      mode,
      providerValidated: 3,
      pool: f.wardrobeItems.length,
      cpuMs: cpu,
      wallMs: performance.now() - wall,
      status: response.status,
      outfits: data.outfits?.length,
      logs,
      stages: metrics,
    };
    reports.push(report);
    console.log(JSON.stringify(report));
    const expected = mode === 'rejected-reserve' ? [200, 422] : [200];
    assert(expected.includes(response.status), JSON.stringify(data));
    if (response.status === 200) assert.equal(data.outfits.length, 3);
    if (process.env.CPU_BUDGET_MS) {
      assert((metrics.validatedQuartets?.calls || 0) <= 1);
      if (mode === 'fresh')
        assert.equal(
          metrics.validatedQuartets?.calls || 0,
          0,
          'fresh valid decks must not expand, including audit'
        );
      if (metrics.expansion) assert(metrics.expansion.choices <= 24);
      assert(logs <= 4, 'no per-candidate production logging');
    }
    for (const look of data.outfits || []) {
      assert.equal(look.items.length, 4);
      assert(look.items.every((i) => f.wardrobeItems.some((p) => p.id === i.id)));
      assert(look.items.reduce((n, i) => n + Number(i.price || 0), 0) <= 250);
      if (anchored) assert(look.items.some((i) => i.id === f.anchorItemId));
    }
    if (response.status === 200 && (mode === 'quality-floor' || mode === 'minimalist-floor')) {
      const signature = (items) =>
        items
          .map((i) => i.id)
          .sort()
          .join('|');
      const shown = data.outfits.map((look) => signature(look.items));
      for (const good of f.aiOutfits.slice(0, 2))
        assert(
          shown.includes(signature(good.items)),
          'good look must be preserved including shoes'
        );
      assert(!shown.includes(signature(f.aiOutfits[2].items)), 'weak look must be replaced');
      assert(data.outfits.every((look) => look.qualityAssessment.acceptance.passed));
    }
    if (response.status === 200) {
      const selected = data.outfits.flatMap((l) => l.items.filter((i) => i.id !== f.anchorItemId));
      for (const item of selected)
        assert(!f.excludeSet.has('id:' + item.id), 'recent products cannot bypass the gate');
      for (const slot of ['top', 'outerwear', 'bottom', 'shoes']) {
        if (anchored && slot === 'outerwear') continue;
        assert.equal(
          new Set(selected.filter((i) => i.macroCategory === slot).map((i) => i.id)).size,
          3
        );
      }
    }
    if (process.env.CPU_BUDGET_MS)
      assert(
        cpu < Number(process.env.CPU_BUDGET_MS),
        `CPU ${cpu} exceeds ${process.env.CPU_BUDGET_MS}`
      );
  }
  if (process.env.CPU_REPORT)
    fs.writeFileSync(
      process.env.CPU_REPORT,
      JSON.stringify(
        {
          scope:
            'actual post-provider handler replay; no remote calls; process CPU excludes module compilation',
          reports,
        },
        null,
        2
      ) + '\n'
    );
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
