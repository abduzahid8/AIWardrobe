import test from 'node:test';
import assert from 'node:assert/strict';
import {
  evaluateFashionQuality,
  rerankFashionCandidates,
  refineQuietOfficeConcept,
  FASHION_QUALITY_VERSION,
} from './fashion_deck.mjs';
import {
  selectCreativeDirectionDeck,
  garmentArchitectureOf,
  deckOfficeMonotonyAdjustment,
  shoeRegisterOf,
} from './creative_directions.mjs';

function makeItem(id, slot, name, { color = 'navy', subtype = 'unknown', texture = 'smooth', price = 200000 } = {}) {
  return {
    id,
    name,
    macroCategory: slot,
    subtype,
    color,
    texture,
    price,
    currency: 'UZS',
    isShopItem: true,
  };
}

const classicWorkContext = {
  style: 'classic',
  occasion: 'Work',
  sourceMode: 'catalog',
};

const oldMoneyWorkContext = {
  style: 'old_money',
  occasion: 'Work',
  sourceMode: 'catalog',
};

test('paletteHarmony rewards classic quiet neutrals and penalizes clashing neons', () => {
  const neutralItems = [
    makeItem('t1', 'top', 'White Cotton Shirt', { color: 'white', subtype: 'dress-shirt' }),
    makeItem('b1', 'bottom', 'Navy Tailored Trousers', { color: 'navy', subtype: 'tailored-trouser' }),
    makeItem('s1', 'shoes', 'Brown Leather Loafers', { color: 'brown', subtype: 'loafer' }),
  ];
  const qNeutral = evaluateFashionQuality(neutralItems, classicWorkContext);
  assert.equal(qNeutral.version, FASHION_QUALITY_VERSION);
  assert.ok(qNeutral.paletteHarmony >= 0.85, `Expected >= 0.85, got ${qNeutral.paletteHarmony}`);

  const neonItems = [
    makeItem('t2', 'top', 'Neon Yellow Tee', { color: 'yellow', subtype: 't-shirt' }),
    makeItem('b2', 'bottom', 'Electric Blue Pants', { color: 'blue', subtype: 'trouser' }),
    makeItem('s2', 'shoes', 'Hot Pink Sneakers', { color: 'pink', subtype: 'sneaker' }),
  ];
  const qNeon = evaluateFashionQuality(neonItems, classicWorkContext);
  assert.ok(qNeutral.paletteHarmony > qNeon.paletteHarmony + 0.15,
    `Neutral palette (${qNeutral.paletteHarmony}) must outrank neon palette (${qNeon.paletteHarmony})`);
});

test('shoeAppropriateness prefers loafers and derbies over sporty sneakers for Classic / Work / Old Money', () => {
  const baseTopsAndBottoms = [
    makeItem('t1', 'top', 'Cotton Poplin Shirt', { color: 'white', subtype: 'dress-shirt' }),
    makeItem('b1', 'bottom', 'Pleated Wool Trousers', { color: 'charcoal', subtype: 'tailored-trouser' }),
  ];

  const loaferLook = evaluateFashionQuality([
    ...baseTopsAndBottoms,
    makeItem('s-loafer', 'shoes', 'Classic Leather Penny Loafers', { color: 'black', subtype: 'loafer' }),
  ], classicWorkContext);

  const derbyLook = evaluateFashionQuality([
    ...baseTopsAndBottoms,
    makeItem('s-derby', 'shoes', 'Smooth Leather Derby Shoes', { color: 'brown', subtype: 'derby' }),
  ], oldMoneyWorkContext);

  const minimalSneakerLook = evaluateFashionQuality([
    ...baseTopsAndBottoms,
    makeItem('s-min-snk', 'shoes', 'Plain Clean Minimal Sneakers', { color: 'white', subtype: 'minimal-sneaker' }),
  ], classicWorkContext);

  const sportySneakerLook = evaluateFashionQuality([
    ...baseTopsAndBottoms,
    makeItem('s-sport-snk', 'shoes', 'Athletic Running Sneakers with Sport Sole', { color: 'grey', subtype: 'sport-sneaker' }),
  ], classicWorkContext);

  // Loafers and derbies should have high appropriateness (0.95 - 1.0)
  assert.ok(loaferLook.shoeAppropriateness >= 0.95, `Loafer appropriateness: ${loaferLook.shoeAppropriateness}`);
  assert.ok(derbyLook.shoeAppropriateness >= 0.95, `Derby appropriateness: ${derbyLook.shoeAppropriateness}`);

  // Minimal sneakers acceptable but lower
  assert.ok(loaferLook.shoeAppropriateness > minimalSneakerLook.shoeAppropriateness);
  assert.ok(minimalSneakerLook.shoeAppropriateness >= 0.60);

  // Sporty sneakers strongly penalized (0.20) in Classic/Work
  assert.equal(sportySneakerLook.shoeAppropriateness, 0.20);
  assert.ok(loaferLook.shoeAppropriateness > sportySneakerLook.shoeAppropriateness + 0.70);
});

test('silhouetteBalance rewards structured tailoring and balanced volume', () => {
  const tailoredOutfit = [
    makeItem('t1', 'top', 'Fitted Oxford Shirt', { subtype: 'dress-shirt' }),
    makeItem('o1', 'outerwear', 'Wool Tailored Blazer', { subtype: 'blazer' }),
    makeItem('b1', 'bottom', 'Tailored Suit Trousers', { subtype: 'tailored-trouser' }),
    makeItem('s1', 'shoes', 'Leather Derby Shoes', { subtype: 'derby' }),
  ];
  const qTailored = evaluateFashionQuality(tailoredOutfit, classicWorkContext);
  assert.ok(qTailored.silhouetteBalance >= 0.85, `Silhouette balance: ${qTailored.silhouetteBalance}`);
});

test('textureCoordination rewards pairing rich textures (knitwear, tailored wool, suede/leather)', () => {
  const richTextures = [
    makeItem('t1', 'top', 'Merino Fine Knit Sweater', { color: 'cream', subtype: 'fine-knit', texture: 'knit' }),
    makeItem('o1', 'outerwear', 'Textured Knit Cardigan', { color: 'beige', subtype: 'cardigan', texture: 'knit' }),
    makeItem('b1', 'bottom', 'Flannel Wool Trousers', { color: 'brown', subtype: 'tailored-trouser', texture: 'flannel' }),
    makeItem('s1', 'shoes', 'Suede Loafers', { color: 'brown', subtype: 'loafer', texture: 'suede' }),
  ];
  const qRich = evaluateFashionQuality(richTextures, oldMoneyWorkContext);

  const flatTextures = [
    makeItem('t2', 'top', 'Plain Synthetic Shirt', { color: 'white', subtype: 'shirt', texture: 'smooth' }),
    makeItem('b2', 'bottom', 'Plain Synthetic Pants', { color: 'grey', subtype: 'trouser', texture: 'smooth' }),
    makeItem('s2', 'shoes', 'Synthetic Loafers', { color: 'black', subtype: 'loafer', texture: 'smooth' }),
  ];
  const qFlat = evaluateFashionQuality(flatTextures, oldMoneyWorkContext);

  assert.ok(qRich.textureCoordination > qFlat.textureCoordination,
    `Rich texture (${qRich.textureCoordination}) must outscore flat texture (${qFlat.textureCoordination})`);
});

test('boringOfficePenalty penalizes unlayered plain shirt-and-trouser formulas for Classic/Work', () => {
  const boringOfficeLook = [
    makeItem('t1', 'top', 'Plain Cotton Shirt', { color: 'white', subtype: 'dress-shirt', texture: 'smooth' }),
    makeItem('b1', 'bottom', 'Plain Chino Trousers', { color: 'navy', subtype: 'tailored-trouser', texture: 'smooth' }),
    makeItem('s1', 'shoes', 'Leather Loafers', { color: 'black', subtype: 'loafer' }),
  ];
  const qBoring = evaluateFashionQuality(boringOfficeLook, classicWorkContext);
  assert.equal(qBoring.boringOfficePenalty, 0.10, 'Unlayered plain shirt+trouser must incur boring office penalty');

  // Adding outerwear (blazer) eliminates boring office penalty
  const layeredLook = [
    ...boringOfficeLook,
    makeItem('o1', 'outerwear', 'Classic Navy Blazer', { color: 'navy', subtype: 'blazer' }),
  ];
  const qLayered = evaluateFashionQuality(layeredLook, classicWorkContext);
  assert.equal(qLayered.boringOfficePenalty, 0, 'Layered blazer look must not incur boring office penalty');

  // Wearing fine-knit knitwear top also eliminates boring office penalty
  const knitLook = [
    makeItem('t2', 'top', 'Fine Knit Merino Sweater', { color: 'cream', subtype: 'fine-knit', texture: 'knit' }),
    makeItem('b1', 'bottom', 'Plain Chino Trousers', { color: 'navy', subtype: 'tailored-trouser', texture: 'smooth' }),
    makeItem('s1', 'shoes', 'Leather Loafers', { color: 'brown', subtype: 'loafer' }),
  ];
  const qKnit = evaluateFashionQuality(knitLook, classicWorkContext);
  assert.equal(qKnit.boringOfficePenalty, 0, 'Knitwear top must not incur boring office penalty');
});

test('rerankFashionCandidates reorders candidates by fashion quality and applies bounded adjustments', () => {
  const boringOfficeLook = {
    id: 'boring-look',
    items: [
      makeItem('t1', 'top', 'Plain Shirt', { color: 'white', subtype: 'dress-shirt' }),
      makeItem('b1', 'bottom', 'Plain Pants', { color: 'grey', subtype: 'trouser' }),
      makeItem('s1', 'shoes', 'Sporty Running Shoes', { color: 'grey', subtype: 'sport-sneaker' }),
    ],
    fashion: { score: 0.88, dimensions: { palette: 0.75, footwear: 0.5, proportion: 0.8 } },
  };

  const polishedLayeredLook = {
    id: 'polished-look',
    items: [
      makeItem('t2', 'top', 'Cotton Oxford Shirt', { color: 'white', subtype: 'dress-shirt' }),
      makeItem('o2', 'outerwear', 'Wool Blazer', { color: 'navy', subtype: 'blazer' }),
      makeItem('b2', 'bottom', 'Tailored Trousers', { color: 'charcoal', subtype: 'tailored-trouser' }),
      makeItem('s2', 'shoes', 'Leather Penny Loafers', { color: 'brown', subtype: 'loafer' }),
    ],
    fashion: { score: 0.86, dimensions: { palette: 0.85, footwear: 0.9, proportion: 0.88 } },
  };

  const pool = [boringOfficeLook, polishedLayeredLook];
  rerankFashionCandidates(pool, classicWorkContext);

  // Polished layered look with loafers should be promoted above the boring look with sporty sneakers
  assert.equal(pool[0].id, 'polished-look');
  assert.equal(pool[1].id, 'boring-look');

  // Verify adjustment bounds [-0.08, +0.06]
  for (const look of pool) {
    const adj = look.fashionQuality.adjustment;
    assert.ok(adj >= -0.08 && adj <= 0.06, `Adjustment ${adj} outside [-0.08, 0.06]`);
  }
});

test('3-look deck selection under Classic / Work enforces architecture diversity without lowering quality', () => {
  const slot = (subtype, color, volume = 'regular', texture = 'smooth', pattern = 'solid') =>
    [subtype, volume, color, texture, pattern];

  function makeLook(id, quality, features, keys = [id + '-top', id + '-bottom', id + '-shoes'], briefId = 'brief-' + id) {
    return {
      id, briefId, features,
      items: keys.map(k => ({ id: k })), fashion: { score: quality },
    };
  }

  const metrics = {
    featuresOf: look => look.features,
    featureDistance: () => 0.35,
    overlap: (a, b) => a.items.filter(x => b.items.some(y => y.id === x.id)).length,
  };

  const SHIRT = [slot('dress-shirt', 'blue', 'slim'), null, slot('tailored-trouser', 'navy', 'slim'), slot('loafer', 'brown')];
  const FRESH_SHIRT = [slot('shirt', 'cream', 'slim'), null, slot('tailored-trouser', 'grey', 'slim'), slot('loafer', 'brown')];
  const THIRD_SHIRT = [slot('shirt', 'white', 'slim'), null, slot('chino', 'beige', 'slim'), slot('derby', 'brown')];
  const KNIT_LOOK = [slot('fine-knit', 'cream', 'regular', 'knit'), null, slot('chino', 'olive', 'wide', 'textured'), slot('chelsea', 'burgundy')];
  const BLAZER_LOOK = [slot('dress-shirt', 'white', 'slim'), slot('blazer', 'navy', 'regular'), slot('tailored-trouser', 'grey', 'slim'), slot('loafer', 'black')];

  // Pool: 3 unlayered shirt+trouser candidates (0.95, 0.94, 0.93) plus knit (0.92) and blazer (0.91)
  const pool = [
    makeLook('shirt-1', 0.95, SHIRT),
    makeLook('shirt-2', 0.94, FRESH_SHIRT),
    makeLook('shirt-3', 0.93, THIRD_SHIRT),
    makeLook('knit', 0.92, KNIT_LOOK),
    makeLook('blazer', 0.91, BLAZER_LOOK),
  ];

  const selection = selectCreativeDirectionDeck(pool, metrics);
  assert.equal(selection.looks.length, 3);

  // Must not allow all 3 looks to share the office-shirt-trouser formula
  const archs = selection.looks.map(l => garmentArchitectureOf(l, metrics.featuresOf));
  const officeFormulaCount = archs.filter(a => a.isOfficeShirtTrouser).length;
  assert.ok(officeFormulaCount < 3, `Expected < 3 office-shirt-trouser looks, but got ${officeFormulaCount}`);
  assert.ok(selection.looks.some(l => l.id === 'knit' || l.id === 'blazer'),
    'Architecture variety must be present in the 3 selected looks');
});

test('quality layer never changes wardrobe, sport, casual, or anchored user selection', () => {
  const fixed = [makeItem('top','top','White Oxford Shirt',{color:'white',subtype:'dress-shirt'}),
    makeItem('bottom','bottom','Navy Tailored Trousers',{color:'navy',subtype:'tailored-trouser'}),
    makeItem('shoe','shoes','Leather Loafer',{color:'brown',subtype:'loafer'})];
  for(const context of [
    {...classicWorkContext,sourceMode:'wardrobe'},
    {...classicWorkContext,sourceMode:'mixed'},
    {...classicWorkContext,style:'sport'},
    {...classicWorkContext,style:'casual'},
    {...classicWorkContext,requiredIds:['top']},
    {...classicWorkContext,occasion:'Weekend'},
  ]) {
    const quality=evaluateFashionQuality(fixed,context);
    assert.equal(quality.adjustment,0);
    assert.equal(quality.active,false);
    const candidate={id:'locked',items:fixed,fashion:{score:.89}};
    const result=rerankFashionCandidates([candidate],context);
    assert.equal(result[0].fashion.score,.89);
    assert.equal(result[0].fashionQuality,undefined);
  }
});

test('quiet-office quality does not penalize requested sneakers or a hot-weather shirt outfit',()=>{
  const office=[makeItem('top','top','Plain White Shirt',{color:'white',subtype:'dress-shirt'}),
    makeItem('bottom','bottom','Navy Tailored Trousers',{color:'navy',subtype:'tailored-trouser'}),
    makeItem('shoe','shoes','Sport Running Sneakers',{color:'grey',subtype:'sport-sneaker'})];
  const natural=evaluateFashionQuality(office,classicWorkContext);
  const requested=evaluateFashionQuality(office,{...classicWorkContext,request:'Please wear sneakers'});
  assert.ok(requested.shoeAppropriateness>natural.shoeAppropriateness);
  const hot=evaluateFashionQuality(office,{...classicWorkContext,weather:{temp:32}});
  const without=evaluateFashionQuality(office,{...classicWorkContext,request:'No blazer or jacket'});
  assert.equal(hot.boringOfficePenalty,0);
  assert.equal(without.boringOfficePenalty,0);
});

test('fashionQualityApplies normalizes style format (casing, hyphens, spaces) and honors request intent', () => {
  const office = [
    makeItem('t1', 'top', 'Cotton Oxford Shirt', { color: 'white', subtype: 'dress-shirt' }),
    makeItem('b1', 'bottom', 'Tailored Wool Trousers', { color: 'navy', subtype: 'tailored-trouser' }),
    makeItem('s1', 'shoes', 'Leather Penny Loafers', { color: 'brown', subtype: 'loafer' }),
  ];

  for (const style of ['Classic', 'Old Money', 'old-money', 'Semi_Classic', 'Business Casual']) {
    const q = evaluateFashionQuality(office, { style, occasion: 'Work', sourceMode: 'catalog' });
    assert.equal(q.version, FASHION_QUALITY_VERSION);
    assert.notEqual(q.score, null, `Quality score must be evaluated for style variant: ${style}`);
    assert.ok(q.score > 0.70, `Expected score > 0.70 for ${style}, got ${q.score}`);
  }

  // Work intent in request with empty occasion
  const qReq = evaluateFashionQuality(office, { style: 'classic', occasion: '', request: 'office work meeting', sourceMode: 'catalog' });
  assert.notEqual(qReq.score, null, 'Quality score must be evaluated when work intent is in request');
  assert.ok(qReq.score > 0.70);

  // Style formats with spaces/hyphens must evaluate palette harmony identically when work intent is in request
  const qOldMoneyUnderscore = evaluateFashionQuality(office, { style: 'old_money', occasion: '', request: 'office meeting', sourceMode: 'catalog' });
  const qOldMoneySpace = evaluateFashionQuality(office, { style: 'Old Money', occasion: '', request: 'office meeting', sourceMode: 'catalog' });
  const qOldMoneyHyphen = evaluateFashionQuality(office, { style: 'old-money', occasion: '', request: 'office meeting', sourceMode: 'catalog' });
  assert.equal(qOldMoneySpace.paletteHarmony, qOldMoneyUnderscore.paletteHarmony);
  assert.equal(qOldMoneyHyphen.paletteHarmony, qOldMoneyUnderscore.paletteHarmony);
  assert.equal(qOldMoneySpace.score, qOldMoneyUnderscore.score);

  // Clashing neon penalty must apply under Old Money with work intent in request
  const neonOffice = [
    makeItem('t1', 'top', 'Neon Green Oxford Shirt', { color: 'green', subtype: 'dress-shirt' }),
    makeItem('b1', 'bottom', 'Tailored Wool Trousers', { color: 'navy', subtype: 'tailored-trouser' }),
    makeItem('s1', 'shoes', 'Leather Penny Loafers', { color: 'brown', subtype: 'loafer' }),
  ];
  const qNeonSpace = evaluateFashionQuality(neonOffice, { style: 'Old Money', occasion: '', request: 'office meeting', sourceMode: 'catalog' });
  const qNeonUnder = evaluateFashionQuality(neonOffice, { style: 'old_money', occasion: '', request: 'office meeting', sourceMode: 'catalog' });
  assert.equal(qNeonSpace.paletteHarmony, 0.50);
  assert.equal(qNeonSpace.paletteHarmony, qNeonUnder.paletteHarmony);

  // refineQuietOfficeConcept honors work intent in request
  const ivyConcept = { id: 'refined-ivy', slots: [['dress-shirt', 'fine-knit'], ['blazer'], ['chino'], ['loafer']] };
  const refined = refineQuietOfficeConcept(ivyConcept, { style: 'Old Money', occasion: '', request: 'office work meeting', sourceMode: 'catalog' });
  assert.notEqual(refined, ivyConcept);
  assert.deepEqual(refined.slots[1], ['blazer', 'cardigan']);
});

test('evaluateFashionQuality and rerankFashionCandidates safely handle null/boundary inputs', () => {
  const ctx = classicWorkContext;
  const qNull = evaluateFashionQuality(null, ctx);
  assert.equal(qNull.version, FASHION_QUALITY_VERSION);
  assert.equal(typeof qNull.score, 'number');

  const qUndefined = evaluateFashionQuality(undefined, ctx);
  assert.equal(qUndefined.version, FASHION_QUALITY_VERSION);

  const qSparse = evaluateFashionQuality({ items: [null, undefined] }, ctx);
  assert.equal(qSparse.version, FASHION_QUALITY_VERSION);

  assert.deepEqual(rerankFashionCandidates(null, ctx), []);
  assert.deepEqual(rerankFashionCandidates([], ctx), []);
  assert.deepEqual(rerankFashionCandidates([null, undefined], ctx), [null, undefined]);
});

test('rerankFashionCandidates updates and reranks candidates with seedScore (without fashion.score)', () => {
  const c1 = {
    id: 'c1-boring-sneaker',
    items: [
      makeItem('t1', 'top', 'Cotton Poplin Shirt', { subtype: 'dress-shirt', texture: 'smooth' }),
      makeItem('b1', 'bottom', 'Tailored Wool Trousers', { subtype: 'tailored-trouser', texture: 'smooth' }),
      makeItem('s1', 'shoes', 'Sport Running Sneakers', { subtype: 'sport-sneaker' }),
    ],
    seedScore: 0.88,
  };
  const c2 = {
    id: 'c2-polished-loafer',
    items: [
      makeItem('t2', 'top', 'Cotton Poplin Shirt', { subtype: 'dress-shirt', texture: 'smooth' }),
      makeItem('o2', 'outerwear', 'Wool Tailored Blazer', { subtype: 'blazer', texture: 'wool' }),
      makeItem('b2', 'bottom', 'Tailored Wool Trousers', { subtype: 'tailored-trouser', texture: 'wool' }),
      makeItem('s2', 'shoes', 'Leather Penny Loafers', { subtype: 'loafer', texture: 'leather' }),
    ],
    seedScore: 0.86,
  };

  const pool = [c1, c2];
  rerankFashionCandidates(pool, classicWorkContext);

  assert.equal(pool[0].id, 'c2-polished-loafer', 'c2 must be promoted above c1 due to fashion quality bonus vs penalty');
  assert.equal(pool[1].id, 'c1-boring-sneaker');
  assert.ok(pool[0].seedScore > pool[1].seedScore, `Expected c2 seedScore (${pool[0].seedScore}) > c1 seedScore (${pool[1].seedScore})`);
});

test('evaluateFashionQuality falls back to item subtype and returns active: true', () => {
  const outfit = [
    { macroCategory: 'top', subtype: 'dress_shirt', name: 'SKU-001' },
    { macroCategory: 'outerwear', subtype: 'blazer', name: 'SKU-002' },
    { macroCategory: 'bottom', subtype: 'tailored_trouser', name: 'SKU-003' },
    { macroCategory: 'shoes', subtype: 'loafer', name: 'SKU-004' },
  ];
  const q = evaluateFashionQuality(outfit, classicWorkContext);
  assert.equal(q.active, true, 'Applicable fashion quality must return active: true');
  assert.equal(q.boringOfficePenalty, 0, 'Blazer identified via item subtype must avoid boring office penalty');
  assert.ok(q.shoeAppropriateness >= 0.95, `Loafer identified via item subtype must score >= 0.95, got ${q.shoeAppropriateness}`);

  // Test garmentArchitectureOf with underscore subtypes
  const arch = garmentArchitectureOf({ items: [
    { macroCategory: 'top', subtype: 'dress_shirt' },
    { macroCategory: 'bottom', subtype: 'tailored_trouser' },
    { macroCategory: 'shoes', subtype: 'loafer' },
  ] });
  assert.equal(arch.isOfficeShirtTrouser, true, 'Underscore subtypes must be recognized as office shirt+trouser formula');
});

test('evaluateFashionQuality falls back to item texture and pattern for uninformative SKU titles', () => {
  // Boring office look where trousers have flannel texture on item property (not in name)
  const flannelTrouserLook = [
    { macroCategory: 'top', subtype: 'dress_shirt', name: 'SKU-001', texture: 'smooth' },
    { macroCategory: 'bottom', subtype: 'tailored_trouser', name: 'SKU-002', texture: 'flannel' },
    { macroCategory: 'shoes', subtype: 'loafer', name: 'SKU-003' },
  ];
  const qFlannel = evaluateFashionQuality(flannelTrouserLook, classicWorkContext);
  assert.equal(qFlannel.boringOfficePenalty, 0.05,
    `Flannel texture from item.texture must reduce boring office penalty to 0.05, got ${qFlannel.boringOfficePenalty}`);

  // Look where texture coordination bonuses rely on item.texture
  const textureLook = [
    { macroCategory: 'top', type: 'fine_knit', name: 'SKU-TOP', texture: 'knit' },
    { macroCategory: 'bottom', type: 'tailored_trouser', name: 'SKU-BOT', texture: 'flannel' },
    { macroCategory: 'shoes', type: 'derby', name: 'SKU-SHOE', texture: 'leather' },
  ];
  const qTex = evaluateFashionQuality(textureLook, classicWorkContext);
  assert.ok(qTex.textureCoordination >= 0.90,
    `Expected textureCoordination >= 0.90 from item.texture and type fallbacks, got ${qTex.textureCoordination}`);
  assert.ok(qTex.shoeAppropriateness >= 0.95,
    `Expected shoeAppropriateness >= 0.95 for type: 'derby', got ${qTex.shoeAppropriateness}`);
});

test('evaluateFashionQuality and garmentArchitectureOf handle capitalized and plural subtypes robustly', () => {
  const lookWithCaps = [
    { macroCategory: 'top', subtype: 'Dress_Shirt', name: 'SKU-CAP-1' },
    { macroCategory: 'bottom', subtype: 'Tailored_Trousers', name: 'SKU-CAP-2' },
    { macroCategory: 'shoes', subtype: 'Loafers', name: 'SKU-CAP-3' },
  ];
  const qCaps = evaluateFashionQuality(lookWithCaps, classicWorkContext);
  assert.equal(qCaps.boringOfficePenalty, 0.10,
    `Capitalized subtypes must incur boring office penalty, got ${qCaps.boringOfficePenalty}`);
  assert.ok(qCaps.shoeAppropriateness >= 0.95,
    `Plural Loafers must be recognized with shoeAppropriateness >= 0.95, got ${qCaps.shoeAppropriateness}`);

  const archCaps = garmentArchitectureOf({ items: lookWithCaps });
  assert.equal(archCaps.isOfficeShirtTrouser, true,
    'Capitalized Dress_Shirt and Tailored_Trousers must be recognized as office shirt+trouser formula');

  // Chinos plural with Derbies plural
  const lookChinos = [
    { macroCategory: 'top', subtype: 'Shirt', name: 'SKU-CH-1' },
    { macroCategory: 'bottom', subtype: 'Chinos', name: 'SKU-CH-2' },
    { macroCategory: 'shoes', subtype: 'Derbies', name: 'SKU-CH-3' },
  ];
  const qChinos = evaluateFashionQuality(lookChinos, classicWorkContext);
  assert.equal(qChinos.boringOfficePenalty, 0.10);
  assert.ok(qChinos.shoeAppropriateness >= 0.95);
  const archChinos = garmentArchitectureOf({ items: lookChinos });
  assert.equal(archChinos.isOfficeShirtTrouser, true);
});

test('paletteHarmony recognizes canonical quiet tokens light-blue, dark-neutral, and indigo', () => {
  const quietOutfit = [
    makeItem('t1', 'top', 'Cotton Poplin Shirt', { color: 'light blue', subtype: 'dress-shirt' }),
    makeItem('b1', 'bottom', 'Tailored Pants', { color: 'dark grey', subtype: 'tailored-trouser' }),
    makeItem('s1', 'shoes', 'Leather Loafers', { color: 'black', subtype: 'loafer' }),
  ];
  const q = evaluateFashionQuality(quietOutfit, classicWorkContext);
  assert.ok(q.paletteHarmony >= 0.85, `Expected paletteHarmony >= 0.85, got ${q.paletteHarmony}`);
});

test('rerankFashionCandidates safely sorts when candidates contain NaN or non-finite scores', () => {
  const c1 = { id: 'c1-valid', fashion: { score: 0.90 } };
  const c2 = { id: 'c2-nan', fashion: { score: NaN } };
  const c3 = { id: 'c3-seed', seedScore: 0.85 };
  const c4 = { id: 'c4-null-score', fashion: { score: null } };

  const pool = [c2, c4, c1, c3];
  rerankFashionCandidates(pool, classicWorkContext);

  assert.equal(pool[0].id, 'c1-valid');
  assert.equal(pool[1].id, 'c3-seed');
  // c2 and c4 must follow predictably without breaking the sort
  assert.ok(pool.slice(2).every(c => c.id === 'c2-nan' || c.id === 'c4-null-score'));
});

test('shoeAppropriateness and textureCoordination recognize monk-straps, brogues, and dress-shoes robustly', () => {
  for (const shoeSub of ['monk-strap', 'monk_strap', 'monk', 'brogue', 'brogues', 'dress-shoe', 'dress-boot']) {
    const outfit = [
      makeItem('t', 'top', 'Cotton Poplin Shirt', { subtype: 'dress-shirt' }),
      makeItem('o', 'outerwear', 'Wool Blazer', { subtype: 'blazer' }),
      makeItem('b', 'bottom', 'Tailored Trousers', { subtype: 'tailored-trouser' }),
      makeItem('s', 'shoes', 'Classic Footwear', { subtype: shoeSub }),
    ];
    const q = evaluateFashionQuality(outfit, classicWorkContext);
    assert.ok(q.shoeAppropriateness >= 0.95,
      `Expected shoeAppropriateness >= 0.95 for ${shoeSub}, got ${q.shoeAppropriateness}`);
    assert.ok(q.textureCoordination >= 0.90,
      `Expected textureCoordination >= 0.90 (tailored + leather/dress footwear) for ${shoeSub}, got ${q.textureCoordination}`);
  }
});

test('shoeAppropriateness recognizes Russian/multilingual dress footwear catalog titles', () => {
  const cases = [
    { name: 'Мужские дерби Canali кожаные', expectedAppropriateness: 1.0 },
    { name: 'Кожаные оксфорды Geox', expectedAppropriateness: 1.0 },
    { name: 'Классические туфли из натуральной кожи', expectedAppropriateness: 1.0 },
    { name: 'Мужские броги замшевые', expectedAppropriateness: 1.0 },
    { name: 'Монки мужские черные', expectedAppropriateness: 0.95 },
  ];

  for (const { name, expectedAppropriateness } of cases) {
    const outfit = [
      makeItem('t', 'top', 'Мужская рубашка под пиджак'),
      makeItem('b', 'bottom', 'Классические брюки шерсть'),
      makeItem('s', 'shoes', name),
    ];
    const q = evaluateFashionQuality(outfit, classicWorkContext);
    assert.ok(q.shoeAppropriateness >= expectedAppropriateness,
      `Expected shoeAppropriateness >= ${expectedAppropriateness} for "${name}", got ${q.shoeAppropriateness}`);
  }
});

test('garmentArchitectureOf infers office formula from item name when subtype property is omitted', () => {
  const lookByNameOnly = {
    items: [
      { macroCategory: 'top', name: 'Classic Cotton Dress Shirt' },
      { macroCategory: 'bottom', name: 'Wool Tailored Trousers' },
      { macroCategory: 'shoes', name: 'Leather Penny Loafers' },
    ],
  };
  const arch = garmentArchitectureOf(lookByNameOnly);
  assert.equal(arch.isOfficeShirtTrouser, true,
    'Must infer office-shirt-trouser formula from item names even without explicit subtype property');
});

test('fashionQualityApplies and refineQuietOfficeConcept honor context.brief', () => {
  const outfit = [
    makeItem('t1', 'top', 'Cotton Oxford Shirt', { color: 'white', subtype: 'dress-shirt' }),
    makeItem('b1', 'bottom', 'Tailored Wool Trousers', { color: 'navy', subtype: 'tailored-trouser' }),
    makeItem('s1', 'shoes', 'Leather Penny Loafers', { color: 'brown', subtype: 'loafer' }),
  ];

  // Brief contains office intent, occasion and request are empty
  const q = evaluateFashionQuality(outfit, {
    style: 'classic',
    occasion: '',
    request: '',
    brief: 'Office work presentation',
    sourceMode: 'catalog',
  });
  assert.equal(q.active, true, 'Fashion quality must activate when office intent is in brief');
  assert.ok(q.score > 0.70);

  const ivyConcept = { id: 'refined-ivy', slots: [['dress-shirt', 'fine-knit'], ['blazer'], ['chino'], ['loafer']] };
  const refined = refineQuietOfficeConcept(ivyConcept, {
    style: 'old_money',
    occasion: '',
    request: '',
    brief: 'Office work presentation',
    sourceMode: 'catalog',
  });
  assert.notEqual(refined, ivyConcept);
  assert.deepEqual(refined.slots[1], ['blazer', 'cardigan']);
});

test('rerankFashionCandidates is idempotent and never double-applies score adjustment', () => {
  const look = {
    id: 'test-look',
    items: [
      makeItem('t', 'top', 'Cotton Poplin Shirt', { subtype: 'dress-shirt' }),
      makeItem('o', 'outerwear', 'Wool Tailored Blazer', { subtype: 'blazer' }),
      makeItem('b', 'bottom', 'Tailored Wool Trousers', { subtype: 'tailored-trouser' }),
      makeItem('s', 'shoes', 'Leather Penny Loafers', { subtype: 'loafer' }),
    ],
    fashion: { score: 0.85 },
  };

  const pool = [look];
  rerankFashionCandidates(pool, classicWorkContext);
  const firstScore = pool[0].fashion.score;
  assert.ok(firstScore > 0.85, 'First call must apply positive fashion adjustment');

  // Second call must NOT re-apply adjustment
  rerankFashionCandidates(pool, classicWorkContext);
  assert.equal(pool[0].fashion.score, firstScore, 'Second call must be idempotent without modifying score');
});

test('garmentArchitectureOf correctly resolves architecture when slots are anchored or feature descriptors are unknown', () => {
  const lookWithBlazer = {
    id: 'look-blazer',
    items: [
      makeItem('shirt-1', 'top', 'Cotton Dress Shirt', { subtype: 'dress-shirt' }),
      makeItem('blazer-1', 'outerwear', 'Tailored Wool Blazer', { subtype: 'blazer' }),
      makeItem('pants-1', 'bottom', 'Tailored Wool Trousers', { subtype: 'tailored-trouser' }),
      makeItem('shoes-1', 'shoes', 'Leather Penny Loafers', { subtype: 'loafer' }),
    ],
  };

  // When blazer is anchored, featuresOf returns null for outerwear slot
  const featuresWithAnchoredBlazer = () => [
    ['dress-shirt', 'slim', 'white', 'smooth', 'solid'],
    null,
    ['trouser', 'slim', 'navy', 'smooth', 'solid'],
    ['loafer', 'regular', 'brown', 'smooth', 'solid'],
  ];

  const archAnchoredBlazer = garmentArchitectureOf(lookWithBlazer, featuresWithAnchoredBlazer);
  assert.equal(archAnchoredBlazer.isOfficeShirtTrouser, false,
    'Anchored blazer must be recognized as outerwear, not unlayered office-shirt-trouser');
  assert.equal(archAnchoredBlazer.outerwear, 'blazer');

  // When dress shirt is anchored, featuresOf returns null for top slot
  const lookShirtTrouser = {
    id: 'look-shirt',
    items: [
      makeItem('shirt-1', 'top', 'Cotton Dress Shirt', { subtype: 'dress-shirt' }),
      makeItem('pants-1', 'bottom', 'Tailored Wool Trousers', { subtype: 'tailored-trouser' }),
      makeItem('shoes-1', 'shoes', 'Leather Penny Loafers', { subtype: 'loafer' }),
    ],
  };
  const featuresWithAnchoredShirt = () => [
    null,
    null,
    ['trouser', 'slim', 'navy', 'smooth', 'solid'],
    ['loafer', 'regular', 'brown', 'smooth', 'solid'],
  ];

  const archAnchoredShirt = garmentArchitectureOf(lookShirtTrouser, featuresWithAnchoredShirt);
  assert.equal(archAnchoredShirt.isOfficeShirtTrouser, true,
    'Anchored dress shirt must be recognized as office shirt-trouser formula');
  assert.equal(archAnchoredShirt.top, 'dress-shirt');

  // When featuresOf returns 'unknown' for top, fall back to item.subtype
  const featuresUnknownTop = () => [
    ['unknown', 'unknown', 'unknown', 'unknown', 'unknown'],
    null,
    ['trouser', 'slim', 'navy', 'smooth', 'solid'],
    ['loafer', 'regular', 'brown', 'smooth', 'solid'],
  ];
  const archUnknownTop = garmentArchitectureOf(lookShirtTrouser, featuresUnknownTop);
  assert.equal(archUnknownTop.isOfficeShirtTrouser, true,
    'Must fall back to item.subtype when featuresOf top is unknown');
  assert.equal(archUnknownTop.top, 'dress-shirt');
});

test('computeBoringOfficePenalty returns 0 whenever an outerwear garment is present', () => {
  const look = {
    items: [
      makeItem('t', 'top', 'Cotton Dress Shirt', { subtype: 'dress-shirt' }),
      makeItem('o', 'outerwear', 'Canali Exclusive Outerwear'), // unclassified subtype
      makeItem('b', 'bottom', 'Tailored Wool Trousers', { subtype: 'tailored-trouser' }),
      makeItem('s', 'shoes', 'Leather Penny Loafers', { subtype: 'loafer' }),
    ],
  };
  const q = evaluateFashionQuality(look, classicWorkContext);
  assert.equal(q.boringOfficePenalty, 0,
    'Look with outerwear item must never incur boring office penalty');
});

test('shoeAppropriateness and textureCoordination recognize chelsea-boot and dress-boot hyphen/underscore variants', () => {
  for (const sub of ['chelsea-boot', 'chelsea_boot', 'dress-boot', 'dress_boot']) {
    const outfit = [
      makeItem('t', 'top', 'Cotton Dress Shirt', { subtype: 'dress-shirt' }),
      makeItem('o', 'outerwear', 'Wool Tailored Blazer', { subtype: 'blazer' }),
      makeItem('b', 'bottom', 'Tailored Wool Trousers', { subtype: 'tailored-trouser' }),
      makeItem('s', 'shoes', 'Fine Leather Footwear', { subtype: sub }),
    ];
    const q = evaluateFashionQuality(outfit, classicWorkContext);
    assert.ok(q.shoeAppropriateness >= 0.95,
      `Expected shoeAppropriateness >= 0.95 for ${sub}, got ${q.shoeAppropriateness}`);
    assert.ok(q.textureCoordination >= 0.90,
      `Expected textureCoordination >= 0.90 for ${sub}, got ${q.textureCoordination}`);
  }
});

test('shoeAppropriateness and textureCoordination recognize compound dress shoe subtypes (penny-loafer, double-monk, cap-toe-oxford)', () => {
  const compoundSubtypes = [
    'penny-loafer', 'penny_loafer', 'tassel-loafer', 'tassel_loafer', 'horsebit-loafer',
    'double-monk', 'double_monk', 'double-monk-strap', 'double_monk_strap', 'single-monk',
    'cap-toe-oxford', 'cap_toe_oxford', 'wholecut-oxford',
    'dress-derby', 'dress_derby',
  ];
  for (const sub of compoundSubtypes) {
    const outfit = [
      makeItem('t', 'top', 'Cotton Dress Shirt', { subtype: 'dress-shirt' }),
      makeItem('o', 'outerwear', 'Wool Tailored Blazer', { subtype: 'blazer' }),
      makeItem('b', 'bottom', 'Tailored Wool Trousers', { subtype: 'tailored-trouser' }),
      makeItem('s', 'shoes', 'Fine Footwear', { subtype: sub }),
    ];
    const q = evaluateFashionQuality(outfit, classicWorkContext);
    assert.ok(q.shoeAppropriateness >= 0.95,
      `Expected shoeAppropriateness >= 0.95 for ${sub}, got ${q.shoeAppropriateness}`);
    assert.ok(q.textureCoordination >= 0.90,
      `Expected textureCoordination >= 0.90 for ${sub}, got ${q.textureCoordination}`);
  }
});

test('unlayered shirt + pants or slacks is recognized as office-shirt-trouser formula and incurs boringOfficePenalty', () => {
  for (const botSub of ['pants', 'pant', 'slacks', 'slack', 'tailored-pants', 'suit-pants']) {
    for (const topSub of ['shirt', 'dress-shirt', 'button-down', 'oxford-shirt']) {
      const outfit = [
        makeItem('t', 'top', 'Office Top', { subtype: topSub, texture: 'smooth' }),
        makeItem('b', 'bottom', 'Office Bottom', { subtype: botSub, texture: 'smooth' }),
        makeItem('s', 'shoes', 'Leather Loafers', { subtype: 'loafer' }),
      ];
      const q = evaluateFashionQuality(outfit, classicWorkContext);
      assert.equal(q.boringOfficePenalty, 0.10,
        `Expected boringOfficePenalty = 0.10 for unlayered ${topSub} + ${botSub}, got ${q.boringOfficePenalty}`);

      const arch = garmentArchitectureOf({ items: outfit });
      assert.equal(arch.isOfficeShirtTrouser, true,
        `Expected isOfficeShirtTrouser = true for unlayered ${topSub} + ${botSub}`);
    }
  }
});

test('garmentArchitectureOf gracefully handles throwing featuresOf by falling back to look.items', () => {
  const look = {
    id: 'look-safe',
    items: [
      makeItem('shirt-1', 'top', 'Cotton Dress Shirt', { subtype: 'dress-shirt' }),
      makeItem('pants-1', 'bottom', 'Wool Flannel Pants', { subtype: 'pants' }),
      makeItem('shoes-1', 'shoes', 'Leather Penny Loafers', { subtype: 'loafer' }),
    ],
  };
  const throwingFeaturesOf = () => {
    throw new Error('Simulated features extraction failure');
  };

  const arch = garmentArchitectureOf(look, throwingFeaturesOf);
  assert.equal(arch.isOfficeShirtTrouser, true);
  assert.equal(arch.top, 'dress-shirt');
  assert.equal(arch.bottom, 'pants');
  assert.equal(arch.outerwear, 'none');
});

test('garmentArchitectureOf and computeBoringOfficePenalty treat outerwear with subtype none as unlayered', () => {
  const look = {
    id: 'look-none-outer',
    items: [
      makeItem('shirt-1', 'top', 'Cotton Poplin Shirt', { subtype: 'shirt', texture: 'smooth' }),
      makeItem('outer-none', 'outerwear', 'No Layer', { subtype: 'none' }),
      makeItem('pants-1', 'bottom', 'Tailored Wool Trousers', { subtype: 'tailored-trouser', texture: 'smooth' }),
      makeItem('shoes-1', 'shoes', 'Leather Penny Loafers', { subtype: 'loafer' }),
    ],
  };

  const archWithoutFeatures = garmentArchitectureOf(look);
  assert.equal(archWithoutFeatures.isOfficeShirtTrouser, true,
    'Outerwear with subtype none must be treated as unlayered by garmentArchitectureOf');
  assert.equal(archWithoutFeatures.outerwear, 'none');

  const archWithFeatures = garmentArchitectureOf(look, () => [
    ['shirt', 'slim', 'white', 'smooth', 'solid'],
    null,
    ['tailored-trouser', 'slim', 'navy', 'smooth', 'solid'],
    ['loafer', 'regular', 'brown', 'smooth', 'solid'],
  ]);
  assert.equal(archWithFeatures.isOfficeShirtTrouser, true);
  assert.equal(archWithFeatures.outerwear, 'none');

  const q = evaluateFashionQuality(look.items, classicWorkContext);
  assert.equal(q.boringOfficePenalty, 0.10,
    `Outerwear with subtype none must incur boring office penalty for shirt+trouser, got ${q.boringOfficePenalty}`);
});

test('looks scored with active fashion quality have qualityAdjustmentApplied set and avoid double adjustment in rerankFashionCandidates', () => {
  const quality = evaluateFashionQuality([
    makeItem('shirt-1', 'top', 'Cotton Poplin Shirt', { subtype: 'dress-shirt' }),
    makeItem('blazer-1', 'outerwear', 'Wool Blazer', { subtype: 'blazer' }),
    makeItem('pants-1', 'bottom', 'Tailored Wool Trousers', { subtype: 'tailored-trouser' }),
    makeItem('shoes-1', 'shoes', 'Leather Penny Loafers', { subtype: 'loafer' }),
  ], classicWorkContext);

  assert.equal(quality.active, true);
  const adj = quality.adjustment;
  assert.ok(adj > 0, `Expected positive adjustment for well-styled look, got ${adj}`);

  const baseScore = 0.85;
  // Look where makeFashionScorer already applied fashionQuality.adjustment and set qualityAdjustmentApplied
  const look = {
    id: 'look-preapplied',
    items: [
      makeItem('shirt-1', 'top', 'Cotton Poplin Shirt', { subtype: 'dress-shirt' }),
      makeItem('blazer-1', 'outerwear', 'Wool Blazer', { subtype: 'blazer' }),
      makeItem('pants-1', 'bottom', 'Tailored Wool Trousers', { subtype: 'tailored-trouser' }),
      makeItem('shoes-1', 'shoes', 'Leather Penny Loafers', { subtype: 'loafer' }),
    ],
    fashion: {
      score: baseScore + adj,
      fashionQuality: quality,
      qualityAdjustmentApplied: true,
    },
  };

  const pool = [look];
  rerankFashionCandidates(pool, classicWorkContext);

  // Score must remain baseScore + adj, never double-adjusted to baseScore + 2*adj
  assert.equal(pool[0].fashion.score, baseScore + adj,
    `Score should not be double-adjusted: expected ${baseScore + adj}, got ${pool[0].fashion.score}`);
  assert.equal(pool[0].fashion.qualityAdjustmentApplied, true);
});

test('shoeRegisterOf and evaluateFashionQuality recognize derby variations (derby, derbies, derbys, dress-derby)', () => {
  for (const variant of ['derby', 'derbies', 'derbys', 'dress-derby', 'dress-derbies', 'dress_derby']) {
    assert.equal(shoeRegisterOf(variant), 'dress',
      `Expected shoeRegisterOf to return 'dress' for variant: ${variant}`);

    const outfit = [
      makeItem('shirt-1', 'top', 'Cotton Poplin Shirt', { subtype: 'dress-shirt' }),
      makeItem('pants-1', 'bottom', 'Tailored Wool Trousers', { subtype: 'tailored-trouser' }),
      makeItem('shoes-1', 'shoes', 'Fine Footwear', { subtype: variant }),
    ];
    const q = evaluateFashionQuality(outfit, classicWorkContext);
    assert.ok(q.shoeAppropriateness >= 0.95,
      `Expected shoeAppropriateness >= 0.95 for variant ${variant}, got ${q.shoeAppropriateness}`);
  }
});
