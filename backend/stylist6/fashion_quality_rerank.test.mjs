import test from 'node:test';
import assert from 'node:assert/strict';
import {
  evaluateFashionQuality,
  rerankFashionCandidates,
  FASHION_QUALITY_VERSION,
} from './fashion_deck.mjs';
import {
  selectCreativeDirectionDeck,
  garmentArchitectureOf,
  deckOfficeMonotonyAdjustment,
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
