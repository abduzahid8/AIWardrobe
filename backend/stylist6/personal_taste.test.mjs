import test from 'node:test';
import assert from 'node:assert/strict';
import { personalTaste, evaluatePersonalItem, evaluatePersonalLook } from './personal_taste.mjs';

const item = (extra = {}) => ({
  id: 'sku-navy-1', name: 'Navy wool blazer', brand: 'Brand House',
  color: 'navy', ...extra,
});
const descriptor = (extra = {}) => ({
  color: 'navy', pattern: 'stripe', volume: 'slim', ...extra,
});

test('an unconfigured or default-balanced profile does not bias a look', () => {
  for (const input of [null, undefined, {}, {fitPreference:'balanced'}, 'bad', []]) {
    const taste = personalTaste(input);
    assert.equal(taste.active, false);
    assert.deepEqual(evaluatePersonalItem(item(), descriptor(), taste), {
      blocked: null, score: 0, matches: [],
    });
    assert.deepEqual(evaluatePersonalLook([{ item: item(), descriptor: descriptor() }], taste),
      { adjustment: 0, matches: [], matchedItems: 0 });
  }
});

test('explicit navy, brand, striped pattern and fitted silhouette help the same real item', () => {
  const taste = personalTaste({
    favoriteColors: [' NAVY ', 'navy'],
    favoriteBrands: ['BRAND HOUSE'],
    likedPatterns: ['stripes'],
    fitPreference: 'fitted',
  });
  const scored = evaluatePersonalItem(item(), descriptor(), taste);
  assert.equal(scored.blocked, null);
  assert.equal(scored.score, 0.12); // capped: no single item may dominate coherence
  assert.deepEqual(scored.matches.map(x => x.type),
    ['favorite_color', 'favorite_brand', 'liked_pattern', 'preferred_fit']);
  assert.ok(scored.matches.every(x=>x.productId==='sku-navy-1'));
  const look = evaluatePersonalLook([
    {item:item(), descriptor:descriptor()},
    {item:item({id:'sku-bottom',brand:'Other',color:'grey'}), descriptor:descriptor({color:'grey',pattern:'solid',volume:'regular'})},
    {item:item({id:'sku-shoes',brand:'Other',color:'brown'}), descriptor:descriptor({color:'brown',pattern:'solid',volume:'unknown'})},
  ], taste);
  assert.ok(look.adjustment > 0 && look.adjustment <= .085);
  assert.equal(look.matchedItems,1);
  assert.ok(look.matches.every(x=>x.productId==='sku-navy-1'));
});

test('avoided brands are exact, case-insensitive exclusions and not fuzzy substrings', () => {
  const taste = personalTaste({avoidBrands:['  BRAND HOUSE  ']});
  assert.equal(evaluatePersonalItem(item(), descriptor(),taste).blocked,'excluded_brand');
  assert.equal(evaluatePersonalItem(item({brand:'Brand House Studio'}),descriptor(),taste).blocked,null);
  assert.equal(evaluatePersonalItem(item({brand:'Other'}),descriptor(),taste).blocked,null);
  assert.equal(evaluatePersonalItem(item({brand:null}),descriptor(),taste).blocked,null);
});

test('fit dislike has bounded soft weight and unknown evidence is neutral', () => {
  const taste=personalTaste({fitPreference:'loose',favoriteColors:['navy'],favoriteBrands:['Brand House']});
  const mismatch = evaluatePersonalItem(item({brand:'Different',color:'grey'}),
    descriptor({color:'grey',volume:'slim'}),taste);
  assert.equal(mismatch.score,-0.025);
  const unknown = evaluatePersonalItem(item({brand:'',color:'unknown'}),
    descriptor({color:'unknown',volume:'unknown',pattern:'unknown'}),taste);
  assert.equal(unknown.score,0);
  const matching = evaluatePersonalItem(item({brand:'Different',color:'grey'}),
    descriptor({color:'grey',volume:'wide'}),taste);
  assert.equal(matching.score,.025);
});

test('hostile and oversized preferences are bounded without guessing body shape or price', () => {
  const taste=personalTaste({
    favoriteColors: Array.from({length:100},(_,i)=>'color'+i),
    avoidBrands:[{},null,44,'Brand House'],
    fitPreference:'tall',
    bodyType:'athletic',priceRange:{min:1,max:1,currency:'USD'},
  });
  assert.ok(taste.favoriteColors.size<=16);
  assert.equal(taste.desiredFit,null);
  assert.equal(evaluatePersonalItem(item(),descriptor(),taste).blocked,'excluded_brand');
  assert.equal(evaluatePersonalLook([],taste).adjustment,0);
});

test('color canonicalization fixes #0A1931 -> navy and gray -> grey across preferences and items', () => {
  const taste = personalTaste({
    favoriteColors: ['#0A1931', 'gray'],
  });
  assert.equal(taste.favoriteColors.has('navy'), true);
  assert.equal(taste.favoriteColors.has('grey'), true);

  const navyHit = evaluatePersonalItem(
    item({ id: 'sku-midnight', color: 'navy' }),
    descriptor({ color: 'navy', pattern: 'solid', volume: 'regular' }),
    taste,
  );
  assert.ok(navyHit.score > 0);
  assert.ok(navyHit.matches.some((m) => m.type === 'favorite_color' && m.value === 'navy'));

  const greyHit = evaluatePersonalItem(
    item({ id: 'sku-gray-trouser', color: 'gray' }),
    descriptor({ color: 'gray', pattern: 'solid', volume: 'regular' }),
    taste,
  );
  assert.ok(greyHit.score > 0);
  assert.ok(greyHit.matches.some((m) => m.type === 'favorite_color' && m.value === 'grey'));
});

test('dislikedPatterns and avoidColors exert decisive negative pressure', () => {
  const taste = personalTaste({
    favoriteColors: ['#0A1931'],
    avoidColors: ['red', 'orange'],
    dislikedPatterns: ['graphic print', 'plaid'],
  });
  assert.equal(taste.richDnaActive, true);

  const printedRed = evaluatePersonalItem(
    item({ id: 'sku-bad', color: 'red', brand: 'Other' }),
    descriptor({ slot: 'top', color: 'red', pattern: 'print', volume: 'regular' }),
    taste,
  );
  assert.ok(printedRed.score <= -0.20, `expected strong penalty for avoided color + disliked pattern, got ${printedRed.score}`);
  assert.ok(printedRed.penalties.some((p) => p.includes('avoided_color:red')));
  assert.ok(printedRed.penalties.some((p) => p.includes('disliked_pattern:print')));

  const cleanNavy = evaluatePersonalItem(
    item({ id: 'sku-good', color: 'navy', brand: 'Other' }),
    descriptor({ slot: 'outerwear', color: 'navy', pattern: 'solid', volume: 'regular' }),
    taste,
  );
  assert.ok(cleanNavy.score >= 0.06);
  assert.ok(cleanNavy.score - printedRed.score >= 0.25);
});

test('PersonalStyleDna + CIELAB colorTheory genuinely drives look selection between Warm Autumn and Cool Winter profiles', () => {
  const earthLook = [
    { item: item({ id: 'e-top', macroCategory: 'top', color: 'cream', brand: 'Loro' }), descriptor: descriptor({ slot: 'top', color: 'cream', pattern: 'solid', volume: 'regular' }) },
    { item: item({ id: 'e-out', macroCategory: 'outerwear', color: 'camel', brand: 'Loro' }), descriptor: descriptor({ slot: 'outerwear', color: 'camel', pattern: 'solid', volume: 'regular' }) },
    { item: item({ id: 'e-bot', macroCategory: 'bottom', color: 'olive', brand: 'Loro' }), descriptor: descriptor({ slot: 'bottom', color: 'olive', pattern: 'solid', volume: 'regular' }) },
    { item: item({ id: 'e-sho', macroCategory: 'shoes', color: 'espresso', brand: 'Loro' }), descriptor: descriptor({ slot: 'shoes', color: 'espresso', pattern: 'solid', volume: 'regular' }) },
  ];
  const icyContrastLook = [
    { item: item({ id: 'c-top', macroCategory: 'top', color: 'crisp_white', brand: 'COS' }), descriptor: descriptor({ slot: 'top', color: 'crisp_white', pattern: 'solid', volume: 'regular' }) },
    { item: item({ id: 'c-out', macroCategory: 'outerwear', color: 'charcoal', brand: 'COS' }), descriptor: descriptor({ slot: 'outerwear', color: 'charcoal', pattern: 'solid', volume: 'regular' }) },
    { item: item({ id: 'c-bot', macroCategory: 'bottom', color: 'slate', brand: 'COS' }), descriptor: descriptor({ slot: 'bottom', color: 'slate', pattern: 'solid', volume: 'regular' }) },
    { item: item({ id: 'c-sho', macroCategory: 'shoes', color: 'black', brand: 'COS' }), descriptor: descriptor({ slot: 'shoes', color: 'black', pattern: 'solid', volume: 'regular' }) },
  ];
  const clashingCognacLook = [
    { item: item({ id: 'x-top', macroCategory: 'top', color: 'crisp_white', brand: 'Other' }), descriptor: descriptor({ slot: 'top', color: 'crisp_white', pattern: 'solid', volume: 'regular' }) },
    { item: item({ id: 'x-out', macroCategory: 'outerwear', color: 'black', brand: 'Other' }), descriptor: descriptor({ slot: 'outerwear', color: 'black', pattern: 'solid', volume: 'regular' }) },
    { item: item({ id: 'x-bot', macroCategory: 'bottom', color: 'cool_grey', brand: 'Other' }), descriptor: descriptor({ slot: 'bottom', color: 'cool_grey', pattern: 'solid', volume: 'regular' }) },
    { item: item({ id: 'x-sho', macroCategory: 'shoes', color: 'cognac', brand: 'Other' }), descriptor: descriptor({ slot: 'shoes', color: 'cognac', pattern: 'solid', volume: 'regular' }) },
  ];

  const warmAutumnTaste = personalTaste({
    appearance: {
      consentGranted: true,
      skinUndertone: 'warm',
      contrastLevel: 'low',
      colorSeason: 'warm_autumn',
      flatteringColors: ['camel', 'olive', 'espresso'],
      powerNeutrals: ['cream'],
      provenance: 'opt_in_photo_analysis',
    },
  });
  const coolWinterTaste = personalTaste({
    appearance: {
      consentGranted: true,
      skinUndertone: 'cool',
      contrastLevel: 'high',
      colorSeason: 'cool_winter',
      flatteringColors: ['crisp_white', 'charcoal', 'black'],
      powerNeutrals: ['slate'],
      provenance: 'opt_in_photo_analysis',
    },
  });

  const autumnOnEarth = evaluatePersonalLook(earthLook, warmAutumnTaste);
  const autumnOnIcy = evaluatePersonalLook(icyContrastLook, warmAutumnTaste);
  const autumnOnClash = evaluatePersonalLook(clashingCognacLook, warmAutumnTaste);

  const winterOnIcy = evaluatePersonalLook(icyContrastLook, coolWinterTaste);
  const winterOnEarth = evaluatePersonalLook(earthLook, coolWinterTaste);

  // Genuine meaningful ranking separation (far larger than legacy ±0.025!)
  assert.ok(autumnOnEarth.adjustment >= 0.14, `expected strong positive Warm Autumn adjustment, got ${autumnOnEarth.adjustment}`);
  assert.ok(autumnOnEarth.adjustment - autumnOnIcy.adjustment >= 0.14);
  assert.ok(autumnOnClash.adjustment < 0);
  assert.equal(autumnOnClash.colorTheory.harmonyMode, 'discordant');

  assert.ok(winterOnIcy.adjustment >= 0.14, `expected strong positive Cool Winter adjustment, got ${winterOnIcy.adjustment}`);
  assert.ok(winterOnIcy.adjustment - winterOnEarth.adjustment >= 0.14);
});

test('opt-in body proportion directives and owned wardrobe DNA drive item scoring while unconsented appearance stays ignored', () => {
  const petiteInvertedTaste = personalTaste({
    body: {
      bodyType: 'inverted_triangle',
      heightCm: 164,
      presentation: 'menswear',
    },
    wardrobeItems: [
      { category: 'outerwear', name: 'Navy Merino Cardigan', primaryColor: 'navy', material: 'merino', brand: 'Suitsupply', wearCount: 15, isFavorite: true },
    ],
  });

  const flatteringLayer = evaluatePersonalItem(
    item({ id: 'sku-soft-crop', macroCategory: 'outerwear', name: 'Unstructured Cropped Soft-Shoulder Merino Jacket', color: 'navy', brand: 'Suitsupply', material: 'merino' }),
    descriptor({ slot: 'outerwear', color: 'navy', pattern: 'solid', volume: 'regular', length: 'cropped', relaxedTailoring: true, materials: ['merino'] }),
    petiteInvertedTaste,
  );
  const unflatteringLayer = evaluatePersonalItem(
    item({ id: 'sku-padded-long', macroCategory: 'outerwear', name: 'Padded Shoulder Longline Overcoat', color: 'stone', brand: 'Other' }),
    descriptor({ slot: 'outerwear', color: 'stone', pattern: 'solid', volume: 'slim', length: 'long' }),
    petiteInvertedTaste,
  );

  assert.ok(flatteringLayer.score >= 0.12, `expected body+wardrobe reward, got ${flatteringLayer.score}`);
  assert.ok(unflatteringLayer.score <= -0.10, `expected body cut penalty, got ${unflatteringLayer.score}`);

  // Unconsented appearance profile must remain inactive
  const unconsented = personalTaste({
    appearance: { consentGranted: false, skinUndertone: 'warm', colorSeason: 'warm_autumn', flatteringColors: ['camel'] },
  });
  assert.equal(unconsented.active, false);
  assert.equal(evaluatePersonalItem(item({ color: 'camel' }), descriptor({ color: 'camel' }), unconsented).score, 0);
});
