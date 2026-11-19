import test from 'node:test';
import assert from 'node:assert/strict';
import {
  PER_DIRECTION_RETRIEVAL_VERSION,
  buildPerDirectionCandidatePools,
  scoreItemForDirection,
} from './per_direction_retrieval.mjs';

function makeSku({
  id, slot, subtype, name, color = 'navy', texture = 'wool', formality = 'smart',
  volume = 'regular', price = 120, currency = 'USD', brand = 'Atelier',
  visualColor = null,
}) {
  const imageUrl = `https://cdn.example.com/photos/${id}.jpg`;
  const sourceHash = 'a'.repeat(32);
  return {
    id,
    name,
    brand,
    category: slot === 'bottom' ? 'bottoms' : slot === 'shoes' ? 'shoes' : 'tops',
    garment_type: slot === 'bottom' ? 'lower_body' : slot === 'shoes' ? 'shoes' : 'upper_body',
    macroCategory: slot,
    fashion_slot: slot,
    subtype,
    fashion_subtype: subtype,
    color,
    texture,
    formality,
    fashion_formality: formality,
    volume,
    price,
    currency,
    is_active: true,
    isShopItem: true,
    imageUrl,
    image_url: imageUrl,
    shopUrl: `https://shop.example.com/products/${id}`,
    source_url: `https://shop.example.com/products/${id}`,
    visualSourceHash: sourceHash,
    visual_source_hash: sourceHash,
    visualFashion: {
      version: 1,
      sourceHash,
      imageUrl,
      imageSha256: 'b'.repeat(64),
      model: 'siglip2',
      revision: 'v1',
      imageKind: { value: 'product', confidence: 0.99, margin: 0.98, evidence: 'image-model' },
      imageQuality: { width: 900, height: 1200 },
      attributes: {
        slot: { value: slot, confidence: 0.96, margin: 0.90, evidence: 'image-model' },
        color: { value: visualColor || color, confidence: 0.94, margin: 0.72, evidence: 'image-model' },
        texture: { value: texture, confidence: 0.90, margin: 0.50, evidence: 'image-model' },
      },
    },
  };
}

function buildDiverseSyntheticCatalog() {
  const items = [];
  const specs = {
    top: [
      { subtype: 'dress-shirt', colors: ['white', 'blue', 'cream'], texture: 'smooth', formality: 'formal', volume: 'slim', basePrice: 80 },
      { subtype: 'fine-knit', colors: ['cream', 'camel', 'olive'], texture: 'knit', formality: 'smart', volume: 'regular', basePrice: 110 },
      { subtype: 'polo', colors: ['navy', 'stone', 'charcoal'], texture: 'knit', formality: 'smart', volume: 'regular', basePrice: 75 },
      { subtype: 't-shirt', colors: ['white', 'black', 'grey'], texture: 'smooth', formality: 'casual', volume: 'wide', basePrice: 45 },
    ],
    outerwear: [
      { subtype: 'blazer', colors: ['navy', 'charcoal', 'grey'], texture: 'wool', formality: 'formal', volume: 'slim', basePrice: 260 },
      { subtype: 'overshirt', colors: ['olive', 'stone', 'tan'], texture: 'cotton', formality: 'casual', volume: 'wide', basePrice: 130 },
      { subtype: 'coat', colors: ['camel', 'black', 'charcoal'], texture: 'wool', formality: 'smart', volume: 'regular', basePrice: 380 },
      { subtype: 'cardigan', colors: ['brown', 'cream', 'navy'], texture: 'knit', formality: 'smart', volume: 'regular', basePrice: 150 },
      { subtype: 'bomber', colors: ['black', 'olive', 'brown'], texture: 'suede', formality: 'casual', volume: 'regular', basePrice: 220 },
    ],
    bottom: [
      { subtype: 'tailored-trouser', colors: ['charcoal', 'navy', 'grey'], texture: 'wool', formality: 'formal', volume: 'regular', basePrice: 160 },
      { subtype: 'chino', colors: ['stone', 'beige', 'olive'], texture: 'cotton', formality: 'smart', volume: 'regular', basePrice: 90 },
      { subtype: 'jeans', colors: ['navy', 'black', 'blue'], texture: 'denim', formality: 'casual', volume: 'wide', basePrice: 115 },
      { subtype: 'trouser', colors: ['cream', 'brown', 'black'], texture: 'wool', formality: 'smart', volume: 'wide', basePrice: 145 },
    ],
    shoes: [
      { subtype: 'oxford', colors: ['black', 'brown', 'burgundy'], texture: 'smooth', formality: 'formal', volume: 'slim', basePrice: 340 },
      { subtype: 'loafer', colors: ['brown', 'tan', 'navy'], texture: 'suede', formality: 'smart', volume: 'regular', basePrice: 190 },
      { subtype: 'chelsea', colors: ['black', 'brown', 'charcoal'], texture: 'smooth', formality: 'smart', volume: 'regular', basePrice: 260 },
      { subtype: 'minimal-sneaker', colors: ['white', 'cream', 'stone'], texture: 'smooth', formality: 'casual', volume: 'regular', basePrice: 95 },
      { subtype: 'derby', colors: ['brown', 'black', 'burgundy'], texture: 'smooth', formality: 'formal', volume: 'regular', basePrice: 140 },
    ],
  };

  for (const [slot, groups] of Object.entries(specs)) {
    let idx = 0;
    for (const g of groups) {
      for (let rep = 0; rep < 3; rep++) {
        for (const color of g.colors) {
          idx++;
          items.push(makeSku({
            id: `PRIVATE_${slot}_${g.subtype}_${color}_${idx}`,
            slot,
            subtype: g.subtype,
            name: `${color} ${g.texture} ${g.subtype} #${idx}`,
            color,
            texture: g.texture,
            formality: g.formality,
            volume: g.volume,
            price: Math.round(g.basePrice * (0.65 + rep * 0.45)),
            currency: 'USD',
            brand: rep === 0 ? 'AccessibleHouse' : rep === 1 ? 'MidAtelier' : 'LuxuryCasa',
          }));
        }
      }
    }
  }
  return items;
}

const threeOrthogonalDirections = [
  {
    id: 'ai-director-1',
    name: 'Architectural Tailored Signature',
    directionType: 'signature_mastery',
    silhouette: 'tailored',
    outerwearArchetype: 'structured_tailoring',
    trouserArchetype: 'tailored_pleated',
    footwearArchetype: 'sleek_lace_up_dress',
    palette: ['navy', 'white', 'charcoal'],
    colorHarmonyMode: 'cool_architectural_contrast',
    materialTexture: 'wool smooth leather',
    formality: 'formal',
    thermalBand: 'transitional',
    roles: { top: ['dress-shirt'], outerwear: ['blazer'], bottom: ['tailored-trouser'], shoes: ['oxford', 'derby'] },
  },
  {
    id: 'ai-director-2',
    name: 'Relaxed Utility Texture Shift',
    directionType: 'silhouette_texture_shift',
    silhouette: 'relaxed',
    outerwearArchetype: 'relaxed_utility_overshirt',
    trouserArchetype: 'relaxed_chino',
    footwearArchetype: 'soft_suede_loafer',
    palette: ['olive', 'stone', 'brown', 'cream'],
    colorHarmonyMode: 'warm_earth_analogous',
    materialTexture: 'cotton suede knit',
    formality: 'smart',
    thermalBand: 'transitional',
    roles: { top: ['fine-knit', 'polo'], outerwear: ['overshirt'], bottom: ['chino'], shoes: ['loafer'] },
  },
  {
    id: 'ai-director-3',
    name: 'Editorial Coat & Boot Statement',
    directionType: 'editorial_statement',
    silhouette: 'balanced',
    outerwearArchetype: 'longline_coat',
    trouserArchetype: 'refined_dark_denim',
    footwearArchetype: 'sculptural_chelsea_boot',
    palette: ['camel', 'black', 'charcoal'],
    colorHarmonyMode: 'tonal_monochrome',
    materialTexture: 'wool denim smooth',
    formality: 'smart',
    thermalBand: 'transitional',
    roles: { top: ['polo', 't-shirt'], outerwear: ['coat', 'bomber'], bottom: ['jeans', 'trouser'], shoes: ['chelsea'] },
  },
];

test('retrieves dedicated, disjoint, direction-aligned pools across 3 orthogonal directions without leaking private IDs', () => {
  assert.equal(PER_DIRECTION_RETRIEVAL_VERSION, 's6-per-direction-retrieval-v1');
  const catalog = buildDiverseSyntheticCatalog();
  const result = buildPerDirectionCandidatePools({
    directions: threeOrthogonalDirections,
    catalog,
    context: { style: 'classic', occasion: 'Date night', weather: { temp: 16 } },
    perSlotLimit: 10,
  });

  assert.equal(result.directions.length, 3);
  assert.equal(result.metrics.crossDirectionOverlapRate, 0, 'zero SKU overlap across 3 directions on a healthy catalog');
  assert.equal(result.metrics.uniqueSkusExposed, 120, '3 directions * 4 slots * 10 SKUs = 120 unique real SKUs');

  // Verify each direction leads with its own archetype (not cardigan+chino monotony!)
  assert.equal(result.directions[0].slots.outerwear[0].public.subtype, 'blazer');
  assert.equal(result.directions[1].slots.outerwear[0].public.subtype, 'overshirt');
  assert.ok(['coat', 'bomber'].includes(result.directions[2].slots.outerwear[0].public.subtype));

  assert.equal(result.directions[0].slots.bottom[0].public.subtype, 'tailored-trouser');
  assert.equal(result.directions[1].slots.bottom[0].public.subtype, 'chino');
  assert.equal(result.directions[2].slots.bottom[0].public.subtype, 'jeans');

  assert.ok(['oxford', 'derby'].includes(result.directions[0].slots.shoes[0].public.subtype));
  assert.equal(result.directions[1].slots.shoes[0].public.subtype, 'loafer');
  assert.equal(result.directions[2].slots.shoes[0].public.subtype, 'chelsea');

  // Verify zero private identifiers or URLs leak into public payloads
  const serializedPublic = JSON.stringify(result.unifiedPool.map(x => x.public));
  assert.doesNotMatch(serializedPublic, /PRIVATE_|https?:\/\/|cdn\.example|shop\.example/);
});

test('balances price tiers and enforces honest budget headroom so $400+ dress shoes never starve a $320 budget', () => {
  const catalog = buildDiverseSyntheticCatalog();
  const budgeted = buildPerDirectionCandidatePools({
    directions: threeOrthogonalDirections,
    catalog,
    context: {
      style: 'classic',
      occasion: 'Work',
      weather: { temp: 18 },
      budget: { amount: 320, currency: 'USD' },
    },
    perSlotLimit: 10,
  });
  const shoes = budgeted.unifiedPool.filter(x => x.slot === 'shoes');
  assert.ok(shoes.length >= 12);
  // Headroom for shoes is 65% of $320 = $208
  assert.ok(shoes.every(s => s.public.price <= 208.01), 'no $400+ shoe may enter a $320 budget pool');
  assert.ok(shoes.some(s => s.priceTier === 'accessible'), 'accessible shoes are guaranteed representation');
});

test('respects Personal Style DNA exclusions, qualityMap quarantines, tasteProfile affinities, and thermal intelligence', () => {
  const catalog = buildDiverseSyntheticCatalog();
  const hotLinen = makeSku({
    id: 'HOT_LINEN_OVERSHIRT',
    slot: 'outerwear',
    subtype: 'overshirt',
    name: 'Breathable Unlined Linen Overshirt',
    color: 'cream',
    texture: 'linen',
    formality: 'smart',
    price: 115,
  });
  const winterPuffer = makeSku({
    id: 'WINTER_DOWN_COAT',
    slot: 'outerwear',
    subtype: 'coat',
    name: 'Heavy Down-Filled Insulated Puffer Coat',
    color: 'cream',
    texture: 'technical',
    formality: 'casual',
    price: 350,
  });
  const quarantinedBlazer = makeSku({
    id: 'QUARANTINED_BLAZER',
    slot: 'outerwear',
    subtype: 'blazer',
    name: 'Navy Wool Blazer Quarantined',
    color: 'navy',
    texture: 'wool',
    formality: 'formal',
    price: 180,
  });

  const qualityMap = new Map([
    ['QUARANTINED_BLAZER', { quarantined: true, photoDefects: 4, qualityScore: 0.1 }],
  ]);

  const dna = {
    presentation: 'menswear',
    avoidColors: ['#0A1931'], // hex navy -> must exclude navy!
    brandExclusions: ['LuxuryCasa'],
    appearance: {
      skinUndertone: 'warm',
      flatteringColors: ['olive', 'cream', 'camel'],
      avoidColors: ['black'],
    },
  };

  const hotResult = buildPerDirectionCandidatePools({
    directions: [
      { ...threeOrthogonalDirections[1], thermalBand: 'hot', roles: { top: ['polo'], outerwear: ['overshirt', 'coat'], bottom: ['chino'], shoes: ['loafer'] } },
    ],
    catalog: [...catalog, hotLinen, winterPuffer, quarantinedBlazer],
    context: { style: 'old_money', occasion: 'Casual', weather: { temp: 31 } },
    personalStyleDna: dna,
    qualityMap,
    perSlotLimit: 10,
  });

  const exposedIds = new Set(hotResult.unifiedPool.map(x => x.item.id));
  assert.equal(exposedIds.has('QUARANTINED_BLAZER'), false, 'quarantined SKU excluded');
  assert.ok(hotResult.unifiedPool.every(x => x.item.brand !== 'LuxuryCasa'), 'excluded brand filtered');
  assert.ok(hotResult.unifiedPool.every(x => x.public.photo.color !== 'navy'), 'hex-canonicalized avoidColor excluded');

  const hotOuter = hotResult.directions[0].slots.outerwear;
  const linenIdx = hotOuter.findIndex(x => x.item.id === 'HOT_LINEN_OVERSHIRT');
  const pufferIdx = hotOuter.findIndex(x => x.item.id === 'WINTER_DOWN_COAT');
  assert.ok(linenIdx >= 0, 'breathable linen overshirt is selected for 31C');
  assert.ok(pufferIdx === -1 || linenIdx < pufferIdx, 'linen outranks winter puffer in 31C heat');

  // Verify direct scoreItemForDirection reflects tasteProfile boosts
  const baseScore = scoreItemForDirection(
    { id: 's1', slot: 'shoes', subtype: 'loafer', color: 'brown', colorVerifiedByPhoto: true, texture: 'suede', formality: 'smart', volume: 'regular', bulk: 'unknown', hemShape: 'unknown', insulated: false, materials: ['suede'], searchText: 'brown suede loafer', brand: 'test', photoFactCount: 3, qualityScore: 0.65 },
    threeOrthogonalDirections[1],
    { context: { occasion: 'Date night', weather: { temp: 18 } } },
  );
  const boostedScore = scoreItemForDirection(
    { id: 's1', slot: 'shoes', subtype: 'loafer', color: 'brown', colorVerifiedByPhoto: true, texture: 'suede', formality: 'smart', volume: 'regular', bulk: 'unknown', hemShape: 'unknown', insulated: false, materials: ['suede'], searchText: 'brown suede loafer', brand: 'test', photoFactCount: 3, qualityScore: 0.90 },
    threeOrthogonalDirections[1],
    {
      context: { occasion: 'Date night', weather: { temp: 18 } },
      tasteProfile: { active: true, itemTally: new Map([['s1', 0.95]]), slotPreferences: { shoes: { subtypes: { loafer: 0.8 } } } },
    },
  );
  assert.ok(boostedScore.score > baseScore.score, 'positive tasteProfile and high qualityScore increase direction score');
});

test('indoor layer priority (>10C) eliminates peacoat/parka monopoly in Work & Date Night while allowing at most 1 tailored coat dir at 10-13C', () => {
  const catalog = buildDiverseSyntheticCatalog();
  // Inject heavy outdoor coats (peacoat, commuter rain trench, parka, wool overcoat)
  const heavyCoats = [
    makeSku({ id: 'PEACOAT_1', slot: 'outerwear', subtype: 'coat', name: 'Dock Peacoat in Wool Blend with PrimaLoft', color: 'navy', texture: 'wool', formality: 'smart', price: 398, brand: 'JCrew' }),
    makeSku({ id: 'TRENCH_COMMUTER', slot: 'outerwear', subtype: 'coat', name: 'Waterproof Commuter Rain Trench Coat', color: 'beige', texture: 'technical', formality: 'smart', price: 320, brand: 'RainHouse' }),
    makeSku({ id: 'PARKA_1', slot: 'outerwear', subtype: 'coat', name: 'Heavy Hooded Winter Parka', color: 'olive', texture: 'technical', formality: 'casual', price: 290, brand: 'NorthGear' }),
    makeSku({ id: 'OVERCOAT_1', slot: 'outerwear', subtype: 'coat', name: 'Camel Tailored Cashmere Overcoat', color: 'camel', texture: 'wool', formality: 'formal', price: 650, brand: 'TailorHouse' }),
  ];
  const fullCatalog = [...catalog, ...heavyCoats];

  // 1) Date Night 15C: ZERO heavy outdoor coats (no peacoat, trench, parka, or overcoat) in ANY direction
  const date15 = buildPerDirectionCandidatePools({
    catalog: fullCatalog,
    context: { style: 'classic', occasion: 'Date night', weather: { temp: 15 } },
    perSlotLimit: 10,
  });
  assert.equal(date15.directions.length, 3);
  assert.equal(date15.metrics.crossDirectionOverlapRate, 0);
  const dateOuterIds = date15.unifiedPool.filter(e => e.slot === 'outerwear').map(e => e.item.id);
  assert.equal(dateOuterIds.includes('PEACOAT_1'), false, 'peacoat must not appear in 15C Date Night');
  assert.equal(dateOuterIds.includes('TRENCH_COMMUTER'), false, 'commuter trench must not appear in 15C Date Night');
  assert.equal(dateOuterIds.includes('PARKA_1'), false, 'parka must not appear in 15C Date Night');
  assert.equal(date15.metrics.tailoredCoatDirectionsUsed, 0);

  // 2) Old Money Work 13C with custom coat direction in Dir 3:
  //    Direction 1 NEVER has heavy outdoor coats, peacoat/parka/commuter trench excluded, at most 1 tailored coat dir!
  const work13 = buildPerDirectionCandidatePools({
    directions: threeOrthogonalDirections,
    catalog: fullCatalog,
    context: { style: 'old_money', occasion: 'Work', weather: { temp: 13 } },
    perSlotLimit: 10,
  });
  const dir1Outer = work13.directions[0].slots.outerwear;
  assert.ok(dir1Outer.every(e => e.public.subtype !== 'coat'), 'Direction 1 indoor Work (13C) must never feature a heavy coat');
  const allWork13OuterIds = work13.unifiedPool.filter(e => e.slot === 'outerwear').map(e => e.item.id);
  assert.equal(allWork13OuterIds.includes('PEACOAT_1'), false, 'PrimaLoft peacoat excluded from 13C indoor Work');
  assert.equal(allWork13OuterIds.includes('PARKA_1'), false, 'parka excluded from 13C indoor Work');
  assert.ok(work13.metrics.tailoredCoatDirectionsUsed <= 1, 'at most 1 direction may feature a tailored coat at 10-13C');

  // 3) Work 16C (default directions): ZERO peacoats or heavy outdoor coats across all 3 pools
  const work16 = buildPerDirectionCandidatePools({
    catalog: fullCatalog,
    context: { style: 'classic', occasion: 'Work', weather: { temp: 16 } },
    perSlotLimit: 10,
  });
  assert.equal(work16.metrics.tailoredCoatDirectionsUsed, 0);
  assert.ok(work16.unifiedPool.filter(e => e.slot === 'outerwear').every(e => !/peacoat|parka|trench/i.test(e.item.name)));
});

test('tight-budget ($350 Work) dynamic headroom caps + fxSnapshot guarantee 3 disjoint affordable pools and fail closed on unconvertible currency', () => {
  const items = [];
  const brands = ['HouseA', 'HouseB', 'HouseC', 'HouseD', 'HouseE', 'HouseF'];
  const colors = ['navy', 'white', 'grey', 'stone', 'cream', 'olive', 'brown', 'charcoal', 'blue', 'tan'];
  // Build rich multi-currency catalog with BOTH expensive ($200-$600) and affordable items
  for (let i = 0; i < 60; i++) {
    const brand = brands[i % brands.length];
    const color = colors[i % colors.length];
    const isCheap = i < 48;
    const currency = i % 3 === 0 ? 'EUR' : 'USD';
    const fxScale = currency === 'EUR' ? 1 / 1.08 : 1;
    items.push(makeSku({ id: `T_${i}`, slot: 'top', subtype: i % 2 === 0 ? 'dress-shirt' : 'polo', name: `${color} Cotton Top ${i}`, color, texture: 'cotton', price: Math.round((isCheap ? 35 + (i % 35) : 180 + i) * fxScale), currency, brand }));
    items.push(makeSku({ id: `O_${i}`, slot: 'outerwear', subtype: ['blazer', 'overshirt', 'cardigan'][i % 3], name: `${color} Tailored Layer ${i}`, color, texture: 'cotton', price: Math.round((isCheap ? 65 + (i % 55) : 340 + i) * fxScale), currency, brand }));
    items.push(makeSku({ id: `B_${i}`, slot: 'bottom', subtype: i % 2 === 0 ? 'tailored-trouser' : 'chino', name: `${color} Smart Trouser ${i}`, color, texture: 'cotton', price: Math.round((isCheap ? 45 + (i % 45) : 220 + i) * fxScale), currency, brand }));
    items.push(makeSku({ id: `S_${i}`, slot: 'shoes', subtype: ['loafer', 'derby', 'oxford', 'minimal-sneaker'][i % 4], name: `${color} Leather Shoe ${i}`, color, texture: 'leather', price: Math.round((isCheap ? 75 + (i % 70) : 320 + i) * fxScale), currency, brand }));
  }
  // Add unconvertible INR shoes that must NEVER sneak into a USD-budgeted pool
  for (let i = 0; i < 10; i++) {
    items.push(makeSku({ id: `S_INR_${i}`, slot: 'shoes', subtype: 'loafer', name: `Hand Welted Loafer INR ${i}`, color: 'brown', texture: 'leather', price: 12800, currency: 'INR', brand: 'ForeignCraft' }));
  }

  const fxSnapshot = {
    id: 'CBU:2026-11-19:12800:13824:16256',
    source: 'CBU',
    asOf: '2026-11-19',
    fetchedAt: new Date().toISOString(),
    uzsPerUnit: { USD: 12800, EUR: 13824, GBP: 16256, UZS: 1 },
  };
  const toUsd = (it) => (it.price * fxSnapshot.uzsPerUnit[it.currency]) / fxSnapshot.uzsPerUnit.USD;

  const result = buildPerDirectionCandidatePools({
    catalog: items,
    context: {
      style: 'classic',
      occasion: 'Work',
      weather: { temp: 18 },
      budget: { amount: 350, currency: 'USD' },
      fxSnapshot,
    },
    perSlotLimit: 14,
  });

  assert.equal(result.metrics.tightBudgetApplied, true);
  assert.equal(result.metrics.crossDirectionOverlapRate, 0, 'zero overlap across all 3 tight-budget pools');
  assert.equal(result.metrics.uniqueSkusExposed, 168, '14 SKUs * 4 slots * 3 directions = 168 unique affordable SKUs');
  assert.ok(result.unifiedPool.every(e => e.item.currency !== 'INR'), 'unconvertible INR SKUs must fail closed under USD budget');

  // Verify tight per-slot caps (shoes <= 44% = $154, outerwear <= 38% = $133, bottom <= 28% = $98, top <= 22% = $77)
  for (const d of result.directions) {
    assert.equal(d.slots.top.length, 14);
    assert.equal(d.slots.outerwear.length, 14);
    assert.equal(d.slots.bottom.length, 14);
    assert.equal(d.slots.shoes.length, 14);
    assert.ok(d.slots.top.every(e => toUsd(e.item) <= 77.01), 'tops <= 22% of $350');
    assert.ok(d.slots.outerwear.every(e => toUsd(e.item) <= 133.01), 'outerwear <= 38% of $350');
    assert.ok(d.slots.bottom.every(e => toUsd(e.item) <= 98.01), 'bottoms <= 28% of $350');
    assert.ok(d.slots.shoes.every(e => toUsd(e.item) <= 154.01), 'shoes <= 44% of $350');
    const cheapestCombo = ['top', 'outerwear', 'bottom', 'shoes'].reduce(
      (sum, slot) => sum + Math.min(...d.slots[slot].map(e => toUsd(e.item))),
      0,
    );
    assert.ok(cheapestCombo <= 350, `direction ${d.directionIndex + 1} must support complete <=$350 outfits (got $${cheapestCombo})`);
  }
});

test('warm/hot weather (>=24C / 31C Summer) expands breathable pools across all 3 orthogonal directions and excludes heavy winter hazards', () => {
  const items = [];
  const brands = ['RivieraA', 'RivieraB', 'RivieraC', 'RivieraD', 'RivieraE'];
  const colors = ['stone', 'cream', 'white', 'olive', 'navy', 'beige', 'blue', 'tan'];
  for (let i = 0; i < 45; i++) {
    const brand = brands[i % brands.length];
    const color = colors[i % colors.length];
    items.push(makeSku({ id: `SUM_T_${i}`, slot: 'top', subtype: ['polo', 't-shirt', 'shirt'][i % 3], name: `${color} Lightweight Linen Cotton Top ${i}`, color, texture: 'linen', formality: 'smart', price: 65 + (i % 40), brand }));
    items.push(makeSku({ id: `SUM_O_${i}`, slot: 'outerwear', subtype: ['blazer', 'overshirt', 'cardigan'][i % 3], name: `${color} Unlined Breathable Linen Cotton Layer ${i}`, color, texture: 'linen', formality: 'smart', price: 110 + (i % 60), brand }));
    items.push(makeSku({ id: `SUM_B_${i}`, slot: 'bottom', subtype: ['chino', 'trouser'][i % 2], name: `${color} Breathable Linen Chino ${i}`, color, texture: 'linen', formality: 'smart', price: 85 + (i % 40), brand }));
    items.push(makeSku({ id: `SUM_S_${i}`, slot: 'shoes', subtype: ['loafer', 'driver', 'minimal-sneaker'][i % 3], name: `${color} Summer Unlined Shoe ${i}`, color, texture: 'suede', formality: 'smart', price: 140 + (i % 60), brand }));
  }
  // Inject winter hazards that must be purged at >=24C
  items.push(
    makeSku({ id: 'HAZARD_PEACOAT', slot: 'outerwear', subtype: 'coat', name: 'Heavy Wool Dock Peacoat', color: 'navy', texture: 'wool', price: 320, brand: 'WinterCo' }),
    makeSku({ id: 'HAZARD_WAXED', slot: 'outerwear', subtype: 'jacket', name: 'Waxed Cotton Field Jacket', color: 'olive', texture: 'cotton', price: 220, brand: 'WinterCo' }),
    makeSku({ id: 'HAZARD_CHUNKY_MERINO', slot: 'outerwear', subtype: 'cardigan', name: 'Chunky Pure Merino Shawl Neck Cardigan', color: 'navy', texture: 'wool', price: 229, brand: 'WinterCo' }),
    makeSku({ id: 'HAZARD_TURTLENECK', slot: 'top', subtype: 'fine-knit', name: 'Heavy Wool Turtleneck Sweater', color: 'charcoal', texture: 'wool', price: 160, brand: 'WinterCo' }),
    makeSku({ id: 'HAZARD_BOOT', slot: 'shoes', subtype: 'chelsea', name: 'Winter Leather Chelsea Boot', color: 'black', texture: 'leather', price: 250, brand: 'WinterCo' }),
  );

  for (const temp of [24, 31]) {
    const res = buildPerDirectionCandidatePools({
      catalog: items,
      context: { style: 'smart_casual', occasion: 'Casual', weather: { temp }, budget: { amount: 750, currency: 'USD' } },
      perSlotLimit: 12,
    });
    assert.equal(res.directions.length, 3);
    assert.equal(res.metrics.crossDirectionOverlapRate, 0, `zero overlap at ${temp}C`);
    assert.equal(res.metrics.uniqueSkusExposed, 144, `3 full orthogonal pools at ${temp}C`);
    const ids = new Set(res.unifiedPool.map(e => e.item.id));
    for (const hazardId of ['HAZARD_PEACOAT', 'HAZARD_WAXED', 'HAZARD_CHUNKY_MERINO', 'HAZARD_TURTLENECK', 'HAZARD_BOOT']) {
      assert.equal(ids.has(hazardId), false, `${hazardId} must be excluded at ${temp}C`);
    }
    assert.equal(res.directions[0].slots.outerwear[0].public.subtype, 'blazer');
    assert.equal(res.directions[1].slots.outerwear[0].public.subtype, 'overshirt');
    assert.equal(res.directions[2].slots.outerwear[0].public.subtype, 'cardigan');
  }
});

test('minimalist & quiet_luxury penalize loud argyle/paisley/camo patterns and >10C indoor Work keeps 0 heavy outdoor coats', () => {
  const items = [];
  const brands = ['RowA', 'RowB', 'RowC', 'RowD', 'RowE'];
  const colors = ['navy', 'cream', 'olive', 'charcoal', 'stone', 'brown'];
  for (let i = 0; i < 42; i++) {
    const brand = brands[i % brands.length];
    const color = colors[i % colors.length];
    items.push(makeSku({ id: `MIN_T_${i}`, slot: 'top', subtype: ['shirt', 'polo', 'fine-knit'][i % 3], name: `${color} Clean Merino Crewneck ${i}`, color, texture: 'knit', formality: 'smart', price: 95, brand }));
    items.push(makeSku({ id: `MIN_O_${i}`, slot: 'outerwear', subtype: ['blazer', 'overshirt', 'cardigan'][i % 3], name: `${color} Tailored Indoor Layer ${i}`, color, texture: 'wool', formality: 'smart', price: 210, brand }));
    items.push(makeSku({ id: `MIN_B_${i}`, slot: 'bottom', subtype: ['tailored-trouser', 'chino'][i % 2], name: `${color} Clean Straight Trouser ${i}`, color, texture: 'wool', formality: 'smart', price: 120, brand }));
    items.push(makeSku({ id: `MIN_S_${i}`, slot: 'shoes', subtype: ['loafer', 'derby', 'minimal-sneaker'][i % 3], name: `${color} Minimalist Shoe ${i}`, color, texture: 'suede', formality: 'smart', price: 180, brand }));
  }
  const argyle = makeSku({ id: 'LOUD_ARGYLE', slot: 'top', subtype: 'fine-knit', name: 'Retro Purple Mustard Argyle Diamond Sweater', color: 'brown', texture: 'knit', formality: 'smart', price: 95, brand: 'RowA' });
  const solid = makeSku({ id: 'CLEAN_SOLID', slot: 'top', subtype: 'fine-knit', name: 'Solid Brown Fine Merino Sweater', color: 'brown', texture: 'knit', formality: 'smart', price: 95, brand: 'RowA' });
  const overcoat = makeSku({ id: 'HEAVY_OVERCOAT_13C', slot: 'outerwear', subtype: 'coat', name: 'Double-Breasted Wool Overcoat', color: 'camel', texture: 'wool', formality: 'smart', price: 320, brand: 'RowB' });

  const minRes = buildPerDirectionCandidatePools({
    catalog: [...items, argyle, solid, overcoat],
    context: { style: 'minimalist', occasion: 'Work', weather: { temp: 13 }, budget: { amount: 1500, currency: 'USD' } },
    perSlotLimit: 10,
  });
  assert.equal(minRes.unifiedPool.some(e => e.item.id === 'HEAVY_OVERCOAT_13C'), false, '13C indoor Work must keep 0 heavy outdoor coats when indoor layers exist');
  const weekendRes = buildPerDirectionCandidatePools({
    catalog: [argyle, solid, ...items.slice(0, 20)],
    context: { style: 'minimalist', occasion: 'Weekend', weather: { temp: 20 }, budget: { amount: 900, currency: 'USD' } },
    perSlotLimit: 10,
  });
  const topEntries = weekendRes.unifiedPool.filter(e => e.slot === 'top');
  const solidEntry = topEntries.find(e => e.item.id === 'CLEAN_SOLID');
  const argyleEntry = topEntries.find(e => e.item.id === 'LOUD_ARGYLE');
  assert.ok(solidEntry, 'clean solid knit must enter minimalist top pool');
  if (argyleEntry) {
    assert.ok(solidEntry.strength > argyleEntry.strength, `solid knit (${solidEntry.strength}) must outscore argyle (${argyleEntry.strength}) in minimalist style`);
  }
});
