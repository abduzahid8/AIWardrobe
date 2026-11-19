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
