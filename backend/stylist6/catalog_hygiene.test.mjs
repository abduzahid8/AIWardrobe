import test from 'node:test';
import assert from 'node:assert/strict';
import {
  CATALOG_HYGIENE_VERSION,
  isValidProductPdpUrl,
  extractKnownImageDimensions,
  isSwimwearItem,
  isCroppedOrDetailPhoto,
  isEveningwearItem,
  classifyGarmentSlot,
  hasVisualSlotContradiction,
  evaluateCatalogItemHygiene,
  filterHygienicCatalog,
} from './catalog_hygiene.mjs';

const validItem = (overrides = {}) => ({
  id: 'sku-001',
  name: 'Navy Merino Wool Blazer',
  brand: 'Atelier',
  category: 'tops',
  garment_type: 'upper_body',
  fashion_slot: 'outerwear',
  fashion_subtype: 'blazer',
  price: 240,
  currency: 'USD',
  is_active: true,
  source_url: 'https://store.example.com/products/navy-merino-wool-blazer',
  image_url: 'https://cdn.example.com/images/blazer-001.jpg',
  visual_source_hash: 'a'.repeat(32),
  visual_fashion: {
    version: 1,
    sourceHash: 'a'.repeat(32),
    imageUrl: 'https://cdn.example.com/images/blazer-001.jpg',
    imageKind: { value: 'product', confidence: 0.99, margin: 0.98, evidence: 'image-model' },
    imageQuality: { width: 960, height: 1200 },
    attributes: {
      slot: { value: 'outerwear', confidence: 0.96, margin: 0.88, evidence: 'image-model' },
      color: { value: 'navy', confidence: 0.94, margin: 0.72, evidence: 'image-model' },
    },
  },
  ...overrides,
});

test('structural PDP URL validator rejects homepages, bare locales, and generic roots without brand lists', () => {
  assert.equal(CATALOG_HYGIENE_VERSION, 's6-catalog-hygiene-v1');
  for (const bad of [
    '',
    null,
    'http://store.example.com/products/shirt',
    'https://www.armani.com/en-us/',
    'https://www.brand.example/',
    'https://www.brand.example/en/',
    'https://www.brand.example/en-gb/men',
    'https://www.brand.example/us/menswear/shop',
    'https://www.brand.example/catalog',
    'https://www.brand.example/new-arrivals',
  ]) {
    const verdict = isValidProductPdpUrl(bad);
    assert.equal(verdict.ok, false, `expected rejection for ${bad}`);
  }
  for (const good of [
    'https://aurelien-online.com/products/beige-flannel-trousers',
    'https://www.allsaints.com/us/menswear/redact-hoodie/USM001NC-162.html',
    'https://www.selected.com/p/seersucker-shirt/16089429',
    'https://shop.example.com/en-us/?pid=SKU12345',
    'https://shop.example.com/wool-overcoat-9921.html',
  ]) {
    const verdict = isValidProductPdpUrl(good);
    assert.equal(verdict.ok, true, `expected acceptance for ${good}`);
  }
});

test('structural media integrity catches diagrams, missing URLs, low-res thumbnails, and qualityMap quarantines', () => {
  const diagram = validItem({
    id: 'sku-diagram',
    visual_fashion: {
      ...validItem().visual_fashion,
      imageKind: { value: 'diagram', confidence: 1, margin: 1, evidence: 'human' },
    },
  });
  const diagramVerdict = evaluateCatalogItemHygiene(diagram);
  assert.equal(diagramVerdict.eligible, false);
  assert.equal(diagramVerdict.mediaValid, false);
  assert.ok(diagramVerdict.reasons.some(r => r.startsWith('non_product_image_kind:diagram')));

  const lowResMeta = validItem({
    id: 'sku-lowres-meta',
    visual_fashion: {
      ...validItem().visual_fashion,
      imageQuality: { width: 262, height: 393 },
    },
  });
  const lowResMetaVerdict = evaluateCatalogItemHygiene(lowResMeta);
  assert.equal(lowResMetaVerdict.eligible, false);
  assert.ok(lowResMetaVerdict.reasons.includes('low_res_image'));

  const lowResUrl = validItem({
    id: 'sku-lowres-url',
    visual_fashion: null,
    image_url: 'https://cdn.example.com/img/q_auto,c_scale,w_262/item.jpg',
  });
  assert.deepEqual(extractKnownImageDimensions(lowResUrl), {
    width: 262, height: null, source: 'url_resize_param',
  });
  const lowResUrlVerdict = evaluateCatalogItemHygiene(lowResUrl);
  assert.equal(lowResUrlVerdict.eligible, false);
  assert.ok(lowResUrlVerdict.reasons.includes('low_res_image'));
  assert.ok(lowResUrlVerdict.warnings.includes('missing_visual_profile'));

  const svgImage = validItem({ id: 'sku-svg', image_url: 'https://cdn.example.com/placeholder.svg' });
  assert.equal(evaluateCatalogItemHygiene(svgImage).eligible, false);

  const qualityMap = new Map([
    ['sku-q1', { quarantined: true, quarantineReason: 'broken_photo', qualityScore: 0.12 }],
    ['sku-q2', { approvals: 1, rejections: 6, photoDefects: 3, qualityScore: 0.18 }],
    ['sku-q3', { approvals: 0, rejections: 2, categoryMismatches: 2, qualityScore: 0.30 }],
    ['sku-good', { approvals: 9, rejections: 1, photoDefects: 0, qualityScore: 0.88 }],
  ]);
  assert.equal(evaluateCatalogItemHygiene(validItem({ id: 'sku-q1' }), qualityMap).eligible, false);
  assert.equal(evaluateCatalogItemHygiene(validItem({ id: 'sku-q2' }), qualityMap).eligible, false);
  assert.equal(evaluateCatalogItemHygiene(validItem({ id: 'sku-q3' }), qualityMap).eligible, false);
  const goodV = evaluateCatalogItemHygiene(validItem({ id: 'sku-good' }), qualityMap);
  assert.equal(goodV.eligible, true);
  assert.equal(goodV.qualityScore, 0.88);
});

test('swimwear is excluded from non-beach occasions (Work/Date/Formal/Casual) and permitted for beach/resort', () => {
  const swimShorts = validItem({
    id: 'sku-swim',
    name: 'Navy Beachside Swim Shorts',
    category: 'bottoms',
    garment_type: 'lower_body',
    fashion_slot: 'bottom',
    fashion_subtype: 'shorts',
    source_url: 'https://store.example.com/products/navy-beachside-swim-shorts',
    visual_fashion: {
      ...validItem().visual_fashion,
      attributes: { slot: { value: 'bottom', confidence: 0.96, margin: 0.9, evidence: 'image-model' } },
    },
  });
  assert.equal(isSwimwearItem(swimShorts), true);
  for (const occasion of ['Work', 'Date night', 'Formal', 'Casual', '']) {
    const v = evaluateCatalogItemHygiene(swimShorts, null, { occasion });
    assert.equal(v.eligible, false, `swim shorts must be excluded for "${occasion}"`);
    assert.ok(v.reasons.includes('swimwear_in_non_beach_occasion'));
  }
  const beachVerdict = evaluateCatalogItemHygiene(swimShorts, null, { occasion: 'Beach resort' });
  assert.equal(beachVerdict.eligible, true);
});

test('dresses and jumpsuits classify cleanly into dress (never top) while dress shirts/shoes/trousers keep their true slots', () => {
  const silkDress = validItem({
    id: 'sku-dress-1',
    name: 'Emerald Silk Midi Dress',
    category: 'tops', // Even if mis-bucketed under tops in legacy feed!
    garment_type: 'upper_body',
    fashion_slot: 'top',
    fashion_subtype: 'unknown',
    source_url: 'https://store.example.com/products/emerald-silk-midi-dress',
    visual_fashion: null,
  });
  assert.equal(classifyGarmentSlot(silkDress).slot, 'dress');
  const dressFiltered = filterHygienicCatalog([silkDress], { occasion: 'Date night' });
  assert.equal(dressFiltered.items.length, 1);
  assert.equal(dressFiltered.items[0].macroCategory, 'dress');

  const jumpsuit = validItem({
    id: 'sku-jumpsuit',
    name: 'Tailored Linen Wide-Leg Jumpsuit',
    category: 'dresses',
    fashion_slot: 'dress',
    source_url: 'https://store.example.com/products/tailored-linen-jumpsuit',
    visual_fashion: null,
  });
  assert.equal(classifyGarmentSlot(jumpsuit).slot, 'dress');

  const dressShirt = validItem({
    id: 'sku-dress-shirt',
    name: 'Crisp White Poplin Dress Shirt',
    category: 'tops',
    fashion_slot: 'top',
    fashion_subtype: 'dress-shirt',
    source_url: 'https://store.example.com/products/white-poplin-dress-shirt',
    visual_fashion: null,
  });
  assert.equal(classifyGarmentSlot(dressShirt).slot, 'top');

  const dressShoes = validItem({
    id: 'sku-dress-shoes',
    name: 'Black Calfskin Oxford Dress Shoes',
    category: 'shoes',
    garment_type: 'shoes',
    fashion_slot: 'shoes',
    fashion_subtype: 'oxford',
    source_url: 'https://store.example.com/products/black-oxford-dress-shoes',
    visual_fashion: null,
  });
  assert.equal(classifyGarmentSlot(dressShoes).slot, 'shoes');

  const dressTrousers = validItem({
    id: 'sku-dress-trousers',
    name: 'Charcoal Wool Dress Trousers',
    category: 'bottoms',
    garment_type: 'lower_body',
    fashion_slot: 'bottom',
    fashion_subtype: 'tailored-trouser',
    source_url: 'https://store.example.com/products/charcoal-dress-trousers',
    visual_fashion: null,
  });
  assert.equal(classifyGarmentSlot(dressTrousers).slot, 'bottom');
});

test('high-confidence visual_fashion vs garment_intelligence cross-family contradictions are caught while top<->outerwear stays flexible', () => {
  const poloInBottoms = validItem({
    id: 'sku-conflict-1',
    name: 'Jeans Terry Towelling Polo Shirt',
    category: 'bottoms',
    garment_type: 'lower_body',
    fashion_slot: 'bottom',
    fashion_subtype: 'jeans',
    source_url: 'https://store.example.com/products/jeans-terry-polo-shirt',
    visual_fashion: {
      ...validItem().visual_fashion,
      attributes: {
        slot: { value: 'top', confidence: 0.999, margin: 0.99, evidence: 'image-model' },
      },
    },
  });
  assert.equal(hasVisualSlotContradiction(poloInBottoms), true);
  const verdict = evaluateCatalogItemHygiene(poloInBottoms);
  assert.equal(verdict.eligible, false);
  assert.ok(verdict.reasons.some(r => r.startsWith('visual_vs_intel_slot_conflict:bottom->top')));

  const knitUpperFlexible = validItem({
    id: 'sku-knit-flex',
    name: 'Fine Merino Pullover Sweater',
    category: 'tops',
    garment_type: 'upper_body',
    fashion_slot: 'top',
    fashion_subtype: 'fine-knit',
    source_url: 'https://store.example.com/products/fine-merino-pullover',
    visual_fashion: {
      ...validItem().visual_fashion,
      attributes: {
        slot: { value: 'outerwear', confidence: 0.94, margin: 0.80, evidence: 'image-model' },
      },
    },
  });
  assert.equal(hasVisualSlotContradiction(knitUpperFlexible), false);
  assert.equal(evaluateCatalogItemHygiene(knitUpperFlexible).eligible, true);
});

test('filterHygienicCatalog preserves immutability, handles owned wardrobe items, and respects presentation', () => {
  const owned = {
    id: 'owned-1',
    name: 'My Favorite Navy Coat',
    macroCategory: 'outerwear',
    isShopItem: false,
    imageUrl: 'https://cdn.example.com/owned/coat.jpg',
  };
  const womensBlazer = validItem({ id: 'w-1', gender: 'womenswear' });
  const mensBlazer = validItem({ id: 'm-1', gender: 'menswear' });
  const unisexBlazer = validItem({ id: 'u-1', gender: 'unisex' });
  const snapshotBefore = JSON.stringify([owned, womensBlazer, mensBlazer, unisexBlazer]);

  const result = filterHygienicCatalog(
    [owned, womensBlazer, mensBlazer, unisexBlazer],
    { occasion: 'Work', presentation: 'womenswear' },
  );
  assert.equal(JSON.stringify([owned, womensBlazer, mensBlazer, unisexBlazer]), snapshotBefore);
  assert.deepEqual(result.items.map(x => x.id), ['owned-1', 'w-1', 'u-1']);
  assert.equal(result.stats.excludedCount, 1);
  assert.equal(result.excluded[0].id, 'm-1');
});

test('evening/black-tie tuxedo and dinner stud shirts are excluded from daytime work/office/casual but allowed for formal/black-tie', () => {
  const msDinnerShirt = validItem({
    id: 'ms-clp60542508-z0',
    brand: 'M&S',
    name: 'Regular Fit Luxury Cotton Double Cuff Dress Shirt',
    category: 'tops',
    garment_type: 'upper_body',
    fashion_slot: 'top',
    fashion_subtype: 'dress-shirt',
    source_url: 'https://www.marksandspencer.com/regular-fit-pure-cotton-dinner-shirt/p/clp60542508',
    visual_fashion: {
      ...validItem().visual_fashion,
      attributes: { slot: { value: 'top', confidence: 0.97, margin: 0.95, evidence: 'image-model' } },
    },
  });
  const tuxedoShirt = validItem({
    id: 'gant-tux-1',
    name: 'Tailored Tuxedo Shirt White',
    category: 'tops',
    garment_type: 'upper_body',
    fashion_slot: 'top',
    fashion_subtype: 'dress-shirt',
    source_url: 'https://gant.example.com/p/tailored-tuxedo-shirt-white',
    visual_fashion: {
      ...validItem().visual_fashion,
      attributes: { slot: { value: 'top', confidence: 0.97, margin: 0.95, evidence: 'image-model' } },
    },
  });
  const studFrontShirt = validItem({
    id: 'marcella-1',
    name: 'Wing-Collar Dinner Marcella Stud-Front Shirt',
    category: 'tops',
    garment_type: 'upper_body',
    fashion_slot: 'top',
    fashion_subtype: 'dress-shirt',
    source_url: 'https://shirtmaker.example.com/p/marcella-shirt',
    visual_fashion: {
      ...validItem().visual_fashion,
      attributes: { slot: { value: 'top', confidence: 0.97, margin: 0.95, evidence: 'image-model' } },
    },
  });
  const regularPoplinShirt = validItem({
    id: 'eton-poplin-1',
    name: 'Slim Fit White Cotton Poplin Dress Shirt',
    category: 'tops',
    garment_type: 'upper_body',
    fashion_slot: 'top',
    fashion_subtype: 'dress-shirt',
    source_url: 'https://eton.example.com/p/white-poplin-dress-shirt',
    visual_fashion: {
      ...validItem().visual_fashion,
      attributes: { slot: { value: 'top', confidence: 0.97, margin: 0.95, evidence: 'image-model' } },
    },
  });

  for (const occasion of ['Work', 'Office', 'Casual']) {
    for (const eveningItem of [msDinnerShirt, tuxedoShirt, studFrontShirt]) {
      const verdict = evaluateCatalogItemHygiene(eveningItem, null, { occasion });
      assert.equal(verdict.eligible, false, `${eveningItem.id} must be excluded for daytime ${occasion}`);
      assert.ok(verdict.reasons.includes('evening_tuxedo_shirt_in_daytime_occasion'));
    }
    assert.equal(evaluateCatalogItemHygiene(regularPoplinShirt, null, { occasion }).eligible, true);
  }

  for (const formalOccasion of ['Formal', 'Black Tie', 'Evening Gala']) {
    for (const eveningItem of [msDinnerShirt, tuxedoShirt, studFrontShirt]) {
      assert.equal(
        evaluateCatalogItemHygiene(eveningItem, null, { occasion: formalOccasion }).eligible,
        true,
        `${eveningItem.id} must stay eligible for ${formalOccasion}`,
      );
    }
  }
});

test('cropped and detail-only product photos are rejected structurally via URL/metadata without false-flagging garment feature words', () => {
  const detailUrlItems = [
    validItem({ id: 'crop-1', image_url: 'https://cdn.example.com/images/shirt_cuff_detail.jpg', visual_fashion: null }),
    validItem({ id: 'crop-2', image_url: 'https://cdn.example.com/images/blazer-closeup.jpg', visual_fashion: null }),
    validItem({ id: 'crop-3', image_url: 'https://cdn.example.com/images/wool-suit.jpg?view=detail', visual_fashion: null }),
    validItem({ id: 'crop-4', image_url: 'https://cdn.example.com/swatch/fabric-navy.jpg', visual_fashion: null }),
    validItem({ id: 'crop-5', image_url: 'https://cdn.example.com/images/trouser_macro1.webp', visual_fashion: null }),
  ];
  for (const item of detailUrlItems) {
    assert.equal(isCroppedOrDetailPhoto(item).detected, true, `${item.id} should be detected as cropped/detail`);
    const v = evaluateCatalogItemHygiene(item);
    assert.equal(v.eligible, false);
    assert.equal(v.mediaValid, false);
    assert.ok(v.reasons.includes('cropped_or_detail_only_photo'));
  }

  const metaCropped = validItem({
    id: 'crop-meta-1',
    visual_fashion: {
      ...validItem().visual_fashion,
      imageQuality: { width: 900, height: 1200, isCropped: true, framing: 'closeup' },
    },
  });
  assert.equal(evaluateCatalogItemHygiene(metaCropped).eligible, false);
  assert.ok(evaluateCatalogItemHygiene(metaCropped).reasons.includes('cropped_or_detail_only_photo'));

  const extremeStrip = validItem({
    id: 'crop-aspect-1',
    visual_fashion: {
      ...validItem().visual_fashion,
      imageQuality: { width: 1600, height: 480 },
    },
  });
  assert.equal(evaluateCatalogItemHygiene(extremeStrip).eligible, false);
  assert.ok(evaluateCatalogItemHygiene(extremeStrip).reasons.includes('cropped_or_detail_only_photo'));

  // Garment feature names like "big_collar" or "double_cuff" in normal product photo filenames must NOT false-trigger
  const normalCollarShirt = validItem({
    id: 'ok-collar-1',
    image_url: 'https://cdn.example.com/files/big_collar_poplin_shirt_double_cuff_navy1.jpg',
    visual_fashion: null,
  });
  assert.equal(isCroppedOrDetailPhoto(normalCollarShirt).detected, false);
  assert.equal(evaluateCatalogItemHygiene(normalCollarShirt).eligible, true);
});

test('non-shirt eveningwear (tuxedo jackets, tuxedo trousers, patent tuxedo shoes) stays out of daytime work but eligible for black-tie', () => {
  const dinnerJacket = validItem({
    id: 'ev-jacket',
    name: 'Shawl Collar Dinner Jacket Cream',
    fashion_slot: 'outerwear',
    fashion_subtype: 'blazer',
    source_url: 'https://store.example.com/p/shawl-collar-dinner-jacket-cream',
  });
  const tuxedoPants = validItem({
    id: 'ev-pants',
    name: 'Regular-Fit Wool-Blend Tuxedo Pants',
    category: 'bottoms',
    garment_type: 'lower_body',
    fashion_slot: 'bottom',
    fashion_subtype: 'tailored-trouser',
    source_url: 'https://store.example.com/p/wool-tuxedo-pants',
    visual_fashion: {
      ...validItem().visual_fashion,
      attributes: { slot: { value: 'bottom', confidence: 0.96, margin: 0.9, evidence: 'image-model' } },
    },
  });
  const patentTuxShoes = validItem({
    id: 'ev-shoes',
    name: 'Whole-cut Oxford - Black Patent | Tuxedo Collection',
    category: 'shoes',
    garment_type: 'shoes',
    fashion_slot: 'shoes',
    fashion_subtype: 'oxford',
    source_url: 'https://store.example.com/p/patent-tuxedo-oxford',
    visual_fashion: {
      ...validItem().visual_fashion,
      attributes: { slot: { value: 'shoes', confidence: 0.96, margin: 0.9, evidence: 'image-model' } },
    },
  });

  for (const item of [dinnerJacket, tuxedoPants, patentTuxShoes]) {
    assert.equal(isEveningwearItem(item), true);
    const workVerdict = evaluateCatalogItemHygiene(item, null, { occasion: 'Work' });
    assert.equal(workVerdict.eligible, false, `${item.id} must be excluded from daytime Work`);
    assert.ok(workVerdict.reasons.includes('eveningwear_in_daytime_occasion'));
    assert.equal(
      evaluateCatalogItemHygiene(item, null, { occasion: 'Black Tie Gala' }).eligible,
      true,
      `${item.id} must remain eligible for Black Tie Gala`,
    );
  }
});
