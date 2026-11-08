/** @jest-environment node */
import { describeGarment } from '../../src/domain/garmentDescriptor';
import { prepareHydratedStylist5Garment } from '../../src/domain/stylist5VisualContract';
import type { VisualFashionProfile } from '../../src/domain/visualFashion';
import { createOutfitPlan } from '../../src/domain/validatedQuartets';
import { stylingContext } from '../../src/domain/stylingContext';

const imageUrl = 'https://catalog.example/cardigan.jpg';
const sourceHash = 'a'.repeat(32);
const profile = (attributes: VisualFashionProfile['attributes']): VisualFashionProfile => ({
  version: 1,
  sourceHash,
  imageUrl,
  imageSha256: 'b'.repeat(64),
  model: 'siglip2',
  revision: 'c'.repeat(40),
  attributes,
});
const observation = (value: string, confidence = 0.95) => ({
  value,
  confidence,
  margin: 0.2,
  evidence: 'image-model' as const,
});

test('accepts the production intelligence shape and applies hydrated visual evidence', () => {
  const item = {
    id: 'shop-cardigan-1',
    name: 'Merino cardigan',
    description: 'Fine knit cardigan',
    macroCategory: 'outerwear',
    type: 'outerwear',
    imageUrl: imageUrl,
    visualSourceHash: sourceHash,
  };
  const hydrated = prepareHydratedStylist5Garment(item, {
    fashion_intelligence: {
      version: 1,
      descriptor: {
        ...describeGarment(item),
        evidence: ['text:catalog'],
      },
    },
    visual_fashion: profile({
      subtype: observation('blazer'),
      color: observation('navy'),
    }),
    visual_source_hash: sourceHash,
    price_peers: { median: 100 },
  });
  expect(hydrated).not.toBeNull();
  expect(Object.isFrozen(hydrated)).toBe(true);
  expect((hydrated as typeof hydrated & { pricePeers?: unknown }).pricePeers).toEqual({ median: 100 });
  expect(describeGarment(hydrated!).color).toBe('navy');
  expect(describeGarment(hydrated!).evidence.some((entry) => entry.startsWith('image:'))).toBe(true);
  expect(describeGarment(hydrated!).evidence.some((entry) => entry.startsWith('image:color:'))).toBe(true);
})

test('preserves the prepared descriptor through planning and selection', () => {
  const cardigan = prepareHydratedStylist5Garment(
    {
      id: 'shop-cardigan-1',
      name: 'Merino cardigan',
      description: 'Fine knit cardigan',
      macroCategory: 'outerwear',
      type: 'outerwear',
      imageUrl: imageUrl,
      visualSourceHash: sourceHash,
    },
    {
      visual_fashion: profile({ color: observation('navy') }),
      visual_source_hash: sourceHash,
      price_peers: { median: 100 },
    }
  );
  const inventory = [
    cardigan!,
    { id: 'shop-shirt-1', name: 'Oxford shirt', macroCategory: 'top' },
    { id: 'shop-trouser-1', name: 'Tailored wool trousers', macroCategory: 'bottom' },
    { id: 'shop-loafer-1', name: 'Brown leather loafers', macroCategory: 'shoes' },
  ];
  const plan = createOutfitPlan(
    inventory,
    stylingContext({ style: 'classic', occasion: 'Work' }),
    () => ({ valid: true, score: 1 }),
    {
      requiredIds: ['shop-cardigan-1'],
      target: 1,
      priority: () => [],
      cost: () => 0,
      maxEvaluations: 64,
      maxSyncMs: 200,
    }
  );
  const selected = plan.seed[0]?.items.find((item) => item.id === 'shop-cardigan-1');
  expect(selected).toBe(cardigan);
  expect(Object.isFrozen(selected)).toBe(true);
  expect(describeGarment(selected!).evidence.some((entry) => entry.startsWith('image:'))).toBe(true);
});

test('quarantines invalid visual source hashes and category conflicts after hydration', () => {
  const item = {
    id: 'shop-cardigan-1',
    name: 'Merino cardigan',
    macroCategory: 'outerwear',
    type: 'outerwear',
    imageUrl,
    visualSourceHash: sourceHash,
  };
  expect(
    prepareHydratedStylist5Garment(item, {
      visual_fashion: profile({ color: observation('navy') }),
      visual_source_hash: 'd'.repeat(32),
    })
  ).not.toBeNull();
  expect(
    prepareHydratedStylist5Garment(item, {
      visual_fashion: {
        ...profile({}),
        imageKind: { ...observation('diagram'), evidence: 'image-ocr' as const },
      },
      visual_source_hash: sourceHash,
    })
  ).toBeNull();
  expect(
    prepareHydratedStylist5Garment(item, {
      visual_fashion: profile({ slot: observation('shoes') }),
      visual_source_hash: sourceHash,
    })
  ).toBeNull();
});
