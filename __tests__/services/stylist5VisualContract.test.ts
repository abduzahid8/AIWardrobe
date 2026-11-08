/** @jest-environment node */
import { describeGarment } from '../../src/domain/garmentDescriptor';
import { prepareHydratedStylist5Garment } from '../../src/domain/stylist5VisualContract';
import type { VisualFashionProfile } from '../../src/domain/visualFashion';

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
  expect(describeGarment(hydrated!).color).toBe('navy');
  expect(describeGarment(hydrated!).evidence.some((entry) => entry.startsWith('image:'))).toBe(true);
})

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
