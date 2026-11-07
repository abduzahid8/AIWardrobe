import {
  assessOutfitQuality,
  styleJudgeReason,
  createQualityMemo,
} from '../../src/domain/occasionRules';
import { outfitGenerationFailure } from '../../src/services/outfitGenerationFailure';
const slots = ['top', 'outerwear', 'bottom', 'shoes'];
const parts = (names: string[]) =>
  names.map((name, i) => ({ id: `item-${i}`, name, macroCategory: slots[i] }));
const refined = [
  'White cotton poplin shirt',
  'Navy structured wool blazer',
  'Grey tailored wool trousers',
  'Brown leather penny loafers',
];

test('a high aggregate score cannot hide weak Work compatibility', () => {
  const look = parts([
    'Bold rugby striped cotton shirt',
    'Technical leisure zip blazer',
    refined[2],
    refined[3],
  ]);
  expect(styleJudgeReason(look, 'old_money', 'Work')).toBeNull();
  const q = assessOutfitQuality(look, 'old_money', 'Work');
  expect(q.score).toBeGreaterThan(0.85);
  expect(q.reasons).toEqual([]);
  expect(q.acceptance.passed).toBe(false);
  expect(q.acceptance.score).toBeLessThan(q.acceptance.minimum);
});
test('severely shredded denim with business tailoring is below the Minimalist Casual floor', () => {
  const look = parts([
    'White cotton T-shirt',
    'Structured business wool blazer',
    'Shredded cotton denim jeans',
    'Minimal white leather sneakers',
  ]);
  expect(styleJudgeReason(look, 'minimalist', 'Casual')).toBeNull();
  expect(assessOutfitQuality(look, 'minimalist', 'Casual').acceptance).toMatchObject({
    passed: false,
    reason: 'business_tailoring_with_severely_distressed_denim',
  });
  // This is a scoped compatibility decision, not a global denim ban.
  expect(assessOutfitQuality(look, 'casual', 'Casual').acceptance.passed).toBe(true);
});
test.each([
  ['classic', 'Formal', refined],
  ['old_money', 'Work', refined],
  [
    'minimalist',
    'Casual',
    [
      'White cotton T-shirt',
      'Structured wool blazer',
      'Dark straight denim jeans',
      'Minimal white leather sneakers',
    ],
  ],
  [
    'minimalist',
    'Casual',
    [
      'White cotton T-shirt',
      'Unstructured casual cotton blazer',
      'Lightly washed straight denim jeans',
      'Minimal white leather sneakers',
    ],
  ],
  [
    'casual',
    'Casual',
    [
      'White cotton T-shirt',
      'Cotton overshirt',
      'Shredded cotton denim jeans',
      'Minimal white leather sneakers',
    ],
  ],
])('retains a coherent %s / %s look', (style, occasion, names) => {
  expect(
    assessOutfitQuality(parts(names as string[]), style as string, occasion as string).acceptance
      .passed
  ).toBe(true);
});
test('the confirmed tropical/drawstring wording remains hard-ineligible, independently of the floor', () => {
  const look = parts([
    'Tropical-pattern short-sleeve shirt',
    refined[1],
    'Drawstring trousers',
    refined[3],
  ]);
  expect(styleJudgeReason(look, 'old_money', 'Work')).not.toBeNull();
  expect(assessOutfitQuality(look, 'old_money', 'Work').acceptance.passed).toBe(false);
});
test('quality exhaustion does not opt into the screen closest-match fallback', async () => {
  const error = await outfitGenerationFailure({ code: 'quality_replacement_unavailable' });
  expect(error.translationKey).toBe('outfitMaker.generateFailed');
  expect(['outfitMaker.noMatchingClothing', 'outfitMaker.aiUnavailable']).not.toContain(
    error.translationKey
  );
});

test('request memo cannot hide changed construction evidence behind the same product id/name', () => {
  const look = parts([
    'White cotton T-shirt',
    'Structured business wool blazer',
    'Dark denim jeans',
    'Minimal white leather sneakers',
  ]);
  const memo = createQualityMemo();
  const good = memo(look, 'minimalist', 'Casual');
  const changed = look.map((item, i) =>
    i === 2 ? { ...item, description: 'Shredded denim with large holes' } : item
  );
  const weak = memo(changed, 'minimalist', 'Casual');
  expect(good.acceptance.passed).toBe(true);
  expect(weak.acceptance.passed).toBe(false);
  expect(weak).not.toBe(good);
});
