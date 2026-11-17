import {
  deriveFitProfile,
  evidenceWeight,
  recommendPurchaseSize,
  garmentPhysicalData,
} from '../../src/services/purchase/fitEvidence';
import { fit, row, now, shoe } from './fixtures';
test('fit evidence is specific to brand/category, uses references and does not create a universal M', () => {
  const profile = deriveFitProfile(
    [
      fit('shoes-1'),
      fit('shirt-1', { category: 'top', sizeLabel: 'M', brand: 'Other' }),
      fit('shoes-2'),
    ],
    now
  );
  expect(profile.referenceGarments).toHaveLength(3);
  const candidate = { ...shoe, availableSizes: ['41', '42', '43'] };
  expect(recommendPurchaseSize(candidate, profile, undefined, now).recommendedSize).toBe('42');
  expect(
    recommendPurchaseSize({ ...candidate, brand: 'Unseen' }, profile, undefined, now)
      .recommendedSize
  ).toBeUndefined();
  expect(
    recommendPurchaseSize(row('top', 'Cotton shirt', 'upper_body'), profile, undefined, now)
      .recommendedSize
  ).toBeUndefined();
});
test('exact reference beats unrelated priors, conflicts and stale evidence lower confidence', () => {
  const candidate = { ...shoe, availableSizes: ['42', '43'] };
  const exact = deriveFitProfile([fit('exact', { productId: shoe.id })], now);
  expect(recommendPurchaseSize(candidate, exact, undefined, now)).toMatchObject({
    recommendedSize: '42',
    confidence: 'high',
  });
  const conflict = deriveFitProfile(
    [fit('positive'), fit('negative', { overallFit: 'too_small', returned: true, kept: false })],
    now
  );
  expect(recommendPurchaseSize(candidate, conflict, undefined, now).confidence).toBe('low');
  expect(evidenceWeight(fit('old', { createdAt: '2020-01-01T00:00:00Z' }), now)).toBeLessThan(0.03);
  expect(
    deriveFitProfile([fit('one', { zones: { shoulders: 'tight' }, overallFit: 'too_small' })], now)
      .preferredEaseByZone.shoulders
  ).toBeUndefined();
});
test('latest same-garment feedback is one observation; unknown evidence and missing charts stay unknown', () => {
  const profile = deriveFitProfile(
    [
      fit('a', { productId: 'one' }),
      fit('b', { productId: 'one', createdAt: '2026-11-03', overallFit: 'too_large' }),
    ],
    now + 86400000
  );
  expect(profile.evidenceCount).toBe(1);
  expect(profile.referenceGarments).toHaveLength(0);
  expect(recommendPurchaseSize(shoe, deriveFitProfile([], now)).recommendedSize).toBeUndefined();
  expect(garmentPhysicalData(shoe, '42').profile).toBeUndefined();
});
test('exact chart outranks stored data without synthesizing centimeters', () => {
  const chart = {
    id: 'chart',
    garmentId: shoe.id,
    sizeLabel: '42',
    category: 'shoes' as const,
    source: 'manufacturer_size_chart' as const,
    measurements: { shoeLength: { valueCm: 27.5, confidence: 'high' as const } },
  };
  const stored = { ...chart, id: 'stored', measurements: { shoeLength: { valueCm: 28 } } };
  const result = garmentPhysicalData({ ...shoe, physicalProfiles: [stored] }, '42', {
    exactCharts: [chart],
  });
  expect(result.source).toBe('exact_chart');
  expect(result.profile).toBe(chart);
  expect(garmentPhysicalData(shoe, '41', { exactCharts: [chart] }).profile).toBeUndefined();
});

test('reported snugness is not silently treated as preferred fit; explicit preference is independent of measurements', () => {
  const observed = deriveFitProfile([fit('snug', { overallFit: 'snug', kept: true })], now);
  expect(observed.preferredEaseByCategory.shoes).toBeUndefined();
  expect(observed.fitPreferenceConfidence).toBe('low');
  const preferred = deriveFitProfile(
    [fit('preference', { overallFit: 'snug', preferredFit: 'relaxed' })],
    now
  );
  expect(preferred.preferredEaseByCategory.shoes).toBe('relaxed');
  const separate = deriveFitProfile(
    [
      fit('sneaker', { subcategory: 'sneaker' }),
      fit('shirt', { category: 'top', subcategory: 'overshirt', sizeLabel: 'M' }),
    ],
    now
  );
  expect(recommendPurchaseSize({ ...shoe, availableSizes: ['42'] }, separate).confidence).toBe(
    'low'
  );
});

test('verified merchant charts convert explicit inch units and reject unverified/unknown-unit inputs', () => {
  const { adaptMerchantMeasurements } = require('../../src/services/purchase/physicalDataAdapter');
  const chart = {
    verified: true,
    sourceUrl: 'https://merchant.test/product',
    productReference: 'sku',
    rows: [
      {
        sizeLabel: 'M',
        category: 'shirt',
        measurements: {
          chest: { value: 42, unit: 'in' },
          waist: { value: 90, unit: 'cm' },
          unknown: { value: 20, unit: 'cm' },
        },
      },
    ],
  };
  const product = { id: 'catalog-row', reference: 'sku', url: chart.sourceUrl };
  const parsed = adaptMerchantMeasurements(chart, product);
  expect(parsed[0].measurements.chest.valueCm).toBe(106.68);
  expect(parsed[0].measurements.waist.valueCm).toBe(90);
  expect(parsed[0].measurements.unknown).toBeUndefined();
  expect(adaptMerchantMeasurements({ ...chart, verified: false }, product)).toEqual([]);
  expect(adaptMerchantMeasurements(chart, { ...product, reference: 'other' })).toEqual([]);
});

test('body-derived observations never become passive garment references or learned fit preferences', () => {
  const profile = deriveFitProfile(
    Array.from({ length: 8 }, (_, i) =>
      fit('body-' + i, { source: 'body_profile', cut: 'regular fit' })
    ),
    now
  );
  expect(profile.referenceGarments).toEqual([]);
  expect(profile.preferredEaseByCategory).toEqual({});
  expect(recommendPurchaseSize({ ...shoe, availableSizes: ['42'] }, profile).confidence).toBe(
    'low'
  );
});

test('incomplete records degrade safely and aggregation never pools different people', () => {
  const broken = {
    id: 'broken',
    userId: 'user-a',
    overallFit: 'perfect',
    source: 'manual',
    confidence: NaN,
    createdAt: new Date(now).toISOString(),
  } as any;
  const aggregate = deriveFitProfile(
    [broken, fit('a'), fit('other-person', { userId: 'user-b', brand: 'Other' })],
    now
  );
  expect(aggregate.userId).toBe('user-a');
  expect(aggregate.evidenceCount).toBe(1);
  expect(aggregate.brandCategorySizePriors.some((prior) => prior.brand === 'other')).toBe(false);
  const exact = deriveFitProfile([fit('exact', { productId: shoe.id })], now);
  expect(
    recommendPurchaseSize(
      { ...shoe, availableSizes: ['42'] },
      exact,
      undefined,
      now + 3 * 365 * 86400000
    ).confidence
  ).toBe('low');
});

test('equal-timestamp repeated feedback follows the last explicit observation', () => {
  const aggregate = deriveFitProfile(
    [
      fit('first', { productId: 'one' }),
      fit('last', { productId: 'one', overallFit: 'too_small' }),
    ],
    now
  );
  expect(aggregate.referenceGarments).toHaveLength(0);
  expect(aggregate.brandCategorySizePriors[0].negativeWeight).toBeGreaterThan(0);
});

test('category/cut and text priors stay separate from trusted physical measurements', () => {
  const item = row('relaxed', 'Relaxed-fit linen shirt', 'upper_body');
  expect(garmentPhysicalData(item, 'M')).toMatchObject({ source: 'text', confidence: 'low' });
  expect(garmentPhysicalData(item, 'M').profile).toBeUndefined();
  const prior = garmentPhysicalData(item, 'M', {
    categoryPrior: { category: 'top', cut: 'relaxed', provenance: 'user reference prior' },
  });
  expect(prior).toMatchObject({ source: 'category_prior', confidence: 'low' });
  expect(prior.profile).toBeUndefined();
});

test('dress and accessory size recommendations cannot borrow unrelated top-size memory', () => {
  const profile = deriveFitProfile([fit('shirt', { category: 'top', sizeLabel: 'M' })], now);
  for (const type of ['dresses', 'accessory'] as const)
    expect(
      recommendPurchaseSize(
        { ...row(type, 'Regular garment', type), availableSizes: ['M'] },
        profile
      ).recommendedSize
    ).toBeUndefined();
});
