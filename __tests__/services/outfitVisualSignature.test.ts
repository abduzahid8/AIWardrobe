import {
  outfitVisualSignature,
  recentStylingPenalty,
} from '../../src/domain/outfitVisualSignature';
import { stylingExplanationKeys } from '../../src/domain/stylingExplanation';
import { stylingContext } from '../../src/domain/stylingContext';
const parts = [
  'White lightweight shirt',
  'Fine navy cardigan',
  'Grey tailored trousers',
  'Brown leather loafers',
].map((name, i) => ({
  name,
  id: String(i),
  macroCategory: ['top', 'outerwear', 'bottom', 'shoes'][i],
}));
it('is independent of IDs and input order, but retains meaningful subtypes and texture', () => {
  const a = outfitVisualSignature(parts);
  expect(outfitVisualSignature(parts.map((p) => ({ ...p, id: 'other-' + p.id })).reverse())).toBe(
    a
  );
  expect(
    outfitVisualSignature(parts.map((p, i) => (i === 1 ? { ...p, name: 'Brown suede blazer' } : p)))
  ).not.toBe(a);
  expect(recentStylingPenalty(parts, [a])).toBeCloseTo(0.03);
  expect(recentStylingPenalty(parts, [])).toBe(0);
});
it('never manufactures certainty from an unknown-only signature', () => {
  const unknown = parts.map((p) => ({ ...p, name: 'Product' }));
  expect(recentStylingPenalty(unknown, [outfitVisualSignature(unknown)])).toBe(0);
});
it('explanation keys exist consistently in all supported locales', () => {
  const keys = stylingExplanationKeys(
    parts,
    stylingContext({ style: 'minimalist', occasion: 'Date night' })
  );
  expect(keys).toHaveLength(2);
  for (const language of ['en', 'ru', 'uz']) {
    const locale = require(`../../i18n/locales/${language}.json`);
    for (const key of keys)
      expect(key.split('.').reduce((value: any, k) => value[k], locale)).toEqual(
        expect.any(String)
      );
  }
});
