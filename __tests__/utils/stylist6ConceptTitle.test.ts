import { outfitConceptDisplayTitle, outfitConceptTitle } from '../../src/services/stylingCopy';

describe('Stylist 6 evidence-led outfit titles', () => {
  test('server distinct titles survive even when three looks share the same native concept ID', () => {
    const native = 'refined-tailoring';
    const titles = [
      outfitConceptDisplayTitle(native, 'Blue tailored contrast', 'stylist-6-dev-local-20261111-r2'),
      outfitConceptDisplayTitle(native, 'Brown knit and suede', 'stylist-6-dev-local-20261111-r2'),
      outfitConceptDisplayTitle(native, 'White shirt and charcoal', 'stylist-6-dev-local-20261111-r2'),
    ];
    expect(new Set(titles).size).toBe(3);
    expect(titles[0]).toBe('Blue tailored contrast');
  });

  test('cloud Stylist 6 preserves its server headline', () => {
    expect(outfitConceptDisplayTitle('refined-tailoring', 'Textured tailoring', 'stylist-6-cloud-staging-1'))
      .toBe('Textured tailoring');
  });

  test('Stylist 5 translations and S6 missing-title fallback remain unchanged', () => {
    expect(outfitConceptDisplayTitle('refined-tailoring', 'Server title', 'stylist-5-editorial-v2'))
      .toBe(outfitConceptTitle('refined-tailoring', 'Server title'));
    expect(outfitConceptDisplayTitle('refined-tailoring', '', 'stylist-6-cloud-staging-1'))
      .toBe(outfitConceptTitle('refined-tailoring', ''));
  });
});
