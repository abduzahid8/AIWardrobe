/**
 * Minimalist print discipline: minimalism vetoes loud pieces in every
 * occasion, so theatre prints (palm, floral, souvenir…) must be rejected
 * for minimalist tops/layers while tailored patterns always survive.
 * Examples below are independent of any vendor fixture.
 */
import { rejectForStyleOccasion } from '../../src/domain/occasionRules';

const top = (name: string, type = 'Shirt') => ({
  name,
  type,
  macroCategory: 'top',
});

describe('minimalist print discipline', () => {
  it('rejects a palm-print camp shirt for minimalist work (minimalist branch)', () => {
    // Shipped engine maps work to the smart dress code, so the formal branch
    // no longer fires; the minimalist overlay rejects the palm print directly.
    // The point stands: the palm token is detected at all (it previously passed).
    expect(
      rejectForStyleOccasion(
        top('Ginger Brown Fan Palm Camp Collar Shirt'),
        'minimalist',
        'work',
      ),
    ).toBe('loud prints break minimalist discipline');
  });

  it('rejects a palm-print shirt for minimalist weekend (minimalist branch)', () => {
    expect(
      rejectForStyleOccasion(top('Ginger Brown Fan Palm Camp Collar Shirt'), 'minimalist', 'weekend'),
    ).toBe('loud prints break minimalist discipline');
  });

  it('rejects a floral shirt for a minimalist date (minimalist branch)', () => {
    // Shipped engine maps date to the smart dress code, so the shared
    // loud-print detector rejects via the minimalist overlay.
    expect(
      rejectForStyleOccasion(top('White Floral Print Linen Shirt'), 'minimalist', 'date'),
    ).toBe('loud prints break minimalist discipline');
  });

  it('rejects a souvenir jacket layer for minimalist travel', () => {
    expect(
      rejectForStyleOccasion(
        { name: 'Black Souvenir Embroidered Jacket', type: 'Jacket', macroCategory: 'outerwear' },
        'minimalist',
        'travel',
      ),
    ).toBe('loud prints break minimalist discipline');
  });

  it('keeps tailored patterns: oxford stripes and Glen checks survive', () => {
    expect(
      rejectForStyleOccasion(top('Light Blue Oxford Stripe Shirt'), 'minimalist', 'work'),
    ).toBeNull();
    expect(
      rejectForStyleOccasion(
        { name: 'Grey Glen Check Wool Blazer', type: 'Blazer', macroCategory: 'outerwear' },
        'minimalist',
        'work',
      ),
    ).toBeNull();
  });

  it('keeps solid knit tops (smart-casual workwear judgment call)', () => {
    expect(
      rejectForStyleOccasion(top('Grey Pique Knit Shirt'), 'minimalist', 'work'),
    ).toBeNull();
  });
});
