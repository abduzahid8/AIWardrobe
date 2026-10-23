import {
  needsLayering,
  resolveOutfitComposition,
} from '../../backend/supabase/functions/_shared/outfitComposition';

describe('outfit composition policy', () => {
  it.each([
    ['casual warm weather', 'casual', { temp: 24, condition: 'clear' }, null, false],
    ['casual cold weather', 'casual', { temp: 7, condition: 'clear' }, null, true],
    ['black-tie request', 'casual', { temp: 24, condition: 'clear' }, 'black-tie wedding', true],
    ['tuxedo request', 'casual', { temp: 24, condition: 'clear' }, 'tuxedo evening', true],
    ['explicit summer request', 'casual', { temp: 7, condition: 'clear' }, 'summer, no jacket', false],
    ['missing weather', 'casual', null, null, false],
    ['old money does not force a layer in warm weather', 'old_money', { temp: 24, condition: 'clear' }, null, false],
    ['business casual does not force a layer in warm weather', 'business_casual', { temp: 24, condition: 'clear' }, null, false],
    ['classic does not force a layer in warm weather', 'classic', { temp: 24, condition: 'clear' }, null, false],
    ['semi classic does not force a layer in warm weather', 'semi_classic', { temp: 24, condition: 'clear' }, null, false],
  ])('%s', (_label, style, weather, prompt, expected) => {
    expect(needsLayering(style, weather, prompt)).toBe(expected);
  });

  it('exposes the exact active slots used by diversity selection', () => {
    expect(resolveOutfitComposition('casual', { temp: 24, condition: 'clear' }).activeSlots)
      .toEqual(['top', 'bottom', 'shoes']);
    expect(resolveOutfitComposition('casual', { temp: 7, condition: 'clear' }).activeSlots)
      .toEqual(['outerwear', 'top', 'bottom', 'shoes']);
  });

  it('formal intent wins over a conflicting no-jacket phrase', () => {
    expect(needsLayering('casual', { temp: 24, condition: 'clear' }, 'black tie, no jacket')).toBe(true);
  });
});
