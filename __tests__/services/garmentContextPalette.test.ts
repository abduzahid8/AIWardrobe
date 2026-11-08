import { garmentContextReason } from '../../src/domain/garmentContext';
import { stylingContext } from '../../src/domain/stylingContext';

const quiet = (style = 'old_money') =>
  stylingContext({ style, occasion: 'Work', preferences: undefined });
const item = (name: string, macroCategory: string, extra: Record<string, unknown> = {}) => ({
  name,
  macroCategory,
  ...extra,
});

describe('vivid red conflicts with quiet classic palettes', () => {
  it('rejects saturated red evidence in the persisted color even when the title omits it', () => {
    expect(
      garmentContextReason(
        item('Classic double-pleated chino pant', 'bottom', { color: 'red' }),
        quiet()
      )
    ).toBe('vivid_red_conflicts_with_quiet_palette');
  });

  it('rejects explicit scarlet or vivid-red titles', () => {
    for (const name of [
      'Scarlet cotton shirt',
      'Crimson wool blazer',
      'Bright red chino trousers',
      'Electric-red knit',
    ]) {
      expect(garmentContextReason(item(name, 'top'), quiet())).toBe(
        'vivid_red_conflicts_with_quiet_palette'
      );
    }
  });

  it('keeps muted reds and explicit favorite colors admissible', () => {
    for (const name of ['Burgundy wool blazer', 'Maroon cotton shirt', 'Oxblood leather loafers']) {
      expect(garmentContextReason(item(name, 'top'), quiet())).toBeNull();
    }
    expect(
      garmentContextReason(
        item('Red cotton shirt', 'top'),
        stylingContext({
          style: 'old_money',
          occasion: 'Work',
          preferences: { favoriteColors: ['red'] },
        })
      )
    ).toBeNull();
  });

  it('is scoped to quiet palettes rather than a global color ban', () => {
    expect(garmentContextReason(item('Red cotton shirt', 'top'), quiet('casual'))).toBeNull();
    expect(garmentContextReason(item('Red cotton shirt', 'top'), quiet('minimalist'))).toBe(
      'vivid_red_conflicts_with_quiet_palette'
    );
  });
});
