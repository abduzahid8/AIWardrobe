/**
 * calendarClassifier.test.ts
 *
 * Tests for the style → calendar-occasion mapping used when saving
 * outfits to the calendar.  The mapping lives in AIOutfitmaker.tsx
 * as getCalendarOccasion() and in the calendar types as OCCASION_IDS.
 */

import {
  OCCASION_IDS,
  isValidOccasion,
  createOutfitLog,
  formatDate,
  getDaysInMonth,
  getFirstDayOfMonth,
  resolveCalendarOccasion,
} from '../../features/calendar/types';

// Exercise the production resolver with no explicit occasion for legacy callers.
const getCalendarOccasion = (styleId?: string) => resolveCalendarOccasion(undefined, styleId);

test.each(OCCASION_IDS)('explicit %s survives every style default', occasion => {
  for (const style of ['classic','old_money','business_casual','casual','minimalist',undefined]) {
    expect(resolveCalendarOccasion(occasion,style)).toBe(occasion);
  }
});

test('an invalid explicit event falls back to a supported occasion',()=>{
  expect(resolveCalendarOccasion('invalid-event','business_casual')).toBe('work');
  expect(resolveCalendarOccasion('invalid-event','unknown-style')).toBe('casual');
});

describe('getCalendarOccasion — style → occasion classifier', () => {
  it('maps business_casual to work', () => {
    expect(getCalendarOccasion('business_casual')).toBe('work');
  });

  it('maps old_money to formal', () => {
    expect(getCalendarOccasion('old_money')).toBe('formal');
  });

  it('maps semi_classic to casual', () => {
    expect(getCalendarOccasion('semi_classic')).toBe('casual');
  });

  it('maps minimalist to casual', () => {
    expect(getCalendarOccasion('minimalist')).toBe('casual');
  });

  it('maps casual to casual', () => {
    expect(getCalendarOccasion('casual')).toBe('casual');
  });

  it('maps unknown styles to casual', () => {
    expect(getCalendarOccasion('bohemian')).toBe('casual');
    expect(getCalendarOccasion(undefined)).toBe('casual');
  });

  it('all classified occasions are valid OCCASION_IDS', () => {
    const styles = ['business_casual', 'old_money', 'semi_classic', 'minimalist', 'casual', 'classic', undefined as string | undefined];
    for (const s of styles) {
      const occasion = getCalendarOccasion(s);
      expect(isValidOccasion(occasion)).toBe(true);
    }
  });
});

describe('isValidOccasion', () => {
  it('accepts all OCCASION_IDS', () => {
    for (const id of OCCASION_IDS) {
      expect(isValidOccasion(id)).toBe(true);
    }
  });

  it('rejects unknown strings', () => {
    expect(isValidOccasion('meeting')).toBe(false);
    expect(isValidOccasion('')).toBe(false);
  });
});

describe('createOutfitLog', () => {
  const baseItems = [
    { id: '1', type: 'shirt', image: 'https://img.test/shirt.jpg' },
    { id: '2', type: 'pants', image: 'https://img.test/pants.jpg' },
    { id: '3', type: 'shoes', image: 'https://img.test/shoes.jpg' },
  ];

  it('creates a valid outfit log with 3 items', () => {
    const log = createOutfitLog('2025-06-15', baseItems, 'casual');
    expect(log.occasion).toBe('casual');
    expect(log.items).toHaveLength(3);
    expect(log.date).toBe('2025-06-15');
  });

  it('throws on invalid occasion', () => {
    expect(() => createOutfitLog('2025-06-15', baseItems, 'meeting')).toThrow(/Invalid occasion/);
  });

  it('throws on empty items', () => {
    expect(() => createOutfitLog('2025-06-15', [], 'casual')).toThrow();
  });

  it('throws when items exceed 6', () => {
    const tooMany = Array.from({ length: 7 }, (_, i) => ({
      id: String(i), type: 'item', image: 'x',
    }));
    expect(() => createOutfitLog('2025-06-15', tooMany, 'casual')).toThrow(/at most 6/);
  });
});

describe('calendar date helpers', () => {
  it('formatDate pads month and day', () => {
    expect(formatDate(2025, 0, 5)).toBe('2025-01-05');
    expect(formatDate(2025, 11, 31)).toBe('2025-12-31');
  });

  it('getDaysInMonth returns correct count', () => {
    expect(getDaysInMonth(2025, 0)).toBe(31);  // Jan
    expect(getDaysInMonth(2025, 1)).toBe(28);  // Feb non-leap
    expect(getDaysInMonth(2024, 1)).toBe(29);  // Feb leap
    expect(getDaysInMonth(2025, 3)).toBe(30);  // Apr
  });

  it('getFirstDayOfMonth returns 0-6', () => {
    const day = getFirstDayOfMonth(2025, 0); // Jan 2025
    expect(day).toBeGreaterThanOrEqual(0);
    expect(day).toBeLessThanOrEqual(6);
  });
});
