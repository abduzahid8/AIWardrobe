import { garmentContextReason } from '../../src/domain/garmentContext';
import { rejectForStyleOccasion } from '../../src/domain/occasionRules';
import { stylingContext } from '../../src/domain/stylingContext';

const formal = stylingContext({ style: 'semi_classic', occasion: 'Formal' });
const item = (name: string, macroCategory: string, type?: string) => ({ name, macroCategory, type });

test('strict formal base rejects jersey knit shirts but keeps woven shirts', () => {
  const knit = item('White Jersey Knit Button Shirt', 'top', 'top');
  expect(garmentContextReason(knit, formal)).toBe('formal_requires_dress_base');
  expect(rejectForStyleOccasion(knit, 'semi_classic', 'Formal')).toBe('formal_requires_dress_base');
  const woven = item('White Cotton Poplin Shirt', 'top', 'top');
  expect(garmentContextReason(woven, formal)).toBeNull();
  expect(rejectForStyleOccasion(woven, 'semi_classic', 'Formal')).toBeNull();
});

test('strict formal rejects casual corduroy tailoring on every slot', () => {
  for (const [name, slot] of [
    ['Olive Corduroy Overshirt Jacket', 'outerwear'],
    ['Tan Pleated Corduroy Trousers', 'bottom'],
  ] as const) {
    const reason = garmentContextReason(item(name, slot), formal);
    expect(reason).toBe('formal_rejects_casual_fabric');
    expect(rejectForStyleOccasion(item(name, slot), 'semi_classic', 'Formal')).toBe(reason);
  }
  const oxford = item('Black Leather Oxford Shoes', 'shoes', 'shoes');
  expect(garmentContextReason(oxford, formal)).toBeNull();
});
