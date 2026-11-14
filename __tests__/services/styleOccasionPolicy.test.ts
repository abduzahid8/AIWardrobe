import { passesStyleJudge } from '../../src/domain/occasionRules';

const item = (id: string, macroCategory: string, name: string) => ({
  id,
  macroCategory,
  name,
  type: name,
  color: 'navy',
});

const workLook = () => [
  item('top', 'top', 'Fine Knit Polo'),
  item('layer', 'outerwear', 'Fine Knit Cardigan'),
  item('bottom', 'bottom', 'Tailored Trousers'),
  item('shoes', 'shoes', 'Leather Loafers'),
];

it.each([
  ['minimalist', 'work'],
  ['classic', 'work'],
  ['old_money', 'work'],
])('%s remains compatible with work knit layering', (style, occasion) => {
  expect(passesStyleJudge(workLook(), style, occasion)).toBe(true);
});

it('keeps classic date looks separate from boardroom-only silhouettes', () => {
  expect(passesStyleJudge([
    item('top', 'top', 'Oxford Shirt'),
    item('bottom', 'bottom', 'Tailored Trousers'),
    item('shoes', 'shoes', 'Leather Loafers'),
  ], 'classic', 'date')).toBe(true);
});
