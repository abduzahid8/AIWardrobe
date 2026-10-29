import { availableStylistCombinations } from '../../features/outfit-generator/utils/stylistAvailability';

const items = [
  { id: 'shirt', name: 'White cotton shirt', macroCategory: 'top' },
  { id: 'pants', name: 'Navy cotton trousers', macroCategory: 'bottom' },
  { id: 'shoes', name: 'Brown leather penny loafers', macroCategory: 'shoes' },
  { id: 'layer', name: 'Beige merino cardigan', macroCategory: 'outerwear' },
];
test('complete Old Money travel inventory is offered', () => {
  expect(availableStylistCombinations(items, ['old_money'], ['travel'])).toEqual([{ style: 'old_money', occasion: 'travel' }]);
});
test.each(['top', 'bottom', 'shoes', 'outerwear'])('missing %s hides an impossible combination', slot => {
  expect(availableStylistCombinations(items.filter(item => item.macroCategory !== slot), ['old_money'], ['travel'])).toEqual([]);
});
test('sport is hidden when the inventory only has refined daywear', () => {
  expect(availableStylistCombinations(items, ['old_money'], ['travel', 'sport'])).toEqual([{ style: 'old_money', occasion: 'travel' }]);
});
test('explicit no-layer requests work with three real categories', () => {
  expect(availableStylistCombinations(items.filter(item => item.macroCategory !== 'outerwear'), ['old_money'], ['travel'], undefined, 'no jacket')).toHaveLength(1);
});
test('a missing locked piece is never replaced to claim availability', () => {
  expect(availableStylistCombinations(items, ['old_money'], ['travel'], undefined, '', 'missing-anchor')).toEqual([]);
});
test('avoided colors cannot count toward an available complete outfit', () => {
  expect(availableStylistCombinations(items, ['old_money'], ['travel'], undefined, '', undefined, { avoidColors: ['navy'] })).toEqual([]);
});
