import { resolveCalendarSlots, stylistCalendarItems } from '../../features/calendar/outfitSlots';
import { validateOutfitItem } from '../../features/calendar/types';

const screenshotItems = [
  { id: 'polo', type: 'tops', macroCategory: 'top', name: 'Polo Piquet 1', image: 'https://test/polo.png' },
  { id: 'jacket', type: 'bottoms', macroCategory: 'outerwear', name: 'Denim Shacket | Washed Black', image: 'https://test/jacket.png' },
  { id: 'shoes', type: 'footwear', macroCategory: 'shoes', name: 'Bexford Leather Oxfords - Black', image: 'https://test/shoes.png' },
  { id: 'pants', type: 'lower_body', macroCategory: 'bottom', name: '5 pockets pant Skinny', image: 'https://test/pants.png' },
];

function expectFourSlots(items: typeof screenshotItems) {
  const slots = resolveCalendarSlots(items);
  expect(slots.layer?.id).toBe('jacket');
  expect(slots.top?.id).toBe('polo');
  expect(slots.pants?.id).toBe('pants');
  expect(slots.shoes?.id).toBe('shoes');
  expect(slots.extras).toEqual([]);
}

test('the screenshot outfit keeps pants in the main grid and shacket in outerwear', () => {
  expectFourSlots(screenshotItems);
});

test('previously saved outfits recover correct slots without needing to regenerate', () => {
  const legacy = screenshotItems.map(({ macroCategory: _macro, ...item }) => item);
  const slots = resolveCalendarSlots(legacy);
  expect(slots.layer?.id).toBe('jacket');
  expect(slots.pants?.id).toBe('pants');
  expect(slots.shoes?.id).toBe('shoes');
  expect(slots.top?.id).toBe('polo');
  expect(slots.extras).toEqual([]);
});

test('save/read validation retains all four item identities and canonical slots', () => {
  const saved = stylistCalendarItems(screenshotItems);
  const loaded = JSON.parse(JSON.stringify(saved)).map(validateOutfitItem);
  expect(loaded).toHaveLength(4);
  expectFourSlots(loaded);
  expect(loaded.map((item: { id: string }) => item.id)).toEqual(screenshotItems.map(item => item.id));
});

test.each([{ items: screenshotItems.slice(0, 3) }, { items: [...screenshotItems, { ...screenshotItems[3], id: 'extra-pants' }] }])('stylist save rejects partial or extra-item outfits', ({ items }) => {
  expect(() => stylistCalendarItems(items)).toThrow('Invalid stylist calendar outfit');
});
