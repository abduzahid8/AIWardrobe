import { outfitCoherence } from '../../src/domain/outfitCoherence';
import { stylingContext } from '../../src/domain/stylingContext';
const look = (names: string[]) =>
  names.map((name, i) => ({
    id: `piece-${i}`,
    name,
    macroCategory: ['top', 'outerwear', 'bottom', 'shoes'][i],
  }));
const formal = [
  'White poplin dress shirt',
  'Unlined wool blazer',
  'Navy tailored wool trousers',
  'Black leather Oxford shoes',
];
it('admits a properly structured formal quartet', () => {
  const q = outfitCoherence(
    look(formal),
    stylingContext({ style: 'semi_classic', occasion: 'Formal' })
  );
  expect(q.valid).toBe(true);
  expect(q.dimensions.dressCode).toBe(1);
  expect(q.dimensions.footwear).toBe(1);
});
it('rejects athletic stripes and drawstrings in strict Formal independent of style', () => {
  for (const style of ['semi_classic', 'minimalist', 'casual'])
    for (const bottom of ['Stripe track pants', 'Drawstring training trousers']) {
      const parts = [...formal];
      parts[2] = bottom;
      expect(
        outfitCoherence(look(parts), stylingContext({ style, occasion: 'Formal' })).reasons
      ).toContain('formal_requires_tailored_bottom');
    }
});
it('hiking boots fail smart date context but remain possible for explicit hiking', () => {
  const parts = [...formal];
  parts[3] = 'Chunky waterproof hiking boots with lug sole';
  expect(
    outfitCoherence(look(parts), stylingContext({ style: 'minimalist', occasion: 'Date night' }))
      .valid
  ).toBe(false);
  expect(
    outfitCoherence(
      look(parts),
      stylingContext({ style: 'casual', occasion: 'Travel', request: 'hiking trail' })
    ).valid
  ).toBe(true);
});
it('does not call cardigan + sweater + loafers strict formal', () => {
  expect(
    outfitCoherence(
      look(['Fine sweater', 'Fine cardigan', 'Tailored trousers', 'Leather loafers']),
      stylingContext({ occasion: 'Formal' })
    ).valid
  ).toBe(false);
});
it('accepts casual tee / overshirt / jeans / sneakers without climate invention', () => {
  const c = stylingContext({ style: 'casual', occasion: 'Weekend' });
  const q = outfitCoherence(
    look(['Cotton T-shirt', 'Lightweight overshirt', 'Straight jeans', 'Clean leather sneakers']),
    c
  );
  expect(q.valid).toBe(true);
  expect(c.weather).toBeNull();
});
it('never uses a locked anchor to waive dress code', () => {
  const parts = look(formal);
  parts[3].name = 'Hiking boots';
  expect(
    outfitCoherence(parts, stylingContext({ occasion: 'Formal', anchorId: 'piece-3' })).valid
  ).toBe(false);
});
it('minimalist monochrome is valid; new listing IDs do not change score', () => {
  const parts = look([
    'Black cotton shirt',
    'Black fine cardigan',
    'Black tailored trousers',
    'Black sleek dress boots',
  ]);
  const c = stylingContext({ style: 'minimalist', occasion: 'Date night' });
  const a = outfitCoherence(parts, c),
    b = outfitCoherence(
      parts.map((p, i) => ({ ...p, id: `another-${i}` })),
      c
    );
  expect(a.valid).toBe(true);
  expect(a.score).toBe(b.score);
});
it('rejects dress shoes and tailoring for workout', () => {
  expect(outfitCoherence(look(formal), stylingContext({ occasion: 'Workout' })).valid).toBe(false);
});
it('insulated puffers cannot pair with summer espadrilles even without a temperature', () => {
  const items = look([
    'Cotton shirt',
    'Down jacket with hood',
    'Straight jeans',
    'Leather espadrilles',
  ]);
  expect(
    outfitCoherence(items, stylingContext({ style: 'casual', occasion: 'Weekend' })).reasons
  ).toContain('insulated_layer_with_summer_footwear');
});
it('corduroy tailoring stays possible for casual dates but is not strict formal suiting', () => {
  const items = look([
    'White dress shirt',
    'Tailored corduroy jacket',
    'Pleated corduroy trousers',
    'Leather Oxford shoes',
  ]);
  expect(
    outfitCoherence(items, stylingContext({ style: 'semi_classic', occasion: 'Formal' })).reasons
  ).toContain('formal_rejects_casual_fabric');
  expect(
    outfitCoherence(items, stylingContext({ style: 'casual', occasion: 'Date night' })).reasons
  ).not.toContain('formal_rejects_casual_fabric');
});
it('strict Formal does not promote every nominal loafer to a dress shoe', () => {
  const generic = look([
    'White dress shirt',
    'Wool blazer',
    'Tailored wool trousers',
    'Leather loafers',
  ]);
  expect(outfitCoherence(generic, stylingContext({ occasion: 'Formal' })).reasons).toContain(
    'formal_requires_dress_footwear'
  );
  const dress = generic.map((i, n) =>
    n === 3 ? { ...i, name: 'Structured leather dress loafers with leather soles' } : i
  );
  expect(outfitCoherence(dress, stylingContext({ occasion: 'Formal' })).valid).toBe(true);
});
it('carries explicit negative preferences into whole-look validation without banning unknown evidence', () => {
  const items = look(['Red cotton shirt', 'Fine cardigan', 'Chino trousers', 'Leather loafers']);
  expect(
    outfitCoherence(items, stylingContext({ preferences: { avoidColors: ['red'] } })).reasons
  ).toContain('excluded_color');
  const striped = items.map((i, n) => (n === 0 ? { ...i, name: 'Blue striped cotton shirt' } : i));
  expect(
    outfitCoherence(striped, stylingContext({ preferences: { dislikedPatterns: ['striped'] } }))
      .reasons
  ).toContain('excluded_pattern');
  expect(
    outfitCoherence(
      items.map((i) => ({ ...i, name: 'Product' })),
      stylingContext({ preferences: { dislikedPatterns: ['solid'] } })
    ).reasons
  ).not.toContain('excluded_pattern');
});
