/** @jest-environment node */
// Whole-outfit formality consistency for refined style + occasion pairs.
// Uses the November 6 simulator cases as fixtures: Classic + Work must not
// ship trainers, athletic bottoms or vacation pieces; Old Money + Formal
// must not ship a checked base or a leisure-cut layer. The same garments
// stay legal in other contexts (no global bans).
const path = require('path');

const domain = path.resolve(__dirname, '../../src/domain');
const coherence = require(path.join(domain, 'outfitCoherence.ts'));
const styling = require(path.join(domain, 'stylingContext.ts'));

const contextFor = (style, occasion) => styling.stylingContext({ style, occasion });

const classicWork = () => contextFor('classic', 'Work');
const oldMoneyFormal = () => contextFor('old_money', 'Formal');

test('classic work rejects trainers while casual keeps them', () => {
  const look = [
    { id: 't1', name: 'White cotton dress shirt', macroCategory: 'top' },
    { id: 'l1', name: 'Navy wool blazer', macroCategory: 'outerwear' },
    { id: 'b1', name: 'Charcoal wool trousers', macroCategory: 'bottom' },
    { id: 's1', name: 'Barefoot Leather Trainers', macroCategory: 'shoes' },
  ];
  expect(coherence.wholeOutfitRegisterClash(look, classicWork()))
    .toBe('classic_work_requires_non_athletic_footwear');
  expect(coherence.wholeOutfitRegisterClash(look, contextFor('casual', 'Casual'))).toBeNull();
});

test('classic work rejects resort layers and drawstring seaside bottoms', () => {
  const resort = [
    { id: 't1', name: 'White linen shirt', macroCategory: 'top' },
    { id: 'l1', name: 'Beige resort camp-collar jacket', macroCategory: 'outerwear' },
    { id: 'b1', name: 'Navy chinos', macroCategory: 'bottom' },
    { id: 's1', name: 'Brown leather loafers', macroCategory: 'shoes' },
  ];
  expect(coherence.wholeOutfitRegisterClash(resort, classicWork()))
    .toBe('classic_work_rejects_vacation_pieces');
  const drawstring = [
    { id: 't2', name: 'White cotton dress shirt', macroCategory: 'top' },
    { id: 'l2', name: 'Navy wool blazer', macroCategory: 'outerwear' },
    { id: 'b2', name: 'Drawstring seaside linen trousers', macroCategory: 'bottom' },
    { id: 's2', name: 'Brown leather loafers', macroCategory: 'shoes' },
  ];
  const bottomClash = coherence.wholeOutfitRegisterClash(drawstring, classicWork());
  expect(['classic_work_requires_non_athletic_bottom', 'classic_work_rejects_vacation_pieces'])
    .toContain(bottomClash);
});

test('old money formal rejects checked base and leisure blazer, keeps dress shirt tailoring', () => {
  const checked = [
    { id: 't1', name: 'Bower Regular-Fit Check Shirt', macroCategory: 'top' },
    { id: 'l1', name: 'Navy Cashmere Blend Leisure Blazer', macroCategory: 'outerwear' },
    { id: 'b1', name: 'Navy high-waisted pleatless wool pants', macroCategory: 'bottom' },
    { id: 's1', name: 'Lincoln Black leather loafers', macroCategory: 'shoes' },
  ];
  const clash = coherence.wholeOutfitRegisterClash(checked, oldMoneyFormal());
  expect(['formal_checked_base_breaks_dress_shirt_line', 'formal_leisure_layer_breaks_structured_tailoring'])
    .toContain(clash);
  const dressy = [
    { id: 't2', name: 'White cotton poplin dress shirt', macroCategory: 'top' },
    { id: 'l2', name: 'Navy structured wool blazer', macroCategory: 'outerwear' },
    { id: 'b2', name: 'Charcoal tailored wool trousers', macroCategory: 'bottom' },
    { id: 's2', name: 'Black leather Oxford shoes', macroCategory: 'shoes' },
  ];
  expect(coherence.wholeOutfitRegisterClash(dressy, oldMoneyFormal())).toBeNull();
});
