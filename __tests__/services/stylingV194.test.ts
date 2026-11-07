/** @jest-environment node */
import {assessOutfitQuality} from '../../src/domain/occasionRules';
import {describeGarment} from '../../src/domain/garmentDescriptor';
import {fixtureLook} from '../fixtures/stylingV193';

const assess = (names: string[], style = 'casual', occasion = 'Travel') =>
  assessOutfitQuality(fixtureLook(names), style, occasion);

test.each([
  ['old_money', 'Work', ['Dark casual Tencel shirt', 'Short wool zip jacket', 'Tailored trousers', 'Black leather dress loafers']],
  ['semi_classic', 'Date night', ['Light Tencel shirt', 'Dark Blue Denim Leisure Blazer', 'Washed skinny jeans', 'Formal black Derby shoes']],
  ['casual', 'Travel', ['Oversized graphic T-shirt', 'Structured navy blazer', 'Brown tailored chinos', 'Bright green retro sneakers']],
  ['casual', 'Travel', ['Black T-shirt', 'Grey knitted zip jacket', 'Blue jeans', 'Technical cross-training sneakers']],
  ['casual', 'Travel', ['Black T-shirt', 'Grey knitted zip jacket', 'Blue jeans', 'GECKO TRAIN CROSS TRAINERS']],
])('observed register conflict: %s / %s / %j', (style, occasion, names) => {
  expect(assess(names, style, occasion).acceptance.passed).toBe(false);
});

test('sparse exact catalog row retains manufacturer-confirmed drawstring evidence', () => {
  const items = fixtureLook(['Navy short-sleeved shirt', 'Structured double-breasted blazer', 'Beige Flannel Trousers', 'Black leather loafers']);
  items[2].id = 'shop_aurelien-15179534139722';
  expect(assessOutfitQuality(items, 'old_money', 'Work').acceptance.passed).toBe(false);
  expect(describeGarment(items[2]).evidence).toContain('catalog:verified-drawstring');
  expect(describeGarment({...items[2], id: 'unrelated-flannel'}).athletic).toBe(false);
});

test('loose denim with checked knit layers and dress loafers ranks below a balanced date look', () => {
  const weak = assess(['Blue checked shirt', 'Navy merino sweater', 'White loose-fit jeans', 'Black dress loafers'], 'semi_classic', 'Date night');
  const good = assess(['Blue subtle checked shirt', 'Navy fine merino sweater', 'Grey straight cotton chinos', 'Brown suede loafers'], 'semi_classic', 'Date night');
  expect(good.acceptance.passed).toBe(true);
  expect(weak.score).toBeLessThan(good.score);
  expect(weak.evidence).toContain('wide_denim_hem_with_dress_footwear');
  expect(describeGarment({name: 'White loose-fit jeans', macroCategory: 'bottom'}).volume).toBe('wide');
});

test.each([
  ['semi_classic', 'Date night', ['Light Tencel shirt', 'Dark Blue Denim Leisure Blazer', 'Dark clean straight jeans', 'Brown suede loafers']],
  ['semi_classic', 'Date night', ['White poplin shirt', 'Unstructured navy cotton blazer', 'Garment-washed dark straight jeans', 'Brown suede loafers']],
  ['old_money', 'Work', ['White poplin shirt', 'Navy wool blazer', 'Grey tailored wool trousers', 'Black leather penny loafers']],
  ['old_money', 'Work', ['Plain white shirt', 'Short wool zip jacket', 'Grey tailored trousers', 'Brown suede loafers']],
  ['casual', 'Travel', ['Plain regular cotton T-shirt', 'Unstructured navy cotton blazer', 'Brown straight chinos', 'White minimal leather sneakers']],
  ['casual', 'Travel', ['Oversized graphic T-shirt', 'Relaxed denim jacket', 'Black straight jeans', 'Bright green retro sneakers']],
  ['casual', 'Travel', ['Black T-shirt', 'Grey knitted zip jacket', 'Blue jeans', 'Black cushioned lifestyle sneakers']],
  ['casual', 'Travel', ['Plain cotton shirt', 'Relaxed cotton jacket', 'Beige drawstring flannel trousers', 'White lifestyle sneakers']],
  ['casual', 'Travel with a gym workout', ['Black athletic T-shirt', 'Grey technical track jacket', 'Black training joggers', 'Technical cross-training sneakers']],
])('positive control: %s / %s / %j', (style, occasion, names) => {
  expect(assess(names, style, occasion).acceptance.passed).toBe(true);
});
