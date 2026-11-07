import type { GarmentEvidence } from '../../src/domain/garmentDescriptor';

const slots = ['top', 'outerwear', 'bottom', 'shoes'];
export const fixtureLook = (names: string[], prefix = 'qa'): (GarmentEvidence & {id: string})[] =>
  names.map((name, i) => ({id: `${prefix}-${i}`, name, macroCategory: slots[i]}));

/** Named garment evidence, not image-derived measurements or human ratings. */
export const stylingV193Fixtures = [
  {id: 'busy-plaid-paisley', accepted: false, names: ['Medium-contrast checked cotton shirt', 'Large busy paisley jacquard cardigan', 'Cream corduroy trousers', 'Black polished Derby shoes']},
  {id: 'competing-large-checks', accepted: false, names: ['Bold large plaid shirt', 'Large high-contrast checked cardigan', 'Cream cotton chinos', 'Black leather penny loafers']},
  {id: 'washed-double-denim-oxford', accepted: false, names: ['Blue denim shirt', 'Navy tailored wool blazer', 'Washed light-blue slim jeans', 'Brown polished Oxford shoes']},
  {id: 'pale-double-denim-oxford', accepted: false, names: ['Blue denim shirt', 'Navy tailored wool blazer', 'Light blue slim jeans', 'Brown polished Oxford shoes']},
  {id: 'airy-heavy-winter', accepted: false, names: ['Lightweight linen popover shirt', 'Heavyweight winter wool-cashmere blazer', 'Cream corduroy trousers', 'Brown leather penny loafers']},
  {id: 'quiet-pattern-mix', accepted: true, names: ['Fine subtle striped poplin shirt', 'Subtle small windowpane wool blazer', 'Charcoal tailored trousers', 'Brown leather penny loafers']},
  {id: 'tonal-jacquard-texture', accepted: true, names: ['Fine subtle striped cotton shirt', 'Tonal textured jacquard cardigan', 'Navy cotton chinos', 'Brown leather penny loafers']},
  {id: 'single-paisley-accent', accepted: true, names: ['Plain white poplin shirt', 'Paisley jacquard cardigan', 'Navy cotton chinos', 'Brown leather penny loafers']},
  {id: 'clean-smart-denim', accepted: true, names: ['White cotton shirt', 'Unstructured navy cotton blazer', 'Dark clean straight denim jeans', 'Brown suede loafers']},
  {id: 'dark-denim-shirt-tailoring', accepted: true, names: ['Dark clean denim shirt', 'Unstructured navy cotton blazer', 'Grey tailored cotton trousers', 'Brown suede loafers']},
  {id: 'linen-medium-wool-unknown-weather', accepted: true, names: ['Linen popover shirt', 'Regular wool-cashmere cardigan', 'Cream corduroy trousers', 'Brown leather penny loafers']},
  {id: 'refined-neutral', accepted: true, names: ['White poplin shirt', 'Navy wool blazer', 'Grey tailored wool trousers', 'Black leather Oxford shoes with slim rubber soles']},
  {id: 'casual-travel', accepted: true, occasion: 'Travel', style: 'casual', names: ['Black plain cotton T-shirt', 'Olive cotton casual jacket', 'Grey straight jeans', 'Black casual leather sneakers']},
  {id: 'graphic-denim-casual', accepted: true, occasion: 'Casual', style: 'casual', names: ['Graphic cotton T-shirt', 'Blue denim jacket', 'Black straight jeans', 'Black casual leather sneakers']},
];
