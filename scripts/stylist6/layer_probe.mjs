// Step-2 probe: how fashion_deck resolves layer policy across request shapes.
// Usage: node scripts/stylist6/layer_probe.mjs
import { recommendFashionDeck, fashionStylingContext } from '../../backend/stylist6/fashion_deck.mjs';

const garment = (id, name, macroCategory, subtype, extra = {}) => ({
  id, name, macroCategory, subtype, isShopItem: true, currency: 'UZS', price: 100,
  imageUrl: 'https://catalog.example/image/' + id + '.jpg', shopUrl: 'https://catalog.example/' + id, ...extra,
});
const catalog = [
  ...['white', 'navy', 'cream'].map((c) => garment('shirt-' + c, c + ' cotton shirt', 'top', 'shirt')),
  ...['polo-white', 'polo-navy'].map((id) => garment(id, id.replace('-', ' ') + ' pique polo shirt', 'top', 'polo')),
  ...['grey', 'navy'].map((c) => garment('trouser-' + c, c + ' tailored suit trousers', 'bottom', 'tailored-trouser')),
  ...['chino-khaki', 'chino-navy'].map((id) => garment(id, id.replace('-', ' ') + ' chino pants', 'bottom', 'chino')),
  ...['brown', 'black'].map((c) => garment('derby-' + c, c + ' leather derby shoes', 'shoes', 'derby')),
  ...['white', 'navy'].map((c) => garment('sneaker-' + c, c + ' minimal leather sneakers', 'shoes', 'minimal-sneaker')),
  garment('locked-outer', 'Navy wool blazer', 'outerwear', 'blazer'),
  garment('blazer-grey', 'Grey wool blazer', 'outerwear', 'blazer'),
  garment('cardigan', 'Camel knitted cardigan', 'outerwear', 'cardigan'),
  garment('parka', 'Insulated navy winter parka', 'outerwear', 'jacket'),
  // Owned wardrobe pieces (no shop link/price) for the mixed-source case.
  { ...garment('own-shirt', 'Owned white linen shirt', 'top', 'shirt'), isShopItem: false, isWardrobe: true },
  { ...garment('own-chino', 'Owned beige chino pants', 'bottom', 'chino'), isShopItem: false, isWardrobe: true },
  { ...garment('own-sneaker', 'Owned white canvas sneakers', 'shoes', 'sneaker'), isShopItem: false, isWardrobe: true },
];
const refs = [{
  id: 'rt_001', referenceId: 'rt_001', name: 'Shirt and trousers',
  roles: { top: ['shirt'], outerwear: ['blazer'], bottom: ['tailored-trouser'], shoes: ['derby'] },
  layerPolicy: 'optional', palette: ['navy', 'white', 'brown', 'grey'],
  garmentNotes: { top: 'white shirt', bottom: 'grey trousers', shoes: 'brown leather derby' },
}];
const base = {
  catalog, referenceBriefs: refs, hardGate: async () => ({ ok: true }),
  priceToBudgetCurrency: (item, currency) => (item.currency === currency ? item.price : NaN),
};

const cases = [
  ['29C casual date, optionalLayer true', { style: 'casual', occasion: 'Date night', weather: { temp: 29 }, optionalLayer: true }],
  ['29C casual date, optionalLayer UNSPECIFIED', { style: 'casual', occasion: 'Date night', weather: { temp: 29 } }],
  ['23C business casual, optionalLayer true', { style: 'business_casual', occasion: 'Work', weather: { temp: 23 }, optionalLayer: true }],
  ['23C business casual, optionalLayer UNSPECIFIED', { style: 'business_casual', occasion: 'Work', weather: { temp: 23 } }],
  ['formal wedding, optionalLayer true', { style: 'classic', occasion: 'Wedding', weather: { temp: 21 }, optionalLayer: true }],
  ['cold outdoor 9C, optionalLayer true', { style: 'casual', occasion: 'Travel', weather: { temp: 9 }, optionalLayer: true }],
  ['cold outdoor 9C, optionalLayer UNSPECIFIED', { style: 'casual', occasion: 'Travel', weather: { temp: 9 } }],
  ['explicit no-layer, optionalLayer true', { style: 'casual', occasion: 'Weekend', weather: { temp: 27 }, brief: 'No outerwear', optionalLayer: true }],
  ['explicit no-layer, optionalLayer UNSPECIFIED', { style: 'casual', occasion: 'Weekend', weather: { temp: 27 }, brief: 'No outerwear' }],
  ['locked outerwear, optionalLayer true warm', { style: 'casual', occasion: 'Weekend', weather: { temp: 27 }, optionalLayer: true, requiredIds: ['locked-outer'] }],
  ['locked outerwear, optionalLayer UNSPECIFIED', { style: 'casual', occasion: 'Weekend', weather: { temp: 27 }, requiredIds: ['locked-outer'] }],
  ['mixed wardrobe+catalog, optionalLayer true', { style: 'casual', occasion: 'Weekend', weather: { temp: 22 }, sourceMode: 'mixed', optionalLayer: true }],
];

for (const [label, overrides] of cases) {
  const request = {
    sourceMode: 'catalog', style: 'casual', occasion: 'Weekend', weather: { temp: 20 },
    preferences: {}, brief: '', referenceId: 'rt_001', ...overrides,
  };
  const result = await recommendFashionDeck({ ...base, request });
  const context = fashionStylingContext(request);
  const layered = result.outfits.filter((o) => o.items.some((i) => i.macroCategory === 'outerwear')).length;
  const locked = result.outfits.filter((o) => o.items.some((i) => i.id === 'locked-outer')).length;
  const slots = result.outfits.map((o) => o.items.map((i) => i.macroCategory).sort().join('+')).join(' , ');
  console.log(
    `${label.padEnd(48)} status=${result.status.padEnd(16)} dressed=${result.outfits.length} layered=${layered} lockedKept=${locked} optionalLayer=${context.optionalLayer}`,
  );
  console.log(`    slots: ${slots || '(none)'}`);
}
