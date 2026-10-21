import type { ShopCatalogItem } from '../../src/types/shop';

/** Synthetic live-row shapes for offline integration tests. Never shipped as catalog data.
 * Live inventory availability must be verified separately against staging.
 */
const groups: Array<[string, string[]]> = [
  ['upper_body', ['White Oxford shirt', 'Light blue cotton shirt', 'Cream knit polo', 'Navy knit polo', 'Regular-fit knit cotton t-shirt', 'White cotton t-shirt', 'Navy cotton t-shirt', 'Beige cotton shirt', 'Black cotton t-shirt']],
  ['lower_body', ['Grey wool trousers', 'Navy wool trousers', 'Beige chinos', 'Navy chinos', 'Charcoal dress trousers', 'Dark rinse jeans', 'Blue jeans', 'Black cotton trousers', 'Khaki shorts']],
  ['shoes', ['Brown penny loafers', 'Black derby shoes', 'Brown leather loafers', 'Black Oxford shoes', 'Brown Chelsea boots', 'White leather sneakers', 'Black canvas sneakers', 'Navy sneakers', 'Tan leather sandals']],
  ['upper_body', ['Navy wool blazer', 'Charcoal wool blazer', 'Camel wool overcoat', 'Navy wool topcoat', 'Beige cotton trench coat', 'Grey merino cardigan', 'Navy bomber jacket', 'Grey hoodie', 'Black quilted jacket']],
];

export const TEST_SHOP_CATALOG = groups.flatMap(([garmentType, names], group) =>
  names.flatMap((name, index) => [0, 1].map(variant => ({
    id: `fixture-${group}-${index}-${variant}`,
    name,
    garmentType,
    description: name,
    brand: 'Test catalog',
    imageUrl: `https://catalog.example.test/${group}-${index}-${variant}.jpg`,
    price: 49,
    currency: 'USD',
    productUrl: `https://catalog.example.test/products/${group}-${index}-${variant}`,
  } as ShopCatalogItem))),
);
