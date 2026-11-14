import type { ShopCatalogItem } from '../types/shop';
import { buildCatalogFamilyViews, selectCatalogVariant } from './catalogFamilyView';
import { rankDiverseShoppingCandidates } from './shoppingIntelligence';
import {
  normalizeCatalogImageKey,
  normalizeCatalogNameKey,
  buildCatalogSimilarityKey,
  getCatalogShoeFamily,
} from '../utils/shopCatalogOrder';
type EssentialSlot = 'outerwear' | 'shirts' | 'knitwear' | 'tees' | 'bottoms' | 'shoes';

const ESSENTIALS_REQUIREMENTS: Record<EssentialSlot, number> = {
  outerwear: 2, // blazers, jackets, coats
  shirts: 2, // dress shirts, casual shirts
  knitwear: 2, // sweaters, cardigans
  tees: 2, // t-shirts, polos
  bottoms: 3, // pants, chinos, jeans, shorts
  shoes: 3, // sneakers, loafers, boots
};

const ESSENTIALS_LIMIT = Object.values(ESSENTIALS_REQUIREMENTS).reduce(
  (sum, count) => sum + count,
  0
);

// Keywords for better classification
const OUTERWEAR_KEYWORDS = [
  'blazer',
  'jacket',
  'coat',
  'overcoat',
  'suit',
  'trench',
  'parka',
  'bomber',
  'puffer',
  'vest',
  'waistcoat',
  'cardigan',
  'hoodie',
];

const SHIRT_KEYWORDS = [
  'shirt',
  'oxford',
  'dress shirt',
  'broadcloth',
  'poplin',
  'linen shirt',
  'flannel',
  'chambray',
  'button-down',
  'button down',
];

const KNITWEAR_KEYWORDS = [
  'sweater',
  'knit',
  'pullover',
  'turtleneck',
  'crewneck',
  'v-neck',
  'cashmere',
  'merino',
  'wool',
  'chunky',
  'cable',
  'cardigan',
];

const TEE_KEYWORDS = ['t-shirt', 'tshirt', 'tee', 'polo', 'henley', 'tank', 'top'];

const BOTTOM_KEYWORDS = [
  'pants',
  'trousers',
  'chinos',
  'jeans',
  'shorts',
  'joggers',
  'slacks',
  'cargos',
  'corduroy',
];

const SHOE_KEYWORDS = [
  'shoes',
  'sneakers',
  'loafers',
  'boots',
  'oxfords',
  'derby',
  'trainers',
  'slip-on',
  'mules',
];

export const classifyUpperBodyItem = (
  item: ShopCatalogItem
): 'outerwear' | 'shirts' | 'knitwear' | 'tees' | null => {
  const text = `${item.name} ${item.description || ''}`.toLowerCase();

  // Check in priority order
  if (OUTERWEAR_KEYWORDS.some((k) => text.includes(k))) return 'outerwear';
  if (SHIRT_KEYWORDS.some((k) => text.includes(k))) return 'shirts';
  if (KNITWEAR_KEYWORDS.some((k) => text.includes(k))) return 'knitwear';
  if (TEE_KEYWORDS.some((k) => text.includes(k))) return 'tees';

  // Default classification based on common patterns
  if (text.includes('blazer') || text.includes('jacket') || text.includes('coat'))
    return 'outerwear';
  if (text.includes('shirt')) return 'shirts';
  if (text.includes('sweater') || text.includes('knit')) return 'knitwear';

  return 'tees'; // default
};

const classifyBottomItem = (item: ShopCatalogItem): 'bottoms' | null => {
  const text = `${item.name} ${item.description || ''}`.toLowerCase();
  if (BOTTOM_KEYWORDS.some((k) => text.includes(k))) return 'bottoms';
  if (item.garmentType === 'lower_body') return 'bottoms';
  return null;
};

const classifyShoeItem = (item: ShopCatalogItem): 'shoes' | null => {
  const text = `${item.name} ${item.description || ''}`.toLowerCase();
  if (SHOE_KEYWORDS.some((k) => text.includes(k))) return 'shoes';
  if (item.garmentType === 'shoes') return 'shoes';
  return null;
};

/**
 * Deterministic seeded PRNG (mulberry32).
 * Returns a pseudo-random number in [0, 1) for a given 32-bit seed integer.
 * Using a stable daily seed (new Date().toDateString()) ensures the shuffle
 * produces the same order for the entire day, keeping useMemo results stable
 * across renders within the same day (fixes Defect 1.1).
 */
const seededRandom = (seed: number): (() => number) => {
  let s = seed >>> 0;
  return () => {
    s += 0x6d2b79f5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) >>> 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

/** Convert a string seed to a 32-bit integer via a simple hash. */
const hashStrToInt = (str: string): number => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (Math.imul(31, hash) + str.charCodeAt(i)) | 0;
  }
  return hash;
};

export const selectEssentialShoppingMix = (
  items: ShopCatalogItem[],
  dailySeed: string
): ShopCatalogItem[] => {
  const pickedCounts: Record<EssentialSlot, number> = {
    outerwear: 0,
    shirts: 0,
    knitwear: 0,
    tees: 0,
    bottoms: 0,
    shoes: 0,
  };

  const selected: ShopCatalogItem[] = [];
  const selectedIds = new Set<string>();
  // Cross-source / near-duplicate guards. Catalog `id` alone is not enough:
  // the same shoe from `manual` + `apify-massimodutti` + `apify-zara-men-uz`
  // (or two static variants like classic-m-shoe-16/17 "Leather loafers with
  // penny strap") has different ids but the same name/image. Without these
  // sets the `shoes: 3` quota fills with 3 visually identical loafers.
  const seenImages = new Set<string>();
  const seenNames = new Set<string>();
  const seenSimilarityKeys = new Set<string>();
  const seenShoeFamilies = new Set<string>();

  // Shuffle items for daily variety using a stable seed so the result is
  // identical across all renders within the same calendar day.
  // Fisher–Yates O(n) — the old `.sort(() => rand() - 0.5)` was O(n log n)
  // with a biased comparator, run on every catalog refresh.
  const rand = seededRandom(hashStrToInt(dailySeed));
  const shuffled = buildCatalogFamilyViews(items).families.map((family) =>
    selectCatalogVariant(family)!
  );
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }

  const getSlot = (item: ShopCatalogItem): EssentialSlot | null => {
    if (item.garmentType === 'upper_body') {
      return classifyUpperBodyItem(item);
    } else if (item.garmentType === 'lower_body') {
      return classifyBottomItem(item);
    } else if (item.garmentType === 'shoes') {
      return classifyShoeItem(item);
    }
    return null;
  };

  const tryPick = (item: ShopCatalogItem, strictVariety: boolean): boolean => {
    if (!item || selectedIds.has(item.id)) return false;

    const slot = getSlot(item);
    if (!slot) return false;
    if (pickedCounts[slot] >= ESSENTIALS_REQUIREMENTS[slot]) return false;

    // Exact + near-duplicate guards (apply in both passes).
    const imageKey = normalizeCatalogImageKey((item as any).imageUrl);
    if (imageKey && seenImages.has(imageKey)) return false;
    const nameKey = normalizeCatalogNameKey((item as any).name);
    if (nameKey && seenNames.has(nameKey)) return false;
    const similarityKey = buildCatalogSimilarityKey({
      id: item.id,
      name: (item as any).name ?? '',
      garmentType: (item as any).garmentType,
    });
    if (seenSimilarityKeys.has(similarityKey)) return false;

    // Strict pass: at most one shoe family (loafer/sneaker/boot/…) so the
    // 3-shoe quota can't fill with 3 loafers that have slightly different
    // names ("oiled leather penny loafer" vs "leather loafer penny strap").
    if (strictVariety && slot === 'shoes') {
      const family = getCatalogShoeFamily((item as any).name);
      if (seenShoeFamilies.has(family)) return false;
    }

    selected.push(item);
    selectedIds.add(item.id);
    if (imageKey) seenImages.add(imageKey);
    if (nameKey) seenNames.add(nameKey);
    seenSimilarityKeys.add(similarityKey);
    if (slot === 'shoes') seenShoeFamilies.add(getCatalogShoeFamily((item as any).name));
    pickedCounts[slot] += 1;
    return true;
  };

  const diverse = rankDiverseShoppingCandidates(
    shuffled.map((item) => ({ item, score: 1 })),
    {
      limit: Math.min(100, shuffled.length),
      eligible: () => true,
    }
  );
  // Pass 1 (strict): enforce shoe-family variety.
  for (const item of diverse) {
    if (selected.length >= ESSENTIALS_LIMIT) break;
    tryPick(item, true);
  }
  // Pass 2 (relaxed fallback): if the catalog has only loafers, still fill
  // the quota rather than showing an incomplete grid — but exact duplicates
  // (same image/name/similarity) stay blocked.
  if (selected.length < ESSENTIALS_LIMIT) {
    for (const item of diverse) {
      if (selected.length >= ESSENTIALS_LIMIT) break;
      if (selectedIds.has(item.id)) continue;
      tryPick(item, false);
    }
  }

  return selected;
};
