/**
 * personaHarness — shared engine for multi-account AI e2e.
 *
 * One implementation drives BOTH:
 *   - __tests__/e2e/aiPersonas.e2e.test.ts  (assertions)
 *   - scripts/ai-persona-report.ts           (HTML report + screenshots)
 * so the screenshots always show exactly what the tests verified.
 *
 * Each persona is a stand-in for a real account: own wardrobe, own style,
 * own occasion/weather. The pipeline uses ONLY production modules:
 * categoryMapper → styleInference (rank) → sanitizeGeneratedOutfit →
 * deckDedupe assembly gate (same rule as AIOutfitmaker).
 */

import { TEST_SHOP_CATALOG as SHOP_CATALOG_ITEMS } from '../fixtures/shopCatalog';
import { CLASSIC_MENS_ITEMS } from '../../data/inspoMensShopItems';
import {
  canonicalizeMacroCategory,
  getMacroCategory,
} from '../../src/utils/categoryMapper';
import {
  rankItemsForStyle,
  scoreItemForStyle,
  needsLayering,
  type StyleId,
} from '../../features/outfit-generator/utils/styleInference';
import { sanitizeGeneratedOutfitItemsDetailed } from '../../features/outfit-generator/utils/sanitizeGeneratedOutfit';
import { createDeckDedupe } from '../../features/outfit-generator/utils/deckDedupe';
import { inferShopRowMacro } from '../../src/utils/shopCatalogOrder';

export interface PersonaItem {
  id: string;
  name: string;
  type: string;
  category: string;
  macroCategory: string;
  color: string;
  brand: string;
  description: string;
  image: string;
  imageUrl: string;
  isWardrobe: boolean;
}

export interface Persona {
  account: string;
  style: StyleId;
  styleLabel: string;
  occasion: string;
  weather: { temp: number; condition: string };
  mode: 'wardrobe' | 'shop';
  wardrobe: PersonaItem[];
}

const img = (id: string) =>
  (SHOP_CATALOG_ITEMS as any[]).find((s) => String(s.id) === id)?.imageUrl ||
  (CLASSIC_MENS_ITEMS as any[]).find((s) => String(s.id) === id)?.imageUrl ||
  '';

function owned(
  id: string,
  name: string,
  type: string,
  category: string,
  macro: string,
  color: string,
  imageUrl: string,
): PersonaItem {
  return {
    id: `w-${id}`,
    name,
    type,
    category,
    macroCategory: macro,
    color,
    brand: 'Own closet',
    description: name,
    image: imageUrl,
    imageUrl,
    isWardrobe: true,
  };
}

/** Same order as AIOutfitmaker.shopRowMacro: explicit accessory type, then
 * outerwear identity, then top/bottom/shoe identity on the NAME only
 * (descriptions mention other garments), then trusted clothing types, then
 * accessory keywords for ambiguous rows only. */
export function slotFor(garmentType: string, name: string, description: string): string {
  return inferShopRowMacro(garmentType, name, description);
}

export function shopPool(): PersonaItem[] {
  const seen = new Set<string>();
  const out: PersonaItem[] = [];
  for (const s of [...(SHOP_CATALOG_ITEMS as any[]), ...(CLASSIC_MENS_ITEMS as any[])] as any[]) {
    const id = String(s?.id || '');
    if (!id || seen.has(id)) continue;
    seen.add(id);
    const macro = slotFor(s.garmentType, s.name || '', s.description || '');
    out.push({
      id,
      name: s.name || 'Shop item',
      type: macro,
      category: macro,
      macroCategory: macro,
      color: 'neutral',
      brand: s.brand || '',
      description: s.description || '',
      image: s.imageUrl || '',
      imageUrl: s.imageUrl || '',
      isWardrobe: false,
    });
  }
  return out;
}

const L = (id: string) => img(id);

export const PERSONAS: Persona[] = [
  {
    account: 'Amir · Old Money executive',
    style: 'old_money',
    styleLabel: 'Old Money',
    occasion: 'Work',
    weather: { temp: 12, condition: 'cold' },
    mode: 'shop',
    wardrobe: [
      owned('blazer', 'Navy Wool Blazer', 'blazer', 'outerwear', 'outerwear', 'navy', L('classic-m-blazer-06') || L('classic-m-shirt-12')),
      owned('oxford', 'White Oxford Shirt', 'shirt', 'tops', 'top', 'white', L('classic-m-shirt-12')),
      owned('polo', 'Navy Polo', 'polo', 'tops', 'top', 'navy', L('classic-m-shirt-13')),
      owned('tee', 'Cream Tee', 't-shirt', 'tops', 'top', 'cream', L('classic-m-shirt-14')),
      owned('chinos', 'Beige Chinos', 'chinos', 'bottoms', 'bottom', 'beige', L('classic-m-trouser-11')),
      owned('trousers', 'Grey Wool Trousers', 'trousers', 'bottoms', 'bottom', 'grey', L('classic-m-trouser-12')),
      owned('jeans', 'Dark Jeans', 'jeans', 'bottoms', 'bottom', 'navy', L('classic-m-jean-07')),
      owned('loafers', 'Brown Penny Loafers', 'loafers', 'shoes', 'shoes', 'brown', L('classic-m-shoe-12')),
      owned('derby', 'Black Derbies', 'derby shoes', 'shoes', 'shoes', 'black', L('classic-m-shoe-13')),
      owned('sneak', 'White Sneakers', 'sneakers', 'shoes', 'shoes', 'white', L('classic-m-shoe-14')),
    ],
  },
  {
    account: 'Sara · Casual student',
    style: 'casual',
    styleLabel: 'Casual',
    occasion: 'Casual',
    weather: { temp: 28, condition: 'clear' },
    mode: 'shop',
    wardrobe: [
      owned('tee1', 'White Cotton Tee', 't-shirt', 'tops', 'top', 'white', L('classic-m-shirt-14')),
      owned('tee2', 'Striped Tee', 't-shirt', 'tops', 'top', 'blue', L('classic-m-shirt-13')),
      owned('shirt', 'Denim Shirt', 'shirt', 'tops', 'top', 'blue', L('classic-m-shirt-12')),
      owned('hoodie', 'Grey Hoodie', 'hoodie', 'tops', 'outerwear', 'grey', L('classic-m-blazer-06')),
      owned('jeans', 'Blue Jeans', 'jeans', 'bottoms', 'bottom', 'blue', L('classic-m-jean-06')),
      owned('shorts', 'Khaki Shorts', 'shorts', 'bottoms', 'bottom', 'beige', L('classic-m-trouser-11')),
      owned('chinos', 'Olive Chinos', 'chinos', 'bottoms', 'bottom', 'olive', L('classic-m-trouser-12')),
      owned('sneak1', 'White Sneakers', 'sneakers', 'shoes', 'shoes', 'white', L('classic-m-shoe-14')),
      owned('sneak2', 'Canvas Sneakers', 'sneakers', 'shoes', 'shoes', 'black', L('classic-m-shoe-15')),
      owned('sandals', 'Leather Sandals', 'sandals', 'shoes', 'shoes', 'brown', L('classic-m-shoe-16')),
    ],
  },
  {
    account: 'Dmitriy · Business casual',
    style: 'business_casual',
    styleLabel: 'Business Casual',
    occasion: 'Work',
    weather: { temp: 15, condition: 'wind' },
    mode: 'wardrobe',
    wardrobe: [
      owned('blazer', 'Charcoal Blazer', 'blazer', 'outerwear', 'outerwear', 'charcoal', L('classic-m-blazer-07')),
      owned('cardigan', 'Camel Cardigan', 'cardigan', 'tops', 'outerwear', 'camel', L('classic-m-knit-03')),
      owned('shirt1', 'Light Blue Shirt', 'shirt', 'tops', 'top', 'blue', L('classic-m-shirt-13')),
      owned('shirt2', 'White Shirt', 'shirt', 'tops', 'top', 'white', L('classic-m-shirt-12')),
      owned('polo', 'Grey Polo', 'polo', 'tops', 'top', 'grey', L('classic-m-shirt-14')),
      owned('chinos', 'Navy Chinos', 'chinos', 'bottoms', 'bottom', 'navy', L('classic-m-trouser-11')),
      owned('trousers', 'Charcoal Trousers', 'trousers', 'bottoms', 'bottom', 'charcoal', L('classic-m-trouser-12')),
      owned('jeans', 'Dark Jeans', 'jeans', 'bottoms', 'bottom', 'navy', L('classic-m-jean-07')),
      owned('loafers', 'Bit Loafers', 'loafers', 'shoes', 'shoes', 'brown', L('classic-m-shoe-12')),
      owned('boots', 'Chelsea Boots', 'boots', 'shoes', 'shoes', 'black', L('classic-m-shoe-17')),
      owned('derby', 'Brown Derbies', 'derby shoes', 'shoes', 'shoes', 'brown', L('classic-m-shoe-13')),
    ],
  },
  {
    account: 'Lola · Minimalist',
    style: 'minimalist',
    styleLabel: 'Minimalist',
    occasion: 'Casual',
    weather: { temp: 20, condition: 'clear' },
    mode: 'wardrobe',
    wardrobe: [
      owned('knit', 'Oatmeal Knit', 'knit', 'tops', 'outerwear', 'beige', L('classic-m-knit-04')),
      owned('tee', 'White Tee', 't-shirt', 'tops', 'top', 'white', L('classic-m-shirt-14')),
      owned('shirt', 'Ecru Shirt', 'shirt', 'tops', 'top', 'cream', L('classic-m-shirt-13')),
      owned('tank', 'Black Tank', 'tank top', 'tops', 'top', 'black', L('classic-m-shirt-15')),
      owned('trousers', 'Black Trousers', 'trousers', 'bottoms', 'bottom', 'black', L('classic-m-trouser-12')),
      owned('jeans', 'Grey Jeans', 'jeans', 'bottoms', 'bottom', 'grey', L('classic-m-jean-05')),
      owned('skirt', 'Pleated Skirt', 'skirt', 'bottoms', 'bottom', 'beige', L('classic-m-trouser-11')),
      owned('sneak', 'White Minimal Sneakers', 'sneakers', 'shoes', 'shoes', 'white', L('classic-m-shoe-18')),
      owned('loafers', 'Black Loafers', 'loafers', 'shoes', 'shoes', 'black', L('classic-m-shoe-19')),
      owned('mules', 'Sand Mules', 'mules', 'shoes', 'shoes', 'beige', L('classic-m-shoe-20')),
    ],
  },
];

// Separate accounts own separate rows: namespace owned ids per persona so
// no two closets ever share an item identity.
PERSONAS.forEach((persona, pi) => {
  persona.wardrobe.forEach((item) => {
    item.id = `w-p${pi}-${item.id.replace(/^w-/, '')}`;
  });
});

export interface PersonaDeckItem extends PersonaItem {
  slot: string;
  score: number;
}

export interface PersonaResult {
  persona: Persona;
  layered: boolean;
  poolSize: number;
  deck: PersonaDeckItem[][];
  checks: { label: string; pass: boolean; detail: string }[];
}

/**
 * Run one account end-to-end through production modules.
 * Wardrobe mode = owned items only (Supabase clothing). Shop mode = live
 * catalog pool, owned items excluded (source-purity rule).
 */
export function runPersona(persona: Persona): PersonaResult {
  const catalog = shopPool();
  const available: PersonaItem[] =
    persona.mode === 'wardrobe'
      ? persona.wardrobe.filter((i) => i.isWardrobe)
      : catalog.filter((i) => !i.isWardrobe);
  const layered = needsLayering(persona.style, persona.weather, null);
  const ranked = rankItemsForStyle(available as any[], persona.style, {
    minKeep: 16,
    dropThreshold: -2,
    perCategoryFloor: 4,
  }) as PersonaItem[];

  const bySlot = (slot: string) =>
    ranked.filter((i) => canonicalizeMacroCategory(i.macroCategory) === slot);
  const tops = bySlot('top');
  const bottoms = bySlot('bottom');
  const shoes = bySlot('shoes');
  const outers = bySlot('outerwear');

  const dedupe = createDeckDedupe();
  const deck: PersonaDeckItem[][] = [];
  const rotations = Math.max(tops.length, bottoms.length, shoes.length, 1);
  for (let r = 0; r < rotations && deck.length < 3; r++) {
    const pick = <T>(arr: T[]): T | undefined =>
      arr.length ? arr[r % arr.length] : undefined;
    const chosen = [
      layered ? pick(outers.length ? outers : tops) : undefined,
      pick(tops),
      pick(bottoms),
      pick(shoes),
    ].filter(Boolean) as PersonaItem[];
    if (!chosen.length) continue;
    const detailed = sanitizeGeneratedOutfitItemsDetailed(
      chosen.map((c) => ({ ...c, image: c.imageUrl }) as any),
      available.map((a) => ({ ...a, image: a.imageUrl }) as any),
      5,
      persona.style,
    );
    const items = detailed.items as PersonaDeckItem[];
    const macros = items.map((i) => canonicalizeMacroCategory(i.macroCategory));
    const complete =
      macros.includes('bottom') &&
      macros.includes('shoes') &&
      (macros.includes('top') || macros.includes('outerwear'));
    if (!complete || !items.length) continue;
    const withMeta = items.map((i) => {
      const src =
        available.find((a) => String(a.id) === String(i.id)) ?? (i as unknown as PersonaItem);
      return {
        ...(src as PersonaItem),
        id: String(i.id),
        slot: canonicalizeMacroCategory(i.macroCategory),
        score: scoreItemForStyle(
          { name: src.name, brand: src.brand, color: src.color, type: src.type, macroCategory: src.macroCategory },
          persona.style,
        ),
      };
    });
    const ids = withMeta.map((i) => i.id);
    if (dedupe.isDupe(ids)) continue;
    dedupe.track(ids);
    deck.push(withMeta);
  }

  const checks: PersonaResult['checks'] = [];
  const push = (label: string, pass: boolean, detail: string) =>
    checks.push({ label, pass, detail });

  push(
    'Deck complete',
    deck.length > 0 &&
      deck.every((o) => {
        const m = o.map((i) => i.slot);
        return m.includes('bottom') && m.includes('shoes') && (m.includes('top') || m.includes('outerwear'));
      }),
    `${deck.length} outfit(s), every card has top-like + bottom + shoes`,
  );

  let uniq = true;
  for (let a = 0; a < deck.length; a++)
    for (let b = a + 1; b < deck.length; b++) {
      const shared = deck[a].filter((x) => deck[b].some((y) => y.id === x.id)).length;
      if (shared >= 3) uniq = false;
    }
  push('No duplicate cards', uniq, 'every pair differs by 2+ items');

  if (persona.mode === 'wardrobe') {
    const pure = deck.every((o) => o.every((i) => i.isWardrobe));
    push('Wardrobe purity', pure, '100% of items are the account’s own closet');
  } else {
    const pure = deck.every((o) => o.every((i) => !i.isWardrobe));
    push('New-clothing purity', pure, '0 owned items — Supabase catalog only');
  }

  const names = deck.flatMap((o) => o.map((i) => i.name)).join(' | ').toLowerCase();
  if (persona.style === 'old_money' || persona.style === 'business_casual') {
    const banned = /(sneaker|hoodie|graphic|cargo|ripped|sweatpants|chunky)/.test(names);
    push('Style discipline', !banned, 'no sneakers/hoodies/graphics in formal decks');
  } else if (persona.style === 'casual') {
    const formal = /(blazer|suit|tuxedo|derby|dress shoe|overcoat|trench)/.test(names);
    push('Style discipline', !formal, 'no suiting/dress shoes in the relaxed deck');
  } else {
    const loud = /(neon|graphic|logo|chunky|ripped)/.test(names);
    push('Style discipline', !loud, 'no loud pieces in minimalist deck');
  }

  return { persona, layered, poolSize: available.length, deck, checks };
}

export function runAllPersonas(): PersonaResult[] {
  return PERSONAS.map(runPersona);
}
