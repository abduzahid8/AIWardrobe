/**
 * 50 distinct Personal Stylist scenarios through the SAME picker + judge
 * the live screen uses (createSlotNormalizer + passesStyleJudge).
 *
 * October weather is 15°C / cool (inferSeasonalWeather) — the simulator
 * currently shows that chip. Each scenario must still produce a 4-slot
 * look that survives taste + climate + coherence rules.
 */
import { createSlotNormalizer, isFourSlotLook } from '../../features/outfit-generator/utils/normalizeSlots';
import {
  passesStyleJudge,
  rejectForStyleOccasion,
  rejectForClimate,
  rejectOutfitClimate,
  rejectOutfitIncoherence,
} from '../../features/outfit-generator/utils/occasionRules';
import { parseStylistIntent } from '../../features/outfit-generator/utils/stylistIntent';
import type { StyleId } from '../../features/outfit-generator/utils/styleInference';

const OCTOBER = { temp: 15, condition: 'cool' };

function item(id: string, macro: string, name: string, extra: Record<string, unknown> = {}) {
  return {
    id,
    name,
    type: macro,
    macroCategory: macro,
    image: `https://cdn.example.com/${id}.jpg`,
    imageUrl: `https://cdn.example.com/${id}.jpg`,
    color: extra.color || 'neutral',
    brand: extra.brand || '',
    description: extra.description || name,
    isShopItem: true,
    price: extra.price ?? 79,
    ...extra,
  };
}

/** Live-catalog shaped pool: enough legal pieces for every style × occasion. */
const POOL = [
  item('ow-budget-coat', 'outerwear', 'Charcoal wool topcoat', { price: 24 }),
  item('ow-overcoat', 'outerwear', 'Navy wool overcoat'),
  item('t-budget-oxford', 'top', 'Slim fit cotton Oxford shirt', { price: 22 }),
  item('b-budget-wool', 'bottom', 'Grey wool dress trousers', { price: 24 }),
  item('s-budget-derby', 'shoes', 'Black leather derby shoes', { price: 24 }),
  item('ow-topcoat', 'outerwear', 'Camel wool topcoat'),
  item('ow-blazer', 'outerwear', 'Navy unstructured blazer'),
  item('ow-suit', 'outerwear', '100% cool wool double-breasted suit blazer'),
  item('ow-trench', 'outerwear', 'Beige cotton trench coat'),
  item('ow-cardigan', 'outerwear', 'Camel merino cardigan'),
  item('ow-quilt', 'outerwear', 'Black quilted jacket'),
  item('ow-hoodie', 'outerwear', 'Grey hoodie'),
  item('ow-bomber', 'outerwear', 'Navy bomber jacket'),
  item('t-oxford', 'top', 'White oxford shirt'),
  item('t-blue-oxford', 'top', 'Light blue oxford shirt'),
  item('t-knit-polo', 'top', 'Navy knit polo'),
  item('t-cream-polo', 'top', 'Cream knit polo'),
  item('t-merino', 'top', 'Grey merino knit sweater'),
  item('t-linen', 'top', '100% linen slim fit shirt'),
  item('t-tee', 'top', 'White cotton t-shirt'),
  item('t-jacquard', 'top', 'Relaxed fit striped jacquard t-shirt'),
  item('t-pajama', 'top', 'Red and white striped pajama set'),
  item('t-plaid', 'top', 'Plaid flannel shirt'),
  item('b-wool', 'bottom', 'Charcoal wool trousers'),
  item('b-darted', 'bottom', 'Navy darted wool trousers'),
  item('b-chinos', 'bottom', 'Beige chinos'),
  item('b-navy-chinos', 'bottom', 'Navy chinos'),
  item('b-jeans', 'bottom', 'Dark rinse jeans'),
  item('b-linen', 'bottom', 'Tapered fit linen pants'),
  item('b-shorts', 'bottom', 'Khaki bermuda shorts'),
  item('b-cargo', 'bottom', 'Cargo joggers'),
  item('s-loafer', 'shoes', 'Brown penny loafers'),
  item('s-nappa', 'shoes', 'Smart nappa loafers'),
  item('s-derby', 'shoes', 'Black derby shoes'),
  item('s-brown-derby', 'shoes', 'Brown derby shoes'),
  item('s-white-sneak', 'shoes', 'White leather sneakers'),
  item('s-chunky', 'shoes', 'Chunky running sneakers'),
  item('s-boat', 'shoes', 'Leather boat shoe'),
  item('s-driver', 'shoes', 'Driving moccasins with rubber nubs'),
  item('s-espadrille', 'shoes', 'Split suede leather espadrille'),
  item('s-mule', 'shoes', 'Backless leather mules'),
];

export const STYLIST_50: Array<{
  id: string;
  comment: string;
  style: StyleId;
  occasion: string;
}> = [
  { id: '01', comment: 'work, old money', style: 'old_money', occasion: 'work' },
  { id: '02', comment: 'work, classic', style: 'classic', occasion: 'work' },
  { id: '03', comment: 'work, semi-classic', style: 'semi_classic', occasion: 'work' },
  { id: '04', comment: 'work, minimalist', style: 'minimalist', occasion: 'work' },
  { id: '05', comment: 'work, casual', style: 'casual', occasion: 'work' },
  { id: '06', comment: 'weekend casual brunch, old money', style: 'old_money', occasion: 'casual' },
  { id: '07', comment: 'weekend casual brunch, classic', style: 'classic', occasion: 'casual' },
  { id: '08', comment: 'weekend casual brunch, semi-classic', style: 'semi_classic', occasion: 'casual' },
  { id: '09', comment: 'weekend casual brunch, minimalist', style: 'minimalist', occasion: 'casual' },
  { id: '10', comment: 'weekend casual brunch', style: 'casual', occasion: 'casual' },
  { id: '11', comment: 'date night, old money', style: 'old_money', occasion: 'date' },
  { id: '12', comment: 'date night, classic', style: 'classic', occasion: 'date' },
  { id: '13', comment: 'date night, semi-classic', style: 'semi_classic', occasion: 'date' },
  { id: '14', comment: 'date night, minimalist', style: 'minimalist', occasion: 'date' },
  { id: '15', comment: 'date night, casual', style: 'casual', occasion: 'date' },
  { id: '16', comment: 'formal wedding, old money', style: 'old_money', occasion: 'formal' },
  { id: '17', comment: 'formal wedding, classic', style: 'classic', occasion: 'formal' },
  { id: '18', comment: 'formal wedding, semi-classic', style: 'semi_classic', occasion: 'formal' },
  { id: '19', comment: 'formal gala, minimalist', style: 'minimalist', occasion: 'formal' },
  { id: '20', comment: 'formal wedding, casual', style: 'casual', occasion: 'formal' },
  { id: '21', comment: 'gym workout, old money', style: 'old_money', occasion: 'sport' },
  { id: '22', comment: 'gym workout, classic', style: 'classic', occasion: 'sport' },
  { id: '23', comment: 'gym workout, semi-classic', style: 'semi_classic', occasion: 'sport' },
  { id: '24', comment: 'gym workout, minimalist', style: 'minimalist', occasion: 'sport' },
  { id: '25', comment: 'gym workout, casual', style: 'casual', occasion: 'sport' },
  { id: '26', comment: 'travel airport, old money', style: 'old_money', occasion: 'travel' },
  { id: '27', comment: 'travel airport, classic', style: 'classic', occasion: 'travel' },
  { id: '28', comment: 'travel airport, semi-classic', style: 'semi_classic', occasion: 'travel' },
  { id: '29', comment: 'travel airport, minimalist', style: 'minimalist', occasion: 'travel' },
  { id: '30', comment: 'travel airport, casual', style: 'casual', occasion: 'travel' },
  { id: '31', comment: 'work, old money, $250', style: 'old_money', occasion: 'work' },
  { id: '32', comment: 'work, minimalist, under $100', style: 'minimalist', occasion: 'work' },
  { id: '33', comment: 'office interview, classic', style: 'classic', occasion: 'work' },
  { id: '34', comment: 'corporate meeting, old money', style: 'old_money', occasion: 'work' },
  { id: '35', comment: 'romantic dinner, old money', style: 'old_money', occasion: 'date' },
  { id: '36', comment: 'black tie wedding, classic', style: 'classic', occasion: 'formal' },
  { id: '37', comment: 'quiet luxury weekend casual', style: 'old_money', occasion: 'casual' },
  { id: '38', comment: 'everyday casual, minimalist', style: 'minimalist', occasion: 'casual' },
  { id: '39', comment: 'airport trip, semi-classic', style: 'semi_classic', occasion: 'travel' },
  { id: '40', comment: 'flight travel, casual', style: 'casual', occasion: 'travel' },
  { id: '41', comment: 'sport run, casual', style: 'casual', occasion: 'sport' },
  { id: '42', comment: 'business casual work', style: 'business_casual', occasion: 'work' },
  { id: '43', comment: 'modern professional office', style: 'business_casual', occasion: 'work' },
  { id: '44', comment: 'date night classic', style: 'classic', occasion: 'date' },
  { id: '45', comment: 'old money work', style: 'old_money', occasion: 'work' },
  { id: '46', comment: 'minimalist date night', style: 'minimalist', occasion: 'date' },
  { id: '47', comment: 'classic work, $200', style: 'classic', occasion: 'work' },
  { id: '48', comment: 'semi-classic brunch weekend', style: 'semi_classic', occasion: 'casual' },
  { id: '49', comment: 'formal old money', style: 'old_money', occasion: 'formal' },
  { id: '50', comment: 'casual everyday, smart casual', style: 'casual', occasion: 'casual' },
];

/** Extra live-style comments: RU, interview, gala, gym, travel variants. */
export const STYLIST_MORE: Array<{
  id: string;
  comment: string;
  style: StyleId;
  occasion: string;
}> = [
  { id: '51', comment: 'офис, олд мани', style: 'old_money', occasion: 'work' },
  { id: '52', comment: 'свидание, классик', style: 'classic', occasion: 'date' },
  { id: '53', comment: 'свадьба, олд мани', style: 'old_money', occasion: 'formal' },
  { id: '54', comment: 'поездка, кэжуал', style: 'casual', occasion: 'travel' },
  { id: '55', comment: 'спортзал, минималист', style: 'minimalist', occasion: 'sport' },
  { id: '56', comment: 'job interview classic', style: 'classic', occasion: 'work' },
  { id: '57', comment: 'gala black tie old money', style: 'old_money', occasion: 'formal' },
  { id: '58', comment: 'weekend brunch semi-classic', style: 'semi_classic', occasion: 'casual' },
  { id: '59', comment: 'airport flight minimalist', style: 'minimalist', occasion: 'travel' },
  { id: '60', comment: 'gym run semi-classic', style: 'semi_classic', occasion: 'sport' },
  { id: '61', comment: 'romantic dinner classic', style: 'classic', occasion: 'date' },
  { id: '62', comment: 'corporate conference old money', style: 'old_money', occasion: 'work' },
  { id: '63', comment: 'wedding guest classic', style: 'classic', occasion: 'formal' },
  { id: '64', comment: 'everyday weekend old money', style: 'old_money', occasion: 'casual' },
  { id: '65', comment: 'smart casual brunch', style: 'casual', occasion: 'casual' },
  { id: '66', comment: 'office work minimalist', style: 'minimalist', occasion: 'work' },
  { id: '67', comment: 'date night semi-classic', style: 'semi_classic', occasion: 'date' },
  { id: '68', comment: 'travel trip classic', style: 'classic', occasion: 'travel' },
  { id: '69', comment: 'formal gala classic', style: 'classic', occasion: 'formal' },
  { id: '70', comment: 'sport gym old money', style: 'old_money', occasion: 'sport' },
];

function runScenario(scenario: (typeof STYLIST_50)[number], cardIndex: number, nonce: number, deckUsed: Set<string>) {
  const normalize = createSlotNormalizer({
    weather: OCTOBER,
    baseItem: null,
    baseItemId: null,
    currentSource: 'shop',
    selectedOccasion: scenario.occasion,
    generationNonceRef: { current: nonce },
    poolFallback: POOL,
    budgetText: scenario.comment,
  });
  return normalize([], scenario.style, cardIndex, nonce, POOL, deckUsed);
}

function judgeFailures(items: any[], style: string, occasion: string): string[] {
  const fails: string[] = [];
  for (const it of items) {
    const r = rejectForStyleOccasion(it, style, occasion);
    if (r) fails.push(`${it.macroCategory}:${it.name} → ${r}`);
    const c = rejectForClimate(it, OCTOBER);
    if (c) fails.push(`${it.macroCategory}:${it.name} → ${c}`);
  }
  const climate = rejectOutfitClimate(items, OCTOBER);
  if (climate) fails.push(`outfit → ${climate}`);
  const inco = rejectOutfitIncoherence(items, style, occasion);
  if (inco) fails.push(`outfit → ${inco}`);
  return fails;
}

describe('parseStylistIntent — 50 comments map to the intended chips', () => {
  it.each(STYLIST_50.map((s) => [s.id, s.comment, s.style, s.occasion] as const))(
    '%s %s',
    (_id, comment, style, occasion) => {
      const parsed = parseStylistIntent(comment);
      expect(parsed.style).toBe(style);
      expect(parsed.occasion).toBe(occasion);
    },
  );
});

describe('parseStylistIntent — extra comments (RU + variants)', () => {
  it.each(STYLIST_MORE.map((s) => [s.id, s.comment, s.style, s.occasion] as const))(
    '%s %s',
    (_id, comment, style, occasion) => {
      const parsed = parseStylistIntent(comment);
      expect(parsed.style).toBe(style);
      expect(parsed.occasion).toBe(occasion);
    },
  );
});

describe('AI Personal Stylist — extra 20 scenarios', () => {
  it.each(STYLIST_MORE.map((s) => [s.id, s] as const))(
    '%s %s',
    (_id, scenario) => {
      const deckUsed = new Set<string>();
      const looks = [0, 1, 2].map((i) => runScenario(scenario, i, 31 + Number(scenario.id), deckUsed));
      expect(looks.every((look) => look.length >= 3)).toBe(true);
      for (const look of looks) {
        expect(judgeFailures(look, scenario.style, scenario.occasion)).toEqual([]);
        expect(passesStyleJudge(look, scenario.style, scenario.occasion, OCTOBER)).toBe(true);
      }
    },
  );
});

describe('AI Personal Stylist — 50 style × occasion scenarios', () => {
  it.each(STYLIST_50.map((s) => [s.id, s] as const))(
    '%s %s',
    (_id, scenario) => {
      const deckUsed = new Set<string>();
      const looks = [0, 1, 2].map((i) => runScenario(scenario, i, 11 + Number(scenario.id), deckUsed));
      expect(looks.every((look) => look.length >= 3)).toBe(true);
      expect(looks.every((look) => isFourSlotLook(look) || look.some((x: any) => x.macroCategory === 'bottom'))).toBe(true);

      for (const look of looks) {
        const fails = judgeFailures(look, scenario.style, scenario.occasion);
        expect(fails).toEqual([]);
        expect(passesStyleJudge(look, scenario.style, scenario.occasion, OCTOBER)).toBe(true);

        const blob = look.map((x: any) => `${x.name} ${x.type}`).join(' | ').toLowerCase();
        expect(blob).not.toMatch(/\bpajamas?\b|\bpyjamas?\b/);
        expect(blob).not.toMatch(/\biphone\b|\bairpods?\b/);
        if (scenario.occasion === 'work' || scenario.occasion === 'date' || scenario.occasion === 'formal') {
          expect(blob).not.toMatch(/\bboat shoe\b|\bespadrilles?\b|\bhoodies?\b|\bchunky running\b/);
          expect(look.some((x: any) => /\bjeans?\b/i.test(x.name) && x.macroCategory === 'bottom')).toBe(false);
        }
        if (scenario.style === 'old_money' || scenario.style === 'classic' || scenario.style === 'business_casual') {
          expect(blob).not.toMatch(/\bdriving moccasins\b|\bbackless\b/);
        }
      }
    },
  );
});
