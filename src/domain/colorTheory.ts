/**
 * colorTheory — Perceptual sRGB -> CIELAB (D65) fashion color engine.
 *
 * Provides:
 *  - True CIE 1976 L*a*b* and L*C*h° metrics for 95+ real fashion colors and
 *    arbitrary #RGB / #RRGGBB hex codes.
 *  - Perceptual color distance (ΔE76) to detect accidental near-miss clashes
 *    (e.g., mismatched dark navies/blacks lacking tonal separation).
 *  - Opt-in personal undertone & contrast harmony scoring (12-season + 4-season
 *    personal color analysis, grounded strictly in provided DNA fields).
 *  - Whole-outfit color theory evaluation rewarding genuine fashion harmonies
 *    and catching subtle clashes (warm cognac leather vs stark icy black/grey,
 *    multi-accent noise, near-miss dark mismatches, flat mid-value palettes).
 *
 * Pure + dependency-free + Node strip-types safe (no enums, no I/O).
 */

export type ColorTemperature = 'warm' | 'cool' | 'neutral';
export type ColorValueTier = 'deep' | 'mid' | 'light' | 'crisp_white';
export type ColorChromaTier = 'achromatic' | 'muted' | 'rich' | 'vivid';

export interface ColorMetrics {
  canonical: string;
  hex: string;
  L: number;
  a: number;
  b: number;
  hue: number;
  chroma: number;
  temperature: ColorTemperature;
  valueTier: ColorValueTier;
  chromaTier: ColorChromaTier;
  known: boolean;
}

export interface ColorProfileAppearance {
  skinUndertone?: string | null;
  skinDepth?: string | null;
  contrastLevel?: string | null;
  colorSeason?: string | null;
  flatteringColors?: readonly string[] | null;
  powerNeutrals?: readonly string[] | null;
  avoidColors?: readonly string[] | null;
  metalTone?: string | null;
  provenance?: string | null;
}

export interface ColorProfileDnaInput {
  appearance?: ColorProfileAppearance | null;
  preferences?: {
    favoriteColors?: readonly string[] | null;
    avoidedColors?: readonly string[] | null;
    avoidColors?: readonly string[] | null;
  } | null;
  // Also allow flat shape for direct/ergonomic callers
  skinUndertone?: string | null;
  skinDepth?: string | null;
  contrastLevel?: string | null;
  colorSeason?: string | null;
  flatteringColors?: readonly string[] | null;
  powerNeutrals?: readonly string[] | null;
  avoidColors?: readonly string[] | null;
  avoidedColors?: readonly string[] | null;
  favoriteColors?: readonly string[] | null;
  metalTone?: string | null;
}

export type UndertoneFitTier = 'flattering' | 'compatible' | 'neutral' | 'clash' | 'avoided';

export interface ColorUndertoneFitResult {
  score: number;
  tier: UndertoneFitTier;
  reasons: string[];
}

export type ColorHarmonyMode =
  | 'tonal_monochrome_gradient'
  | 'neutral_anchor_plus_accent'
  | 'warm_earth_analogous'
  | 'cool_architectural_contrast'
  | 'muted_complementary'
  | 'balanced_neutral_classic'
  | 'discordant'
  | 'insufficient_evidence';

export interface OutfitColorTheoryResult {
  score: number;
  harmonyMode: ColorHarmonyMode;
  temperatureCoherence: number;
  valueContrast: number;
  undertoneFit: number;
  flaws: string[];
  strengths: string[];
}

interface ColorDictionaryEntry {
  hex: string;
  temperature: ColorTemperature;
  role: 'core_neutral' | 'workhorse_neutral' | 'earth' | 'chromatic';
}

/**
 * 96 real fashion color definitions with curated sRGB reference pigments,
 * stylist-grade temperature classification, and outfit-role taxonomy.
 */
export const FASHION_COLOR_DICTIONARY: Readonly<Record<string, ColorDictionaryEntry>> = Object.freeze({
  // ── Achromatics, Blacks, Greys & Whites (26) ────────────────────────────
  black: { hex: '#141416', temperature: 'neutral', role: 'core_neutral' },
  jet_black: { hex: '#0B0B0D', temperature: 'neutral', role: 'core_neutral' },
  obsidian: { hex: '#181A1F', temperature: 'cool', role: 'core_neutral' },
  charcoal: { hex: '#36393F', temperature: 'cool', role: 'core_neutral' },
  anthracite: { hex: '#2E3136', temperature: 'cool', role: 'core_neutral' },
  graphite: { hex: '#43464B', temperature: 'cool', role: 'core_neutral' },
  dark_grey: { hex: '#4A4D52', temperature: 'neutral', role: 'core_neutral' },
  slate: { hex: '#5A6772', temperature: 'cool', role: 'core_neutral' },
  steel: { hex: '#687683', temperature: 'cool', role: 'core_neutral' },
  pewter: { hex: '#767B80', temperature: 'cool', role: 'core_neutral' },
  grey: { hex: '#808285', temperature: 'neutral', role: 'core_neutral' },
  mid_grey: { hex: '#7A7D82', temperature: 'neutral', role: 'core_neutral' },
  heather_grey: { hex: '#9A9EA4', temperature: 'neutral', role: 'core_neutral' },
  cool_grey: { hex: '#8D95A0', temperature: 'cool', role: 'core_neutral' },
  warm_grey: { hex: '#8F8982', temperature: 'warm', role: 'core_neutral' },
  ash: { hex: '#B2B5B8', temperature: 'cool', role: 'core_neutral' },
  silver: { hex: '#C0C5CC', temperature: 'cool', role: 'core_neutral' },
  light_grey: { hex: '#C7C9CC', temperature: 'neutral', role: 'core_neutral' },
  dove_grey: { hex: '#BDBFC2', temperature: 'cool', role: 'core_neutral' },
  white: { hex: '#FFFFFF', temperature: 'neutral', role: 'core_neutral' },
  crisp_white: { hex: '#FCFDFF', temperature: 'cool', role: 'core_neutral' },
  off_white: { hex: '#F6F4EE', temperature: 'warm', role: 'core_neutral' },
  ivory: { hex: '#F5EFE1', temperature: 'warm', role: 'core_neutral' },
  cream: { hex: '#F2E8D5', temperature: 'warm', role: 'core_neutral' },
  ecru: { hex: '#E8DEC8', temperature: 'warm', role: 'core_neutral' },
  bone: { hex: '#E3DAC9', temperature: 'warm', role: 'core_neutral' },

  // ── Beiges, Stones, Tans, Browns & Warm Earth (26) ──────────────────────
  alabaster: { hex: '#EDE8DF', temperature: 'warm', role: 'core_neutral' },
  oatmeal: { hex: '#DDD2BF', temperature: 'warm', role: 'workhorse_neutral' },
  sand: { hex: '#D6C4A8', temperature: 'warm', role: 'workhorse_neutral' },
  stone: { hex: '#C9C0B1', temperature: 'neutral', role: 'workhorse_neutral' },
  greige: { hex: '#B5ACA0', temperature: 'neutral', role: 'workhorse_neutral' },
  taupe: { hex: '#8B7D72', temperature: 'warm', role: 'workhorse_neutral' },
  mushroom: { hex: '#9E9286', temperature: 'warm', role: 'workhorse_neutral' },
  beige: { hex: '#D4C4A8', temperature: 'warm', role: 'workhorse_neutral' },
  khaki: { hex: '#B6A47E', temperature: 'warm', role: 'earth' },
  camel: { hex: '#C19A6B', temperature: 'warm', role: 'earth' },
  tan: { hex: '#C49A70', temperature: 'warm', role: 'earth' },
  biscuit: { hex: '#D2B48C', temperature: 'warm', role: 'earth' },
  honey: { hex: '#C9963B', temperature: 'warm', role: 'earth' },
  ochre: { hex: '#BE842C', temperature: 'warm', role: 'earth' },
  caramel: { hex: '#A96B3A', temperature: 'warm', role: 'earth' },
  cognac: { hex: '#9A4D28', temperature: 'warm', role: 'earth' },
  saddle: { hex: '#8B4A23', temperature: 'warm', role: 'earth' },
  chestnut: { hex: '#7A3E24', temperature: 'warm', role: 'earth' },
  cinnamon: { hex: '#995432', temperature: 'warm', role: 'earth' },
  terracotta: { hex: '#B85A3F', temperature: 'warm', role: 'earth' },
  rust: { hex: '#A44222', temperature: 'warm', role: 'earth' },
  burnt_orange: { hex: '#BE511E', temperature: 'warm', role: 'chromatic' },
  tobacco: { hex: '#724E32', temperature: 'warm', role: 'earth' },
  brown: { hex: '#6E4730', temperature: 'warm', role: 'workhorse_neutral' },
  chocolate: { hex: '#4B2E20', temperature: 'warm', role: 'workhorse_neutral' },
  espresso: { hex: '#34221A', temperature: 'warm', role: 'workhorse_neutral' },
  mocha: { hex: '#6C5142', temperature: 'warm', role: 'earth' },
  bronze: { hex: '#8C6239', temperature: 'warm', role: 'earth' },
  copper: { hex: '#B5633B', temperature: 'warm', role: 'earth' },

  // ── Blues, Navies & Teals (17) ──────────────────────────────────────────
  navy: { hex: '#1B2845', temperature: 'cool', role: 'workhorse_neutral' },
  midnight_blue: { hex: '#0A1931', temperature: 'cool', role: 'workhorse_neutral' },
  ink: { hex: '#182238', temperature: 'cool', role: 'workhorse_neutral' },
  indigo: { hex: '#283862', temperature: 'cool', role: 'workhorse_neutral' },
  denim: { hex: '#3B567C', temperature: 'cool', role: 'workhorse_neutral' },
  light_denim: { hex: '#7C9AC0', temperature: 'cool', role: 'chromatic' },
  cobalt: { hex: '#1D4ED8', temperature: 'cool', role: 'chromatic' },
  royal_blue: { hex: '#2342A6', temperature: 'cool', role: 'chromatic' },
  sapphire: { hex: '#1E3F8A', temperature: 'cool', role: 'chromatic' },
  blue: { hex: '#2B5EA7', temperature: 'cool', role: 'chromatic' },
  steel_blue: { hex: '#4D7092', temperature: 'cool', role: 'chromatic' },
  chambray: { hex: '#7693B5', temperature: 'cool', role: 'chromatic' },
  powder_blue: { hex: '#A5C1DC', temperature: 'cool', role: 'chromatic' },
  sky_blue: { hex: '#82B5E2', temperature: 'cool', role: 'chromatic' },
  light_blue: { hex: '#A7C7E7', temperature: 'cool', role: 'chromatic' },
  ice_blue: { hex: '#D6E6F4', temperature: 'cool', role: 'chromatic' },
  petrol: { hex: '#1D525E', temperature: 'cool', role: 'chromatic' },
  teal: { hex: '#1F6F78', temperature: 'cool', role: 'chromatic' },
  turquoise: { hex: '#2FA4A9', temperature: 'cool', role: 'chromatic' },

  // ── Greens & Olives (12) ────────────────────────────────────────────────
  forest_green: { hex: '#214231', temperature: 'cool', role: 'chromatic' },
  hunter_green: { hex: '#2A4B38', temperature: 'cool', role: 'chromatic' },
  pine: { hex: '#1F4E3D', temperature: 'cool', role: 'chromatic' },
  emerald: { hex: '#197553', temperature: 'cool', role: 'chromatic' },
  green: { hex: '#2E7D47', temperature: 'cool', role: 'chromatic' },
  moss: { hex: '#586B3B', temperature: 'warm', role: 'earth' },
  olive: { hex: '#556B2F', temperature: 'warm', role: 'earth' },
  dark_olive: { hex: '#3E4B26', temperature: 'warm', role: 'earth' },
  army_green: { hex: '#4B5334', temperature: 'warm', role: 'earth' },
  sage: { hex: '#8EA08A', temperature: 'cool', role: 'chromatic' },
  mint: { hex: '#A5D6BF', temperature: 'cool', role: 'chromatic' },
  pistachio: { hex: '#A9BE88', temperature: 'warm', role: 'chromatic' },
  chartreuse: { hex: '#B9C93A', temperature: 'warm', role: 'chromatic' },

  // ── Reds, Burgundies, Pinks & Corals (14) ───────────────────────────────
  burgundy: { hex: '#800020', temperature: 'cool', role: 'chromatic' },
  bordeaux: { hex: '#5C1A28', temperature: 'cool', role: 'chromatic' },
  oxblood: { hex: '#5A1B20', temperature: 'warm', role: 'earth' },
  maroon: { hex: '#6E1E26', temperature: 'warm', role: 'chromatic' },
  crimson: { hex: '#A81C2E', temperature: 'cool', role: 'chromatic' },
  red: { hex: '#C62828', temperature: 'warm', role: 'chromatic' },
  brick: { hex: '#9E3A2E', temperature: 'warm', role: 'earth' },
  coral: { hex: '#E06D58', temperature: 'warm', role: 'chromatic' },
  peach: { hex: '#F0B697', temperature: 'warm', role: 'chromatic' },
  blush: { hex: '#E8C5C4', temperature: 'warm', role: 'chromatic' },
  dusty_rose: { hex: '#B98389', temperature: 'cool', role: 'chromatic' },
  mauve: { hex: '#9B7285', temperature: 'cool', role: 'chromatic' },
  pink: { hex: '#D95B86', temperature: 'cool', role: 'chromatic' },
  magenta: { hex: '#B82570', temperature: 'cool', role: 'chromatic' },

  // ── Purples (6) ─────────────────────────────────────────────────────────
  plum: { hex: '#58294B', temperature: 'cool', role: 'chromatic' },
  aubergine: { hex: '#47223B', temperature: 'cool', role: 'chromatic' },
  purple: { hex: '#6A358A', temperature: 'cool', role: 'chromatic' },
  violet: { hex: '#6B48A8', temperature: 'cool', role: 'chromatic' },
  lavender: { hex: '#B5A6D4', temperature: 'cool', role: 'chromatic' },
  lilac: { hex: '#C6A9CC', temperature: 'cool', role: 'chromatic' },

  // ── Yellows, Oranges & Metallics (8) ────────────────────────────────────
  mustard: { hex: '#C79B2E', temperature: 'warm', role: 'earth' },
  gold: { hex: '#C59B3B', temperature: 'warm', role: 'earth' },
  amber: { hex: '#CE8220', temperature: 'warm', role: 'chromatic' },
  yellow: { hex: '#E5C13B', temperature: 'warm', role: 'chromatic' },
  butter: { hex: '#F3DF99', temperature: 'warm', role: 'chromatic' },
  champagne: { hex: '#E6D6B8', temperature: 'warm', role: 'workhorse_neutral' },
  orange: { hex: '#D96B27', temperature: 'warm', role: 'chromatic' },
  rose_gold: { hex: '#C99488', temperature: 'warm', role: 'earth' },
});

/** Explicit alias & hex map (prioritizing legacy app tokens like #0A1931 and American spellings). */
export const COLOR_SYNONYM_MAP: Readonly<Record<string, string>> = Object.freeze({
  '#0a1931': 'navy',
  '0a1931': 'navy',
  '#1a237e': 'navy',
  '#d4c4a8': 'beige',
  '#757575': 'grey',
  '#800020': 'burgundy',
  '#556b2f': 'olive',
  '#8b4513': 'brown',
  '#e91e63': 'pink',
  '#2196f3': 'blue',
  '#f44336': 'red',
  '#4caf50': 'green',
  midnight: 'navy',
  midnight_blue: 'navy',
  dark_navy: 'navy',
  marine: 'navy',
  marine_blue: 'navy',
  navy_blue: 'navy',
  gray: 'grey',
  dark_gray: 'dark_grey',
  light_gray: 'light_grey',
  mid_gray: 'mid_grey',
  heather_gray: 'heather_grey',
  cool_gray: 'cool_grey',
  warm_gray: 'warm_grey',
  dove_gray: 'dove_grey',
  silver_gray: 'silver',
  silver_grey: 'silver',
  charcoal_grey: 'charcoal',
  charcoal_gray: 'charcoal',
  slate_grey: 'slate',
  slate_gray: 'slate',
  offwhite: 'off_white',
  optical_white: 'crisp_white',
  chalk: 'off_white',
  snow: 'crisp_white',
  wine: 'burgundy',
  merlot: 'burgundy',
  claret: 'burgundy',
  dark_red: 'burgundy',
  ruby: 'crimson',
  scarlet: 'red',
  cherry: 'crimson',
  dark_brown: 'chocolate',
  coffee: 'espresso',
  cocoa: 'chocolate',
  walnut: 'brown',
  mahogany: 'chestnut',
  whiskey: 'cognac',
  sienna: 'terracotta',
  burnt_sienna: 'rust',
  dark_green: 'forest_green',
  bottle_green: 'forest_green',
  british_racing_green: 'forest_green',
  khaki_green: 'olive',
  loden: 'dark_olive',
  eucalyptus: 'sage',
  celadon: 'sage',
  seafoam: 'mint',
  lime: 'chartreuse',
  baby_blue: 'light_blue',
  cornflower: 'blue',
  french_blue: 'blue',
  dark_blue: 'navy',
  dark_denim: 'indigo',
  mid_denim: 'denim',
  cyan: 'turquoise',
  aqua: 'turquoise',
  peacock: 'teal',
  eggplant: 'aubergine',
  mulberry: 'plum',
  amethyst: 'purple',
  periwinkle: 'lavender',
  rose: 'dusty_rose',
  old_rose: 'dusty_rose',
  salmon: 'coral',
  apricot: 'peach',
  berry: 'magenta',
  fuchsia: 'magenta',
  hot_pink: 'magenta',
  canary: 'yellow',
  lemon: 'yellow',
  saffron: 'mustard',
  marigold: 'amber',
  tangerine: 'orange',
  buff: 'beige',
  fawn: 'beige',
  wheat: 'sand',
  putty: 'stone',
  natural: 'ecru',
  raw_white: 'ecru',
  cream_white: 'cream',
  // Common Russian/Uzbek catalog terms
  chernyj: 'black',
  чёрный: 'black',
  черный: 'black',
  belyj: 'white',
  белый: 'white',
  seryj: 'grey',
  серый: 'grey',
  sinij: 'blue',
  синий: 'blue',
  темно_синий: 'navy',
  коричневый: 'brown',
  korichnevyj: 'brown',
  бежевый: 'beige',
  bezhevyj: 'beige',
  зеленый: 'green',
  зелёный: 'green',
  zelenyj: 'green',
  бордовый: 'burgundy',
  красный: 'red',
  оливковый: 'olive',
});

const HEX_6_RE = /^#?([0-9a-f]{6})$/i;
const HEX_3_RE = /^#?([0-9a-f]{3})$/i;

function round3(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function normalizeHex(raw: string): string | null {
  const trimmed = raw.trim();
  const m6 = HEX_6_RE.exec(trimmed);
  if (m6) return `#${m6[1].toUpperCase()}`;
  const m3 = HEX_3_RE.exec(trimmed);
  if (m3) {
    const [r, g, b] = m3[1].toUpperCase().split('');
    return `#${r}${r}${g}${g}${b}${b}`;
  }
  return null;
}

function hexToRgb(hex: string): [number, number, number] {
  const clean = hex.startsWith('#') ? hex.slice(1) : hex;
  return [
    parseInt(clean.slice(0, 2), 16),
    parseInt(clean.slice(2, 4), 16),
    parseInt(clean.slice(4, 6), 16),
  ];
}

/** True sRGB -> linear RGB -> CIE XYZ (D65/2°) -> CIELAB conversion. */
function rgbToLab(r8: number, g8: number, b8: number): { L: number; a: number; b: number; hue: number; chroma: number } {
  const linearize = (channel8: number): number => {
    const c = channel8 / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  const r = linearize(r8);
  const g = linearize(g8);
  const b = linearize(b8);

  // IEC 61966-2-1 sRGB to CIE XYZ (D65)
  const x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
  const y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
  const z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;

  // D65 reference white
  const Xn = 0.95047;
  const Yn = 1.00000;
  const Zn = 1.08883;

  const EPSILON = 216 / 24389; // 0.008856451679...
  const KAPPA = 24389 / 27;    // 903.296296...
  const f = (t: number): number => (t > EPSILON ? Math.cbrt(t) : (KAPPA * t + 16) / 116);

  const fx = f(x / Xn);
  const fy = f(y / Yn);
  const fz = f(z / Zn);

  const L = 116 * fy - 16;
  const a = 500 * (fx - fy);
  const bLab = 200 * (fy - fz);
  const chroma = Math.hypot(a, bLab);
  const hue = chroma < 1e-6 ? 0 : ((Math.atan2(bLab, a) * 180) / Math.PI + 360) % 360;

  return {
    L: round3(L),
    a: round3(a),
    b: round3(bLab),
    hue: round3(hue),
    chroma: round3(chroma),
  };
}

function classifyValueTier(L: number, chroma: number): ColorValueTier {
  if (L >= 94 && chroma <= 4.5) return 'crisp_white';
  if (L >= 73) return 'light';
  if (L >= 36) return 'mid';
  return 'deep';
}

function classifyChromaTier(chroma: number): ColorChromaTier {
  if (chroma < 6.5) return 'achromatic';
  if (chroma < 24) return 'muted';
  if (chroma < 52) return 'rich';
  return 'vivid';
}

function classifyTemperatureFromLab(a: number, b: number, hue: number, chroma: number): ColorTemperature {
  if (chroma < 5) {
    if (b > 2.8) return 'warm';
    if (b < -2.8) return 'cool';
    return 'neutral';
  }
  if (hue >= 22 && hue <= 118) return 'warm';
  if (hue >= 165 && hue <= 335) return 'cool';
  if (hue < 22 || hue > 335) {
    // Crimson/burgundy vs warm tomato red
    if (b >= 14) return 'warm';
    if (b <= -2 || a > 25 && b < 12) return 'cool';
    return 'neutral';
  }
  // Transitional green (118..165)
  return b >= 20 ? 'warm' : 'cool';
}

// Precompute CIELAB coordinates for the entire fashion dictionary once for O(N) nearest-color hex mapping.
const DICTIONARY_LAB_TABLE: ReadonlyArray<{
  canonical: string;
  entry: ColorDictionaryEntry;
  lab: { L: number; a: number; b: number; hue: number; chroma: number };
}> = Object.entries(FASHION_COLOR_DICTIONARY).map(([canonical, entry]) => {
  const [r, g, b] = hexToRgb(entry.hex);
  return { canonical, entry, lab: rgbToLab(r, g, b) };
});

function nearestDictionaryColor(L: number, a: number, b: number): { canonical: string; entry: ColorDictionaryEntry } {
  let best = DICTIONARY_LAB_TABLE[0];
  let bestDist = Infinity;
  for (const row of DICTIONARY_LAB_TABLE) {
    const dL = L - row.lab.L;
    const da = a - row.lab.a;
    const db = b - row.lab.b;
    const dist = dL * dL + da * da + db * db;
    if (dist < bestDist) {
      bestDist = dist;
      best = row;
    }
  }
  return { canonical: best.canonical, entry: best.entry };
}

function normalizeColorKey(raw: unknown): string {
  if (typeof raw !== 'string') return '';
  return raw
    .normalize('NFKC')
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 64);
}

/**
 * Resolve any fashion color name, alias, or #HEX code to perceptual CIELAB metrics.
 * Unknown or invalid inputs return a safe neutral fallback with `known: false`.
 */
export function getColorMetrics(colorOrHex: unknown): ColorMetrics {
  const rawStr = typeof colorOrHex === 'string' ? colorOrHex.trim() : '';
  if (!rawStr) {
    return {
      canonical: 'unknown',
      hex: '#808080',
      L: 53.585,
      a: 0,
      b: 0,
      hue: 0,
      chroma: 0,
      temperature: 'neutral',
      valueTier: 'mid',
      chromaTier: 'achromatic',
      known: false,
    };
  }

  // 1. Check explicit hex alias first (e.g. '#0A1931' -> canonical 'navy', exact #0A1931 Lab)
  const parsedHex = normalizeHex(rawStr);
  if (parsedHex) {
    const lowerHex = parsedHex.toLowerCase();
    const [r, g, b] = hexToRgb(parsedHex);
    const lab = rgbToLab(r, g, b);
    const explicitAlias = COLOR_SYNONYM_MAP[lowerHex] || COLOR_SYNONYM_MAP[lowerHex.slice(1)];
    const nearest = nearestDictionaryColor(lab.L, lab.a, lab.b);
    const canonical = explicitAlias || (nearest.canonical === 'midnight_blue' ? 'navy' : nearest.canonical);
    const dictEntry = FASHION_COLOR_DICTIONARY[canonical];
    const temperature =
      explicitAlias && dictEntry
        ? dictEntry.temperature
        : classifyTemperatureFromLab(lab.a, lab.b, lab.hue, lab.chroma);

    return {
      canonical,
      hex: parsedHex,
      L: lab.L,
      a: lab.a,
      b: lab.b,
      hue: lab.hue,
      chroma: lab.chroma,
      temperature,
      valueTier: classifyValueTier(lab.L, lab.chroma),
      chromaTier: classifyChromaTier(lab.chroma),
      known: true,
    };
  }

  // 2. Named fashion color or alias lookup
  const key = normalizeColorKey(rawStr);
  const resolvedKey = COLOR_SYNONYM_MAP[key] || key;
  const entry = FASHION_COLOR_DICTIONARY[resolvedKey];
  if (entry) {
    const canonical = resolvedKey === 'midnight_blue' ? 'navy' : resolvedKey;
    const [r, g, b] = hexToRgb(entry.hex);
    const lab = rgbToLab(r, g, b);
    return {
      canonical,
      hex: entry.hex,
      L: lab.L,
      a: lab.a,
      b: lab.b,
      hue: lab.hue,
      chroma: lab.chroma,
      temperature: entry.temperature,
      valueTier: classifyValueTier(lab.L, lab.chroma),
      chromaTier: classifyChromaTier(lab.chroma),
      known: true,
    };
  }

  // 3. Compound fallback (e.g. "dark forest green", "warm camel wool")
  const tokens = key.split('_').filter(Boolean);
  for (let len = Math.min(2, tokens.length); len >= 1; len--) {
    for (let i = tokens.length - len; i >= 0; i--) {
      const slice = tokens.slice(i, i + len).join('_');
      const mapped = COLOR_SYNONYM_MAP[slice] || slice;
      const found = FASHION_COLOR_DICTIONARY[mapped];
      if (found) {
        const canonical = mapped === 'midnight_blue' ? 'navy' : mapped;
        const [r, g, b] = hexToRgb(found.hex);
        const lab = rgbToLab(r, g, b);
        return {
          canonical,
          hex: found.hex,
          L: lab.L,
          a: lab.a,
          b: lab.b,
          hue: lab.hue,
          chroma: lab.chroma,
          temperature: found.temperature,
          valueTier: classifyValueTier(lab.L, lab.chroma),
          chromaTier: classifyChromaTier(lab.chroma),
          known: true,
        };
      }
    }
  }

  return {
    canonical: 'unknown',
    hex: '#808080',
    L: 53.585,
    a: 0,
    b: 0,
    hue: 0,
    chroma: 0,
    temperature: 'neutral',
    valueTier: 'mid',
    chromaTier: 'achromatic',
    known: false,
  };
}

/** CIE76 perceptual color distance ΔE*ab between two fashion colors or metrics. */
export function deltaE76(c1: string | ColorMetrics, c2: string | ColorMetrics): number {
  const m1 = typeof c1 === 'string' ? getColorMetrics(c1) : c1;
  const m2 = typeof c2 === 'string' ? getColorMetrics(c2) : c2;
  return round3(Math.hypot(m1.L - m2.L, m1.a - m2.a, m1.b - m2.b));
}

function extractColorProfile(dna?: ColorProfileDnaInput | null) {
  if (!dna || typeof dna !== 'object') {
    return {
      skinUndertone: 'unknown',
      skinDepth: 'unknown',
      contrastLevel: 'unknown',
      colorSeason: 'unknown',
      flattering: new Set<string>(),
      powerNeutrals: new Set<string>(),
      avoid: new Set<string>(),
      favorites: new Set<string>(),
      hasAppearance: false,
    };
  }
  const app = dna.appearance ?? dna;
  const prefs = dna.preferences ?? dna;
  const normVal = (v: unknown): string => (typeof v === 'string' ? v.trim().toLowerCase().replace(/[\s-]+/g, '_') : 'unknown') || 'unknown';
  const toColorSet = (arr?: readonly string[] | null): Set<string> => {
    const out = new Set<string>();
    if (!Array.isArray(arr)) return out;
    for (const raw of arr.slice(0, 32)) {
      const m = getColorMetrics(raw);
      if (m.known && m.canonical !== 'unknown') out.add(m.canonical);
    }
    return out;
  };

  const skinUndertone = normVal(app.skinUndertone);
  const skinDepth = normVal(app.skinDepth);
  const contrastLevel = normVal(app.contrastLevel);
  const colorSeason = normVal(app.colorSeason);
  const flattering = toColorSet(app.flatteringColors);
  const powerNeutrals = toColorSet(app.powerNeutrals);
  const avoid = new Set<string>([
    ...toColorSet(app.avoidColors),
    ...toColorSet(prefs.avoidedColors),
    ...toColorSet(prefs.avoidColors),
  ]);
  const favorites = toColorSet(prefs.favoriteColors);

  const hasAppearance =
    skinUndertone !== 'unknown' ||
    skinDepth !== 'unknown' ||
    contrastLevel !== 'unknown' ||
    colorSeason !== 'unknown' ||
    flattering.size > 0 ||
    powerNeutrals.size > 0;

  return {
    skinUndertone,
    skinDepth,
    contrastLevel,
    colorSeason,
    flattering,
    powerNeutrals,
    avoid,
    favorites,
    hasAppearance,
  };
}

const SEASON_PROFILES: Readonly<
  Record<
    string,
    {
      temp: ColorTemperature;
      preferredValues: ReadonlySet<ColorValueTier>;
      preferredChromas: ReadonlySet<ColorChromaTier>;
      idealColors: ReadonlySet<string>;
      clashColors: ReadonlySet<string>;
    }
  >
> = Object.freeze({
  winter: {
    temp: 'cool',
    preferredValues: new Set<ColorValueTier>(['deep', 'crisp_white', 'light']),
    preferredChromas: new Set<ColorChromaTier>(['achromatic', 'rich', 'vivid']),
    idealColors: new Set(['black', 'crisp_white', 'white', 'navy', 'charcoal', 'emerald', 'cobalt', 'royal_blue', 'crimson', 'burgundy', 'ice_blue', 'silver', 'plum', 'magenta']),
    clashColors: new Set(['camel', 'mustard', 'ochre', 'burnt_orange', 'terracotta', 'beige', 'oatmeal', 'orange', 'honey', 'khaki', 'olive', 'dark_olive', 'moss', 'rust', 'cognac', 'tan', 'cream', 'bronze']),
  },
  autumn: {
    temp: 'warm',
    preferredValues: new Set<ColorValueTier>(['deep', 'mid', 'light']),
    preferredChromas: new Set<ColorChromaTier>(['muted', 'rich']),
    idealColors: new Set(['camel', 'olive', 'dark_olive', 'espresso', 'chocolate', 'brown', 'cream', 'ivory', 'ecru', 'oatmeal', 'sand', 'beige', 'cognac', 'terracotta', 'rust', 'forest_green', 'moss', 'tobacco', 'mustard', 'oxblood', 'bronze', 'warm_grey']),
    clashColors: new Set(['crisp_white', 'ice_blue', 'silver', 'magenta', 'hot_pink', 'cool_grey', 'lavender', 'lilac', 'mint', 'jet_black']),
  },
  summer: {
    temp: 'cool',
    preferredValues: new Set<ColorValueTier>(['light', 'mid']),
    preferredChromas: new Set<ColorChromaTier>(['achromatic', 'muted']),
    idealColors: new Set(['slate', 'dove_grey', 'cool_grey', 'heather_grey', 'powder_blue', 'steel_blue', 'chambray', 'navy', 'charcoal', 'dusty_rose', 'mauve', 'sage', 'lavender', 'off_white', 'plum', 'burgundy']),
    clashColors: new Set(['orange', 'burnt_orange', 'mustard', 'amber', 'camel', 'rust', 'chartreuse', 'jet_black', 'yellow']),
  },
  spring: {
    temp: 'warm',
    preferredValues: new Set<ColorValueTier>(['light', 'mid']),
    preferredChromas: new Set<ColorChromaTier>(['rich', 'vivid', 'muted']),
    idealColors: new Set(['ivory', 'cream', 'camel', 'sand', 'tan', 'coral', 'peach', 'navy', 'turquoise', 'emerald', 'butter', 'honey', 'gold', 'light_blue', 'pistachio', 'warm_grey']),
    clashColors: new Set(['jet_black', 'charcoal', 'anthracite', 'cool_grey', 'silver', 'aubergine', 'slate', 'bordeaux']),
  },
});

function macroSeasonOf(colorSeason: string): string | null {
  if (!colorSeason || colorSeason === 'unknown') return null;
  for (const macro of ['winter', 'autumn', 'summer', 'spring']) {
    if (colorSeason === macro || colorSeason.endsWith(`_${macro}`) || colorSeason.includes(macro)) {
      return macro;
    }
  }
  return null;
}

/**
 * Score how well a garment color harmonizes with the user's opt-in undertone,
 * skin depth, contrast level, color season, and explicit color preferences.
 */
export function scoreColorUndertoneFit(
  color: string | ColorMetrics,
  dna?: ColorProfileDnaInput | null,
): ColorUndertoneFitResult {
  const m = typeof color === 'string' ? getColorMetrics(color) : color;
  if (!m.known) {
    return { score: 0.55, tier: 'neutral', reasons: ['unknown_garment_color'] };
  }

  const profile = extractColorProfile(dna);
  const reasons: string[] = [];

  if (profile.avoid.has(m.canonical)) {
    return { score: 0.12, tier: 'avoided', reasons: [`explicit_avoid_color:${m.canonical}`] };
  }

  if (!profile.hasAppearance && profile.favorites.size === 0) {
    return { score: 0.55, tier: 'neutral', reasons: ['appearance_profile_unspecified'] };
  }

  let score = 0.55;

  // Direct opt-in flattering / power neutral lists
  if (profile.flattering.has(m.canonical)) {
    score += 0.30;
    reasons.push(`opt_in_flattering_color:${m.canonical}`);
  } else if (profile.powerNeutrals.has(m.canonical)) {
    score += 0.24;
    reasons.push(`opt_in_power_neutral:${m.canonical}`);
  }

  if (profile.favorites.has(m.canonical)) {
    score += 0.12;
    reasons.push(`favorite_color:${m.canonical}`);
  }

  // Undertone harmony
  const undertone = profile.skinUndertone;
  if (undertone === 'warm') {
    if (m.temperature === 'warm') {
      score += 0.16;
      reasons.push('warm_undertone_harmony');
    } else if (m.temperature === 'cool') {
      // Deep workhorse navy/forest remain wearable; icy cool tones clash more strongly
      if (m.canonical === 'navy' || m.canonical === 'forest_green' || m.canonical === 'teal' || m.canonical === 'burgundy') {
        score += 0.02;
        reasons.push('universal_cool_bridge');
      } else {
        score -= m.valueTier === 'crisp_white' || m.canonical === 'ice_blue' || m.canonical === 'silver' ? 0.22 : 0.14;
        reasons.push('cool_tone_against_warm_undertone');
      }
    } else if (m.valueTier === 'crisp_white' || m.canonical === 'black' || m.canonical === 'jet_black') {
      score -= 0.08;
      reasons.push('stark_achromatic_against_warm_undertone');
    }
  } else if (undertone === 'cool') {
    if (m.temperature === 'cool' || m.valueTier === 'crisp_white' || m.canonical === 'black') {
      score += 0.16;
      reasons.push('cool_undertone_harmony');
    } else if (m.temperature === 'warm') {
      if (m.canonical === 'taupe' || m.canonical === 'espresso' || m.canonical === 'off_white') {
        score -= 0.02;
      } else {
        score -= m.chromaTier === 'rich' || m.chromaTier === 'vivid' ? 0.22 : 0.14;
        reasons.push('warm_tone_against_cool_undertone');
      }
    }
  } else if (undertone === 'olive') {
    const oliveStars = new Set([
      'olive', 'dark_olive', 'forest_green', 'espresso', 'chocolate', 'cream', 'ivory',
      'navy', 'charcoal', 'burgundy', 'terracotta', 'camel', 'teal', 'aubergine', 'rust', 'warm_grey',
    ]);
    const oliveClashes = new Set(['chartreuse', 'pistachio', 'butter', 'beige', 'sand', 'ash', 'ice_blue']);
    if (oliveStars.has(m.canonical)) {
      score += 0.18;
      reasons.push('olive_undertone_harmony');
    } else if (oliveClashes.has(m.canonical)) {
      score -= 0.18;
      reasons.push('sallow_or_washing_tone_for_olive_undertone');
    }
  } else if (undertone === 'neutral') {
    if (m.chromaTier !== 'vivid') {
      score += 0.06;
      reasons.push('balanced_neutral_undertone_fit');
    }
  }

  // Color season specifics
  const macroSeason = macroSeasonOf(profile.colorSeason);
  if (macroSeason && SEASON_PROFILES[macroSeason]) {
    const sp = SEASON_PROFILES[macroSeason];
    if (sp.idealColors.has(m.canonical)) {
      score += 0.16;
      reasons.push(`color_season_ideal:${profile.colorSeason}`);
    } else if (sp.clashColors.has(m.canonical)) {
      score -= 0.22;
      reasons.push(`color_season_clash:${profile.colorSeason}`);
    } else {
      if (sp.preferredChromas.has(m.chromaTier)) score += 0.04;
      if (sp.preferredValues.has(m.valueTier)) score += 0.03;
    }
  }

  // Contrast level modulation
  if (profile.contrastLevel === 'high') {
    if (m.valueTier === 'deep' || m.valueTier === 'crisp_white' || m.chromaTier === 'rich' || m.chromaTier === 'vivid') {
      score += 0.08;
      reasons.push('high_contrast_crisp_saturation');
    } else if (m.valueTier === 'mid' && m.chromaTier === 'muted') {
      score -= 0.08;
      reasons.push('dusty_midtone_underwhelms_high_contrast');
    }
  } else if (profile.contrastLevel === 'low') {
    if (m.chromaTier === 'muted' || m.valueTier === 'mid' || m.canonical === 'cream' || m.canonical === 'oatmeal' || m.canonical === 'camel' || m.canonical === 'olive' || m.canonical === 'espresso') {
      score += 0.07;
      reasons.push('low_contrast_soft_harmony');
    } else if (m.valueTier === 'crisp_white' || m.canonical === 'jet_black' || m.chromaTier === 'vivid') {
      score -= 0.12;
      reasons.push('stark_tone_overpowers_low_contrast');
    }
  }

  // Skin depth modulation (strictly optical value contrast, never inferring identity)
  if (profile.skinDepth === 'deep' || profile.skinDepth === 'tan') {
    if (m.valueTier === 'light' || m.valueTier === 'crisp_white' || m.chromaTier === 'rich' || m.canonical === 'camel' || m.canonical === 'cream' || m.canonical === 'ivory') {
      score += 0.05;
      reasons.push('luminous_value_contrast_for_depth');
    }
  } else if (profile.skinDepth === 'fair' || profile.skinDepth === 'light') {
    if (m.canonical === 'beige' || m.canonical === 'sand' || m.canonical === 'biscuit') {
      score -= 0.05;
      reasons.push('low_separation_pale_beige_on_light_depth');
    }
  }

  const finalScore = round3(clamp01(score));
  const tier: UndertoneFitTier =
    finalScore >= 0.78
      ? 'flattering'
      : finalScore >= 0.62
        ? 'compatible'
        : finalScore >= 0.44
          ? 'neutral'
          : 'clash';

  return { score: finalScore, tier, reasons: [...new Set(reasons)] };
}

type SlotColorInput =
  | Readonly<Record<string, string | null | undefined>>
  | ReadonlyArray<{ slot?: string; color?: string | null } | string | null | undefined>;

interface NormalizedSlotColor {
  slot: string;
  metrics: ColorMetrics;
  role: 'core_neutral' | 'workhorse_neutral' | 'earth' | 'chromatic';
}

function normalizeSlotColors(input: SlotColorInput): NormalizedSlotColor[] {
  const entries: Array<{ slot: string; raw: unknown }> = [];
  if (Array.isArray(input)) {
    input.forEach((entry, index) => {
      if (typeof entry === 'string') {
        entries.push({ slot: `slot_${index}`, raw: entry });
      } else if (entry && typeof entry === 'object') {
        entries.push({ slot: String(entry.slot || `slot_${index}`).toLowerCase(), raw: entry.color });
      }
    });
  } else if (input && typeof input === 'object') {
    for (const [slot, raw] of Object.entries(input)) {
      entries.push({ slot: slot.toLowerCase(), raw });
    }
  }

  const resolved: NormalizedSlotColor[] = [];
  for (const { slot, raw } of entries) {
    const metrics = getColorMetrics(raw);
    if (!metrics.known) continue;
    const dictRole = FASHION_COLOR_DICTIONARY[metrics.canonical]?.role;
    const role: NormalizedSlotColor['role'] =
      dictRole ||
      (metrics.chromaTier === 'achromatic'
        ? 'core_neutral'
        : metrics.chromaTier === 'muted'
          ? 'workhorse_neutral'
          : 'chromatic');
    resolved.push({ slot, metrics, role });
  }
  return resolved;
}

function hueDifferenceDeg(h1: number, h2: number): number {
  const diff = Math.abs(h1 - h2) % 360;
  return diff > 180 ? 360 - diff : diff;
}

const SLOT_FACE_WEIGHTS: Readonly<Record<string, number>> = Object.freeze({
  top: 0.40,
  outerwear: 0.35,
  dress: 0.55,
  bottom: 0.15,
  shoes: 0.10,
  accessory: 0.15,
});

const WARM_LEATHER_SHOE_HUES = new Set([
  'cognac', 'tan', 'caramel', 'saddle', 'honey', 'ochre', 'amber', 'orange', 'burnt_orange', 'rust', 'cinnamon', 'copper',
]);

const STARK_COOL_ANCHORS = new Set([
  'black', 'jet_black', 'obsidian', 'cool_grey', 'silver', 'ice_blue', 'slate', 'crisp_white', 'anthracite', 'graphite',
]);

const DARK_NAVY_FAMILY = new Set(['navy', 'midnight_blue', 'ink', 'indigo']);
const DARK_BLACK_FAMILY = new Set(['black', 'jet_black', 'obsidian']);

/**
 * Evaluate complete-outfit color theory across slots, combining perceptual
 * CIELAB distance, temperature coherence, value stepping, harmony taxonomy,
 * and opt-in personal undertone/contrast alignment.
 */
export function scoreOutfitColorTheory(
  slotColors: SlotColorInput,
  dna?: ColorProfileDnaInput | null,
): OutfitColorTheoryResult {
  const items = normalizeSlotColors(slotColors);
  const profile = extractColorProfile(dna);

  if (items.length < 2) {
    const fallbackUndertone =
      items.length === 1 ? scoreColorUndertoneFit(items[0].metrics, dna).score : 0.55;
    return {
      score: 0.6,
      harmonyMode: 'insufficient_evidence',
      temperatureCoherence: 0.6,
      valueContrast: 0.5,
      undertoneFit: round3(fallbackUndertone),
      flaws: [],
      strengths: ['insufficient_color_evidence'],
    };
  }

  const flaws: string[] = [];
  const strengths: string[] = [];

  // ── 1. Undertone & personal color alignment (weighted toward near-face slots) ──
  let weightedUndertoneSum = 0;
  let weightTotal = 0;
  let avoidedCount = 0;
  for (const item of items) {
    const fit = scoreColorUndertoneFit(item.metrics, dna);
    const w = SLOT_FACE_WEIGHTS[item.slot] ?? 0.25;
    weightedUndertoneSum += fit.score * w;
    weightTotal += w;
    if (fit.tier === 'avoided') {
      avoidedCount += 1;
      flaws.push(`avoided_color_in_${item.slot}:${item.metrics.canonical}`);
    }
  }
  const undertoneFit = round3(weightTotal > 0 ? weightedUndertoneSum / weightTotal : 0.55);
  if (profile.hasAppearance && undertoneFit >= 0.74) {
    strengths.push('strong_personal_undertone_alignment');
  } else if (profile.hasAppearance && undertoneFit <= 0.42) {
    flaws.push('weak_personal_undertone_alignment');
  }

  // ── 2. Near-miss dark/tonal clash detection via CIELAB ΔE76 ──────────────
  const isAdjacentGarmentPair = (s1: string, s2: string): boolean => {
    if (s1.startsWith('slot_') || s2.startsWith('slot_')) return true;
    const pair = [s1, s2].sort().join('|');
    return (
      pair === 'bottom|outerwear' ||
      pair === 'bottom|top' ||
      pair === 'outerwear|top' ||
      pair === 'bottom|shoes' ||
      pair === 'dress|outerwear' ||
      pair === 'dress|shoes'
    );
  };
  let nearMissPenalty = 0;
  const seenNearMiss = new Set<string>();
  for (let i = 0; i < items.length; i++) {
    for (let j = i + 1; j < items.length; j++) {
      const a = items[i];
      const b = items[j];
      const dE = deltaE76(a.metrics, b.metrics);
      const dL = Math.abs(a.metrics.L - b.metrics.L);
      const bothDeep = a.metrics.valueTier === 'deep' && b.metrics.valueTier === 'deep';
      const adjacent = a.slot !== 'accessory' && b.slot !== 'accessory' && isAdjacentGarmentPair(a.slot, b.slot);

      // Case A: Accidental dark navy directly paired with black (e.g. navy jacket + black trousers)
      // Note: navy blazer + mid-grey trousers + black dress shoes is separated & classic.
      const navyVsBlack =
        (DARK_NAVY_FAMILY.has(a.metrics.canonical) && DARK_BLACK_FAMILY.has(b.metrics.canonical)) ||
        (DARK_BLACK_FAMILY.has(a.metrics.canonical) && DARK_NAVY_FAMILY.has(b.metrics.canonical));

      const sameFamilyNearMiss =
        a.metrics.hex !== b.metrics.hex &&
        bothDeep &&
        dE >= 4.0 &&
        dE <= 18.0 &&
        dL < 12.0;

      if (adjacent && navyVsBlack && dE < 24) {
        const key = 'near_miss_navy_black_clash';
        if (!seenNearMiss.has(key)) {
          seenNearMiss.add(key);
          flaws.push(key);
          nearMissPenalty += 0.18;
        }
      } else if (adjacent && sameFamilyNearMiss) {
        const key = `near_miss_dark_mismatch:${a.slot}_${b.slot}`;
        if (!seenNearMiss.has(key)) {
          seenNearMiss.add(key);
          flaws.push('near_miss_dark_tonal_clash');
          nearMissPenalty += 0.20;
        }
      }
    }
  }

  // ── 3. Temperature coherence & warm-cognac-vs-icy-black/grey check ───────
  const warmItems = items.filter((it) => it.metrics.temperature === 'warm');
  const coolItems = items.filter((it) => it.metrics.temperature === 'cool');
  const starkCoolNonShoe = items.filter(
    (it) => it.slot !== 'shoes' && STARK_COOL_ANCHORS.has(it.metrics.canonical),
  );
  const hasWarmNonShoe = items.some(
    (it) => it.slot !== 'shoes' && it.metrics.temperature === 'warm' && it.metrics.chroma >= 8,
  );
  const hasNavyAnchor = items.some((it) => DARK_NAVY_FAMILY.has(it.metrics.canonical) || it.metrics.canonical === 'denim');

  let tempCoherence = 0.85;
  let warmCognacVsIcyClash = false;

  const shoeItem = items.find((it) => it.slot === 'shoes');
  if (
    shoeItem &&
    WARM_LEATHER_SHOE_HUES.has(shoeItem.metrics.canonical) &&
    starkCoolNonShoe.length >= 2 &&
    !hasWarmNonShoe &&
    !hasNavyAnchor
  ) {
    warmCognacVsIcyClash = true;
    tempCoherence -= 0.42;
    flaws.push('warm_cognac_leather_clashes_with_icy_cool_palette');
  }

  // Check harsh unbridged warm chromatic vs icy cool chromatic clash
  const warmChromatics = warmItems.filter((it) => it.role === 'chromatic' || (it.role === 'earth' && it.metrics.chroma >= 26));
  const coolChromatics = coolItems.filter((it) => it.role === 'chromatic');
  const isClassicComplementary = (w: NormalizedSlotColor, c: NormalizedSlotColor): boolean => {
    const hDiff = hueDifferenceDeg(w.metrics.hue, c.metrics.hue);
    return hDiff >= 130 && hDiff <= 220 && (w.metrics.chromaTier !== 'vivid' || c.metrics.chromaTier !== 'vivid');
  };

  if (warmChromatics.length > 0 && coolChromatics.length > 0 && !warmCognacVsIcyClash) {
    const harmoniousComp = warmChromatics.length === 1 && coolChromatics.length === 1 &&
      isClassicComplementary(warmChromatics[0], coolChromatics[0]);
    if (!harmoniousComp) {
      tempCoherence -= 0.25;
      flaws.push('mixed_warm_and_cool_temperature_clash');
    } else {
      tempCoherence = Math.max(tempCoherence, 0.84);
    }
  } else if (warmItems.length >= 2 && coolItems.length === 0) {
    tempCoherence = 0.96;
    strengths.push('cohesive_warm_temperature_story');
  } else if (coolItems.length >= 2 && warmItems.length === 0) {
    tempCoherence = 0.95;
    strengths.push('cohesive_cool_temperature_story');
  }

  // Classic navy + brown/cognac/camel bridge
  if (hasNavyAnchor && warmItems.some((it) => it.role === 'earth' || it.metrics.canonical === 'brown' || it.metrics.canonical === 'chocolate')) {
    if (!warmCognacVsIcyClash) {
      tempCoherence = Math.max(tempCoherence, 0.88);
      strengths.push('classic_navy_and_warm_earth_pairing');
    }
  }
  tempCoherence = round3(clamp01(tempCoherence));

  // ── 4. Multi-accent visual noise detection ───────────────────────────────
  const accentItems = items.filter(
    (it) =>
      it.role === 'chromatic' &&
      (it.metrics.chromaTier === 'vivid' || it.metrics.chromaTier === 'rich' || it.metrics.chroma >= 22),
  );
  // Group distinct hue families among accents (>= 35° apart)
  const distinctAccentFamilies: NormalizedSlotColor[] = [];
  for (const acc of accentItems) {
    const existing = distinctAccentFamilies.find(
      (f) => hueDifferenceDeg(f.metrics.hue, acc.metrics.hue) < 35,
    );
    if (!existing) distinctAccentFamilies.push(acc);
  }

  let accentPenalty = 0;
  if (distinctAccentFamilies.length >= 3) {
    accentPenalty = 0.34;
    flaws.push('multi_accent_visual_noise');
  } else if (distinctAccentFamilies.length === 2) {
    const hDiff = hueDifferenceDeg(distinctAccentFamilies[0].metrics.hue, distinctAccentFamilies[1].metrics.hue);
    const bothMutedOrDeep = distinctAccentFamilies.every(
      (x) => x.metrics.chromaTier !== 'vivid' && x.metrics.L < 62,
    );
    if (hDiff >= 135 && hDiff <= 215 && bothMutedOrDeep) {
      strengths.push('refined_muted_complementary_accents');
    } else {
      accentPenalty = 0.22;
      flaws.push('competing_accent_hues');
    }
  } else if (distinctAccentFamilies.length === 1) {
    strengths.push(`focused_single_accent:${distinctAccentFamilies[0].metrics.canonical}`);
  }

  // ── 5. Value contrast & tonal stepping ───────────────────────────────────
  const lightnessValues = items.map((it) => it.metrics.L);
  const minL = Math.min(...lightnessValues);
  const maxL = Math.max(...lightnessValues);
  const lRange = maxL - minL;
  const allDarkMonochrome = maxL <= 30 && nearMissPenalty === 0;

  let valueContrast = 0.72;
  if (allDarkMonochrome) {
    valueContrast = 0.84;
    strengths.push('intentional_dark_monochrome_column');
  } else if (lRange >= 32 && lRange <= 88) {
    valueContrast = 0.92;
    strengths.push('balanced_light_dark_value_stepping');
  } else if (lRange > 88) {
    valueContrast = profile.contrastLevel === 'low' ? 0.62 : 0.90;
    if (profile.contrastLevel === 'low') {
      flaws.push('high_value_contrast_exceeds_low_contrast_profile');
    } else {
      strengths.push('crisp_architectural_value_contrast');
    }
  } else if (lRange >= 18) {
    valueContrast = 0.78;
    strengths.push('subtle_tonal_value_gradient');
  } else {
    valueContrast = 0.48;
    flaws.push('flat_mid_value_separation');
  }

  // Match contrast preference if supplied
  if (profile.contrastLevel === 'high' && lRange >= 45) {
    valueContrast = Math.min(1, valueContrast + 0.06);
    strengths.push('matches_high_contrast_profile');
  } else if (profile.contrastLevel === 'low' && lRange >= 16 && lRange <= 58) {
    valueContrast = Math.min(1, valueContrast + 0.06);
    strengths.push('matches_low_contrast_tonal_profile');
  }
  valueContrast = round3(clamp01(valueContrast));

  // ── 6. Classify Fashion Harmony Mode ─────────────────────────────────────
  const chromaticAngles = items
    .filter((it) => it.metrics.chroma >= 10)
    .map((it) => it.metrics.hue);

  let maxHueSpread = 0;
  for (let i = 0; i < chromaticAngles.length; i++) {
    for (let j = i + 1; j < chromaticAngles.length; j++) {
      maxHueSpread = Math.max(maxHueSpread, hueDifferenceDeg(chromaticAngles[i], chromaticAngles[j]));
    }
  }

  const neutralCount = items.filter((it) => it.role === 'core_neutral' || it.role === 'workhorse_neutral').length;
  const earthCount = items.filter(
    (it) => it.role === 'earth' || (it.metrics.temperature === 'warm' && it.role === 'workhorse_neutral'),
  ).length;
  const coolArchitecturalCount = items.filter(
    (it) =>
      it.metrics.temperature === 'cool' ||
      it.metrics.canonical === 'black' ||
      it.metrics.canonical === 'white' ||
      it.metrics.canonical === 'crisp_white' ||
      it.metrics.canonical === 'grey',
  ).length;

  let harmonyMode: ColorHarmonyMode = 'balanced_neutral_classic';
  let harmonyBonus = 0.04;

  const hasSevereClash =
    warmCognacVsIcyClash ||
    accentPenalty >= 0.22 ||
    nearMissPenalty >= 0.18 ||
    avoidedCount >= 2 ||
    tempCoherence < 0.55;

  if (hasSevereClash) {
    harmonyMode = 'discordant';
    harmonyBonus = -0.14;
  } else if (
    coolArchitecturalCount === items.length &&
    lRange >= 65 &&
    maxL >= 90 &&
    minL <= 28 &&
    distinctAccentFamilies.length <= 1 &&
    tempCoherence >= 0.85 &&
    (items.some((it) => it.metrics.temperature === 'cool') || profile.contrastLevel === 'high')
  ) {
    harmonyMode = 'cool_architectural_contrast';
    harmonyBonus = 0.11;
    strengths.push('cool_architectural_contrast_harmony');
  } else if (
    distinctAccentFamilies.length === 0 &&
    (chromaticAngles.length <= 1 || maxHueSpread <= 32) &&
    (lRange >= 18 || allDarkMonochrome)
  ) {
    // Distinguish warm earth tonal vs pure monochrome gradient
    if (earthCount >= 2 && warmItems.length >= 2 && maxHueSpread > 12) {
      harmonyMode = 'warm_earth_analogous';
      harmonyBonus = 0.12;
      strengths.push('warm_earth_analogous_harmony');
    } else {
      harmonyMode = 'tonal_monochrome_gradient';
      harmonyBonus = 0.12;
      strengths.push('tonal_monochrome_gradient_harmony');
    }
  } else if (
    earthCount >= 2 &&
    coolChromatics.length === 0 &&
    tempCoherence >= 0.85 &&
    maxHueSpread <= 75
  ) {
    harmonyMode = 'warm_earth_analogous';
    harmonyBonus = 0.12;
    strengths.push('warm_earth_analogous_harmony');
  } else if (
    coolArchitecturalCount === items.length &&
    lRange >= 38 &&
    distinctAccentFamilies.length <= 1 &&
    tempCoherence >= 0.85
  ) {
    harmonyMode = 'cool_architectural_contrast';
    harmonyBonus = 0.11;
    strengths.push('cool_architectural_contrast_harmony');
  } else if (distinctAccentFamilies.length === 1 && neutralCount >= 2) {
    harmonyMode = 'neutral_anchor_plus_accent';
    harmonyBonus = 0.11;
    strengths.push('neutral_anchor_plus_accent_harmony');
  } else if (
    (distinctAccentFamilies.length === 2 && accentPenalty === 0) ||
    (hasNavyAnchor && earthCount >= 1 && tempCoherence >= 0.85)
  ) {
    harmonyMode = 'muted_complementary';
    harmonyBonus = 0.10;
    strengths.push('muted_complementary_harmony');
  }

  // ── 7. Composite score (0..1) ────────────────────────────────────────────
  const baseStructural = profile.hasAppearance
    ? 0.34 * tempCoherence + 0.28 * valueContrast + 0.38 * undertoneFit
    : 0.42 * tempCoherence + 0.36 * valueContrast + 0.22 * 0.72;
  const effectiveHarmonyBonus =
    profile.hasAppearance && undertoneFit < 0.42 && harmonyBonus > 0
      ? harmonyBonus * 0.35
      : harmonyBonus;
  let score =
    baseStructural +
    effectiveHarmonyBonus -
    nearMissPenalty -
    accentPenalty -
    avoidedCount * 0.22;

  if (harmonyMode === 'discordant') {
    score = Math.min(score, 0.46);
  }

  return {
    score: round3(clamp01(score)),
    harmonyMode,
    temperatureCoherence: tempCoherence,
    valueContrast,
    undertoneFit,
    flaws: [...new Set(flaws)],
    strengths: [...new Set(strengths)],
  };
}
