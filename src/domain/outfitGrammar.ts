/**
 * outfitGrammar — 4-real-purchasable-item outfit grammar & thermal suitability engine.
 *
 * Preserves the binding AIWardrobe North Star 4-item contract across all climates
 * and presentations without ever inventing synthetic slots or dressing a 30°C
 * request in a heavy winter coat:
 *
 *   - `layered_separates`      -> ['top', 'outerwear', 'bottom', 'shoes']
 *   - `warm_weather_separates` -> ['top', 'bottom', 'shoes', 'accessory'] (only when real accessory inventory exists)
 *   - `one_piece_editorial`    -> ['dress', 'outerwear', 'shoes', 'accessory'] (only when real dress + accessory inventory exists)
 *
 * When real accessory/dress inventory is absent in the current catalog, falls back
 * honestly to `layered_separates` with explicit breathable thermal constraints.
 *
 * Pure + dependency-free + Node strip-types safe (no enums, no I/O).
 */

export type OutfitGrammarId =
  | 'layered_separates'
  | 'warm_weather_separates'
  | 'one_piece_editorial';

export type GrammarSlot =
  | 'top'
  | 'outerwear'
  | 'bottom'
  | 'shoes'
  | 'accessory'
  | 'dress';

export type ThermalBand =
  | 'freezing'
  | 'cold'
  | 'cool'
  | 'mild'
  | 'warm'
  | 'hot'
  | 'unknown';

export interface GrammarWeatherInput {
  temp?: number | null;
  condition?: string | null;
  humidity?: number | null;
}

export interface OutfitGrammarContext {
  weather?: GrammarWeatherInput | null;
  occasion?: string | null;
  style?: string | null;
  request?: string | null;
  presentation?: 'menswear' | 'womenswear' | 'unisex' | string | null;
  preferredGrammar?: OutfitGrammarId | null;
  minItemsPerSlot?: number | null;
}

export interface CatalogSlotCounts {
  top?: number | null;
  outerwear?: number | null;
  bottom?: number | null;
  shoes?: number | null;
  accessory?: number | null;
  dress?: number | null;
}

export interface ThermalConstraints {
  band: ThermalBand;
  maxOuterwearWeight: 'ultralight_breathable' | 'light_transitional' | 'midweight' | 'insulating';
  requireBreathableLayer: boolean;
  requireInsulatingOuterwear: boolean;
  preferredLayerSubtypes: string[];
  forbiddenOuterwearTypes: string[];
  note: string;
}

export interface ResolvedOutfitGrammar {
  grammarId: OutfitGrammarId;
  slots: readonly [GrammarSlot, GrammarSlot, GrammarSlot, GrammarSlot];
  fourItemContract: true;
  thermalBand: ThermalBand;
  thermalConstraints: ThermalConstraints;
  fallbackApplied: boolean;
  fallbackReason: string | null;
  requiredCountsMet: boolean;
}

export interface ThermalGarmentInput {
  macroCategory?: string | null;
  category?: string | null;
  slot?: string | null;
  name?: string | null;
  type?: string | null;
  subtype?: string | null;
  description?: string | null;
  material?: string | readonly string[] | null;
  materials?: readonly string[] | null;
  bulk?: string | null;
  insulated?: boolean | null;
  layering?: string | null;
}

export interface ThermalSuitabilityResult {
  score: number;
  band: ThermalBand;
  suitable: boolean;
  reasons: string[];
}

export const OUTFIT_GRAMMAR_SLOTS: Readonly<
  Record<OutfitGrammarId, readonly [GrammarSlot, GrammarSlot, GrammarSlot, GrammarSlot]>
> = Object.freeze({
  layered_separates: Object.freeze(['top', 'outerwear', 'bottom', 'shoes'] as const),
  warm_weather_separates: Object.freeze(['top', 'bottom', 'shoes', 'accessory'] as const),
  one_piece_editorial: Object.freeze(['dress', 'outerwear', 'shoes', 'accessory'] as const),
});

function round3(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

export function resolveThermalBand(
  weather?: GrammarWeatherInput | null,
  request?: string | null,
): ThermalBand {
  const temp = typeof weather?.temp === 'number' && Number.isFinite(weather.temp)
    ? weather.temp
    : null;
  const text = `${weather?.condition || ''} ${request || ''}`.toLowerCase();

  if (temp != null) {
    if (temp <= 2) return 'freezing';
    if (temp <= 10) return 'cold';
    if (temp <= 17) return 'cool';
    if (temp < 23) return 'mild';
    if (temp < 26) return 'warm';
    return 'hot';
  }

  if (/\b(freezing|snow|sub[ -]?zero|blizzard|icy)\b|мороз|снег|қиш/i.test(text)) return 'freezing';
  if (/\b(winter|cold|chilly winter)\b|зим|холод/i.test(text)) return 'cold';
  if (/\b(autumn|fall|chilly|crisp|brisk)\b|осен|прохлад/i.test(text)) return 'cool';
  if (/\b(spring|mild|temperate)\b|весен|весна/i.test(text)) return 'mild';
  if (/\b(hot|heatwave|tropical|scorching|summer)\b|жар|лет[оа]|yoz/i.test(text)) return 'hot';
  if (/\b(warm|sunny)\b|тепл/i.test(text)) return 'warm';
  return 'unknown';
}

function buildThermalConstraints(band: ThermalBand, grammarId: OutfitGrammarId): ThermalConstraints {
  if (band === 'hot') {
    return {
      band,
      maxOuterwearWeight: 'ultralight_breathable',
      requireBreathableLayer: grammarId !== 'warm_weather_separates',
      requireInsulatingOuterwear: false,
      preferredLayerSubtypes:
        grammarId === 'warm_weather_separates'
          ? []
          : ['linen-overshirt', 'unlined-overshirt', 'open-weave-cardigan', 'unlined-linen-blazer', 'cotton-shirt-jacket'],
      forbiddenOuterwearTypes: ['puffer', 'parka', 'down', 'shearling', 'quilted', 'overcoat', 'topcoat', 'peacoat', 'heavy-wool-coat'],
      note:
        grammarId === 'warm_weather_separates'
          ? 'Hot weather (>=26°C): unlayered breathable top + bottom + shoes anchored with a real purchasable accessory.'
          : 'Hot weather (>=26°C) under 4-item layered contract: prioritize ultralight breathable linen/cotton unlined open layers; strictly forbid heavy winter outerwear.',
    };
  }
  if (band === 'warm') {
    return {
      band,
      maxOuterwearWeight: 'light_transitional',
      requireBreathableLayer: true,
      requireInsulatingOuterwear: false,
      preferredLayerSubtypes: ['linen-blazer', 'overshirt', 'light-cardigan', 'cotton-chore-jacket', 'unlined-blazer'],
      forbiddenOuterwearTypes: ['puffer', 'parka', 'down', 'shearling', 'heavy-wool-coat'],
      note: 'Warm weather (23–25°C): keep outer layer lightweight, unlined, and breathable.',
    };
  }
  if (band === 'cold' || band === 'freezing') {
    return {
      band,
      maxOuterwearWeight: 'insulating',
      requireBreathableLayer: false,
      requireInsulatingOuterwear: true,
      preferredLayerSubtypes:
        band === 'freezing'
          ? ['puffer', 'parka', 'down-coat', 'shearling', 'wool-overcoat', 'insulated-coat']
          : ['overcoat', 'topcoat', 'peacoat', 'wool-coat', 'puffer', 'quilted-jacket', 'trench-lined'],
      forbiddenOuterwearTypes: ['linen-overshirt', 'sleeveless-vest', 'sheer-cardigan'],
      note:
        band === 'freezing'
          ? 'Freezing weather (<=2°C): require real insulating winter coat and weather-protective footwear.'
          : 'Cold weather (<=10°C): reward insulating wool coats, puffers, and lined winter outerwear.',
    };
  }
  if (band === 'cool') {
    return {
      band,
      maxOuterwearWeight: 'midweight',
      requireBreathableLayer: false,
      requireInsulatingOuterwear: false,
      preferredLayerSubtypes: ['blazer', 'trench', 'wool-cardigan', 'leather-jacket', 'suede-jacket', 'overshirt', 'bomber'],
      forbiddenOuterwearTypes: ['heavy-arctic-parka'],
      note: 'Cool transitional weather (11–17°C): tailored midweight layers, trenches, blazers, and fine knitwear excel.',
    };
  }
  return {
    band,
    maxOuterwearWeight: 'midweight',
    requireBreathableLayer: false,
    requireInsulatingOuterwear: false,
    preferredLayerSubtypes: ['blazer', 'overshirt', 'cardigan', 'trench', 'light-jacket'],
    forbiddenOuterwearTypes: [],
    note: 'Mild or unspecified temperature: versatile balanced layering.',
  };
}

function slotCount(counts: CatalogSlotCounts | null | undefined, slot: GrammarSlot): number {
  const val = Number(counts?.[slot] ?? 0);
  return Number.isFinite(val) && val > 0 ? Math.floor(val) : 0;
}

function hasInventoryForGrammar(
  grammarId: OutfitGrammarId,
  counts: CatalogSlotCounts | null | undefined,
  minPerSlot: number,
): boolean {
  return OUTFIT_GRAMMAR_SLOTS[grammarId].every((slot) => slotCount(counts, slot) >= minPerSlot);
}

/**
 * Resolve the active 4-item outfit grammar based on climate, user context,
 * presentation, and REAL verified catalog slot counts. Never drops below 4 items
 * and never claims accessory/dress grammars when catalog inventory is missing.
 */
export function resolveOutfitGrammar(
  context?: OutfitGrammarContext | null,
  catalogSlotCounts?: CatalogSlotCounts | null,
): ResolvedOutfitGrammar {
  const ctx = context && typeof context === 'object' ? context : {};
  const counts = catalogSlotCounts && typeof catalogSlotCounts === 'object' ? catalogSlotCounts : {};
  const minPerSlot = Math.max(1, Math.floor(Number(ctx.minItemsPerSlot) || 1));

  const request = String(ctx.request || '');
  const occasion = String(ctx.occasion || '').toLowerCase();
  const presentation = String(ctx.presentation || 'menswear').toLowerCase();
  const band = resolveThermalBand(ctx.weather, request);

  const explicitNoLayerIntent =
    /\b(?:no|without|skip)\s+(?:(?:an?|any|extra|additional|heavy)\s+){0,2}(?:layers?|outerwear|jacket|blazer|coat|cardigan)\b/i.test(request) ||
    /без\s+(?:пиджака|куртки|пальто|кардигана|верхней\s+одежды)/i.test(request);

  const explicitDressIntent =
    /\b(dress|gown|jumpsuit|one[ -]?piece|slip dress|midi dress|maxi dress)\b|плать/i.test(request);

  const isWomenswear = presentation === 'womenswear' || presentation === 'female';
  const isWarmOrHot = band === 'hot' || (band === 'warm' && explicitNoLayerIntent);

  // 1. One-piece editorial grammar (dress + outerwear + shoes + accessory)
  const wantsOnePiece =
    ctx.preferredGrammar === 'one_piece_editorial' ||
    explicitDressIntent ||
    (isWomenswear &&
      ctx.preferredGrammar !== 'layered_separates' &&
      ctx.preferredGrammar !== 'warm_weather_separates' &&
      /\b(formal|date|gala|cocktail|evening|editorial|wedding)\b/.test(occasion) &&
      slotCount(counts, 'dress') >= minPerSlot &&
      slotCount(counts, 'accessory') >= minPerSlot);

  if (wantsOnePiece) {
    if (hasInventoryForGrammar('one_piece_editorial', counts, minPerSlot)) {
      return {
        grammarId: 'one_piece_editorial',
        slots: OUTFIT_GRAMMAR_SLOTS.one_piece_editorial,
        fourItemContract: true,
        thermalBand: band,
        thermalConstraints: buildThermalConstraints(band, 'one_piece_editorial'),
        fallbackApplied: false,
        fallbackReason: null,
        requiredCountsMet: true,
      };
    }
    const missingSlots = OUTFIT_GRAMMAR_SLOTS.one_piece_editorial.filter(
      (s) => slotCount(counts, s) < minPerSlot,
    );
    return {
      grammarId: 'layered_separates',
      slots: OUTFIT_GRAMMAR_SLOTS.layered_separates,
      fourItemContract: true,
      thermalBand: band,
      thermalConstraints: buildThermalConstraints(band, 'layered_separates'),
      fallbackApplied: true,
      fallbackReason: `insufficient_inventory_for_one_piece_editorial:missing_${missingSlots.join('_')}`,
      requiredCountsMet: hasInventoryForGrammar('layered_separates', counts, minPerSlot),
    };
  }

  // 2. Warm-weather separates (top + bottom + shoes + accessory)
  const wantsWarmSeparates =
    ctx.preferredGrammar === 'warm_weather_separates' ||
    (isWarmOrHot && ctx.preferredGrammar !== 'layered_separates') ||
    (explicitNoLayerIntent && band !== 'freezing' && band !== 'cold');

  if (wantsWarmSeparates) {
    if (hasInventoryForGrammar('warm_weather_separates', counts, minPerSlot)) {
      return {
        grammarId: 'warm_weather_separates',
        slots: OUTFIT_GRAMMAR_SLOTS.warm_weather_separates,
        fourItemContract: true,
        thermalBand: band,
        thermalConstraints: buildThermalConstraints(band, 'warm_weather_separates'),
        fallbackApplied: false,
        fallbackReason: null,
        requiredCountsMet: true,
      };
    }
    return {
      grammarId: 'layered_separates',
      slots: OUTFIT_GRAMMAR_SLOTS.layered_separates,
      fourItemContract: true,
      thermalBand: band,
      thermalConstraints: buildThermalConstraints(band, 'layered_separates'),
      fallbackApplied: true,
      fallbackReason:
        slotCount(counts, 'accessory') < minPerSlot
          ? 'no_accessory_inventory_for_warm_weather_separates'
          : 'insufficient_inventory_for_warm_weather_separates',
      requiredCountsMet: hasInventoryForGrammar('layered_separates', counts, minPerSlot),
    };
  }

  // 3. Default binding 4-item layered separates (top + outerwear + bottom + shoes)
  return {
    grammarId: 'layered_separates',
    slots: OUTFIT_GRAMMAR_SLOTS.layered_separates,
    fourItemContract: true,
    thermalBand: band,
    thermalConstraints: buildThermalConstraints(band, 'layered_separates'),
    fallbackApplied: false,
    fallbackReason: null,
    requiredCountsMet: hasInventoryForGrammar('layered_separates', counts, minPerSlot),
  };
}

const HEAVY_INSULATING_RE = /\b(puffers?|parkas?|down[ -]filled|goose[ -]down|duck[ -]down|down\s+(?:jacket|coat|parka)|shearling|quilted|padded|insulated|fur[ -]lined|fleece[ -]lined|thermal|snow\s+boots?|winter\s+boots?|winter\s+coat)\b|пухов|утепл|ст[её]ган|зимн/i;
const HEAVY_WOOL_RE = /\b(heavy\s+wool|boiled\s+wool|thick\s+wool|overcoats?|topcoats?|peacoats?|trench\s+coat|cashmere\s+coat|wool\s+coat|tweed|heavy\s+flannel|chunky\s+knit|cable[ -]knit|turtleneck|corduroy|velvet|melton)\b|пальто|шерст/i;
const BREATHABLE_SUMMER_RE = /\b(linen|seersucker|poplin|chambray|unlined|open[ -]weave|tropical\s+wool|lightweight|sheer|mesh|ramie|lyocell|tencel|viscose|silk|cotton[ -]linen|breathable|short[ -]sleeve|sleeveless|camp[ -]collar|resort|espadrilles?|sandals?|slides?|driving\s+shoes?|unlined\s+loafers?)\b|лён|лен|льнян|легк/i;
const MIDWEIGHT_COTTON_RE = /\b(cotton|overshirt|shacket|cardigan|fine[ -]gauge|merino|twill|gabardine|chino|denim|jersey|polo|shirt[ -]jacket|blazer|suede|loafer|derby|oxford|sneaker|chelsea)\b/i;
const BOOT_RE = /\b(boots?|chelsea|chukka|lace[ -]up\s+boot|combat\s+boot|dress\s+boot)\b|ботинк|сапог/i;
const OPEN_SUMMER_SHOE_RE = /\b(sandals?|slides?|flip[ -]flops?|espadrilles?|mules?)\b|сандал|шлепан/i;

function itemTextBlob(item: ThermalGarmentInput): string {
  const mat =
    typeof item.material === 'string'
      ? item.material
      : Array.isArray(item.material)
        ? item.material.filter((x) => typeof x === 'string').join(' ')
        : '';
  const mats = Array.isArray(item.materials)
    ? item.materials.filter((x) => typeof x === 'string').join(' ')
    : '';
  return [item.name, item.type, item.subtype, item.description, mat, mats, item.layering]
    .filter((x) => typeof x === 'string' && x.trim().length > 0)
    .join(' ')
    .toLowerCase();
}

/**
 * Score how thermally appropriate an individual garment is for the target weather.
 *  - Warm/hot weather (>=26°C) strongly prefers breathable linen/cotton/unlined
 *    lightweight open layers and penalizes heavy wool/puffer/insulated coats.
 *  - Cold/freezing weather (<=10°C) rewards real insulating outerwear and boots
 *    while penalizing unlined linen summer layers and open summer footwear.
 */
export function scoreThermalSuitability(
  item: ThermalGarmentInput | null | undefined,
  weather?: GrammarWeatherInput | null,
): ThermalSuitabilityResult {
  const band = resolveThermalBand(weather);
  if (!item || typeof item !== 'object') {
    return { score: 0.5, band, suitable: true, reasons: ['missing_item'] };
  }
  if (band === 'unknown') {
    return { score: 0.65, band: 'unknown', suitable: true, reasons: ['weather_unknown'] };
  }

  const slot = String(item.macroCategory || item.slot || item.category || 'unknown').toLowerCase();
  const text = itemTextBlob(item);
  const isInsulated = item.insulated === true || HEAVY_INSULATING_RE.test(text);
  const isHeavyWool = HEAVY_WOOL_RE.test(text) || item.bulk === 'heavy';
  const isBreathable = BREATHABLE_SUMMER_RE.test(text) || item.bulk === 'light';
  const isMidweight = MIDWEIGHT_COTTON_RE.test(text);
  const isBoot = BOOT_RE.test(text);
  const isOpenSummerShoe = OPEN_SUMMER_SHOE_RE.test(text);

  const reasons: string[] = [];
  let score = 0.65;
  let suitable = true;

  if (band === 'hot') {
    if (isInsulated) {
      score = 0.06;
      suitable = false;
      reasons.push('heavy_insulation_unsuitable_for_hot_weather');
    } else if (isHeavyWool) {
      score = slot === 'outerwear' ? 0.16 : 0.24;
      suitable = false;
      reasons.push('heavy_winter_fabric_unsuitable_for_hot_weather');
    } else if (slot === 'shoes' && isBoot) {
      score = 0.25;
      suitable = false;
      reasons.push('heavy_boots_unsuitable_for_hot_weather');
    } else if (isBreathable) {
      score = slot === 'outerwear' ? 0.94 : 0.96;
      reasons.push(
        slot === 'outerwear'
          ? 'breathable_lightweight_open_layer_ideal_for_hot_weather'
          : 'breathable_lightweight_fabric_ideal_for_hot_weather',
      );
    } else if (slot === 'outerwear' && /\b(overshirt|shirt[ -]jacket|cardigan)\b/.test(text) && /\bcotton\b/.test(text)) {
      score = 0.86;
      reasons.push('light_cotton_layer_suitable_for_hot_weather');
    } else if (isMidweight) {
      score = slot === 'outerwear' ? 0.56 : 0.72;
      reasons.push('midweight_acceptable_in_warm_weather');
    }
  } else if (band === 'warm') {
    if (isInsulated) {
      score = 0.12;
      suitable = false;
      reasons.push('insulated_outerwear_too_heavy_for_warm_weather');
    } else if (isHeavyWool) {
      score = slot === 'outerwear' ? 0.30 : 0.42;
      suitable = slot !== 'outerwear';
      reasons.push('heavy_wool_too_warm_for_warm_weather');
    } else if (isBreathable) {
      score = 0.92;
      reasons.push('breathable_fabric_ideal_for_warm_weather');
    } else if (isMidweight) {
      score = 0.78;
      reasons.push('light_to_midweight_well_suited_for_warm_weather');
    }
  } else if (band === 'cold' || band === 'freezing') {
    const freezing = band === 'freezing';
    if (slot === 'outerwear') {
      if (isInsulated) {
        score = 0.96;
        reasons.push('insulating_outerwear_ideal_for_cold_weather');
      } else if (isHeavyWool) {
        score = freezing ? 0.86 : 0.92;
        reasons.push('wool_tailored_outerwear_well_suited_for_cold_weather');
      } else if (isBreathable) {
        score = freezing ? 0.08 : 0.16;
        suitable = false;
        reasons.push('breathable_summer_layer_insufficient_for_cold_weather');
      } else if (/\b(blazer|cardigan|overshirt|denim\s+jacket)\b/.test(text)) {
        score = freezing ? 0.28 : 0.52;
        suitable = !freezing;
        reasons.push(
          freezing
            ? 'uninsulated_mid_layer_insufficient_for_freezing_weather'
            : 'midweight_layer_borderline_for_cold_weather',
        );
      }
    } else if (slot === 'shoes') {
      if (isOpenSummerShoe) {
        score = 0.05;
        suitable = false;
        reasons.push('open_summer_footwear_unsuitable_for_cold_weather');
      } else if (isBoot || isInsulated) {
        score = 0.94;
        reasons.push('protective_boots_ideal_for_cold_weather');
      } else if (/\b(leather|suede|derby|oxford|loafer)\b/.test(text)) {
        score = freezing ? 0.55 : 0.76;
        reasons.push('closed_leather_footwear_acceptable_for_cold_weather');
      }
    } else {
      if (isHeavyWool || isInsulated || /\b(merino|cashmere|flannel|knit|wool|turtleneck)\b/.test(text)) {
        score = 0.90;
        reasons.push('warm_winter_fabric_ideal_for_cold_weather');
      } else if (/\b(linen|seersucker|sleeveless|shorts?|short[ -]sleeve)\b/.test(text)) {
        score = freezing ? 0.14 : 0.24;
        suitable = false;
        reasons.push('summer_garment_unsuitable_for_cold_weather');
      } else {
        score = 0.68;
      }
    }
  } else if (band === 'cool') {
    if (isInsulated && /\b(arctic|heavy\s+parka|snow\s+boot)\b/.test(text)) {
      score = 0.48;
      reasons.push('heavy_freezing_gear_slightly_warm_for_cool_weather');
    } else if (slot === 'shoes' && isOpenSummerShoe) {
      score = 0.22;
      suitable = false;
      reasons.push('open_summer_footwear_too_cold_for_cool_weather');
    } else if (isHeavyWool || isMidweight || isBoot) {
      score = 0.88;
      reasons.push('transitional_construction_ideal_for_cool_weather');
    } else if (isBreathable) {
      score = slot === 'outerwear' ? 0.45 : 0.60;
      reasons.push('light_summer_fabric_under_insulated_for_cool_weather');
    }
  } else if (band === 'mild') {
    if (isInsulated) {
      score = 0.22;
      suitable = false;
      reasons.push('heavy_winter_insulation_too_warm_for_mild_weather');
    } else if (isHeavyWool && slot === 'outerwear') {
      score = 0.42;
      reasons.push('heavy_overcoat_warm_for_mild_weather');
    } else if (isMidweight || isBreathable) {
      score = 0.86;
      reasons.push('comfortable_for_mild_weather');
    }
  }

  return {
    score: round3(clamp01(score)),
    band,
    suitable,
    reasons: [...new Set(reasons)],
  };
}
