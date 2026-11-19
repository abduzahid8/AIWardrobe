/**
 * Stylist 6 — Surgical Single-Slot Visual Refiner & Interactive 1-Tap Swap Engine.
 *
 * Solves the 39.4% partial-deck failure rate without weakening quality bars:
 * When a 4-item outfit has 3 strong pieces and 1 weak/clashing slot
 * (`critique.weakestSlot`), LOCK the 3 strong pieces, search the full
 * `catalogPool` for replacement SKUs matching `critique.repairDirective`,
 * verify hard gates (SKU/budget/weather/anchors), re-render the editorial
 * board (hitting the photo cache at 0ms CDN cost for the 3 locked items),
 * and re-evaluate with the independent Visual Critic.
 *
 * Commit ONLY if hard gates pass AND the composite visual score strictly
 * improves and clears the editorial threshold — NEVER lower the quality floor.
 *
 * Also exposes `swapSingleSlot` for instant interactive 1-tap "Swap This Item"
 * from the mobile UI.
 */

export const SURGICAL_REFINER_VERSION = 's6-surgical-refiner-v1';
export const DEFAULT_MIN_EDITORIAL_SCORE = 7.0;
export const DEFAULT_MIN_VISUAL_QUALITY = 0.68;

const CANONICAL_SLOTS = ['top', 'outerwear', 'bottom', 'shoes'];
const ALL_SWAPPABLE_SLOTS = ['top', 'outerwear', 'bottom', 'shoes', 'accessory', 'dress'];

const COLOR_ALIASES = Object.freeze({
  gray: 'grey',
  charcoal: 'grey',
  slate: 'grey',
  '#0a1931': 'navy',
  midnight: 'navy',
  ecru: 'cream',
  offwhite: 'ivory',
  'off-white': 'ivory',
  chocolate: 'brown',
  espresso: 'brown',
  cognac: 'tan',
  chestnut: 'brown',
  bordeaux: 'burgundy',
  wine: 'burgundy',
  oxblood: 'burgundy',
});

const WARM_COLORS = new Set([
  'camel', 'tan', 'cognac', 'brown', 'chocolate', 'espresso', 'beige',
  'sand', 'cream', 'ivory', 'olive', 'khaki', 'rust', 'terracotta',
  'burgundy', 'gold', 'mustard', 'amber', 'taupe', 'stone', 'ochre',
]);
const COOL_COLORS = new Set([
  'black', 'white', 'grey', 'charcoal', 'slate', 'navy', 'blue', 'sky',
  'cobalt', 'silver', 'icy', 'lavender', 'plum', 'emerald', 'teal', 'cool',
]);

const KNOWN_COLORS = [
  'black', 'white', 'ivory', 'cream', 'beige', 'sand', 'stone', 'taupe',
  'camel', 'tan', 'cognac', 'brown', 'espresso', 'chocolate', 'grey', 'gray',
  'charcoal', 'navy', 'blue', 'sky', 'green', 'olive', 'sage', 'khaki',
  'burgundy', 'bordeaux', 'wine', 'oxblood', 'red', 'pink', 'purple', 'yellow', 'orange',
];

const KNOWN_SUBTYPES = [
  'dress-shirt', 'shirt', 'polo', 'fine-knit', 'turtleneck', 'mockneck', 't-shirt',
  'blazer', 'coat', 'overcoat', 'trench', 'overshirt', 'jacket', 'cardigan', 'bomber',
  'tailored-trouser', 'pleated-trouser', 'trouser', 'chino', 'jeans',
  'oxford', 'derby', 'loafer', 'monk', 'chelsea', 'boot', 'driver',
  'minimal-sneaker', 'sneaker', 'sport-sneaker', 'espadrille',
];

const KNOWN_TEXTURES = [
  'suede', 'leather', 'wool', 'cashmere', 'merino', 'linen', 'cotton',
  'knit', 'silk', 'velvet', 'corduroy', 'denim', 'twill', 'flannel', 'poplin', 'smooth',
];

const FORMALITY_RANK = Object.freeze({
  formal: 4,
  smart: 3,
  smart_casual: 2.5,
  casual: 2,
  sport: 1,
  athletic: 1,
  beach: 0,
  unknown: 2.2,
});

const SUBTYPE_FORMALITY = Object.freeze({
  oxford: 4,
  monk: 3.8,
  derby: 3.5,
  loafer: 3.1,
  chelsea: 3.0,
  boot: 2.7,
  driver: 2.4,
  'minimal-sneaker': 2.0,
  sneaker: 1.5,
  'sport-sneaker': 1.0,
  espadrille: 1.5,
  'dress-shirt': 3.8,
  shirt: 3.0,
  'fine-knit': 2.9,
  polo: 2.4,
  't-shirt': 1.6,
  blazer: 3.8,
  coat: 3.5,
  overcoat: 3.7,
  trench: 3.3,
  cardigan: 2.7,
  overshirt: 2.3,
  jacket: 2.4,
  bomber: 2.0,
  'tailored-trouser': 3.7,
  'pleated-trouser': 3.6,
  trouser: 3.1,
  chino: 2.5,
  jeans: 1.9,
});

function canonicalColor(raw) {
  if (typeof raw !== 'string') return '';
  const clean = raw.trim().toLowerCase();
  if (!clean) return '';
  return COLOR_ALIASES[clean] || clean;
}

function itemSlotOf(item) {
  return String(item?.slot || item?.macroCategory || '').trim().toLowerCase();
}

function itemColorOf(item) {
  const explicit = canonicalColor(item?.color || item?.primaryColor || item?.primary_color || item?.visualColor);
  if (explicit && explicit !== 'unknown' && explicit !== 'neutral') return explicit;
  const text = String(item?.name || '').toLowerCase();
  for (const c of KNOWN_COLORS) {
    if (new RegExp(`\\b${c}\\b`).test(text)) return canonicalColor(c);
  }
  return '';
}

function itemSubtypeOf(item) {
  const raw = String(item?.subtype || item?.fashion_subtype || '').trim().toLowerCase().replace(/_/g, '-');
  if (raw && raw !== 'unknown') return raw;
  const text = String(item?.name || '').toLowerCase();
  for (const s of KNOWN_SUBTYPES) {
    const pattern = s.replace(/-/g, '[\\s-]?');
    if (new RegExp(`\\b${pattern}s?\\b`).test(text)) return s;
  }
  return '';
}

function itemTextureOf(item) {
  const raw = String(item?.texture || item?.material || '').trim().toLowerCase();
  if (raw && raw !== 'unknown' && raw !== 'smooth') return raw;
  const text = `${raw} ${String(item?.name || '').toLowerCase()}`;
  for (const t of KNOWN_TEXTURES) {
    if (new RegExp(`\\b${t}\\b`).test(text)) return t;
  }
  return raw || '';
}

function itemFormalityScore(item) {
  const f = String(item?.formality || '').trim().toLowerCase();
  if (f && f in FORMALITY_RANK && f !== 'unknown') return FORMALITY_RANK[f];
  const sub = itemSubtypeOf(item);
  if (sub in SUBTYPE_FORMALITY) return SUBTYPE_FORMALITY[sub];
  return 2.4;
}

function colorTemperatureOf(color) {
  const c = canonicalColor(color);
  if (!c) return 'unknown';
  if (WARM_COLORS.has(c)) return 'warm';
  if (COOL_COLORS.has(c)) return 'cool';
  return 'neutral';
}

/**
 * Parse a Visual Critic repairDirective (or user 1-tap swap reason) into
 * structured fashion targeting attributes. Safe against prompt injection or
 * arbitrary free-text input.
 */
export function parseRepairDirective(directive = '', defectCategory = '') {
  const rawText = `${String(directive || '')} ${String(defectCategory || '')}`
    .replace(/https?:\/\/[^\s]+/gi, ' ')
    .replace(/[\u0000-\u001f]/g, ' ')
    .slice(0, 400)
    .toLowerCase();

  const targetColors = new Set();
  const avoidColors = new Set();
  const targetSubtypes = new Set();
  const avoidSubtypes = new Set();
  const targetTextures = new Set();

  // Split around replacement cues ("replace X with Y", "swap X for Y", "instead of X")
  const avoidSegmentMatches = [
    ...rawText.matchAll(/(?:avoid|not|no|remove|replace|swap out|instead of|ditch|clash(?:es|ing)? with)\s+([^,.;]{3,65})/gi),
  ].map((m) => m[1]);
  const targetSegmentMatches = [
    ...rawText.matchAll(/(?:with|for|use|choose|prefer|try|switch to|into|need(?:s)?|add|opt for)\s+([^,.;]{3,80})/gi),
  ].map((m) => m[1]);

  const avoidZone = avoidSegmentMatches.join(' ');
  const targetZone = targetSegmentMatches.length ? targetSegmentMatches.join(' ') : rawText;

  for (const c of KNOWN_COLORS) {
    const re = new RegExp(`\\b${c}\\b`);
    if (re.test(targetZone)) targetColors.add(canonicalColor(c));
    if (re.test(avoidZone) && !re.test(targetZone)) avoidColors.add(canonicalColor(c));
  }

  for (const sub of KNOWN_SUBTYPES) {
    const re = new RegExp(`\\b${sub.replace(/-/g, '[\\s-]?')}s?\\b`);
    if (re.test(targetZone)) targetSubtypes.add(sub);
    if (re.test(avoidZone) && !re.test(targetZone)) avoidSubtypes.add(sub);
  }
  // Common synonym mappings in fashion critique prose
  if (/\b(?:trainer|running shoe|chunky sneaker|athletic shoe)\b/.test(avoidZone)) {
    avoidSubtypes.add('sneaker');
    avoidSubtypes.add('sport-sneaker');
  }
  if (/\b(?:dress shoe|sleek leather shoe|polished leather)\b/.test(targetZone)) {
    targetSubtypes.add('oxford');
    targetSubtypes.add('derby');
    targetSubtypes.add('loafer');
  }

  for (const tex of KNOWN_TEXTURES) {
    if (new RegExp(`\\b${tex}\\b`).test(targetZone)) targetTextures.add(tex);
  }

  let targetTemperature = 'neutral';
  if (/\b(?:warm(?:er)?|earth(?:y)?|golden|camel|brown|cognac|olive|cream|espresso|autumn)\b/.test(targetZone)) {
    targetTemperature = 'warm';
  } else if (/\b(?:cool(?:er)?|crisp|charcoal|navy|silver|icy|slate|monochrome|black)\b/.test(targetZone)) {
    targetTemperature = 'cool';
  }

  let formalityShift = 'neutral';
  if (/\b(?:too[ _-]?casual|more formal|dressier|elevat(?:e|ed)|tailored|sleeker|polished|refined|formal(?:ity)?_mismatch|work|office|formal)\b/.test(rawText)
      && !/\b(?:too[ _-]?formal|more casual|relaxed|dress down)\b/.test(rawText)) {
    formalityShift = 'more_formal';
  } else if (/\b(?:too[ _-]?formal|too[ _-]?stiff|more casual|relaxed|softer|less formal|dress down|weekend)\b/.test(rawText)) {
    formalityShift = 'more_casual';
  }

  const budgetDirection = /\b(?:over[ _-]?budget|cheaper|affordable|lower[ _-]?price|less expensive)\b/.test(rawText)
    ? 'cheaper'
    : null;

  const fitShift = /\b(?:not[ _-]?my[ _-]?fit|slimmer|tailored fit|sharper cut)\b/.test(rawText)
    ? 'structured'
    : /\b(?:relaxed fit|wider|looser|drapey|flowy)\b/.test(rawText)
      ? 'relaxed'
      : null;

  const isPhotoDefect = String(defectCategory || '').trim().toLowerCase() === 'photo_quality_defect'
    || /\b(?:crop(?:ped)?|close[ -]?up|cuff|sleeve zoom|collar[ -]?only|partial[ -]?garment|blurry|broken|low[ -]?res|swatch|full[ -]?garment)\b/.test(rawText);

  const lighterLayer = String(defectCategory || '').trim().toLowerCase() === 'weather_mismatch'
    || /\b(?:lighter|breathable|unlined|summer|warm[ -]?weather|too heavy|heavy coat|overcoat|peacoat|parka|indoor)\b/.test(rawText);

  if (lighterLayer) {
    for (const heavy of ['overcoat', 'coat', 'peacoat', 'parka', 'puffer']) {
      if (!targetSubtypes.has(heavy)) avoidSubtypes.add(heavy);
    }
    if (targetSubtypes.size === 0) {
      targetSubtypes.add('overshirt');
      targetSubtypes.add('cardigan');
      targetSubtypes.add('blazer');
    }
  }

  return {
    targetColors: [...targetColors],
    avoidColors: [...avoidColors],
    targetTemperature,
    targetSubtypes: [...targetSubtypes],
    avoidSubtypes: [...avoidSubtypes],
    targetTextures: [...targetTextures],
    formalityShift,
    budgetDirection,
    fitShift,
    isPhotoDefect,
    lighterLayer,
  };
}

/**
 * Normalize composite visual score on BOTH 0–10 (`score10`) and 0–1 (`score01`)
 * scales so surgical refinement never lowers existing quality thresholds.
 */
export function computeCritiqueCompositeScore(critique, {
  minEditorialScore = DEFAULT_MIN_EDITORIAL_SCORE,
  minVisualQuality = DEFAULT_MIN_VISUAL_QUALITY,
} = {}) {
  if (!critique || typeof critique !== 'object') {
    return { score10: 0, score01: 0, passedFloor: false, rejected: true, source: 'missing' };
  }
  const hasEditorialScore = typeof critique.editorialScore === 'number'
    && Number.isFinite(critique.editorialScore);
  const hasLegacyScores = ['visualCohesion', 'styleFit', 'occasionFit', 'formalityFit']
    .every((k) => typeof critique[k] === 'number' && Number.isFinite(critique[k]));

  let score01 = 0;
  if (hasLegacyScores) {
    score01 = critique.referenceless === true || typeof critique.referenceFidelity !== 'number'
      ? 0.45 * critique.visualCohesion + 0.20 * critique.styleFit + 0.20 * critique.occasionFit + 0.15 * critique.formalityFit
      : 0.30 * critique.visualCohesion + 0.30 * critique.referenceFidelity + 0.15 * critique.styleFit + 0.15 * critique.occasionFit + 0.10 * critique.formalityFit;
  } else if (hasEditorialScore) {
    score01 = Math.max(0, Math.min(1, critique.editorialScore / 10));
  }

  const score10 = hasEditorialScore
    ? Math.max(0, Math.min(10, critique.editorialScore))
    : Number((score01 * 10).toFixed(2));

  const confidenceOk = critique.confidence === undefined || (typeof critique.confidence === 'number' && critique.confidence >= 0.6);
  const notRejected = critique.reject !== true;
  const editorialOk = !hasEditorialScore || score10 >= minEditorialScore;
  const legacyOk = !hasLegacyScores || score01 >= minVisualQuality;
  const passedFloor = notRejected && confidenceOk && editorialOk && legacyOk && (hasEditorialScore || hasLegacyScores);

  return {
    score10: Number(score10.toFixed(2)),
    score01: Number(score01.toFixed(4)),
    passedFloor,
    rejected: critique.reject === true,
    source: hasEditorialScore ? '6axis_editorial' : hasLegacyScores ? 'legacy_visual' : 'missing',
  };
}

function normalizeColorList(list) {
  if (!Array.isArray(list)) return new Set();
  return new Set(list.map((c) => canonicalColor(String(c || ''))).filter(Boolean));
}

function normalizeStringSet(list) {
  if (!Array.isArray(list)) return new Set();
  return new Set(list.map((s) => String(s || '').trim().toLowerCase()).filter(Boolean));
}

function computeBudgetHeadroom(lockedItems, context, priceToBudgetCurrency) {
  const amount = Number(context?.budget?.amount);
  const currency = typeof context?.budget?.currency === 'string' ? context.budget.currency.trim().toUpperCase() : null;
  if (!Number.isFinite(amount) || amount <= 0) return null;

  const convert = (item) => {
    if (typeof priceToBudgetCurrency === 'function') {
      const val = Number(priceToBudgetCurrency(item, currency));
      return Number.isFinite(val) ? val : null;
    }
    const p = Number(item?.price);
    const c = typeof item?.currency === 'string' ? item.currency.trim().toUpperCase() : null;
    if (!Number.isFinite(p) || p <= 0) return 0;
    if (!currency || !c || c === currency) return p;
    return null; // Unknown FX without converter: do not fabricate conversion!
  };

  let lockedSum = 0;
  for (const item of lockedItems) {
    const p = convert(item);
    if (p === null) return null;
    lockedSum += p;
  }
  return {
    maxSlotPrice: Math.max(0, amount - lockedSum),
    currency,
    convert,
  };
}

function dominantTemperatureOfItems(items) {
  let warm = 0;
  let cool = 0;
  for (const item of items) {
    const t = colorTemperatureOf(itemColorOf(item));
    if (t === 'warm') warm++;
    else if (t === 'cool') cool++;
  }
  if (warm > cool) return 'warm';
  if (cool > warm) return 'cool';
  return 'neutral';
}

/**
 * Rank replacement candidates in `targetSlot` from the full `catalogPool`
 * against the critique repair directive, locked-item harmony, Personal Style DNA,
 * budget headroom, zero SKU reuse, and catalog quality scores.
 */
export function rankSlotReplacementCandidates({
  look,
  targetSlot,
  directive = '',
  defectCategory = '',
  catalogPool = [],
  context = {},
  personalStyleDna = null,
  excludedIds = new Set(),
  qualityMap = null,
  priceToBudgetCurrency = null,
  limit = 8,
} = {}) {
  const slot = String(targetSlot || '').trim().toLowerCase();
  if (!ALL_SWAPPABLE_SLOTS.includes(slot) || !Array.isArray(catalogPool)) return [];

  const lookItems = Array.isArray(look?.items) ? look.items : [];
  const currentItem = lookItems.find((i) => itemSlotOf(i) === slot) || null;
  const lockedItems = lookItems.filter((i) => itemSlotOf(i) !== slot);

  const blockedIds = new Set(Array.from(excludedIds || [], String));
  for (const item of lookItems) {
    if (item?.id != null) blockedIds.add(String(item.id));
  }

  const parsed = parseRepairDirective(directive, defectCategory);
  const dnaAvoidColors = new Set([
    ...normalizeColorList(personalStyleDna?.avoidedColors),
    ...normalizeColorList(personalStyleDna?.avoidColors),
    ...normalizeColorList(personalStyleDna?.appearance?.avoidColors),
    ...normalizeColorList(context?.preferences?.avoidColors),
    ...normalizeColorList(context?.avoidColors),
  ]);
  const dnaFavoriteColors = new Set([
    ...normalizeColorList(personalStyleDna?.favoriteColors),
    ...normalizeColorList(personalStyleDna?.appearance?.flatteringColors),
    ...normalizeColorList(personalStyleDna?.appearance?.powerNeutrals),
    ...normalizeColorList(context?.preferences?.favoriteColors),
  ]);
  const excludedBrands = new Set([
    ...normalizeStringSet(personalStyleDna?.brandExclusions),
    ...normalizeStringSet(context?.preferences?.brandExclusions),
  ]);
  const preferredBrands = new Set([
    ...normalizeStringSet(personalStyleDna?.preferredBrands),
    ...normalizeStringSet(context?.preferences?.favoriteBrands),
  ]);
  const dislikedPatterns = new Set([
    ...normalizeStringSet(personalStyleDna?.dislikedPatterns),
    ...normalizeStringSet(context?.preferences?.dislikedPatterns),
    ...normalizeStringSet(context?.dislikedPatterns),
  ]);

  const budget = computeBudgetHeadroom(lockedItems, context, priceToBudgetCurrency);
  const lockedTemp = dominantTemperatureOfItems(lockedItems);
  const lockedAvgFormality = lockedItems.length
    ? lockedItems.reduce((sum, i) => sum + itemFormalityScore(i), 0) / lockedItems.length
    : 2.5;
  const currentColor = itemColorOf(currentItem);
  const currentSubtype = itemSubtypeOf(currentItem);
  const currentTexture = itemTextureOf(currentItem);
  const currentImageUrl = String(currentItem?.imageUrl || currentItem?.image_url || '').trim();
  const currentPrice = Number(currentItem?.price);
  const weatherTemp = Number(context?.weather?.temp ?? context?.weather?.temperature);
  const isWarmWeather = Number.isFinite(weatherTemp) && weatherTemp >= 24;
  const isIndoorMildWeather = Number.isFinite(weatherTemp) && weatherTemp > 10;

  const occasion = String(context?.occasion || '').toLowerCase();
  const occasionTargetFormality = /\b(formal|wedding|gala|black[ -]?tie)\b/.test(occasion)
    ? 3.8
    : /\b(work|office|business|meeting)\b/.test(occasion)
      ? 3.2
      : /\b(date|dinner|evening)\b/.test(occasion)
        ? 3.0
        : /\b(workout|gym|sport)\b/.test(occasion)
          ? 1.0
          : 2.3;

  const targetFormality = parsed.formalityShift === 'more_formal'
    ? Math.min(4.0, Math.max(occasionTargetFormality, lockedAvgFormality + 0.5))
    : parsed.formalityShift === 'more_casual'
      ? Math.max(1.5, Math.min(occasionTargetFormality, lockedAvgFormality - 0.5))
      : (lockedAvgFormality * 0.6 + occasionTargetFormality * 0.4);

  const scored = [];
  for (const item of catalogPool) {
    if (!item || item.id == null) continue;
    const id = String(item.id);
    if (blockedIds.has(id)) continue;
    if (itemSlotOf(item) !== slot) continue;
    const candidateImageUrl = String(item.imageUrl || item.image_url || '').trim();
    if (currentImageUrl && candidateImageUrl && candidateImageUrl === currentImageUrl) continue;

    // Quality flywheel quarantine gate
    const qRow = qualityMap instanceof Map ? qualityMap.get(id) : (qualityMap && typeof qualityMap === 'object' ? qualityMap[id] : null);
    if (qRow?.quarantined === true || item.quarantined === true) continue;

    const color = itemColorOf(item);
    const brand = String(item.brand || '').trim().toLowerCase();
    const pattern = String(item.pattern || '').trim().toLowerCase();

    // Hard personal exclusions
    if (color && dnaAvoidColors.has(color)) continue;
    if (brand && excludedBrands.has(brand)) continue;
    if (pattern && dislikedPatterns.has(pattern)) continue;

    // Hard budget headroom check
    let candidateBudgetPrice = null;
    if (budget) {
      candidateBudgetPrice = budget.convert(item);
      if (candidateBudgetPrice !== null && candidateBudgetPrice > budget.maxSlotPrice) continue;
    }

    const subtype = itemSubtypeOf(item);
    const texture = itemTextureOf(item);
    const formality = itemFormalityScore(item);
    const temp = colorTemperatureOf(color);
    const reasons = [];
    let score = 0.40;

    // 1. Repair directive color match / avoidance
    if (parsed.targetColors.length > 0 && color) {
      if (parsed.targetColors.includes(color)) {
        score += 0.28;
        reasons.push(`color_match:${color}`);
      }
    }
    if (parsed.avoidColors.length > 0 && color && parsed.avoidColors.includes(color)) {
      score -= 0.30;
      reasons.push(`avoided_color:${color}`);
    }

    // 2. Temperature harmony (directive temperature or locked-item temperature)
    const desiredTemp = parsed.targetTemperature !== 'neutral' ? parsed.targetTemperature : lockedTemp;
    if (desiredTemp !== 'neutral' && temp !== 'unknown') {
      if (temp === desiredTemp) {
        score += 0.14;
        reasons.push(`temp_match:${temp}`);
      } else if (temp !== 'neutral') {
        score -= 0.08;
      }
    }

    // 3. Subtype alignment
    if (parsed.targetSubtypes.length > 0 && subtype) {
      if (parsed.targetSubtypes.includes(subtype)) {
        score += 0.26;
        reasons.push(`subtype_match:${subtype}`);
      }
    }
    if (parsed.avoidSubtypes.length > 0 && subtype && parsed.avoidSubtypes.includes(subtype)) {
      score -= 0.32;
      reasons.push(`avoided_subtype:${subtype}`);
    }

    // 4. Formality alignment
    const formalityDelta = Math.abs(formality - targetFormality);
    score += Math.max(-0.18, 0.18 - formalityDelta * 0.11);
    if (formalityDelta <= 0.45) reasons.push('formality_aligned');

    // 5. Texture alignment
    if (parsed.targetTextures.length > 0 && texture && parsed.targetTextures.includes(texture)) {
      score += 0.14;
      reasons.push(`texture_match:${texture}`);
    }

    // 6. Personal Style DNA bonus
    if (color && dnaFavoriteColors.has(color)) {
      score += 0.10;
      reasons.push(`dna_color:${color}`);
    }
    if (brand && preferredBrands.has(brand)) {
      score += 0.06;
      reasons.push(`dna_brand:${brand}`);
    }

    // 7. Catalog quality flywheel score & verified photo bonus for photo_quality_defect rescue
    const qScore = Number.isFinite(Number(qRow?.qualityScore))
      ? Number(qRow.qualityScore)
      : Number.isFinite(Number(item.qualityScore))
        ? Number(item.qualityScore)
        : 0.65;
    score += (Math.max(0, Math.min(1, qScore)) - 0.5) * (parsed.isPhotoDefect ? 0.30 : 0.20);
    if (parsed.isPhotoDefect && item.visualEvidenceVerified === true) {
      score += 0.12;
      reasons.push('verified_full_studio_photo');
    }

    // 7b. Climate & lightweight layer suitability (hot >=24C or weather_mismatch rescue)
    const itemNameLower = String(item.name || '').toLowerCase();
    if (slot === 'outerwear' && (isWarmWeather || parsed.lighterLayer)) {
      const isHeavyOuter = /\b(?:overcoat|peacoat|parka|puffer|down|shearling|quilted|heavy wool|winter coat)\b/.test(itemNameLower)
        || ['overcoat', 'coat'].includes(subtype);
      const isBreathableOuter = /\b(?:linen|cotton|seersucker|unlined|lightweight|chambray|summer|overshirt|shacket|chore|fine[ -]?knit)\b/.test(itemNameLower)
        || ['overshirt', 'cardigan'].includes(subtype);
      if (isHeavyOuter && (isWarmWeather || isIndoorMildWeather || parsed.lighterLayer)) {
        score -= 0.35;
        reasons.push('heavy_outerwear_climate_mismatch');
      } else if (isBreathableOuter) {
        score += 0.22;
        reasons.push('breathable_layer_match');
      }
    } else if (slot === 'shoes' && isWarmWeather) {
      if (['boot', 'chelsea'].includes(subtype) || /\b(?:winter boot|snow boot|shearling|lug boot)\b/.test(itemNameLower)) {
        score -= 0.28;
        reasons.push('heavy_boot_hot_weather');
      } else if (['loafer', 'driver', 'minimal-sneaker', 'espadrille'].includes(subtype)) {
        score += 0.16;
        reasons.push('warm_weather_footwear');
      }
    }

    // 8. Anti-clone penalty vs the rejected/replaced item (skipped when replacing a cropped/broken photo
    // so a clean full-garment studio photo in the same color/subtype can directly rescue the look!)
    if (currentItem && !parsed.isPhotoDefect) {
      const sameSub = currentSubtype && subtype === currentSubtype;
      const sameCol = currentColor && color === currentColor;
      const sameTex = currentTexture && texture === currentTexture;
      if (sameSub && sameCol) {
        score -= sameTex ? 0.24 : 0.16;
        reasons.push('near_clone_of_replaced');
      }
    }

    // 9. Budget reduction preference when user requested cheaper replacement
    if (parsed.budgetDirection === 'cheaper' && Number.isFinite(currentPrice) && currentPrice > 0) {
      const p = Number(item.price);
      if (Number.isFinite(p) && p > 0) {
        if (p < currentPrice * 0.85) {
          score += 0.22;
          reasons.push('meaningfully_cheaper');
        } else if (p >= currentPrice) {
          score -= 0.25;
        }
      }
    }

    scored.push({
      item,
      score: Number(score.toFixed(4)),
      reasons,
      budgetPrice: candidateBudgetPrice,
    });
  }

  scored.sort((a, b) => b.score - a.score || String(a.item.id).localeCompare(String(b.item.id)));
  return scored.slice(0, Math.max(1, limit));
}

function buildReplacedLook(look, targetSlot, replacementItem, newLookId) {
  const slot = String(targetSlot).toLowerCase();
  const existingItems = Array.isArray(look?.items) ? look.items : [];
  let replaced = false;
  const nextItems = existingItems.map((item) => {
    if (!replaced && itemSlotOf(item) === slot) {
      replaced = true;
      // Preserve slot/macroCategory shape expected by downstream consumers
      return {
        ...replacementItem,
        ...(item.slot ? { slot: item.slot } : {}),
        ...(item.macroCategory ? { macroCategory: item.macroCategory } : { macroCategory: slot }),
      };
    }
    return item;
  });
  if (!replaced) {
    nextItems.push({ ...replacementItem, slot, macroCategory: slot });
  }

  // Honest total price recalculation when all shop items share one currency
  const shopItems = nextItems.filter((i) => Number.isFinite(Number(i?.price)) && Number(i.price) > 0);
  const currencies = new Set(shopItems.map((i) => String(i.currency || '').trim().toUpperCase()).filter(Boolean));
  const totalPrice = currencies.size === 1
    ? Number(shopItems.reduce((sum, i) => sum + Number(i.price), 0).toFixed(2))
    : undefined;

  return {
    ...look,
    id: newLookId || look?.id || 'refined_look',
    items: nextItems,
    ...(totalPrice !== undefined ? { totalPrice, currency: [...currencies][0] } : {}),
  };
}

async function runHardGate(hardGateValidator, candidateLook, context) {
  if (typeof hardGateValidator !== 'function') return { ok: true, reason: null };
  try {
    const res = await Promise.resolve(hardGateValidator(candidateLook.items, candidateLook, context));
    if (res === true) return { ok: true, reason: null };
    if (res && typeof res === 'object' && res.ok === true) return { ok: true, reason: null };
    return {
      ok: false,
      reason: (res && typeof res === 'object' && (res.reason || res.code)) || 'hard_gate_failed',
    };
  } catch (err) {
    return { ok: false, reason: err?.message || 'hard_gate_error' };
  }
}

async function evaluateWithVisualCritic(visualCritic, candidateLook, board, context, personalStyleDna, signal) {
  if (typeof visualCritic !== 'function') return null;
  const candidatePayload = {
    id: candidateLook.id,
    referenceId: candidateLook.referenceId || null,
    products: candidateLook.items.map((item) => ({
      id: String(item.id),
      slot: itemSlotOf(item),
      imageUrl: item.imageUrl || item.image_url || '',
      name: item.name || '',
      subtype: item.subtype || '',
    })),
    ...(board?.jpegBuffer ? { editorialBoardJpeg: board } : {}),
  };
  const criticReq = {
    ...(context || {}),
    ...(personalStyleDna?.promptBlock ? { personalStyleDnaPromptBlock: personalStyleDna.promptBlock } : {}),
  };
  const raw = await Promise.resolve(visualCritic([candidatePayload], criticReq, { signal, board, candidateLook }));
  if (Array.isArray(raw)) {
    return raw.find((v) => v?.candidateId === candidateLook.id) || raw[0] || null;
  }
  return raw && typeof raw === 'object' ? raw : null;
}

async function evaluateBatchWithVisualCritic(visualCritic, entries, context, personalStyleDna, signal) {
  if (typeof visualCritic !== 'function' || !Array.isArray(entries) || entries.length === 0) return new Map();
  const payloads = [];
  const editorialBoards = new Map();
  for (const entry of entries) {
    const { candidateLook, board } = entry;
    payloads.push({
      id: candidateLook.id,
      referenceId: candidateLook.referenceId || null,
      products: candidateLook.items.map((item) => ({
        id: String(item.id),
        slot: itemSlotOf(item),
        imageUrl: item.imageUrl || item.image_url || '',
        name: item.name || '',
        subtype: item.subtype || '',
      })),
      ...(board?.jpegBuffer ? { editorialBoardJpeg: board } : {}),
    });
    if (board?.jpegBuffer) editorialBoards.set(candidateLook.id, board);
  }
  const criticReq = {
    ...(context || {}),
    ...(personalStyleDna?.promptBlock ? { personalStyleDnaPromptBlock: personalStyleDna.promptBlock } : {}),
  };
  const raw = await Promise.resolve(visualCritic(payloads, criticReq, {
    signal,
    ...(editorialBoards.size ? { editorialBoards } : {}),
  }));
  const byId = new Map();
  if (Array.isArray(raw)) {
    for (const v of raw) {
      if (v && typeof v.candidateId === 'string') byId.set(v.candidateId, v);
    }
  } else if (raw && typeof raw === 'object' && typeof raw.candidateId === 'string') {
    byId.set(raw.candidateId, raw);
  }
  return byId;
}

/**
 * Surgically refine an imperfect 4-item look by locking its 3 strong pieces
 * and swapping ONLY `critique.weakestSlot` against the full `catalogPool`,
 * re-checking hard gates, re-rendering the studio board, and re-verifying with
 * the independent Visual Critic.
 *
 * Commits ONLY if hard gates pass AND the composite visual score strictly
 * improves AND clears the editorial threshold (never weakens quality floors).
 */
export async function refineLookSurgically({
  look,
  critique,
  catalogPool = [],
  lockedSlots = [],
  context = {},
  personalStyleDna = null,
  hardGateValidator = null,
  visualCritic = null,
  boardRenderer = null,
  maxRounds = 2,
  deadlineMs = 12_000,
  excludedItemIds = [],
  qualityMap = null,
  priceToBudgetCurrency = null,
  minEditorialScore = DEFAULT_MIN_EDITORIAL_SCORE,
  minVisualQuality = DEFAULT_MIN_VISUAL_QUALITY,
  topCandidatesPerRound = 3,
  parallelCandidates = false,
  signal = null,
} = {}) {
  const startedAt = Date.now();
  const boundedRounds = Math.max(1, Math.min(3, Math.floor(Number(maxRounds) || 2)));
  const deadlineAt = startedAt + Math.max(500, Math.min(45_000, Number(deadlineMs) || 12_000));
  const baseline = computeCritiqueCompositeScore(critique, { minEditorialScore, minVisualQuality });

  if (!look || !Array.isArray(look.items) || look.items.length < 3) {
    return {
      status: 'invalid_look',
      improved: false,
      committed: false,
      approved: false,
      look,
      critique,
      previousScore: baseline.score10,
      newScore: baseline.score10,
      rounds: 0,
      attempts: [],
      diagnostic: 'invalid_look_structure',
      version: SURGICAL_REFINER_VERSION,
    };
  }

  const userLockedSlots = new Set(
    Array.from(lockedSlots || [], (s) => String(s || '').trim().toLowerCase()).filter(Boolean),
  );
  const pinnedItemIds = new Set(Array.from(context?.requiredIds || [], String));
  const triedItemIds = new Set(Array.from(excludedItemIds || [], String));
  for (const item of look.items) {
    if (item?.id != null) triedItemIds.add(String(item.id));
  }

  let workingLook = look;
  let workingCritique = critique;
  let workingScore = baseline;
  let bestImprovement = null;
  const attempts = [];
  let roundsExecuted = 0;

  for (let round = 1; round <= boundedRounds; round++) {
    if (signal?.aborted || Date.now() >= deadlineAt) {
      return {
        status: bestImprovement ? 'improved_below_threshold' : 'deadline',
        improved: Boolean(bestImprovement),
        committed: false,
        approved: false,
        look,
        critique,
        bestVerifiedImprovement: bestImprovement,
        previousScore: baseline.score10,
        newScore: bestImprovement ? bestImprovement.score10 : baseline.score10,
        rounds: roundsExecuted,
        attempts,
        diagnostic: 'deadline_exceeded',
        version: SURGICAL_REFINER_VERSION,
      };
    }

    const weakestSlot = String(workingCritique?.weakestSlot || '').trim().toLowerCase();
    if (!CANONICAL_SLOTS.includes(weakestSlot)) {
      return {
        status: bestImprovement ? 'improved_below_threshold' : 'no_actionable_slot',
        improved: Boolean(bestImprovement),
        committed: false,
        approved: false,
        look,
        critique,
        bestVerifiedImprovement: bestImprovement,
        previousScore: baseline.score10,
        newScore: bestImprovement ? bestImprovement.score10 : baseline.score10,
        rounds: roundsExecuted,
        attempts,
        diagnostic: 'weakest_slot_unspecified',
        version: SURGICAL_REFINER_VERSION,
      };
    }

    const slotItem = workingLook.items.find((i) => itemSlotOf(i) === weakestSlot);
    if (!slotItem || userLockedSlots.has(weakestSlot) || pinnedItemIds.has(String(slotItem.id))) {
      return {
        status: bestImprovement ? 'improved_below_threshold' : 'weakest_slot_locked',
        improved: Boolean(bestImprovement),
        committed: false,
        approved: false,
        look,
        critique,
        bestVerifiedImprovement: bestImprovement,
        previousScore: baseline.score10,
        newScore: bestImprovement ? bestImprovement.score10 : baseline.score10,
        rounds: roundsExecuted,
        attempts,
        diagnostic: `slot_${weakestSlot}_locked_or_pinned`,
        version: SURGICAL_REFINER_VERSION,
      };
    }

    roundsExecuted = round;
    const ranked = rankSlotReplacementCandidates({
      look: workingLook,
      targetSlot: weakestSlot,
      directive: workingCritique?.repairDirective || '',
      defectCategory: workingCritique?.defectCategory || '',
      catalogPool,
      context,
      personalStyleDna,
      excludedIds: triedItemIds,
      qualityMap,
      priceToBudgetCurrency,
      limit: Math.max(topCandidatesPerRound * 2, 6),
    });

    if (!ranked.length) {
      return {
        status: bestImprovement ? 'improved_below_threshold' : 'no_eligible_replacements',
        improved: Boolean(bestImprovement),
        committed: false,
        approved: false,
        look,
        critique,
        bestVerifiedImprovement: bestImprovement,
        previousScore: baseline.score10,
        newScore: bestImprovement ? bestImprovement.score10 : baseline.score10,
        rounds: roundsExecuted,
        attempts,
        diagnostic: `no_eligible_candidates_for_${weakestSlot}`,
        version: SURGICAL_REFINER_VERSION,
      };
    }

    let evaluatedInRound = 0;
    let roundBest = null;

    if (parallelCandidates === true && topCandidatesPerRound > 1 && typeof visualCritic === 'function') {
      // Fast parallel batch mode: validate hard gates, render all candidate boards
      // in parallel (~50ms with warm photoCache), then evaluate in ONE parallel critic wave (~2s)!
      const gatePassed = [];
      for (let idx = 0; idx < ranked.length && gatePassed.length < topCandidatesPerRound; idx++) {
        if (signal?.aborted || Date.now() >= deadlineAt) break;
        const candidateEntry = ranked[idx];
        const replacement = candidateEntry.item;
        const replacementId = String(replacement.id);
        triedItemIds.add(replacementId);

        const candidateLookId = `${String(look.id || 'look').replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 44)}_sr${round}_${idx}`;
        const candidateLook = buildReplacedLook(workingLook, weakestSlot, replacement, candidateLookId);

        const gate = await runHardGate(hardGateValidator, candidateLook, context);
        if (!gate.ok) {
          attempts.push({
            round,
            slot: weakestSlot,
            replacedItemId: String(slotItem.id),
            candidateItemId: replacementId,
            rankScore: candidateEntry.score,
            hardGatePassed: false,
            hardGateReason: gate.reason,
            criticRan: false,
            score10: null,
            committed: false,
          });
          continue;
        }
        gatePassed.push({ candidateEntry, replacement, replacementId, candidateLook });
      }

      const renderedReady = [];
      if (gatePassed.length > 0 && !signal?.aborted && Date.now() < deadlineAt) {
        const rendered = await Promise.all(gatePassed.map(async (entry) => {
          if (typeof boardRenderer !== 'function') return { ...entry, board: null };
          let board = null;
          try {
            board = await Promise.resolve(boardRenderer(entry.candidateLook.items, {
              lookToken: entry.candidateLook.id,
              title: entry.candidateLook.title || 'SURGICAL REFINEMENT BOARD',
              anonymizeLabels: true,
              requireAllPhotos: true,
              deadlineMs: Math.max(400, deadlineAt - Date.now()),
              signal,
            }));
          } catch {
            board = { ok: false, error: 'board_render_error' };
          }
          return { ...entry, board };
        }));

        for (const entry of rendered) {
          if (entry.board && entry.board.ok === false) {
            attempts.push({
              round,
              slot: weakestSlot,
              replacedItemId: String(slotItem.id),
              candidateItemId: entry.replacementId,
              rankScore: entry.candidateEntry.score,
              hardGatePassed: true,
              boardRendered: false,
              boardError: entry.board.error || 'board_failed',
              criticRan: false,
              score10: null,
              committed: false,
            });
            continue;
          }
          renderedReady.push(entry);
        }
      }

      if (renderedReady.length > 0 && !signal?.aborted && Date.now() < deadlineAt) {
        evaluatedInRound += renderedReady.length;
        let verdictById = new Map();
        try {
          verdictById = await evaluateBatchWithVisualCritic(
            visualCritic, renderedReady, context, personalStyleDna, signal,
          );
        } catch {
          verdictById = new Map();
        }

        let bestWinning = null;
        for (const entry of renderedReady) {
          const newCritique = verdictById.get(entry.candidateLook.id) || null;
          if (!newCritique) {
            attempts.push({
              round,
              slot: weakestSlot,
              replacedItemId: String(slotItem.id),
              candidateItemId: entry.replacementId,
              rankScore: entry.candidateEntry.score,
              hardGatePassed: true,
              boardRendered: Boolean(entry.board?.jpegBuffer),
              criticRan: true,
              criticError: 'critic_abstained',
              score10: null,
              committed: false,
            });
            continue;
          }
          const nextScore = computeCritiqueCompositeScore(newCritique, { minEditorialScore, minVisualQuality });
          const strictlyImproved = nextScore.score10 > baseline.score10
            || (baseline.rejected && !nextScore.rejected && nextScore.score10 >= baseline.score10);
          const clearedAndImproved = nextScore.passedFloor && strictlyImproved;

          attempts.push({
            round,
            slot: weakestSlot,
            replacedItemId: String(slotItem.id),
            candidateItemId: entry.replacementId,
            rankScore: entry.candidateEntry.score,
            hardGatePassed: true,
            boardRendered: Boolean(entry.board?.jpegBuffer),
            criticRan: true,
            score10: nextScore.score10,
            score01: nextScore.score01,
            passedFloor: nextScore.passedFloor,
            rejected: nextScore.rejected,
            committed: clearedAndImproved,
          });

          if (clearedAndImproved && (!bestWinning || nextScore.score10 > bestWinning.nextScore.score10)) {
            bestWinning = { entry, newCritique, nextScore };
          } else if (nextScore.score10 > workingScore.score10) {
            const candidateRecord = {
              look: entry.candidateLook,
              critique: newCritique,
              board: entry.board,
              swappedSlot: weakestSlot,
              replacedItemId: String(slotItem.id),
              selectedItemId: entry.replacementId,
              score10: nextScore.score10,
              score01: nextScore.score01,
              passedFloor: nextScore.passedFloor,
            };
            if (!roundBest || candidateRecord.score10 > roundBest.score10) roundBest = candidateRecord;
            if (!bestImprovement || candidateRecord.score10 > bestImprovement.score10) bestImprovement = candidateRecord;
          }
        }

        if (bestWinning) {
          return {
            status: 'refined_and_approved',
            improved: true,
            committed: true,
            approved: true,
            look: bestWinning.entry.candidateLook,
            critique: bestWinning.newCritique,
            board: bestWinning.entry.board,
            swappedSlot: weakestSlot,
            replacedItemId: String(slotItem.id),
            selectedItemId: bestWinning.entry.replacementId,
            previousScore: baseline.score10,
            newScore: bestWinning.nextScore.score10,
            scoreDelta: Number((bestWinning.nextScore.score10 - baseline.score10).toFixed(2)),
            rounds: roundsExecuted,
            attempts,
            diagnostic: null,
            latencyMs: Date.now() - startedAt,
            version: SURGICAL_REFINER_VERSION,
          };
        }
      }
    } else for (let idx = 0; idx < ranked.length && evaluatedInRound < topCandidatesPerRound; idx++) {
      if (signal?.aborted || Date.now() >= deadlineAt) break;

      const candidateEntry = ranked[idx];
      const replacement = candidateEntry.item;
      const replacementId = String(replacement.id);
      triedItemIds.add(replacementId);

      const candidateLookId = `${String(look.id || 'look').replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 44)}_sr${round}_${idx}`;
      const candidateLook = buildReplacedLook(workingLook, weakestSlot, replacement, candidateLookId);

      // 1. Server hard gates ALWAYS run before board render / VLM call
      const gate = await runHardGate(hardGateValidator, candidateLook, context);
      if (!gate.ok) {
        attempts.push({
          round,
          slot: weakestSlot,
          replacedItemId: String(slotItem.id),
          candidateItemId: replacementId,
          rankScore: candidateEntry.score,
          hardGatePassed: false,
          hardGateReason: gate.reason,
          criticRan: false,
          score10: null,
          committed: false,
        });
        continue;
      }

      // 2. Re-render proportion-aware editorial board (3 locked items hit cache at 0ms!)
      let board = null;
      if (typeof boardRenderer === 'function') {
        try {
          board = await Promise.resolve(boardRenderer(candidateLook.items, {
            lookToken: candidateLook.id,
            title: candidateLook.title || 'SURGICAL REFINEMENT BOARD',
            requireAllPhotos: true,
            deadlineMs: Math.max(400, deadlineAt - Date.now()),
            signal,
          }));
        } catch {
          board = { ok: false, error: 'board_render_error' };
        }
        if (board && board.ok === false) {
          attempts.push({
            round,
            slot: weakestSlot,
            replacedItemId: String(slotItem.id),
            candidateItemId: replacementId,
            rankScore: candidateEntry.score,
            hardGatePassed: true,
            boardRendered: false,
            boardError: board.error || 'board_failed',
            criticRan: false,
            score10: null,
            committed: false,
          });
          continue;
        }
      }

      // 3. Independent Visual Critic re-evaluation
      evaluatedInRound++;
      let newCritique = null;
      try {
        newCritique = await evaluateWithVisualCritic(
          visualCritic, candidateLook, board, context, personalStyleDna, signal,
        );
      } catch {
        newCritique = null;
      }

      if (!newCritique) {
        attempts.push({
          round,
          slot: weakestSlot,
          replacedItemId: String(slotItem.id),
          candidateItemId: replacementId,
          rankScore: candidateEntry.score,
          hardGatePassed: true,
          boardRendered: Boolean(board?.jpegBuffer),
          criticRan: true,
          criticError: 'critic_abstained',
          score10: null,
          committed: false,
        });
        continue;
      }

      const nextScore = computeCritiqueCompositeScore(newCritique, { minEditorialScore, minVisualQuality });
      const strictlyImproved = nextScore.score10 > baseline.score10
        || (baseline.rejected && !nextScore.rejected && nextScore.score10 >= baseline.score10);
      const clearedAndImproved = nextScore.passedFloor && strictlyImproved;

      attempts.push({
        round,
        slot: weakestSlot,
        replacedItemId: String(slotItem.id),
        candidateItemId: replacementId,
        rankScore: candidateEntry.score,
        hardGatePassed: true,
        boardRendered: Boolean(board?.jpegBuffer),
        criticRan: true,
        score10: nextScore.score10,
        score01: nextScore.score01,
        passedFloor: nextScore.passedFloor,
        rejected: nextScore.rejected,
        committed: clearedAndImproved,
      });

      if (clearedAndImproved) {
        return {
          status: 'refined_and_approved',
          improved: true,
          committed: true,
          approved: true,
          look: candidateLook,
          critique: newCritique,
          board,
          swappedSlot: weakestSlot,
          replacedItemId: String(slotItem.id),
          selectedItemId: replacementId,
          previousScore: baseline.score10,
          newScore: nextScore.score10,
          scoreDelta: Number((nextScore.score10 - baseline.score10).toFixed(2)),
          rounds: roundsExecuted,
          attempts,
          diagnostic: null,
          latencyMs: Date.now() - startedAt,
          version: SURGICAL_REFINER_VERSION,
        };
      }

      // Track best verified improvement (still NOT committed because threshold was not cleared!)
      if (nextScore.score10 > workingScore.score10) {
        const candidateRecord = {
          look: candidateLook,
          critique: newCritique,
          board,
          swappedSlot: weakestSlot,
          replacedItemId: String(slotItem.id),
          selectedItemId: replacementId,
          score10: nextScore.score10,
          score01: nextScore.score01,
          passedFloor: nextScore.passedFloor,
        };
        if (!roundBest || candidateRecord.score10 > roundBest.score10) {
          roundBest = candidateRecord;
        }
        if (!bestImprovement || candidateRecord.score10 > bestImprovement.score10) {
          bestImprovement = candidateRecord;
        }
      }
    }

    // Carry forward the best verified intermediate state into Round 2 if another slot remains weak
    if (roundBest) {
      workingLook = roundBest.look;
      workingCritique = roundBest.critique;
      workingScore = computeCritiqueCompositeScore(roundBest.critique, { minEditorialScore, minVisualQuality });
    } else {
      break;
    }
  }

  return {
    status: bestImprovement ? 'improved_below_threshold' : 'unimproved',
    improved: Boolean(bestImprovement),
    committed: false,
    approved: false,
    look,
    critique,
    bestVerifiedImprovement: bestImprovement,
    previousScore: baseline.score10,
    newScore: bestImprovement ? bestImprovement.score10 : baseline.score10,
    scoreDelta: bestImprovement ? Number((bestImprovement.score10 - baseline.score10).toFixed(2)) : 0,
    rounds: roundsExecuted,
    attempts,
    diagnostic: bestImprovement
      ? 'best_candidate_below_editorial_threshold'
      : attempts.every((a) => !a.hardGatePassed)
        ? 'hard_gate_blocked_all_candidates'
        : 'all_candidates_rejected_or_unimproved',
    latencyMs: Date.now() - startedAt,
    version: SURGICAL_REFINER_VERSION,
  };
}

/**
 * Interactive 1-tap "Swap This Item" engine for the mobile UI.
 * Locks all other slots, finds the best hard-gate-verified (and optionally
 * visually verified) replacement in `targetSlot` matching the user's `reason`.
 */
export async function swapSingleSlot({
  look,
  targetSlot,
  reason = '',
  catalogPool = [],
  context = {},
  personalStyleDna = null,
  hardGateValidator = null,
  visualCritic = null,
  boardRenderer = null,
  excludedItemIds = [],
  qualityMap = null,
  priceToBudgetCurrency = null,
  minEditorialScore = DEFAULT_MIN_EDITORIAL_SCORE,
  minVisualQuality = DEFAULT_MIN_VISUAL_QUALITY,
  maxCandidates = 5,
  deadlineMs = 8_000,
  signal = null,
} = {}) {
  const startedAt = Date.now();
  const slot = String(targetSlot || '').trim().toLowerCase();
  if (!look || !Array.isArray(look.items) || !ALL_SWAPPABLE_SLOTS.includes(slot)) {
    return {
      status: 'invalid_slot',
      swapped: false,
      targetSlot: slot || null,
      look,
      previousItem: null,
      newItem: null,
      critique: null,
      visualVerified: false,
      attempts: [],
      latencyMs: Date.now() - startedAt,
      version: SURGICAL_REFINER_VERSION,
    };
  }

  const previousItem = look.items.find((i) => itemSlotOf(i) === slot) || null;
  if (!previousItem) {
    return {
      status: 'invalid_slot',
      swapped: false,
      targetSlot: slot,
      look,
      previousItem: null,
      newItem: null,
      critique: null,
      visualVerified: false,
      attempts: [],
      latencyMs: Date.now() - startedAt,
      version: SURGICAL_REFINER_VERSION,
    };
  }

  const deadlineAt = startedAt + Math.max(400, Math.min(30_000, Number(deadlineMs) || 8_000));
  const excludedSet = new Set(Array.from(excludedItemIds || [], String));
  excludedSet.add(String(previousItem.id));

  const ranked = rankSlotReplacementCandidates({
    look,
    targetSlot: slot,
    directive: String(reason || ''),
    defectCategory: String(reason || ''),
    catalogPool,
    context,
    personalStyleDna,
    excludedIds: excludedSet,
    qualityMap,
    priceToBudgetCurrency,
    limit: Math.max(1, Math.min(12, maxCandidates * 2)),
  });

  if (!ranked.length) {
    return {
      status: 'no_eligible_replacement',
      swapped: false,
      targetSlot: slot,
      look,
      previousItem,
      newItem: null,
      critique: null,
      visualVerified: false,
      attempts: [],
      latencyMs: Date.now() - startedAt,
      version: SURGICAL_REFINER_VERSION,
    };
  }

  const attempts = [];
  let checked = 0;

  for (let i = 0; i < ranked.length && checked < maxCandidates; i++) {
    if (signal?.aborted || Date.now() >= deadlineAt) {
      return {
        status: 'deadline',
        swapped: false,
        targetSlot: slot,
        look,
        previousItem,
        newItem: null,
        critique: null,
        visualVerified: false,
        attempts,
        latencyMs: Date.now() - startedAt,
        version: SURGICAL_REFINER_VERSION,
      };
    }

    const entry = ranked[i];
    const replacement = entry.item;
    const candidateLook = buildReplacedLook(
      look,
      slot,
      replacement,
      `${String(look.id || 'look').replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 44)}_swap_${i}`,
    );

    const gate = await runHardGate(hardGateValidator, candidateLook, context);
    if (!gate.ok) {
      attempts.push({
        candidateItemId: String(replacement.id),
        rankScore: entry.score,
        hardGatePassed: false,
        hardGateReason: gate.reason,
        visualVerified: false,
      });
      continue;
    }

    checked++;
    let board = null;
    if (typeof boardRenderer === 'function') {
      try {
        board = await Promise.resolve(boardRenderer(candidateLook.items, {
          lookToken: candidateLook.id,
          requireAllPhotos: Boolean(visualCritic),
          deadlineMs: Math.max(400, deadlineAt - Date.now()),
          signal,
        }));
      } catch {
        board = { ok: false, error: 'board_render_error' };
      }
      if (visualCritic && board && board.ok === false) {
        attempts.push({
          candidateItemId: String(replacement.id),
          rankScore: entry.score,
          hardGatePassed: true,
          boardRendered: false,
          visualVerified: false,
        });
        continue;
      }
    }

    if (typeof visualCritic === 'function') {
      let critique = null;
      try {
        critique = await evaluateWithVisualCritic(
          visualCritic, candidateLook, board, context, personalStyleDna, signal,
        );
      } catch {
        critique = null;
      }
      const scoreInfo = computeCritiqueCompositeScore(critique, { minEditorialScore, minVisualQuality });
      attempts.push({
        candidateItemId: String(replacement.id),
        rankScore: entry.score,
        hardGatePassed: true,
        visualVerified: scoreInfo.passedFloor,
        score10: scoreInfo.score10,
        rejected: scoreInfo.rejected,
      });

      if (scoreInfo.passedFloor) {
        return {
          status: 'swapped',
          swapped: true,
          targetSlot: slot,
          previousItem,
          newItem: replacement,
          look: candidateLook,
          critique,
          board,
          visualVerified: true,
          editorialScore: scoreInfo.score10,
          attempts,
          latencyMs: Date.now() - startedAt,
          version: SURGICAL_REFINER_VERSION,
        };
      }
      continue;
    }

    // No visualCritic supplied: return top hard-gate-verified swap with honest visualVerified: false
    attempts.push({
      candidateItemId: String(replacement.id),
      rankScore: entry.score,
      hardGatePassed: true,
      visualVerified: false,
    });
    return {
      status: 'swapped',
      swapped: true,
      targetSlot: slot,
      previousItem,
      newItem: replacement,
      look: candidateLook,
      critique: null,
      board,
      visualVerified: false,
      attempts,
      latencyMs: Date.now() - startedAt,
      version: SURGICAL_REFINER_VERSION,
    };
  }

  return {
    status: typeof visualCritic === 'function' ? 'no_approved_replacement' : 'no_eligible_replacement',
    swapped: false,
    targetSlot: slot,
    look,
    previousItem,
    newItem: null,
    critique: null,
    visualVerified: false,
    attempts,
    latencyMs: Date.now() - startedAt,
    version: SURGICAL_REFINER_VERSION,
  };
}

/**
 * Execute multiple surgical 1-slot rescues IN PARALLEL (`Promise.all`) while
 * pre-partitioning replacement SKU exclusions per target so concurrent rescues
 * targeting the same slot never collide on the same replacement SKU.
 */
export async function refineLooksSurgicallyInParallel({
  targets = [],
  catalogPool = [],
  context = {},
  personalStyleDna = null,
  hardGateValidator = null,
  visualCritic = null,
  boardRenderer = null,
  maxRounds = 1,
  deadlineMs = 8_000,
  excludedItemIds = [],
  qualityMap = null,
  priceToBudgetCurrency = null,
  minEditorialScore = DEFAULT_MIN_EDITORIAL_SCORE,
  minVisualQuality = DEFAULT_MIN_VISUAL_QUALITY,
  topCandidatesPerRound = 2,
  parallelCandidates = true,
  signal = null,
} = {}) {
  if (!Array.isArray(targets) || targets.length === 0) return [];
  const boundedTargets = targets.slice(0, 4);
  const baseExcluded = new Set(Array.from(excludedItemIds || [], String));
  // Pre-reserve disjoint top replacement SKUs per target when targets share a weakestSlot
  const reservedAcrossTargets = new Set(baseExcluded);
  const perTargetExcluded = [];

  for (const target of boundedTargets) {
    const targetExcluded = new Set(reservedAcrossTargets);
    const slot = String(target?.critique?.weakestSlot || '').trim().toLowerCase();
    if (CANONICAL_SLOTS.includes(slot)) {
      const topPicks = rankSlotReplacementCandidates({
        look: target.look,
        targetSlot: slot,
        directive: target?.critique?.repairDirective || '',
        defectCategory: target?.critique?.defectCategory || '',
        catalogPool,
        context: target.context || context,
        personalStyleDna,
        excludedIds: targetExcluded,
        qualityMap,
        priceToBudgetCurrency,
        limit: Math.max(2, topCandidatesPerRound),
      });
      for (const pick of topPicks) {
        if (pick?.item?.id != null) reservedAcrossTargets.add(String(pick.item.id));
      }
    }
    perTargetExcluded.push([...targetExcluded]);
  }

  return Promise.all(boundedTargets.map((target, idx) => refineLookSurgically({
    look: target.look,
    critique: target.critique,
    catalogPool,
    lockedSlots: target.lockedSlots || [],
    context: target.context || context,
    personalStyleDna,
    hardGateValidator: target.hardGateValidator || hardGateValidator,
    visualCritic,
    boardRenderer,
    maxRounds,
    deadlineMs,
    excludedItemIds: perTargetExcluded[idx],
    qualityMap,
    priceToBudgetCurrency,
    minEditorialScore,
    minVisualQuality,
    topCandidatesPerRound,
    parallelCandidates,
    signal,
  })));
}
