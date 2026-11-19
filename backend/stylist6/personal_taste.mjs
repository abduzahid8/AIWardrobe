/**
 * Stylist 6 — explicit, evidence-grounded personal taste & Personal Style DNA scorer.
 *
 * Grounded ONLY in explicit user preferences, opt-in appearance/body profiles,
 * and owned wardrobe DNA. Unknown catalog evidence stays neutral.
 *
 * Integrates:
 *  - Color canonicalization ('#0A1931' -> 'navy', 'gray' -> 'grey', hex/alias normalization)
 *  - Perceptual CIELAB color theory (undertone harmony, value contrast, near-miss clash detection)
 *  - Disliked patterns & avoided colors
 *  - Opt-in body & proportion styling directives
 *  - Owned wardrobe DNA affinity
 *
 * No I/O; no profile persistence; never logs preferences. Bounded and typed so
 * malformed client payloads cannot amplify taste over hard feasibility/safety gates.
 */

import {
  canonicalizeStyleColor,
  normalizePersonalStyleDna,
} from '../../src/domain/personalStyleDna.ts';
import {
  scoreColorUndertoneFit,
  scoreOutfitColorTheory,
} from '../../src/domain/colorTheory.ts';

const norm = (value) => typeof value === 'string'
  ? value.normalize('NFKC').trim().toLowerCase().replace(/\s+/g, ' ').slice(0, 64)
  : '';

const list = (value) => Array.isArray(value)
  ? [...new Set(value.slice(0, 32).map(norm).filter((x) => x.length >= 2))].slice(0, 16)
  : [];

const canonicalColorToken = (value) => {
  const canonical = canonicalizeStyleColor(value);
  if (canonical !== 'unknown') return canonical;
  const fallback = norm(value);
  return fallback && fallback !== 'unknown' && fallback !== 'neutral' ? fallback : '';
};

const colorList = (value) => Array.isArray(value)
  ? [...new Set(value.slice(0, 32).map(canonicalColorToken).filter((x) => x.length >= 2))].slice(0, 16)
  : [];

const included = (choices, value) => Boolean(value && choices.has(norm(value)));

const canonicalPattern = (value) => {
  const word = norm(value);
  if (!word) return '';
  if (/\b(solid|plain|unpatterned)\b/.test(word)) return 'solid';
  if (/\b(stripes?|striped|pinstripes?|pinstriped)\b/.test(word)) return 'stripe';
  if (/\b(check|checked|plaid|tartan|gingham|houndstooth|windowpane|argyle)\b/.test(word)) return 'check';
  if (/\b(herringbone|tweed|jacquard|textured|texture|ribbed|cable|boucle)\b/.test(word)) return 'texture';
  if (/\b(print|printed|graphic|floral|logo|camo|patterned|geometric|animal|paisley)\b/.test(word)) return 'print';
  return word;
};

const patternList = (value) => Array.isArray(value)
  ? [...new Set(value.slice(0, 32).map(canonicalPattern).filter((x) => x.length >= 2))].slice(0, 16)
  : [];

function round4(value) {
  return Math.round(value * 10000) / 10000;
}

function isPreNormalizedDna(candidate) {
  return Boolean(
    candidate &&
    typeof candidate === 'object' &&
    !Array.isArray(candidate) &&
    candidate.preferences &&
    candidate.appearance &&
    candidate.body &&
    candidate.ownedWardrobeDna &&
    typeof candidate.active === 'boolean',
  );
}

/**
 * Compile user preferences (and optional PersonalStyleDna / opt-in appearance,
 * body, wardrobe DNA) into a fast, bounded scorer profile.
 */
export function personalTaste(preferences) {
  const source = preferences && typeof preferences === 'object' &&
    !Array.isArray(preferences) ? preferences : {};

  // Support passing a pre-normalized PersonalStyleDna directly or via `personalStyleDna`
  const explicitDnaObj = isPreNormalizedDna(source)
    ? source
    : isPreNormalizedDna(source.personalStyleDna)
      ? source.personalStyleDna
      : (source.personalStyleDna && typeof source.personalStyleDna === 'object'
        ? normalizePersonalStyleDna(source.personalStyleDna)
        : null);

  const prefSrc = explicitDnaObj ? explicitDnaObj.preferences : source;

  const favoriteColors = new Set([
    ...colorList(source.favoriteColors),
    ...(explicitDnaObj ? explicitDnaObj.preferences.favoriteColors : []),
  ].slice(0, 16));

  const avoidColors = new Set([
    ...colorList(source.avoidColors),
    ...colorList(source.avoidedColors),
    ...(explicitDnaObj ? explicitDnaObj.preferences.avoidedColors : []),
    ...(explicitDnaObj ? explicitDnaObj.appearance.avoidColors : []),
  ].slice(0, 16));

  const favoriteBrands = new Set([
    ...list(source.favoriteBrands),
    ...(explicitDnaObj ? explicitDnaObj.preferences.favoriteBrands.map(norm) : []),
  ].slice(0, 16));

  const avoidBrands = new Set([
    ...list(source.avoidBrands),
    ...list(source.brandExclusions),
    ...(explicitDnaObj ? explicitDnaObj.preferences.brandExclusions.map(norm) : []),
  ].slice(0, 16));

  const likedPatterns = new Set([
    ...patternList(source.likedPatterns),
    ...(explicitDnaObj ? explicitDnaObj.preferences.likedPatterns : []),
  ].slice(0, 16));

  const dislikedPatterns = new Set([
    ...patternList(source.dislikedPatterns),
    ...(explicitDnaObj ? explicitDnaObj.preferences.dislikedPatterns : []),
  ].slice(0, 16));

  const fit = norm(prefSrc?.fitPreference ?? source.fitPreference);
  // The app's persisted default is "balanced": never claim the user chose
  // it or quietly bias the engine without explicit fit evidence.
  const desiredFit = fit === 'fitted' ? 'slim' : fit === 'loose' ? 'wide' : null;

  const hasOptInAppearance = Boolean(source.appearance || source.appearanceProfile || explicitDnaObj?.appearance?.provenance !== 'none');
  const hasOptInBody = Boolean(
    (source.body && typeof source.body === 'object') ||
    (source.bodyProfile && typeof source.bodyProfile === 'object') ||
    explicitDnaObj?.body?.hasExplicitBodyEvidence,
  );
  const hasWardrobeDna = Boolean(
    (Array.isArray(source.wardrobeItems) && source.wardrobeItems.length > 0) ||
    (source.ownedWardrobeDna && typeof source.ownedWardrobeDna === 'object') ||
    (source.ownedWardrobeSummary && typeof source.ownedWardrobeSummary === 'object') ||
    (explicitDnaObj && (explicitDnaObj.ownedWardrobeDna.totalItems > 0 || explicitDnaObj.ownedWardrobeDna.dominantPalette.length > 0)),
  );

  // Build unified PersonalStyleDna ONLY from explicit/opt-in sources (ignoring unverified flat legacy bodyType strings).
  const dna = explicitDnaObj || normalizePersonalStyleDna({
    presentation: source.presentation ?? source.body?.presentation ?? source.bodyProfile?.presentation,
    gender: source.gender ?? source.body?.gender ?? source.bodyProfile?.gender,
    preferences: {
      favoriteColors: [...favoriteColors],
      avoidedColors: [...avoidColors],
      likedPatterns: [...likedPatterns],
      dislikedPatterns: [...dislikedPatterns],
      preferredFits: Array.isArray(source.preferredFits) ? source.preferredFits : undefined,
      fitPreference: fit || undefined,
      favoriteBrands: [...favoriteBrands],
      brandExclusions: [...avoidBrands],
      styles: Array.isArray(source.styles) ? source.styles : undefined,
      stylePersonality: typeof source.stylePersonality === 'string' ? source.stylePersonality : undefined,
    },
    appearance: source.appearance ?? source.appearanceProfile ?? undefined,
    body: source.body ?? source.bodyProfile ?? undefined,
    wardrobeItems: Array.isArray(source.wardrobeItems) ? source.wardrobeItems : undefined,
    ownedWardrobeDna: source.ownedWardrobeDna ?? undefined,
    ownedWardrobeSummary: source.ownedWardrobeSummary ?? undefined,
  });

  // Rich DNA scoring activates when any extended DNA dimension is provided
  // (appearance, opt-in body directives, dislikedPatterns, avoidColors,
  // owned wardrobe DNA, preferredFits, brandExclusions, or explicit personalStyleDna).
  // Plain legacy quiz payloads keep exact legacy bounded weights for 100% compatibility.
  const richDnaActive = Boolean(
    explicitDnaObj ||
    avoidColors.size > 0 ||
    dislikedPatterns.size > 0 ||
    (hasOptInAppearance && dna.appearance.provenance !== 'none') ||
    (hasOptInBody && dna.body.hasExplicitBodyEvidence) ||
    (hasWardrobeDna && (dna.ownedWardrobeDna.totalItems > 0 || dna.ownedWardrobeDna.dominantPalette.length > 0)) ||
    (Array.isArray(source.preferredFits) && source.preferredFits.length > 0) ||
    (Array.isArray(source.brandExclusions) && source.brandExclusions.length > 0) ||
    source.useDnaScoring === true,
  );

  const active =
    favoriteColors.size > 0 ||
    avoidColors.size > 0 ||
    favoriteBrands.size > 0 ||
    avoidBrands.size > 0 ||
    likedPatterns.size > 0 ||
    dislikedPatterns.size > 0 ||
    Boolean(desiredFit) ||
    richDnaActive;

  return {
    favoriteColors,
    avoidColors,
    favoriteBrands,
    avoidBrands,
    likedPatterns,
    dislikedPatterns,
    desiredFit,
    dna,
    richDnaActive,
    active,
  };
}

function evaluateBodyAlignment(item, descriptor, body, desiredFit) {
  let delta = 0;
  const matches = [];
  const penalties = [];
  const fit = norm(descriptor?.volume || item?.fit || item?.silhouette);
  const slot = norm(descriptor?.slot || item?.macroCategory || item?.category);
  const length = norm(descriptor?.length);
  const collar = norm(descriptor?.collar);
  const text = `${item?.name || ''} ${item?.description || ''} ${item?.fit || ''} ${item?.silhouette || ''} ${descriptor?.subtype || ''}`.toLowerCase();

  if (desiredFit && fit && fit !== 'unknown') {
    if (fit === desiredFit) {
      delta += 0.06;
      matches.push({ type: 'preferred_fit', value: fit });
    } else if (fit === 'slim' || fit === 'wide') {
      delta -= 0.07;
      penalties.push(`fit_mismatch:${fit}_vs_${desiredFit}`);
    }
  } else if (body?.hasExplicitBodyEvidence && fit && fit !== 'unknown') {
    if (body.preferredGarmentVolumes?.includes(fit)) {
      delta += 0.045;
      matches.push({ type: 'body_proportion_fit', value: fit });
    } else if (body.avoidGarmentVolumes?.includes(fit)) {
      delta -= 0.055;
      penalties.push(`body_volume_avoid:${fit}`);
    }
  }

  if (!body?.hasExplicitBodyEvidence) return { delta, matches, penalties };

  // Height-band proportion calibration
  if (body.heightBand === 'petite' || body.heightBand === 'short') {
    if (slot === 'outerwear' && (length === 'long' || /\b(longline|overcoat|maxi|trench|floor[ -]length)\b/.test(text))) {
      delta -= 0.07;
      penalties.push('height_avoid:longline_outerwear_on_petite_frame');
    } else if (slot === 'outerwear' && (length === 'cropped' || /\b(cropped|hip[ -]length|bomber|blouson)\b/.test(text))) {
      delta += 0.055;
      matches.push({ type: 'body_proportion_cut', value: 'cropped_proportion_layer' });
    }
    if (slot === 'bottom' && /\b(high[ -]?rise|high[ -]?waist|cropped|ankle)\b/.test(text)) {
      delta += 0.05;
      matches.push({ type: 'body_proportion_cut', value: 'high_rise_elongating_trouser' });
    }
  } else if (body.heightBand === 'tall') {
    if (slot === 'outerwear' && (length === 'long' || /\b(overcoat|topcoat|trench|longline)\b/.test(text))) {
      delta += 0.055;
      matches.push({ type: 'body_proportion_cut', value: 'longline_layer_for_tall_frame' });
    } else if (slot === 'outerwear' && length === 'cropped') {
      delta -= 0.045;
      penalties.push('height_avoid:overly_cropped_layer_on_tall_frame');
    }
  }

  // Body-type silhouette rules
  const bt = body.bodyType;
  if (bt === 'inverted_triangle') {
    if (slot === 'outerwear' && /\b(padded\s+shoulder|strong\s+shoulder|boxy\s+double[ -]breasted)\b/.test(text)) {
      delta -= 0.07;
      penalties.push('body_cut_avoid:padded_shoulders_inverted_triangle');
    } else if (slot === 'outerwear' && (descriptor?.relaxedTailoring || /\b(unstructured|raglan|soft[ -]shoulder)\b/.test(text))) {
      delta += 0.055;
      matches.push({ type: 'body_proportion_cut', value: 'soft_shoulder_outerwear' });
    }
    if (slot === 'bottom' && (fit === 'slim' || /\b(skinny|spray[ -]on|super[ -]slim)\b/.test(text))) {
      delta -= 0.065;
      penalties.push('body_cut_avoid:skinny_bottom_inverted_triangle');
    } else if (slot === 'bottom' && (fit === 'wide' || /\b(pleated|straight[ -]leg|relaxed)\b/.test(text))) {
      delta += 0.055;
      matches.push({ type: 'body_proportion_cut', value: 'balanced_lower_line_trouser' });
    }
  } else if (bt === 'triangle_pear') {
    if (slot === 'bottom' && /\b(cargo|patch\s+pocket|skinny|low[ -]rise)\b/.test(text)) {
      delta -= 0.07;
      penalties.push('body_cut_avoid:bulky_or_tight_hip_bottom');
    } else if (slot === 'bottom' && /\b(high[ -]rise|straight|wide[ -]leg|a[ -]line|fluid)\b/.test(text)) {
      delta += 0.055;
      matches.push({ type: 'body_proportion_cut', value: 'fluid_elongating_bottom' });
    }
    if (slot === 'outerwear' && descriptor?.structured) {
      delta += 0.05;
      matches.push({ type: 'body_proportion_cut', value: 'structured_upper_balance' });
    }
  } else if (bt === 'oval_apple') {
    if (slot === 'outerwear' && descriptor?.closure === 'openable' && !/\bdouble[ -]breasted\b/.test(text)) {
      delta += 0.055;
      matches.push({ type: 'body_proportion_cut', value: 'open_vertical_lapel_line' });
    } else if (slot === 'top' && (collar === 'open' || /\bv[ -]?neck\b/.test(text))) {
      delta += 0.05;
      matches.push({ type: 'body_proportion_cut', value: 'vertical_open_neckline' });
    } else if (slot === 'top' && (fit === 'slim' || /\b(tight|clingy|cropped|compression)\b/.test(text))) {
      delta -= 0.065;
      penalties.push('body_cut_avoid:clingy_midsection_top');
    }
  } else if (bt === 'hourglass') {
    if (slot === 'outerwear' && /\b(boxy|oversized|drop[ -]shoulder|tent)\b/.test(text)) {
      delta -= 0.065;
      penalties.push('body_cut_avoid:boxy_outerwear_erases_waist');
    } else if (/\b(belted|waisted|darted|wrap|tailored|high[ -]rise)\b/.test(text)) {
      delta += 0.055;
      matches.push({ type: 'body_proportion_cut', value: 'waist_defining_cut' });
    }
  } else if (bt === 'athletic') {
    if (slot === 'bottom' && /\b(skinny|ultra[ -]slim|super[ -]slim)\b/.test(text)) {
      delta -= 0.07;
      penalties.push('body_cut_avoid:skinny_trouser_athletic_thighs');
    } else if (slot === 'bottom' && /\b(athletic|tapered|pleated|straight)\b/.test(text)) {
      delta += 0.05;
      matches.push({ type: 'body_proportion_cut', value: 'athletic_taper_trouser' });
    }
  } else if (bt === 'rectangle') {
    if (slot === 'outerwear' && (descriptor?.structured || /\b(structured|belted|blazer|tailored)\b/.test(text))) {
      delta += 0.05;
      matches.push({ type: 'body_proportion_cut', value: 'sculptural_structured_layer' });
    } else if (slot === 'bottom' && /\b(pleated|tapered|high[ -]rise)\b/.test(text)) {
      delta += 0.045;
      matches.push({ type: 'body_proportion_cut', value: 'dimension_building_trouser' });
    }
  }

  return { delta, matches, penalties };
}

function itemSignalsLegacy(item, descriptor, taste) {
  const brand = norm(item?.brand);
  const color = canonicalColorToken(descriptor?.color || item?.color);
  const pattern = canonicalPattern(descriptor?.pattern || item?.pattern);
  const fit = norm(descriptor?.volume);
  const productId = String(item?.id || '');
  if (included(taste.avoidBrands, brand)) {
    return { blocked: 'excluded_brand', score: 0, matches: [] };
  }
  let score = 0;
  const matches = [];
  const add = (type, value, weight) => {
    score += weight;
    if (productId) matches.push({ type, value, productId });
  };
  if (color && taste.favoriteColors.has(color) && color !== 'unknown' && color !== 'neutral') {
    add('favorite_color', color, 0.04);
  }
  if (included(taste.favoriteBrands, brand)) {
    add('favorite_brand', String(item.brand).slice(0, 64), 0.06);
  }
  if (pattern && pattern !== 'unknown' && included(taste.likedPatterns, pattern)) {
    add('liked_pattern', pattern, 0.02);
  }
  if (taste.desiredFit && fit !== 'unknown' && fit) {
    if (fit === taste.desiredFit) add('preferred_fit', fit, 0.025);
    else if (fit === 'slim' || fit === 'wide') score -= 0.025;
  }
  return { blocked: null, score: Math.max(-0.025, Math.min(0.12, score)), matches };
}

function itemSignalsRichDna(item, descriptor, taste) {
  const brand = norm(item?.brand);
  const color = canonicalColorToken(descriptor?.color || item?.color);
  const pattern = canonicalPattern(descriptor?.pattern || item?.pattern);
  const productId = String(item?.id || '');

  if (included(taste.avoidBrands, brand)) {
    return { blocked: 'excluded_brand', score: 0, matches: [], penalties: ['excluded_brand'] };
  }

  let score = 0;
  const matches = [];
  const penalties = [];
  const addMatch = (type, value, weight) => {
    score += weight;
    if (productId) matches.push({ type, value, productId });
  };

  // 1. Explicit color preferences & strict avoids
  if (color && color !== 'unknown' && color !== 'neutral') {
    if (taste.avoidColors.has(color)) {
      score -= 0.16;
      penalties.push(`avoided_color:${color}`);
    } else if (taste.favoriteColors.has(color)) {
      addMatch('favorite_color', color, 0.07);
    }

    // 2. Opt-in personal color / undertone fit via CIELAB colorTheory
    if (taste.dna?.appearance?.provenance && taste.dna.appearance.provenance !== 'none') {
      const undertone = scoreColorUndertoneFit(color, taste.dna);
      if (undertone.tier === 'avoided' && !taste.avoidColors.has(color)) {
        score -= 0.16;
        penalties.push(`appearance_avoid_color:${color}`);
      } else if (undertone.tier === 'flattering') {
        addMatch('undertone_flattering_color', color, 0.10);
      } else if (undertone.tier === 'compatible') {
        addMatch('undertone_compatible_color', color, 0.05);
      } else if (undertone.tier === 'clash') {
        score -= 0.10;
        penalties.push(`undertone_clash:${color}`);
      }
    }
  }

  // 3. Explicit favorite brands
  if (included(taste.favoriteBrands, brand)) {
    addMatch('favorite_brand', String(item.brand).slice(0, 64), 0.07);
  }

  // 4. Pattern likes & dislikes
  if (pattern && pattern !== 'unknown') {
    if (taste.dislikedPatterns.has(pattern)) {
      score -= 0.14;
      penalties.push(`disliked_pattern:${pattern}`);
    } else if (included(taste.likedPatterns, pattern)) {
      addMatch('liked_pattern', pattern, 0.04);
    }
  }

  // 5. Fit preference & opt-in body proportion directives
  const bodyEval = evaluateBodyAlignment(item, descriptor, taste.dna?.body, taste.desiredFit);
  score += bodyEval.delta;
  for (const m of bodyEval.matches) {
    if (productId) matches.push({ ...m, productId });
  }
  penalties.push(...bodyEval.penalties);

  // 6. Owned Wardrobe DNA continuity (palette, textures, trusted brands)
  const wd = taste.dna?.ownedWardrobeDna;
  if (wd && (wd.totalItems > 0 || wd.dominantPalette?.length > 0)) {
    if (color && !taste.favoriteColors.has(color) && !taste.avoidColors.has(color) && wd.dominantPalette.includes(color)) {
      addMatch('wardrobe_palette_dna', color, 0.035);
    }
    if (brand && !taste.favoriteBrands.has(brand) && wd.topBrands.some((b) => norm(b) === brand)) {
      addMatch('wardrobe_brand_dna', String(item.brand).slice(0, 64), 0.035);
    }
    const itemTex = [
      descriptor?.texture || '',
      Array.isArray(descriptor?.materials) ? descriptor.materials.join(' ') : '',
      typeof item?.material === 'string' ? item.material : '',
      item?.name || '',
    ].join(' ').toLowerCase();
    const matchedTex = wd.texturePreferences.find((t) => itemTex.includes(t.toLowerCase()));
    if (matchedTex) {
      addMatch('wardrobe_texture_dna', matchedTex, 0.03);
    }
  }

  return {
    blocked: null,
    score: round4(Math.max(-0.25, Math.min(0.25, score))),
    matches,
    penalties,
  };
}

export function evaluatePersonalItem(item, descriptor, taste) {
  if (!taste?.active) return { blocked: null, score: 0, matches: [] };
  return taste.richDnaActive
    ? itemSignalsRichDna(item, descriptor, taste)
    : itemSignalsLegacy(item, descriptor, taste);
}

/**
 * Evaluate whole-outfit alignment with Personal Style DNA.
 *
 * - Legacy simple preferences preserve bounded tie-breaker semantics ([-0.025, 0.085])
 *   so existing Stylist 6 calibration stays backwards-compatible.
 * - Rich Personal Style DNA profiles integrate perceptual whole-outfit color theory
 *   (`scoreOutfitColorTheory`), undertone & contrast fit, disliked patterns, avoided
 *   colors, body proportion directives, and wardrobe DNA with meaningful selection
 *   authority ([-0.25, +0.25]) while still never overriding hard feasibility/safety gates.
 */
export function evaluatePersonalLook(items, taste) {
  if (!taste?.active || !Array.isArray(items) || !items.length) {
    return { adjustment: 0, matches: [], matchedItems: 0 };
  }

  const scored = items.map(({ item, descriptor }) => evaluatePersonalItem(item, descriptor, taste));
  const meanItemScore = scored.reduce((sum, x) => sum + x.score, 0) / scored.length;
  const matches = scored.flatMap((x) => x.matches).slice(0, 12);
  const matchedItems = scored.filter((x) => x.matches.length).length;

  if (!taste.richDnaActive) {
    const adjustment = Math.max(-0.025, Math.min(0.085, meanItemScore));
    return { adjustment, matches: matches.slice(0, 8), matchedItems };
  }

  // Whole-outfit CIELAB color theory + undertone harmony
  const slotColors = items.map(({ item, descriptor }, index) => ({
    slot: String(descriptor?.slot || item?.macroCategory || item?.category || `slot_${index}`).toLowerCase(),
    color: descriptor?.color || item?.color || '',
  }));
  const colorTheory = scoreOutfitColorTheory(slotColors, taste.dna);
  const colorTheoryDelta =
    colorTheory.harmonyMode === 'insufficient_evidence'
      ? 0
      : (colorTheory.score - 0.62) * 0.30;

  const adjustment = round4(Math.max(-0.25, Math.min(0.25, meanItemScore + colorTheoryDelta)));
  const penalties = [
    ...new Set([
      ...scored.flatMap((x) => x.penalties || []),
      ...colorTheory.flaws,
    ]),
  ].slice(0, 12);

  return {
    adjustment,
    matches,
    matchedItems,
    penalties,
    colorTheory,
    dnaActive: true,
  };
}
