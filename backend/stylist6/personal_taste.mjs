/**
 * Stylist 6 — explicit, evidence-grounded personal taste.
 *
 * This is NOT body-shape inference, visual AI judgment, stock verification or
 * a substitute for coherence/feasibility. Only preferences the user actually
 * supplied can affect selection. Unknown catalog evidence is neutral.
 *
 * No I/O; no profile persistence; do not log preferences. Capped and typed
 * so a malformed client payload cannot amplify taste over safety/quality.
 */
const norm = (value) => typeof value === 'string'
  ? value.normalize('NFKC').trim().toLowerCase().replace(/\s+/g, ' ').slice(0, 64)
  : '';
const list = (value) => Array.isArray(value)
  ? [...new Set(value.slice(0, 32).map(norm).filter((x) => x.length >= 2))].slice(0, 16)
  : [];
const included = (choices, value) => value && choices.has(norm(value));
const own = (object, key) => Object.prototype.hasOwnProperty.call(object || {}, key);

export function personalTaste(preferences) {
  const source = preferences && typeof preferences === 'object' &&
    !Array.isArray(preferences) ? preferences : {};
  const favoriteColors = new Set(list(source.favoriteColors));
  const favoriteBrands = new Set(list(source.favoriteBrands));
  const avoidBrands = new Set(list(source.avoidBrands));
  const likedPatterns = new Set(list(source.likedPatterns));
  const fit = norm(source.fitPreference);
  // The app's persisted default is "balanced": never claim the user chose
  // it or quietly bias the engine without explicit fit evidence.
  const desiredFit = fit === 'fitted' ? 'slim' : fit === 'loose' ? 'wide' : null;
  return {
    favoriteColors, favoriteBrands, avoidBrands, likedPatterns, desiredFit,
    active: favoriteColors.size > 0 || favoriteBrands.size > 0 ||
      avoidBrands.size > 0 || likedPatterns.size > 0 || !!desiredFit,
  };
}

function itemSignals(item, descriptor, taste) {
  const brand = norm(item?.brand);
  const color = norm(descriptor?.color || item?.color);
  const pattern = norm(descriptor?.pattern);
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
  if (included(taste.favoriteColors, color) && color !== 'unknown' && color !== 'neutral') {
    add('favorite_color', color, 0.04);
  }
  if (included(taste.favoriteBrands, brand)) add('favorite_brand', String(item.brand).slice(0, 64), 0.06);
  if (pattern && pattern !== 'unknown' && included(taste.likedPatterns, pattern)) {
    add('liked_pattern', pattern, 0.02);
  }
  if (taste.desiredFit && fit !== 'unknown' && fit) {
    if (fit === taste.desiredFit) add('preferred_fit', fit, 0.025);
    else if (fit === 'slim' || fit === 'wide') score -= 0.025;
  }
  return { blocked: null, score: Math.max(-0.025, Math.min(0.12, score)), matches };
}

export function evaluatePersonalItem(item, descriptor, taste) {
  return !taste?.active ? { blocked: null, score: 0, matches: [] }
    : itemSignals(item, descriptor, taste);
}

/**
 * A small bounded tie-breaker applied only AFTER all hard quality checks.
 * Personal taste must never rescue invalid composition or erase the chosen
 * concept. The score is mean item alignment, not item-count preference.
 */
export function evaluatePersonalLook(items, taste) {
  if (!taste?.active || !Array.isArray(items) || !items.length)
    return { adjustment: 0, matches: [], matchedItems: 0 };
  const scored = items.map(({ item, descriptor }) => evaluatePersonalItem(item, descriptor, taste));
  const adjustment = Math.max(-0.025, Math.min(0.085,
    scored.reduce((sum, x) => sum + x.score, 0) / scored.length));
  const matches = scored.flatMap((x) => x.matches).slice(0, 8);
  return { adjustment, matches, matchedItems: scored.filter((x) => x.matches.length).length };
}
