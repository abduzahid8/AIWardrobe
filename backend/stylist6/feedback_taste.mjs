/**
 * Stylist 6 — explicit feedback taste (mission gate D).
 *
 * Turns the user's own thumbs up/down rows (stylist_look_feedback) into a
 * small, bounded ranking adjustment for later decks. This is NOT a hard
 * block, a quality floor, a coherence gate or a substitute for any admission
 * rule: adjustments apply only to already-feasible candidates and can never
 * rescue invalid composition or invert a quality floor. Only EXPLICIT ±1
 * ratings count — no weak signals, no inferred taste, no profile building.
 *
 * Linking contract: a row is usable when it carries rating ±1 plus at least
 * one of (a) the item ids of the rated look, (b) the concept id + direction
 * label it was displayed under, or (c) a detailed versioned styling signature.
 * The new cross-SKU transfer is soft and evidence-gated, never a hard filter.
 *
 * No I/O; no persistence; do not log taste. Capped, decayed and typed so 60
 * hostile rows cannot amplify feedback over safety/quality.
 */
export const FEEDBACK_TASTE_VERSION = 's6-feedback-taste-v2';
// Exponential recency decay: a rating loses half of its influence every 30
// days. Rows carry only their created_at; `nowMs` is injected so the math is
// fully deterministic for a fixed input set.
export const FEEDBACK_TASTE_HALF_LIFE_DAYS = 30;
// The cloud handler reads at most this many newest rows per request.
export const FEEDBACK_TASTE_MAX_ROWS = 60;
// Per-look total cap — same magnitude as personal_taste's ±0.085 look cap,
// kept slightly tighter because feedback is cross-request and persistent.
export const FEEDBACK_TASTE_LOOK_CAP = 0.06;
// Components of the capped look adjustment: mean saturated item tally over
// the look's items, plus the direction-weighted concept/direction transfer.
const ITEM_COMPONENT_SCALE = 0.045;
const PAIR_COMPONENT_SCALE = 0.03;
const FEATURE_COMPONENT_SCALE = 0.02;
const STYLE_FEATURE_WEIGHTS = [0.35, 0.2, 0.2, 0.15, 0.1];
const MIN_STYLE_SIMILARITY = 0.72;
// Item ids are opaque server keys: trimmed and length-bounded, never
// case-folded (two ids differing in case are two products).
const MAX_ITEM_ID_LENGTH = 128;
const MAX_ITEM_IDS_PER_ROW = 8;
const MAX_TOKEN_LENGTH = 64;
// Direction labels are the deck's own vocabulary. Transfer weight = how much
// a rating of that deck slot predicts the reception of a future candidate of
// the same concept: the signature look is the concept's primary presentation
// (full weight); refresh/discovery variations transfer at half; an honestly
// labelled alternative at a quarter. Unknown labels contribute nothing.
const DIRECTION_TRANSFER = { signature: 1, refresh: 0.5, discovery: 0.5, alternative: 0.25 };
const MAX_AGE_DAYS = 365;
const DAY_MS = 86_400_000;

const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const token = (value) => typeof value === 'string'
  ? value.normalize('NFKC').trim().toLowerCase().replace(/\s+/g, ' ').slice(0, MAX_TOKEN_LENGTH)
  : '';
const itemId = (value) => {
  const id = typeof value === 'string' ? value.trim().slice(0, MAX_ITEM_ID_LENGTH) : '';
  return id.length >= 1 ? id : null;
};
const createdMs = (value) => {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (value instanceof Date && !Number.isNaN(value.getTime())) return value.getTime();
  if (typeof value === 'string' && value.length >= 4) {
    const parsed = Date.parse(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
};

/** Ratings are exactly ±1. Booleans, strings, zero and floats are weak or
 * malformed signals and are dropped, never clamped into evidence. */
const explicitRating = (value) => (value === 1 ? 1 : value === -1 ? -1 : null);

function rowField(row, snake, camel) {
  if (!row || typeof row !== 'object' || Array.isArray(row)) return undefined;
  if (row[snake] !== undefined) return row[snake];
  return row[camel];
}


/** Validated client sq1 styling features: four slots, five semantic fields.
 * Unknown metadata is not evidence; no body, skin, or user identity inference.
 */
function normalizeFeatureMatrix(raw) {
  if (!Array.isArray(raw) || raw.length !== 4) return null;
  let present = 0;
  const features = raw.map(slot => {
    if (slot === null) return null;
    if (!Array.isArray(slot) || slot.length !== 5) return false;
    const fields = slot.map(value => {
      if (value === null || value === undefined) return null;
      if (typeof value !== 'string' || value.length > MAX_TOKEN_LENGTH) return false;
      const word = value.normalize('NFKC').trim().toLowerCase().replace(/\s+/g, '_');
      return !word || ['unknown', 'neutral', 'missing'].includes(word) ? null : word;
    });
    if (fields.includes(false)) return false;
    if (fields[0]) present++;
    return fields;
  });
  return features.includes(false) || present < 2 ? null : features;
}
function signedFeedbackFeatures(raw) {
  return raw && typeof raw === 'object' && !Array.isArray(raw) && raw.version === 1
    ? normalizeFeatureMatrix(raw.features) : null;
}
// Cross-SKU transfer needs two matching garment subtypes in their own slots
// and enough comparable whole-look evidence. One color or garment alone
// cannot create an inferred preference or an exclusion.
function styleSimilarity(a, b) {
  if (!a || !b) return 0;
  let evidence = 0, agreements = 0, sharedSlots = 0, matchingTypes = 0;
  for (let slot = 0; slot < 4; slot++) {
    const x = a[slot], y = b[slot];
    if (!x || !y) continue;
    let compared = 0;
    for (let field = 0; field < 5; field++) {
      if (!x[field] || !y[field]) continue;
      const weight = STYLE_FEATURE_WEIGHTS[field];
      evidence += weight;
      compared++;
      if (x[field] === y[field]) agreements += weight;
    }
    if (compared >= 2) sharedSlots++;
    if (x[0] && y[0] && x[0] === y[0]) matchingTypes++;
  }
  if (sharedSlots < 2 || matchingTypes < 2 || evidence < 1.2) return 0;
  const similarity = agreements / evidence;
  return similarity >= MIN_STYLE_SIMILARITY
    ? Math.min(1, (similarity - MIN_STYLE_SIMILARITY) / (1 - MIN_STYLE_SIMILARITY)) : 0;
}

/**
 * Normalized, decayed model over the user's explicit feedback rows. Accepts
 * the DB projection ({rating, concept_id, item_ids, look_signature,
 * created_at}) and the camelCase test/fixture spelling alike. The styling
 * signature document transfers only high-similarity cross-SKU evidence
 * at a small weight. Returns an inert model for empty or hostile input.
 */
export function feedbackTasteModel(rows, nowMs = Date.now()) {
  const model = {
    version: FEEDBACK_TASTE_VERSION,
    halfLifeDays: FEEDBACK_TASTE_HALF_LIFE_DAYS,
    active: false,
    rows: 0,
    itemTally: new Map(),
    pairTally: new Map(),
    featureEvidence: [],
  };
  if (!Array.isArray(rows)) return model;
  const now = typeof nowMs === 'number' && Number.isFinite(nowMs) ? nowMs : Date.now();
  const itemSums = new Map();
  const pairSums = new Map();
  for (const row of rows.slice(0, FEEDBACK_TASTE_MAX_ROWS)) {
    const rating = explicitRating(rowField(row, 'rating', 'rating'));
    const created = createdMs(rowField(row, 'created_at', 'createdAt'));
    if (rating === null || created === null) continue;
    // Future timestamps are treated as fresh, never as negative age.
    const ageDays = Math.max(0, (now - created) / DAY_MS);
    if (ageDays > MAX_AGE_DAYS) continue;
    const weight = rating * 0.5 ** (ageDays / FEEDBACK_TASTE_HALF_LIFE_DAYS);
    let contributed = false;
    const rawItemIds = rowField(row, 'item_ids', 'itemIds');
    if (Array.isArray(rawItemIds)) {
      for (const entry of rawItemIds.slice(0, MAX_ITEM_IDS_PER_ROW)) {
        const id = itemId(entry);
        if (!id) continue;
        itemSums.set(id, (itemSums.get(id) || 0) + weight);
        contributed = true;
      }
    }
    const features = signedFeedbackFeatures(rowField(row, 'look_signature', 'lookSignature'));
    if (features) { model.featureEvidence.push({ features, weight }); contributed = true; }
    const concept = token(rowField(row, 'concept_id', 'conceptId'));
    const direction = token(rowField(row, 'direction', 'directionLabel'));
    if (concept && direction && DIRECTION_TRANSFER[direction] !== undefined) {
      const key = `${concept}|${direction}`;
      pairSums.set(key, (pairSums.get(key) || 0) + weight);
      contributed = true;
    }
    if (contributed) model.rows += 1;
  }
  // Net signals saturate at ±1: repeated identical ratings say "strong",
  // not "stronger"; a like and a dislike of the same key cancel.
  for (const [id, sum] of itemSums) {
    const tally = clamp(sum, -1, 1);
    if (tally !== 0) model.itemTally.set(id, tally);
  }
  for (const [key, sum] of pairSums) {
    const tally = clamp(sum, -1, 1);
    if (tally !== 0) model.pairTally.set(key, tally);
  }
  model.active = model.itemTally.size > 0 || model.pairTally.size > 0 ||
    model.featureEvidence.length > 0;
  return model;
}

/**
 * Saturated per-item signal in [-1, 1] for the retrieval seed. Unseen or
 * malformed items are exactly neutral; nothing here blocks or rejects.
 */
export function evaluateFeedbackItem(item, model) {
  if (!model?.active || !item || typeof item !== 'object') return { score: 0, match: null };
  const id = itemId(item.id);
  const tally = id !== null ? model.itemTally.get(id) : undefined;
  return tally === undefined ? { score: 0, match: null } : { score: tally, match: id };
}

/**
 * Capped per-look adjustment in ±FEEDBACK_TASTE_LOOK_CAP: the mean item
 * tally across the look's items plus the direction-weighted transfer of the
 * concept's own (concept, direction) tallies. conceptId is the candidate's
 * generating brief id (the native concept slug the deck displays). Malformed
 * items and unknown concepts contribute zero. Pure and deterministic.
 */
export function evaluateFeedbackLook(items, conceptId, model, stylingFeatures) {
  const inert = {
    adjustment: 0, matches: [], likedItems: 0, dislikedItems: 0, conceptId: token(conceptId),
  };
  if (!model?.active || !Array.isArray(items) || !items.length) return inert;
  const concept = token(conceptId);
  let itemSum = 0;
  const matches = [];
  for (const item of items.slice(0, MAX_ITEM_IDS_PER_ROW)) {
    const { score, match } = evaluateFeedbackItem(item, model);
    itemSum += score;
    if (score !== 0 && match) {
      matches.push({ type: 'feedback_item', value: score > 0 ? 'liked' : 'disliked', productId: match });
    }
  }
  // Mean over the look's items: unseen items are exactly neutral, so a
  // single disliked piece cannot dominate a four-piece look on its own.
  const itemComponent = itemSum / Math.max(1, Math.min(items.length, MAX_ITEM_IDS_PER_ROW));
  let pairRaw = 0;
  if (concept) {
    for (const [key, tally] of model.pairTally) {
      if (!key.startsWith(`${concept}|`)) continue;
      const direction = key.slice(concept.length + 1);
      pairRaw += (DIRECTION_TRANSFER[direction] ?? 0) * tally;
    }
  }
  const pairComponent = clamp(pairRaw, -1, 1);
  const candidateFeatures = normalizeFeatureMatrix(stylingFeatures);
  const featureRaw = candidateFeatures && Array.isArray(model.featureEvidence)
    ? model.featureEvidence.reduce((sum, row) =>
        sum + row.weight * styleSimilarity(row.features, candidateFeatures), 0) : 0;
  const featureComponent = clamp(featureRaw, -1, 1);
  if (featureComponent !== 0) matches.push({
    type: 'feedback_style', value: featureComponent > 0 ? 'liked' : 'disliked',
  });
  const adjustment = clamp(
    ITEM_COMPONENT_SCALE * itemComponent + PAIR_COMPONENT_SCALE * pairComponent +
      FEATURE_COMPONENT_SCALE * featureComponent,
    -FEEDBACK_TASTE_LOOK_CAP, FEEDBACK_TASTE_LOOK_CAP,
  );
  return {
    adjustment,
    matches: matches.slice(0, 8),
    likedItems: matches.filter((entry) => entry.value === 'liked').length,
    dislikedItems: matches.filter((entry) => entry.value === 'disliked').length,
    conceptId: concept,
  };
}
