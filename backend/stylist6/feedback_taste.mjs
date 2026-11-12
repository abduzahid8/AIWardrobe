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
 * label it was displayed under. Rows missing both degrade to no-op; unknown
 * or malformed fields never crash and never guess.
 *
 * No I/O; no persistence; do not log taste. Capped, decayed and typed so 60
 * hostile rows cannot amplify feedback over safety/quality.
 */
export const FEEDBACK_TASTE_VERSION = 's6-feedback-taste-v1';
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

/**
 * Normalized, decayed model over the user's explicit feedback rows. Accepts
 * the DB projection ({rating, concept_id, item_ids, look_signature,
 * created_at}) and the camelCase test/fixture spelling alike. The styling
 * signature document is accepted in the row contract but never scored: only
 * explicit ratings, item ids, concept ids and direction labels influence
 * ranking. Returns an inert model (active:false) for empty/hostile input.
 */
export function feedbackTasteModel(rows, nowMs = Date.now()) {
  const model = {
    version: FEEDBACK_TASTE_VERSION,
    halfLifeDays: FEEDBACK_TASTE_HALF_LIFE_DAYS,
    active: false,
    rows: 0,
    itemTally: new Map(),
    pairTally: new Map(),
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
  model.active = model.itemTally.size > 0 || model.pairTally.size > 0;
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
export function evaluateFeedbackLook(items, conceptId, model) {
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
  const adjustment = clamp(
    ITEM_COMPONENT_SCALE * itemComponent + PAIR_COMPONENT_SCALE * pairComponent,
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
