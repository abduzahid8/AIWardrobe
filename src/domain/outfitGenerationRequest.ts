/** Bound caller-controlled work without silently truncating user constraints. */
export const MAX_GENERATION_BODY_BYTES = 16 * 1024 * 1024;
export class OutfitRequestError extends Error {
  constructor(
    readonly status = 400,
    readonly field = 'body',
    readonly reason = 'invalid_type'
  ) {
    super('invalid_generation_request');
  }
}
const object = (value: unknown): value is Record<string, any> =>
  !!value && typeof value === 'object' && !Array.isArray(value);
function text(value: unknown, max: number, field = 'body') {
  if (value !== undefined && (typeof value !== 'string' || value.length > max))
    throw new OutfitRequestError(
      400,
      field,
      typeof value !== 'string' ? 'invalid_type' : 'too_long'
    );
}
function list(value: unknown, count: number, max: number, field = 'body') {
  if (value === undefined) return;
  if (!Array.isArray(value)) throw new OutfitRequestError(400, field, 'invalid_type');
  if (value.length > count) throw new OutfitRequestError(400, field, 'too_many');
  if (value.some((item) => typeof item !== 'string' || !item.trim() || item.length > max))
    throw new OutfitRequestError(400, field, 'invalid_element');
}
export function validateOutfitGenerationRequest(body: unknown): Record<string, any> {
  if (!object(body)) throw new OutfitRequestError();
  text(body.prompt, 1000, 'prompt');
  text(body.stylePreferences, 80, 'stylePreferences');
  text(body.occasion, 100, 'occasion');
  text(body.conversationContext, 4000, 'conversationContext');
  text(body.anchorItemId, 200, 'anchorItemId');
  if (
    body.limit !== undefined &&
    (!Number.isInteger(body.limit) || body.limit < 1 || body.limit > 15)
  )
    throw new OutfitRequestError(400, 'limit', 'out_of_range');
  // deckTarget = the deck size the caller DISPLAYS. It may be smaller than
  // limit (the Personal Stylist requests extra candidates but shows three
  // cards). Capacity is judged against deckTarget, so it must be a sane
  // integer no larger than the requested limit.
  if (
    body.deckTarget !== undefined &&
    (!Number.isInteger(body.deckTarget) ||
      body.deckTarget < 1 ||
      body.deckTarget > Math.min(15, body.limit ?? 3))
  )
    throw new OutfitRequestError(400, 'deckTarget', 'out_of_range');
  if (body.compositionProtocol !== undefined && body.compositionProtocol !== 'optional-layer-v1')
    throw new OutfitRequestError(400, 'compositionProtocol', 'unsupported');
  list(body.recentLookSignatures, 36, 1000, 'recentLookSignatures');
  list(body.selectedItemIds, 200, 200, 'selectedItemIds');
  list(body.excludeItemIds, 200, 200, 'excludeItemIds');
  list(body.excludeProductKeys, 450, 2048, 'excludeProductKeys');
  list(body.previousDeckProductKeys, 60, 2048, 'previousDeckProductKeys');
  list(body.olderRecentProductKeys, 450, 2048, 'olderRecentProductKeys');
  list(body.previousShoeProductKeys, 60, 2048, 'previousShoeProductKeys');
  list(body.recentShoeProductKeys, 450, 2048, 'recentShoeProductKeys');
  text(body.rotationScope, 500, 'rotationScope');
  text(body.generationRequestId, 100, 'generationRequestId');
  for (const key of ['rotationDecks', 'shoeRotationDecks'])
    if (body[key] !== undefined) {
      if (!Array.isArray(body[key]) || body[key].length > 12)
        throw new OutfitRequestError(400, key, 'invalid_decks');
      for (const deck of body[key]) {
        if (!object(deck)) throw new OutfitRequestError(400, key, 'invalid_deck');
        text(deck.requestId, 100, key + '.requestId');
        list(deck.productKeys, 60, 2048, key + '.productKeys');
      }
    }
  for (const key of ['wardrobeOnly', 'useProvidedWardrobeOnly', 'catalogOnly'])
    if (body[key] !== undefined && typeof body[key] !== 'boolean')
      throw new OutfitRequestError(400, key, 'invalid_type');
  if (body.weather !== undefined && body.weather !== null) {
    if (!object(body.weather)) throw new OutfitRequestError(400, 'weather', 'invalid_type');
    if (
      body.weather.temp !== undefined &&
      (typeof body.weather.temp !== 'number' ||
        !Number.isFinite(body.weather.temp) ||
        body.weather.temp < -100 ||
        body.weather.temp > 100)
    )
      throw new OutfitRequestError(400, 'weather.temp', 'out_of_range');
    text(body.weather.condition, 100, 'weather.condition');
  }
  if (body.preferences !== undefined && body.preferences !== null) {
    if (!object(body.preferences)) throw new OutfitRequestError(400, 'preferences', 'invalid_type');
    for (const key of [
      'favoriteColors',
      'avoidColors',
      'likedPatterns',
      'dislikedPatterns',
      'styleGoals',
    ])
      list(body.preferences[key] ?? undefined, 20, 100, 'preferences.' + key);
    text(body.preferences.fitPreference ?? undefined, 100, 'preferences.fitPreference');
    text(body.preferences.budgetRatesId, 200, 'preferences.budgetRatesId');
  }
  if (body.wardrobeItems !== undefined) {
    if (!Array.isArray(body.wardrobeItems) || body.wardrobeItems.length > 200)
      throw new OutfitRequestError(400, 'wardrobeItems', 'invalid_pool');
    for (const item of body.wardrobeItems) {
      if (!object(item) || typeof item.id !== 'string' || !item.id.trim())
        throw new OutfitRequestError(400, 'wardrobeItems.id', 'invalid_element');
      text(item.id, 200, 'wardrobeItems.id');
      text(item.name ?? undefined, 300, 'wardrobeItems.name');
      text(item.description ?? undefined, 2000, 'wardrobeItems.description');
      for (const key of [
        'brand',
        'color',
        'primary_color',
        'pattern',
        'type',
        'category',
        'macroCategory',
      ])
        text(item[key] ?? undefined, 150, 'wardrobeItems.' + key);
      list(item.styleTags ?? undefined, 20, 100, 'wardrobeItems.styleTags');
      if (Array.isArray(item.material)) list(item.material, 20, 100, 'wardrobeItems.material');
      else text(item.material ?? undefined, 500, 'wardrobeItems.material');
      for (const key of ['imageUrl', 'image', 'thumbnailUrl'])
        if (item[key] !== undefined && typeof item[key] !== 'string')
          throw new OutfitRequestError(400, 'wardrobeItems.' + key, 'invalid_type');
    }
  }
  return body;
}

/** Only optional history is compacted. Explicit user intent, exclusions, selected
 * garments and budgets are never silently changed by this helper. */
export function boundedGenerationHistory(history: Record<string, any>) {
  const strings = (value: unknown, count: number, max: number): string[] =>
    Array.isArray(value)
      ? [
          ...new Set(
            value.filter((x): x is string => typeof x === 'string' && !!x.trim() && x.length <= max)
          ),
        ].slice(-count)
      : [];
  const decks = (value: unknown) =>
    Array.isArray(value)
      ? value
          .slice(-12)
          .filter((d) => object(d) && typeof d.requestId === 'string' && d.requestId.length <= 100)
          .map((d) => ({ requestId: d.requestId, productKeys: strings(d.productKeys, 60, 2048) }))
      : [];
  const previous = strings(history.previousDeckProductKeys, 60, 2048),
    older = strings(history.olderRecentProductKeys, 450 - previous.length, 2048);
  return {
    recentLookSignatures: strings(history.recentLookSignatures, 36, 1000),
    previousDeckProductKeys: previous,
    olderRecentProductKeys: older,
    excludeProductKeys: strings([...older, ...previous], 450, 2048),
    rotationDecks: decks(history.rotationDecks),
    previousShoeProductKeys: strings(history.previousShoeProductKeys, 60, 2048),
    recentShoeProductKeys: strings(history.recentShoeProductKeys, 450, 2048),
    shoeRotationDecks: decks(history.shoeRotationDecks),
  };
}
