/** Bound caller-controlled work without silently truncating user constraints. */
export const MAX_GENERATION_BODY_BYTES = 16 * 1024 * 1024;
export class OutfitRequestError extends Error {
  constructor(readonly status = 400) { super('invalid_generation_request'); }
}
const object = (value: unknown): value is Record<string, any> => !!value && typeof value === 'object' && !Array.isArray(value);
function text(value: unknown, max: number) {
  if (value !== undefined && (typeof value !== 'string' || value.length > max)) throw new OutfitRequestError();
}
function list(value: unknown, count: number, max: number) {
  if (value === undefined) return;
  if (!Array.isArray(value) || value.length > count || value.some(item => typeof item !== 'string' || !item.trim() || item.length > max)) throw new OutfitRequestError();
}
export function validateOutfitGenerationRequest(body: unknown): Record<string, any> {
  if (!object(body)) throw new OutfitRequestError();
  text(body.prompt, 1000); text(body.stylePreferences, 80); text(body.occasion, 100);
  text(body.conversationContext, 4000); text(body.anchorItemId, 200);
  if (body.limit !== undefined && (!Number.isInteger(body.limit) || body.limit < 1 || body.limit > 15)) throw new OutfitRequestError();
  // deckTarget = the deck size the caller DISPLAYS. It may be smaller than
  // limit (the Personal Stylist requests extra candidates but shows three
  // cards). Capacity is judged against deckTarget, so it must be a sane
  // integer no larger than the requested limit.
  if (body.deckTarget !== undefined && (!Number.isInteger(body.deckTarget) || body.deckTarget < 1 || body.deckTarget > Math.min(15, body.limit ?? 3))) throw new OutfitRequestError();
  list(body.recentLookSignatures,36,1000);
  list(body.selectedItemIds, 200, 200); list(body.excludeItemIds, 200, 200);
  list(body.excludeProductKeys, 450, 2048);
  list(body.previousDeckProductKeys, 60, 2048); list(body.olderRecentProductKeys, 450, 2048);
  list(body.previousShoeProductKeys, 60, 2048); list(body.recentShoeProductKeys, 450, 2048);
  text(body.rotationScope, 500); text(body.generationRequestId, 100);
  for(const key of ['rotationDecks','shoeRotationDecks']) if(body[key] !== undefined) {
    if(!Array.isArray(body[key])||body[key].length>12)throw new OutfitRequestError();
    for(const deck of body[key]){if(!object(deck))throw new OutfitRequestError();text(deck.requestId,100);list(deck.productKeys,60,2048);}
  }
  for (const key of ['wardrobeOnly', 'useProvidedWardrobeOnly', 'catalogOnly']) if (body[key] !== undefined && typeof body[key] !== 'boolean') throw new OutfitRequestError();
  if (body.weather !== undefined && body.weather !== null) {
    if (!object(body.weather)) throw new OutfitRequestError();
    if (body.weather.temp !== undefined && (typeof body.weather.temp !== 'number' || !Number.isFinite(body.weather.temp) || body.weather.temp < -100 || body.weather.temp > 100)) throw new OutfitRequestError();
    text(body.weather.condition, 100);
  }
  if (body.preferences !== undefined && body.preferences !== null) {
    if (!object(body.preferences)) throw new OutfitRequestError();
    for (const key of ['favoriteColors', 'avoidColors', 'likedPatterns', 'dislikedPatterns', 'styleGoals']) list(body.preferences[key] ?? undefined, 20, 100);
    text(body.preferences.fitPreference ?? undefined, 100); text(body.preferences.budgetRatesId, 200);
  }
  if (body.wardrobeItems !== undefined) {
    if (!Array.isArray(body.wardrobeItems) || body.wardrobeItems.length > 200) throw new OutfitRequestError();
    for (const item of body.wardrobeItems) {
      if (!object(item) || typeof item.id !== 'string' || !item.id.trim()) throw new OutfitRequestError();
      text(item.id, 200); text(item.name ?? undefined, 300); text(item.description ?? undefined, 2000);
      for (const key of ['brand', 'color', 'primary_color', 'pattern', 'type', 'category', 'macroCategory']) text(item[key] ?? undefined, 150);
      list(item.styleTags ?? undefined, 20, 100);
      if (Array.isArray(item.material)) list(item.material, 20, 100); else text(item.material ?? undefined, 500);
      for (const key of ['imageUrl', 'image', 'thumbnailUrl']) if (item[key] !== undefined && typeof item[key] !== 'string') throw new OutfitRequestError();
    }
  }
  return body;
}
