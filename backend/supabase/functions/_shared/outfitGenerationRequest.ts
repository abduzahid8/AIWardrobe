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
  list(body.selectedItemIds, 200, 200); list(body.excludeItemIds, 200, 200);
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
export async function readOutfitGenerationRequest(request: Request): Promise<Record<string, any>> {
  const declared = request.headers.get('content-length');
  if (declared !== null && Number(declared) > MAX_GENERATION_BODY_BYTES) throw new OutfitRequestError(413);
  const reader = request.body?.getReader();
  if (!reader) throw new OutfitRequestError();
  let timer: ReturnType<typeof setTimeout>;
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(() => { reject(new OutfitRequestError(408)); void reader.cancel().catch(() => undefined); }, 15000);
  });
  const chunks: Uint8Array[] = [];
  let bytes = 0;
  try {
    for (;;) {
      const part = await Promise.race([reader.read(), deadline]);
      if (part.done) break;
      bytes += part.value.byteLength;
      if (bytes > MAX_GENERATION_BODY_BYTES) throw new OutfitRequestError(413);
      chunks.push(part.value);
    }
    const buffer = new Uint8Array(bytes);
    let offset = 0;
    for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.byteLength; }
    let body: unknown;
    try { body = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(buffer)); } catch { throw new OutfitRequestError(); }
    return validateOutfitGenerationRequest(body);
  } finally {
    clearTimeout(timer!);
    void reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
