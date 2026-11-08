export * from "../../../../src/domain/outfitGenerationRequest.ts";
import { OutfitRequestError, MAX_GENERATION_BODY_BYTES, validateOutfitGenerationRequest } from "../../../../src/domain/outfitGenerationRequest.ts";
export async function readOutfitGenerationRequest(request: Request): Promise<Record<string, any>> {
  const declared = request.headers.get('content-length');
  if (declared !== null && Number(declared) > MAX_GENERATION_BODY_BYTES) throw new OutfitRequestError(413,'body','too_large');
  const reader = request.body?.getReader();
  if (!reader) throw new OutfitRequestError();
  let timer: ReturnType<typeof setTimeout>;
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(() => { reject(new OutfitRequestError(408,'body','timeout')); void reader.cancel().catch(() => undefined); }, 15000);
  });
  const chunks: Uint8Array[] = [];
  let bytes = 0;
  try {
    for (;;) {
      const part = await Promise.race([reader.read(), deadline]);
      if (part.done) break;
      bytes += part.value.byteLength;
      if (bytes > MAX_GENERATION_BODY_BYTES) throw new OutfitRequestError(413,'body','too_large');
      chunks.push(part.value);
    }
    const buffer = new Uint8Array(bytes);
    let offset = 0;
    for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.byteLength; }
    let body: unknown;
    try { body = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(buffer)); } catch { throw new OutfitRequestError(400,'body','invalid_json'); }
    // Legacy mobile JS may send an oversized OPTIONAL displayed-product history.
    // Keep recent valid keys and enforce the original 450-item bound; do not alter
    // user budgets, selected items, wardrobe items, or other request constraints.
    if (body && typeof body === 'object' && !Array.isArray(body)) {
      const legacy = (body as Record<string, unknown>).excludeProductKeys;
      if (Array.isArray(legacy) && legacy.length > 450) {
        if (legacy.length > 2048)
          throw new OutfitRequestError(400, 'excludeProductKeys', 'too_many');
        if (legacy.some((key) => typeof key !== 'string' || !key.trim() || key.length > 2048))
          throw new OutfitRequestError(400, 'excludeProductKeys', 'invalid_element');
        (body as Record<string, unknown>).excludeProductKeys = [...new Set(legacy)].slice(-450);
      }
    }
    return validateOutfitGenerationRequest(body);
  } finally {
    clearTimeout(timer!);
    void reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
