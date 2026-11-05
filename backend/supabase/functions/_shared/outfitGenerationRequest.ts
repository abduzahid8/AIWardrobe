export * from "../../../../src/domain/outfitGenerationRequest.ts";
import { OutfitRequestError, MAX_GENERATION_BODY_BYTES, validateOutfitGenerationRequest } from "../../../../src/domain/outfitGenerationRequest.ts";
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
