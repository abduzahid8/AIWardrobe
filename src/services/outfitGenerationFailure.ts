import { withRequestDeadline } from '../lib/requestDeadline';

const messages = {
  ai_generation_unavailable: 'outfitMaker.aiUnavailable',
  catalog_lookup_unavailable: 'outfitMaker.catalogUnavailable',
  invalid_generation_request: 'outfitMaker.invalidRequest',
  budget_rates_unavailable: 'outfitMaker.budgetRatesUnavailable',
  unverified_shop_pricing: 'outfitMaker.shopPricesUnavailable',
  no_outfit_within_budget: 'outfitMaker.budgetTooLow',
  no_items_within_budget: 'outfitMaker.noItemsWithinBudget',
  no_matching_clothing: 'outfitMaker.noMatchingClothing',
  invalid_outfit_budget: 'outfitMaker.invalidBudget',
} as const;
type MessageKey = typeof messages[keyof typeof messages] | 'outfitMaker.generateFailed'
  | 'outfitMaker.requestTooLarge' | 'outfitMaker.requestUploadTimedOut';

export class OutfitGenerationFailure extends Error {
  constructor(readonly translationKey: MessageKey = 'outfitMaker.generateFailed') {
    super('outfit_generation_failed');
    this.name = 'OutfitGenerationFailure';
  }
}

function messageFrom(payload: unknown, status?: number): MessageKey | undefined {
  if (!payload || typeof payload !== 'object') return undefined;
  const row = payload as Record<string, unknown>;
  for (const code of [row.code, row.error]) {
    if (typeof code === 'string' && Object.prototype.hasOwnProperty.call(messages, code)) {
      if (code === 'invalid_generation_request' && status === 413) return 'outfitMaker.requestTooLarge';
      if (code === 'invalid_generation_request' && status === 408) return 'outfitMaker.requestUploadTimedOut';
      return messages[code as keyof typeof messages];
    }
  }
  return undefined;
}

/** Keep operational failure categories without logging input, headers or server text. */
function reportUnmappedFailure(error: unknown, phase: 'transport' | 'http_body' | 'http_decode') {
  const sdkName = (error as { name?: unknown } | null)?.name;
  const knownNames = ['FunctionsFetchError', 'FunctionsHttpError', 'FunctionsRelayError', 'AbortError', 'Error'];
  const status = (error as { context?: { status?: unknown } } | null)?.context?.status;
  console.warn('[stylist-request-failure]', {
    phase,
    sdkError: typeof sdkName === 'string' && knownNames.includes(sdkName) ? sdkName : 'unknown',
    httpStatus: typeof status === 'number' && Number.isInteger(status) && status >= 100 && status <= 599 ? status : null,
  });
}

/** Decode only known server codes; never display raw provider/server messages. */
export async function outfitGenerationFailure(data: unknown, error?: unknown): Promise<OutfitGenerationFailure> {
  const direct = messageFrom(data);
  if (direct) return new OutfitGenerationFailure(direct);
  const context = (error as { context?: Response } | null)?.context;
  if (context && typeof context.clone === 'function') {
    try {
      // Supabase HTTP errors retain the Response, rather than returning its JSON
      // as data. Bound this read too so an error body cannot stall the screen.
      const payload = await withRequestDeadline(() => context.clone().json(), 1500);
      // eslint-disable-next-line no-console
      if (__DEV__) console.log('[stylist-http-response]', { httpStatus: context.status, payload });
      const message = messageFrom(payload, context.status);
      if (!message) reportUnmappedFailure(error, 'http_body');
      return new OutfitGenerationFailure(message);
    } catch { reportUnmappedFailure(error, 'http_decode'); return new OutfitGenerationFailure(); }
  }
  reportUnmappedFailure(error, 'transport');
  return new OutfitGenerationFailure();
}
