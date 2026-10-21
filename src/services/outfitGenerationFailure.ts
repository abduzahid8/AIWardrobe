import { withRequestDeadline } from '../lib/requestDeadline';

const messages = {
  ai_generation_unavailable: 'outfitMaker.aiUnavailable',
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
      return new OutfitGenerationFailure(messageFrom(payload, context.status));
    } catch { /* Malformed/slow errors use the localized generic message. */ }
  }
  return new OutfitGenerationFailure();
}
