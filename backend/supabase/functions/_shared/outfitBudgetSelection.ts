export interface OutfitBudget { amount: number; currency: string }

/** A prompt may tighten an explicit spending limit, but cannot raise it. */
export function strictestOutfitBudget(
  selected: unknown, parsed: OutfitBudget | null,
  convert: (budget: OutfitBudget, currency: string) => OutfitBudget | null,
): OutfitBudget | null {
  const validate = (value: unknown): OutfitBudget | null => {
    if (value == null) return null;
    if (typeof value !== 'object' || Array.isArray(value)) throw new Error('invalid_outfit_budget');
    const candidate = value as Record<string, unknown>;
    const currency = typeof candidate.currency === 'string' ? candidate.currency.trim().toUpperCase() : '';
    if (typeof candidate.amount !== 'number' || !Number.isFinite(candidate.amount) || candidate.amount < 0
      || !['USD', 'EUR', 'GBP', 'UZS'].includes(currency)) throw new Error('invalid_outfit_budget');
    return { amount: candidate.amount, currency };
  };
  const explicit = validate(selected), text = validate(parsed);
  if (!explicit) return text;
  if (!text) return explicit;
  const comparable = validate(convert(text, explicit.currency));
  if (!comparable || comparable.currency !== explicit.currency) throw new Error('invalid_outfit_budget');
  return { amount: Math.min(explicit.amount, comparable.amount), currency: explicit.currency };
}
