export interface OutfitBudget { amount: number; currency: string }

/** Empty means no cap. Reject incomplete and ambiguous amounts before network I/O. */
export function parseBudgetControl(input: string, currency: string): OutfitBudget | null {
  const value = input.trim();
  if (!value) return null;
  const compact = value.replace(/[\s\u00a0\u202f]/g, '');
  const grouped = /^\d{1,3}(?:[,.]\d{3})+$/.test(compact);
  if (!grouped && !/^\d+(?:[.,]\d{1,2})?$/.test(compact)) throw new Error('invalid_outfit_budget');
  const amount = Number(grouped ? compact.replace(/[,.]/g, '') : compact.replace(',', '.'));
  if (!Number.isFinite(amount) || amount <= 0 || !['UZS', 'USD', 'EUR', 'GBP'].includes(currency))
    throw new Error('invalid_outfit_budget');
  return { amount, currency };
}

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
