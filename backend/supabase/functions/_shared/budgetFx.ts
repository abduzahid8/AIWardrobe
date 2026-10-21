/** CBU reference rates, expressed as UZS per ONE currency unit. No guessed rates. */
export const BUDGET_CURRENCIES = ['USD', 'EUR', 'GBP', 'UZS'] as const;
export type BudgetCurrency = typeof BUDGET_CURRENCIES[number];
export interface BudgetFxSnapshot {
  readonly id: string;
  readonly source: 'CBU';
  readonly asOf: string;
  readonly fetchedAt: string;
  readonly uzsPerUnit: Readonly<Record<BudgetCurrency, number>>;
}
export const BUDGET_FX_TTL_MS = 60 * 60 * 1000;
const MAX_REFERENCE_AGE_DAYS = 7;
export class BudgetFxError extends Error {
  constructor() { super('budget_rates_unavailable'); this.name = 'BudgetFxError'; }
}
export function uzbekDateKey(now: number): string {
  return new Date(now + 5 * 60 * 60 * 1000).toISOString().slice(0, 10);
}
function validDate(value: string): boolean {
  return /^\d{4}-\d{2}-\d{2}$/.test(value) && Number.isFinite(Date.parse(value)) && new Date(value).toISOString().slice(0, 10) === value;
}
function snapshotId(asOf: string, rates: Record<BudgetCurrency, number>): string {
  return `CBU:${asOf}:${rates.USD}:${rates.EUR}:${rates.GBP}`;
}
export function validateBudgetFxSnapshot(value: unknown, now = Date.now()): BudgetFxSnapshot {
  const row = value as BudgetFxSnapshot | null;
  if (!row || row.source !== 'CBU' || typeof row.asOf !== 'string' || !validDate(row.asOf) || typeof row.fetchedAt !== 'string') throw new BudgetFxError();
  const referenceAge = (Date.parse(uzbekDateKey(now)) - Date.parse(row.asOf)) / 86400000;
  const fetchedAge = now - Date.parse(row.fetchedAt);
  if (!Number.isFinite(fetchedAge) || fetchedAge < -60000 || fetchedAge > BUDGET_FX_TTL_MS || referenceAge < 0 || referenceAge > MAX_REFERENCE_AGE_DAYS) throw new BudgetFxError();
  const rates = row.uzsPerUnit;
  if (!rates || rates.UZS !== 1 || !BUDGET_CURRENCIES.every(currency => typeof rates[currency] === 'number' && Number.isFinite(rates[currency]) && rates[currency] > 0 && rates[currency] <= 1e12)) throw new BudgetFxError();
  if (row.id !== snapshotId(row.asOf, rates)) throw new BudgetFxError();
  return Object.freeze({id: row.id, source: 'CBU' as const, asOf: row.asOf, fetchedAt: row.fetchedAt, uzsPerUnit: Object.freeze({USD: rates.USD, EUR: rates.EUR, GBP: rates.GBP, UZS: 1})});
}
/** Nominal is the number of currency units quoted, not a multiplier. */
export function parseCbuBudgetRates(payload: unknown, now = Date.now()): BudgetFxSnapshot {
  if (!Array.isArray(payload) || payload.length > 1000) throw new BudgetFxError();
  const rates = {UZS: 1} as Record<BudgetCurrency, number>;
  let asOf = '';
  for (const currency of ['USD', 'EUR', 'GBP'] as const) {
    const rows = payload.filter(row => row && row.Ccy === currency);
    if (rows.length !== 1) throw new BudgetFxError();
    const row = rows[0];
    if (typeof row.Date !== 'string' || !/^\d{2}\.\d{2}\.\d{4}$/.test(row.Date)) throw new BudgetFxError();
    const [day, month, year] = row.Date.split('.'), date = `${year}-${month}-${day}`;
    if (asOf && date !== asOf) throw new BudgetFxError();
    asOf = date;
    if (typeof row.Rate !== 'string' || typeof row.Nominal !== 'string' || !/^\d+(?:\.\d+)?$/.test(row.Rate) || !/^\d+(?:\.\d+)?$/.test(row.Nominal)) throw new BudgetFxError();
    const nominal = Number(row.Nominal), rate = Number(row.Rate);
    if (!Number.isFinite(nominal) || nominal <= 0 || !Number.isFinite(rate) || rate <= 0) throw new BudgetFxError();
    rates[currency] = rate / nominal;
  }
  return validateBudgetFxSnapshot({id: snapshotId(asOf, rates), source: 'CBU', asOf, fetchedAt: new Date(now).toISOString(), uzsPerUnit: rates}, now);
}
/** Return Infinity when a value cannot safely be admitted under a spending cap. */
export function referenceAmount(amount: number, from: string, to: string, snapshot: BudgetFxSnapshot, now = Date.now()): number {
  const source = from.trim().toUpperCase() as BudgetCurrency, target = to.trim().toUpperCase() as BudgetCurrency;
  if (!Number.isFinite(amount) || amount < 0 || !BUDGET_CURRENCIES.includes(source) || !BUDGET_CURRENCIES.includes(target)) return Infinity;
  if (source === target) return amount;
  const rates = validateBudgetFxSnapshot(snapshot, now).uzsPerUnit;
  const converted = amount * rates[source] / rates[target];
  return Number.isFinite(converted) ? converted : Infinity;
}

/** A generation pins the snapshot at its initial validation time. */
export interface BudgetFxContext { readonly snapshot: BudgetFxSnapshot; readonly checkedAt: number }
export function budgetFxContext(snapshot: BudgetFxSnapshot, now = Date.now()): BudgetFxContext {
  return Object.freeze({snapshot: validateBudgetFxSnapshot(snapshot, now), checkedAt: now});
}

/** Floating-point tolerance only; never grant an extra fraction of a cent. */
export function fitsBudgetAmount(spend: number, cap: number): boolean {
  if (!Number.isFinite(spend) || !Number.isFinite(cap) || spend < 0 || cap < 0) return false;
  const noise = Math.min(1e-6, Number.EPSILON * Math.max(1, spend, cap) * 8);
  return spend <= cap + noise;
}
