import type {BudgetFxSnapshot} from '../../backend/supabase/functions/_shared/budgetFx';
import {formatUzsPrice} from './uzsPrice';
/** Customer prices are UZS; original quotes remain available for validation. */
export function shopPriceLabel(item: {price?: number; currency?: string}, _locale?: string, rates?: BudgetFxSnapshot | null): string | null {
  return formatUzsPrice(item.price,item.currency,rates);
}
