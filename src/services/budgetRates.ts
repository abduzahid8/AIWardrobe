import {supabase} from '../../lib/supabase';
import {withRequestDeadline} from '../lib/requestDeadline';
import {BudgetFxError, type BudgetFxSnapshot, validateBudgetFxSnapshot} from '../../backend/supabase/functions/_shared/budgetFx';

export type {BudgetFxSnapshot} from '../../backend/supabase/functions/_shared/budgetFx';
let cached: BudgetFxSnapshot | null = null;
let pending: Promise<BudgetFxSnapshot> | null = null;
/** Public reference data may be shared across sessions; callers pin the immutable return value. */
export async function getBudgetRates(): Promise<BudgetFxSnapshot> {
    if (cached) {
        try { return validateBudgetFxSnapshot(cached); } catch { cached = null; }
    }
    if (pending) return pending;
    const work = withRequestDeadline(async signal => {
        const {data, error} = await supabase.functions.invoke('budget-rates', {signal, body: {}});
        if (signal.aborted || error || !data?.success) throw new BudgetFxError();
        const snapshot = validateBudgetFxSnapshot(data.snapshot);
        if (signal.aborted) throw new BudgetFxError();
        cached = snapshot;
        return snapshot;
    }, 15000);
    pending = work;
    try { return await work; }
    finally { if (pending === work) pending = null; }
}
