import {budgetFxContext, type BudgetFxSnapshot, uzbekDateKey} from '../../backend/supabase/functions/_shared/budgetFx';
/** Controlled numerical fixture; live validation fetches CBU independently. */
export function makeReferenceSnapshot(now = Date.now()): BudgetFxSnapshot {
    const asOf = uzbekDateKey(now);
    return {id:`CBU:${asOf}:12044.72:13525.02:16062.84`,source:'CBU',asOf,fetchedAt:new Date(now).toISOString(),uzsPerUnit:{USD:12044.72,EUR:13525.02,GBP:16062.84,UZS:1}};
}
const now = Date.parse('2026-10-20T12:00:00Z');
export const referenceFx = budgetFxContext(makeReferenceSnapshot(now), now);
