import {referenceAmount, type BudgetFxSnapshot} from '../../backend/supabase/functions/_shared/budgetFx';
import {formatMoney} from './outfitBudget';
/** Never relabel an unconverted foreign amount as UZS. */
export function formatUzsPrice(amount?: number, currency?: string, rates?: BudgetFxSnapshot | null): string | null {
    if(typeof amount!=='number'||!Number.isFinite(amount)||amount<=0)return null;
    const from=String(currency||'').trim().toUpperCase();
    if(from==='UZS')return formatMoney(amount,'UZS');
    if(!rates)return null;
    try {
        const converted=referenceAmount(amount,from,'UZS',rates);
        return Number.isFinite(converted)?formatMoney(Math.round(converted),'UZS'):null;
    }catch{return null;}
}
