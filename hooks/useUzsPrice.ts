import {useEffect,useState} from 'react';
import type {BudgetFxSnapshot} from '../backend/supabase/functions/_shared/budgetFx';
import {getDisplayExchangeRates} from '../src/services/displayExchangeRates';
import {formatUzsPrice} from '../src/utils/uzsPrice';
export function useUzsDisplayRates(enabled = true): BudgetFxSnapshot | null {
    const [rates,setRates]=useState<BudgetFxSnapshot|null>(null);
    useEffect(()=>{
        if(!enabled)return;
        let active=true;
        void getDisplayExchangeRates().then(value=>{if(active)setRates(value);},()=>{if(active)setRates(null);});
        return()=>{active=false;};
    },[enabled]);
    return rates;
}
export function useUzsPrice(amount?: number,currency?: string): string | null {
    const foreign=!!amount&&!!currency&&currency.toUpperCase()!=='UZS';
    return formatUzsPrice(amount,currency,useUzsDisplayRates(foreign));
}
