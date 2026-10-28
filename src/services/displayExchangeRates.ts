import AsyncStorage from '@react-native-async-storage/async-storage';
import {parseCbuBudgetRates, uzbekDateKey, validateBudgetFxSnapshot, type BudgetFxSnapshot} from '../../backend/supabase/functions/_shared/budgetFx';
import {withRequestDeadline} from '../lib/requestDeadline';
import {getBudgetRates} from './budgetRates';
const key='public-cbu-display-rates-v1';
let cached: BudgetFxSnapshot | null=null;
let pending: Promise<BudgetFxSnapshot> | null=null;
/** Display conversions use published reference rates; retailer quotes stay unchanged. */
export async function getDisplayExchangeRates(): Promise<BudgetFxSnapshot> {
    if(cached){try{return validateBudgetFxSnapshot(cached);}catch{cached=null;}}
    if(pending)return pending;
    const work=(async()=>{
        try {const stored=await AsyncStorage.getItem(key);if(stored){cached=validateBudgetFxSnapshot(JSON.parse(stored));return cached;}}catch{/* An expired offline quote cannot price a foreign product. */}
        let snapshot: BudgetFxSnapshot;
        try {
            snapshot=await withRequestDeadline(async signal=>{
                const response=await fetch(`https://cbu.uz/uz/arkhiv-kursov-valyut/json/all/${uzbekDateKey(Date.now())}/`,{signal,headers:{Accept:'application/json'}});
                if(!response.ok)throw Error('reference_unavailable');
                return parseCbuBudgetRates(await response.json());
            },8000);
        }catch{snapshot=await getBudgetRates();}
        cached=validateBudgetFxSnapshot(snapshot);
        await AsyncStorage.setItem(key,JSON.stringify(cached)).catch(()=>undefined);
        return cached;
    })();
    pending=work;try{return await work;}finally{if(pending===work)pending=null;}
}
