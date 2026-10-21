import {budgetCandidatesFromText,stripBudgetText} from '../../backend/supabase/functions/_shared/outfitBudgetText';
import {parseBudgetFromText} from '../../src/utils/outfitBudget';
import {referenceFx} from '../fixtures/budgetFx';
test.each([
 ['budget 80 euros',80,'EUR'],['under 80 pounds',80,'GBP'],['under $1,000.50',1000.5,'USD'],
 ['budget UZS 1 000 000',1000000,'UZS'],['under 1.000.000 UZS',1000000,'UZS'],['under 1.000,50 euros',1000.5,'EUR'],
])('shared parser retains the units in %s',(text,amount,currency)=>{expect(parseBudgetFromText(String(text))).toEqual({amount,currency});expect(new Set(budgetCandidatesFromText(String(text)).map(x=>x.currency))).toEqual(new Set([currency]));});
test('budget removal uses the same recognized spans and preserves weather units',()=>{
 expect(stripBudgetText('Work under 1 000 000 UZS, navy')).toBe('Work, navy');
 expect(stripBudgetText('Wear layers under 20 C')).toBe('Wear layers under 20 C');
});
test('mixed limits cannot silently use guessed rates',()=>{
 expect(parseBudgetFromText('under $100 or €105')).toBeNull();expect(parseBudgetFromText('under $100 or €105',referenceFx)).toMatchObject({amount:100,currency:'USD'});
});
