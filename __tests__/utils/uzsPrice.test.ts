import {formatUzsPrice} from '../../src/utils/uzsPrice';
import {makeReferenceSnapshot} from '../fixtures/budgetFx';
test('150 USD becomes its reference UZS equivalent, never 150 UZS',()=>{
 expect(formatUzsPrice(150,'USD',makeReferenceSnapshot())).toBe('1 806 708 UZS');
});
test('native UZS amounts require no exchange rate',()=>expect(formatUzsPrice(150000,'UZS')).toBe('150 000 UZS'));
test('unknown or unavailable currency conversion never displays a falsely labeled price',()=>{
 expect(formatUzsPrice(150,'USD')).toBeNull();expect(formatUzsPrice(150,undefined,makeReferenceSnapshot())).toBeNull();
 expect(formatUzsPrice(150,'USD',makeReferenceSnapshot(Date.now()-3600001))).toBeNull();
});
