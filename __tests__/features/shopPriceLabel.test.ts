import {makeReferenceSnapshot} from '../fixtures/budgetFx';
import { shopPriceLabel } from '../../src/utils/shopPriceLabel';
test('converts the full listed dollar price to UZS',()=>expect(shopPriceLabel({price:12.49,currency:'USD'},'en-US',makeReferenceSnapshot())).toBe('150 439 UZS'));
test('does not relabel a known som price as dollars',()=>{
 const label=shopPriceLabel({price:800000,currency:'UZS'},'en-US');expect(label).toContain('800 000');expect(label).toContain('UZS');expect(label).not.toContain('$');
});
test.each([undefined,'','BAD-CURRENCY'])('unknown currency %s is not guessed as USD',currency=>expect(shopPriceLabel({price:800000,currency},'en-US')).toBeNull());
test.each([NaN,Infinity,-5,0])('invalid listed price %s is not rendered',price=>expect(shopPriceLabel({price,currency:'USD'},'en-US')).toBeNull());
test('converts decimal euros and normalizes the provided currency code',()=>expect(shopPriceLabel({price:14.99,currency:' eur '},'en-US',makeReferenceSnapshot())).toBe('202 740 UZS'));
test('does not round away a listed som fractional amount',()=>expect(shopPriceLabel({price:800000.5,currency:'UZS'},'en-US')).toContain('800 000.50'));
