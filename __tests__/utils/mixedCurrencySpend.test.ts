import {referenceFx} from '../fixtures/budgetFx';
import { outfitShopSpend, itemShopPrice, resolveShopPrice, resolveShopQuote, buildAgentClothingContext } from '../../src/utils/outfitBudget';
const local = { id: 'shop_local', price: 100000, currency: 'UZS', isShopItem: true };
const imported = { id: 'shop_import', price: 20, currency: 'USD', isShopItem: true };
test('mixed currencies are converted before adding and cannot pass a 200000 UZS cap', () => {
    expect(outfitShopSpend([local, imported], undefined, 'UZS', referenceFx)).toBeCloseTo(340894.4);
    expect(outfitShopSpend([local, imported], undefined, undefined, referenceFx)).toBeCloseTo(340894.4);
    expect(outfitShopSpend([imported, local], undefined, 'UZS', referenceFx)).toBeCloseTo(340894.4);
    expect(outfitShopSpend([local, imported], undefined, 'USD', referenceFx)).toBeCloseTo(100000/12044.72+20);
});
test('GBP and EUR use dated reference units without rounding each line', () => {
    expect(itemShopPrice({ ...imported, price: 1.08, currency: 'EUR' }, 'GBP', referenceFx)).toBeCloseTo(1.08*13525.02/16062.84);
});
test.each(['ZZZ', '', null])('unknown currency %s fails closed under an explicit cap', currency => {
    expect(itemShopPrice({ ...imported, currency }, 'UZS')).toBe(Infinity);
    expect(outfitShopSpend([{ ...imported, currency }], undefined, 'UZS', referenceFx)).toBe(Infinity);
});
test('a matching complete catalog quote is used without changing the listed snapshot', () => {
    const incomplete = { ...imported, currency: null, price: 1 };
    expect(resolveShopPrice(incomplete, [imported], 'UZS', referenceFx)).toBeCloseTo(240894.4);
    expect(resolveShopQuote(incomplete, [imported])).toEqual({ price: 20, currency: 'USD' });
    expect(incomplete.price).toBe(1);
});
test('owned items stay free even when a catalog id matches', () => {
    expect(resolveShopPrice({ ...local, isShopItem: false, isWardrobe: true }, [local], 'USD')).toBe(0);
});
test('chat candidate filtering compares converted amounts and preserves listed units', () => {
    const context = buildAgentClothingContext({ shop: [{ ...local, name: 'local shirt' }, { ...imported, name: 'imported shirt' }], userMessage: 'under $10', budgetFx: referenceFx });
    expect(context).toContain('local shirt 100 000 UZS');
    expect(context).not.toContain('imported shirt');
});

test.each([undefined, null, 0, -1, NaN, Infinity])('an unverified shop price %s cannot be counted as free under a cap', price => {
    expect(itemShopPrice({ ...imported, price }, 'USD')).toBe(Infinity);
    expect(outfitShopSpend([{ ...imported, price }], undefined, 'USD', referenceFx)).toBe(Infinity);
    expect(itemShopPrice({ ...imported, price, isShopItem: false, isWardrobe: true }, 'USD')).toBe(0);
});
test('a matching unpriced catalog row cannot silently become free when reference flags are absent', () => {
    expect(resolveShopPrice({id:'product'}, [{id:'product',currency:'USD'}], 'USD')).toBe(Infinity);
});
test('a matching owned row keeps a retail value out of budget spending', () => {
    expect(resolveShopPrice({id:'owned'}, [{id:'owned',isWardrobe:true,price:200,currency:'USD'}], 'USD')).toBe(0);
});
