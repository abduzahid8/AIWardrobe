import {resolveShopQuote} from '../../src/utils/outfitBudget';
it('uses only the matching catalog product, never the first unrelated currency',()=>{
 expect(resolveShopQuote({id:'shop_coat',isShopItem:true},[{id:'shoes',price:50,currency:'USD'},{id:'coat',price:90,currency:'EUR'}])).toEqual({price:90,currency:'EUR'});
});
it('keeps a complete validated item snapshot together',()=>{
 expect(resolveShopQuote({id:'shop_coat',isShopItem:true,price:100,currency:' uzs '},[{id:'coat',price:90,currency:'EUR'}])).toEqual({price:100,currency:'UZS'});
});
it('replaces an incomplete price snapshot with a complete matching quote instead of mixing fields',()=>{
 expect(resolveShopQuote({id:'shop_coat',isShopItem:true,price:100},[{id:'coat',price:90,currency:'EUR'}])).toEqual({price:90,currency:'EUR'});
});
it('does not invent a currency for an unmatched amount',()=>{
 expect(resolveShopQuote({id:'shop_missing',isShopItem:true,price:100},[{id:'coat',price:90,currency:'EUR'}])).toEqual({price:100,currency:null});
});
it('owned items have no shopping price even when their ID matches a catalog row',()=>{
 expect(resolveShopQuote({id:'coat',isWardrobe:true,price:100,currency:'USD'},[{id:'coat',price:90,currency:'EUR'}])).toBeNull();
});
it.each([NaN,Infinity,-1,0])('does not expose invalid listed price %s',price=>{
 expect(resolveShopQuote({id:'shop_coat',price,currency:'USD'})).toBeNull();
});
