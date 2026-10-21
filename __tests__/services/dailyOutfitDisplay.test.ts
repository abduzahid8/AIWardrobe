import { dailyOutfitDisplayItems } from '../../src/services/dailyOutfitDisplay';
const product={id:'shop_exact',name:'Original product',brand:'Original brand',price:800000,currency:'UZS',macroCategory:'top',isShopItem:true,imageUrl:'https://catalog.invalid/exact.jpg',shopUrl:'https://retailer.invalid/exact'};
test('Home display preserves accepted product attributes and link',()=>{
 const result=dailyOutfitDisplayItems({items:[product]});
 expect(result).toEqual([{...product,image:product.imageUrl}]);
 expect(result[0]).not.toBe(product);
});
test('a missing image does not substitute a different catalog product or invent a slot',()=>{
 const result=dailyOutfitDisplayItems({items:[{...product,imageUrl:undefined}]});
 expect(result).toHaveLength(1);expect(result[0].id).toBe(product.id);expect(result[0].name).toBe(product.name);expect(result[0].price).toBe(product.price);expect(result[0].image).toBe('');
});
test('empty generation remains empty rather than becoming a curated deck',()=>{
 expect(dailyOutfitDisplayItems({items:[]})).toEqual([]);expect(dailyOutfitDisplayItems({items:null})).toEqual([]);
});
