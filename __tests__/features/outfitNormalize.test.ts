/** Exercise production helpers, rather than copied layering implementations. */
import { isFormalLayer as isFormalLayerItem, isShortsBottom as isShortsItem, repairEdgeItems } from '../../features/outfit-generator/utils/normalizeSlots';
import { resolveOutfitComposition } from '../../backend/supabase/functions/_shared/outfitComposition';
import { fourSlotOutfitReason } from '../../backend/supabase/functions/_shared/outfitStructure';

describe('isFormalLayerItem', () => {
  it('identifies a blazer as formal', () => {
    expect(isFormalLayerItem({ name: 'Navy Blazer', macroCategory: 'outerwear' })).toBe(true);
  });

  it('identifies a suit jacket as formal', () => {
    expect(isFormalLayerItem({ name: 'Charcoal Suit Jacket', macroCategory: 'outerwear' })).toBe(true);
  });

  it('identifies a trench coat as formal', () => {
    expect(isFormalLayerItem({ name: 'Belted Trench Coat', type: 'coat', macroCategory: 'outerwear' })).toBe(true);
  });

  it('identifies a peacoat as formal', () => {
    expect(isFormalLayerItem({ name: 'Wool Peacoat', macroCategory: 'outerwear' })).toBe(true);
  });

  it('does NOT flag a denim jacket as formal', () => {
    expect(isFormalLayerItem({ name: 'Denim Jacket', macroCategory: 'outerwear' })).toBe(false);
  });

  it('does NOT flag a hoodie as formal', () => {
    expect(isFormalLayerItem({ name: 'Zip Hoodie', macroCategory: 'outerwear' })).toBe(false);
  });

  it('does NOT flag a puffer as formal', () => {
    expect(isFormalLayerItem({ name: 'Puffer Jacket', macroCategory: 'outerwear' })).toBe(false);
  });

  it('does NOT flag a bomber jacket as formal', () => {
    expect(isFormalLayerItem({ name: 'Bomber Jacket', macroCategory: 'outerwear' })).toBe(false);
  });

  it('does NOT flag a cardigan as formal', () => {
    expect(isFormalLayerItem({ name: 'Cashmere Cardigan', macroCategory: 'outerwear' })).toBe(false);
  });

  it('returns false for null/undefined', () => {
    expect(isFormalLayerItem(null)).toBe(false);
    expect(isFormalLayerItem(undefined)).toBe(false);
  });
});

describe('isShortsItem', () => {
  it('identifies chino shorts as shorts', () => {
    expect(isShortsItem({ name: 'Chino Shorts', macroCategory: 'bottom' })).toBe(true);
  });

  it('identifies athletic shorts as shorts', () => {
    expect(isShortsItem({ name: 'Athletic Shorts', macroCategory: 'bottom' })).toBe(true);
  });

  it('identifies bermudas as shorts', () => {
    expect(isShortsItem({ name: 'Bermudas', macroCategory: 'bottom' })).toBe(true);
  });

  it('does NOT flag chinos as shorts', () => {
    expect(isShortsItem({ name: 'Slim Chinos', macroCategory: 'bottom' })).toBe(false);
  });

  it('does NOT flag jeans as shorts', () => {
    expect(isShortsItem({ name: 'Straight Jeans', macroCategory: 'bottom' })).toBe(false);
  });

  it('does NOT flag trousers as shorts', () => {
    expect(isShortsItem({ name: 'Wide Leg Trousers', macroCategory: 'bottom' })).toBe(false);
  });

  it('does NOT flag short-sleeved tees mislabeled as bottoms', () => {
    // Feed rows mistag tops as bottoms; 'short-sleeved' describes sleeves.
    expect(isShortsItem({ name: 'Front view of a light beige short-sleeved ecru T-shirt', macroCategory: 'bottom' })).toBe(false);
    expect(isShortsItem({ name: 'Navy blue short sleeve shirt', macroCategory: 'bottom' })).toBe(false);
  });

  it('returns false for null/undefined', () => {
    expect(isShortsItem(null)).toBe(false);
    expect(isShortsItem(undefined)).toBe(false);
  });
});

describe('composition by climate and intent',()=>{
  test.each([undefined,{temp:5,condition:'snow'}])('retains every category: %j',weather=>{
    expect(resolveOutfitComposition('casual',weather,'summer beach')).toEqual({layered:true,activeSlots:['outerwear','top','bottom','shoes']});
  });
  test('explicit no-jacket intent retains all four categories',()=>{
    expect(resolveOutfitComposition('casual',{temp:35,condition:'sunny'},'summer beach')).toEqual({layered:true,activeSlots:['outerwear','top','bottom','shoes']});
    expect(resolveOutfitComposition('casual',{temp:28,condition:'sunny'},'28C, short sleeves, no jacket')).toEqual({layered:true,activeSlots:['outerwear','top','bottom','shoes']});
  });
  const look=[{id:'jacket',name:'Denim jacket',macroCategory:'outerwear'},{id:'shirt',name:'Cotton T-shirt',macroCategory:'top'},{id:'pants',name:'Cotton chinos',macroCategory:'bottom'},{id:'shoes',name:'Leather loafers',macroCategory:'shoes'}];
  test('requires pants even when a casual jacket would match shorts',()=>{
    expect(fourSlotOutfitReason(look)).toBeNull();
    expect(fourSlotOutfitReason(look.map(i=>i.id==='pants'?{...i,name:'Chino shorts'}:i))).not.toBeNull();
  });
  test('a coat cannot replace the base shirt',()=>expect(fourSlotOutfitReason(look.filter(i=>i.id!=='shirt'))).not.toBeNull());
});

it('keeps an authoritative cap-toe shoe through client repair, preserving all four slots',()=>{
 const items=[
  {id:'layer',name:'Navy cotton blazer',macroCategory:'outerwear'},
  {id:'shirt',name:'White cotton shirt',macroCategory:'top'},
  {id:'pants',name:'Navy cotton trousers',macroCategory:'bottom'},
  {id:'shop_thursday-253390159881',name:'Executive | Black',type:'shoes',macroCategory:'shoes',description:'Black leather oxford dress shoe. style Cap Toe style Lace-Up.'},
 ];
 const repaired=repairEdgeItems(items);
 expect(repaired).toHaveLength(4);expect(fourSlotOutfitReason(repaired)).toBeNull();
 expect(repaired[3].id).toBe('shop_thursday-253390159881');
});
it('still drops a real accessory even when a catalog row labels it as shoes',()=>{
 expect(repairEdgeItems([{id:'cap',name:'Cotton baseball cap',type:'shoes',macroCategory:'shoes'}])).toEqual([]);
});
