import { fourSlotOutfitReason, outfitStructureReason } from '../../backend/supabase/functions/_shared/outfitStructure';
const top = { id: 'top', name: 'White dress shirt', macroCategory: 'top' };
const bottom = { id: 'bottom', name: 'Navy trousers', macroCategory: 'bottom' };
const shoes = { id: 'shoes', name: 'Oxford shoes', macroCategory: 'shoes' };
const dress = { id: 'dress', name: 'Silk evening dress', macroCategory: 'top' };
const layer = { id: 'layer', name: 'Wool coat', macroCategory: 'outerwear' };
it('accepts a dress and shoes, with a layer only when required', () => {
  expect(outfitStructureReason([dress, shoes])).toBeNull();
  expect(outfitStructureReason([dress, shoes], true)).toBe('missing_layer');
  expect(outfitStructureReason([dress, shoes, layer], true)).toBeNull();
});
it('does not mistake a dress shirt for a dress', () => {
  expect(outfitStructureReason([top, shoes])).toBe('missing_slot');
  expect(outfitStructureReason([top, bottom, shoes])).toBeNull();
});
it('rejects duplicate slots, repeated IDs and a dress plus trousers', () => {
  expect(outfitStructureReason([top, bottom, shoes, { ...bottom, id: 'second' }])).toBe('duplicate_slot');
  expect(outfitStructureReason([top, { ...bottom, id: top.id }, shoes])).toBe('duplicate_item');
  expect(outfitStructureReason([dress, bottom, shoes])).toBe('dress_with_bottom');
});
it.each([null, {}, [null], [{ ...top, id: '' }, bottom, shoes], [{ ...top, id: 'placeholder_top' }, bottom, shoes]])('rejects malformed product identity without repairing it: %p', items => {
  expect(outfitStructureReason(items)).not.toBeNull();
});

it.each([
 {name:'DRESS PENNY LOAFERS',macroCategory:'shoes'},
 {name:'Dress leather loafers',macroCategory:'top'},
 {name:'Dress Oxford shoes',type:'shoes'},
 {name:'Dress trousers',macroCategory:'bottom'},
 {name:'Dress coat',macroCategory:'outerwear'},
])('does not let formal clothing masquerade as a dress: %p',item=>{
 const {isDressGarment}=require('../../backend/supabase/functions/_shared/outfitStructure');
 expect(isDressGarment(item)).toBe(false);
});
it.each(['Cotton shirt dress','T-shirt dress','Polo dress'])('preserves a real %s',name=>{
 const {isDressGarment}=require('../../backend/supabase/functions/_shared/outfitStructure');
 expect(isDressGarment({name,macroCategory:'top'})).toBe(true);
});

it.each(['43Mm Silicone Strap Sport Watch','SUIT CARRIER','SMALL TRAINING TOWEL','LARGE TRAINING TOWEL'])('rejects a non-garment mislabeled as a top: %s',name=>{
 expect(outfitStructureReason([{...top,name},bottom,shoes])).toBe('non_outfit_item');
});
it.each(['Towel-knit cotton shirt','Watch-print cotton shirt'])('does not confuse a garment texture or print with an accessory: %s',name=>{
 expect(outfitStructureReason([{...top,name},bottom,shoes])).toBeNull();
});

it('rejects a plain knit garment mislabeled as a shirt but accepts a knit polo',()=>{
 const fourSlots=[layer,bottom,shoes];
 expect(fourSlotOutfitReason([...fourSlots,{id:'plain-knit',name:'Fine gauge knit',type:'shirt',macroCategory:'top'}])).toBe('invalid_required_category');
 expect(fourSlotOutfitReason([...fourSlots,{id:'knit-polo',name:'Navy knit polo',type:'shirt',macroCategory:'top'}])).toBeNull();
});

it.each(['METAL KEY RING WITH KEYS','METAL CHAIN KEY RING','TOTE BAG','Textured Mini Reporter Bag'])('rejects another accessory labelled as a top: %s',name=>{
 expect(outfitStructureReason([{...top,name},bottom,shoes])).toBe('non_outfit_item');
});
it.each(['Watch-print dress','COTTON-VISCOSE SHIRT WITH TIES','Baggy cotton shirt'])('preserves an actual garment named %s',name=>{
 const {isNonOutfitItem}=require('../../backend/supabase/functions/_shared/outfitStructure');
 expect(isNonOutfitItem({name})).toBe(false);
});
