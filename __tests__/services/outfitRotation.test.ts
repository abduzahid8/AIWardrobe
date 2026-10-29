import {rotationPool,wasRecentlyShown,freshProductCount} from '../../backend/supabase/functions/_shared/outfitRotation';
import {productIdentityKeys} from '../../backend/supabase/functions/_shared/outfitDiversity';
const product=(id:string,macroCategory='shoes')=>({id,name:`Leather loafers ${id}`,imageUrl:`https://cdn.test/${id}.jpg`,macroCategory});
it('blocks aliases with the same normalized name or photo, even when their IDs change',()=>{
 const original=product('old');const seen=new Set(productIdentityKeys(original));
 expect(wasRecentlyShown({...product('new'),name:'LEATHER LOAFERS OLD'},seen)).toBe(true);
 expect(wasRecentlyShown({...product('another'),imageUrl:'http://cdn.test/old.jpg?width=800'},seen)).toBe(true);
 expect(wasRecentlyShown(product('unused'),seen)).toBe(false);
});
it('keeps a rich fresh slot strictly unused, while allowing a scarce slot to recycle',()=>{
 const shoes=[product('s0'),product('s1'),product('s2'),product('s3')];
 const tops=Array.from({length:5},(_,i)=>product(`t${i}`,'top'));
 const seen=new Set([...productIdentityKeys(shoes[0]),...productIdentityKeys(shoes[1]),...productIdentityKeys(tops[0])]);
 const pool=rotationPool([...shoes,...tops],seen,3);
 expect(pool.recycledSlots).toEqual(['shoes']);expect(pool.items.filter(i=>i.macroCategory==='top').map(i=>i.id)).not.toContain('t0');
 expect(pool.items.filter(i=>i.macroCategory==='shoes').slice(0,2).map(i=>i.id)).toEqual(['s2','s3']);
 expect(freshProductCount(pool.items,seen)).toBe(6);
});
it('keeps a locked anchor even when history contains it',()=>{
 const rows=Array.from({length:5},(_,i)=>product(`p${i}`));const seen=new Set(productIdentityKeys(rows[0]));
 expect(rotationPool(rows,seen,3,new Set(['p0'])).items).toContain(rows[0]);
});
