import {assessOutfitQuality,refinedDateProductReason} from '../../backend/supabase/functions/_shared/occasionRules';
const piece=(name:string,macroCategory:string,extra:any={})=>({name,macroCategory,...extra});
const refined=[piece('Fine merino cardigan','outerwear'),piece('White cotton dress shirt','top'),piece('Navy tailored wool trousers','bottom'),piece('Brown leather loafers','shoes')];
it('rejects observed Western denim and leather trucker signals only in their request context',()=>{
 for(const item of [piece('Western Denim Shirt','top'),piece('Chaiyo Leather Trucker Jacket','outerwear'),piece('Xylon Shearling Biker Jacket','outerwear')]){
  expect(refinedDateProductReason(item,'old_money','date')).not.toBeNull();
  expect(refinedDateProductReason(item,'casual','casual')).toBeNull();
 }
 expect(refinedDateProductReason(piece('Clean suede blazer','outerwear'),'old_money','date')).toBeNull();
});
it('uses actual rugged lug-sole evidence rather than globally banning leather boots',()=>{
 expect(refinedDateProductReason(piece('President StormKing Bull Brown','shoes',{description:'Rugged leather reinforced eyelets rubber lug outsoles'}),'old_money','date')).not.toBeNull();
 expect(refinedDateProductReason(piece('Duke black leather Chelsea boots','shoes'),'old_money','date')).toBeNull();
});
it('scores all six explained dimensions and accepts a coherent refined look',()=>{
 const q=assessOutfitQuality(refined,'old_money','date',{temp:18});
 expect(q.reasons).toEqual([]);expect(q.score).toBe(1);expect(Object.keys(q.dimensions)).toHaveLength(6);
});
it('evaluates complementary colors between actual outfit pairs, preserving muted mixed palettes',()=>{
 const bright=refined.map((p,i)=>({...p,color:['bright red','bright green','navy','brown'][i]}));
 expect(assessOutfitQuality(bright,'old_money','date').reasons).toContain('competing vivid complementary colors across outfit pairs');
 const muted=refined.map((p,i)=>({...p,color:['burgundy','cream','navy','brown'][i]}));
 expect(assessOutfitQuality(muted,'old_money','date').reasons).toEqual([]);
});
it('rejects an observed heavy rugged upper and airy linen combination without banning linen',()=>{
 const items=refined.map((p,i)=>i===0?piece('Heavy shearling leather coat','outerwear'):i===2?piece('Brown linen trousers','bottom'):p);
 expect(assessOutfitQuality(items,'old_money','date').dimensions.materialHarmony).toBe(0);
 expect(assessOutfitQuality(refined.map((p,i)=>i===0?piece('Lightweight linen blazer','outerwear'):i===2?piece('Cotton chino trousers','bottom'):p),'old_money','date',{temp:24}).reasons).toEqual([]);
});
it.each(['Silvo Cotton Flight Jacket','Kirk Waxed-Cotton Racer Jacket','Nylon seersucker jacket zip pocket','Contrasting bands jacket with zip'])('rejects observed sport/utility construction: %s',name=>{
 expect(refinedDateProductReason(piece(name,'outerwear'),'old_money','date')).not.toBeNull();
 expect(refinedDateProductReason(piece(name,'outerwear'),'casual','casual')).toBeNull();
});
it('rejects the observed beach towelling polo rather than every polo or every textured shirt',()=>{
 expect(refinedDateProductReason(piece('Emerald Terry Towelling Polo Shirt','top'),'old_money','date')).not.toBeNull();
 expect(refinedDateProductReason(piece('Fine cotton knit polo shirt','top'),'old_money','date')).toBeNull();
});
it('requires evidence of refined jacket construction, without banning clean suede tailoring',()=>{
 expect(refinedDateProductReason(piece('Miller Leather Jacket','outerwear'),'old_money','date')).not.toBeNull();
 expect(refinedDateProductReason(piece('Single-breasted suede blazer jacket','outerwear'),'old_money','date')).toBeNull();
});
it('detects the real alpaca sweater / linen base mismatch at complete-look level',()=>{
 const items=refined.map((p,i)=>i===0?piece('Soren Alpaca-Blend Crew-Neck Sweater','outerwear'):i===1?piece('Nut Linen Pop-over Shirt','top'):p);
 expect(assessOutfitQuality(items,'old_money','date').dimensions.materialHarmony).toBe(0);
});
it('rejects the displayed interlock knit-trouser construction for the refined date, while leaving casual requests alone',()=>{
 const p=piece('Interlock trousers','bottom');
 expect(refinedDateProductReason(p,'old_money','date')).not.toBeNull();
 expect(refinedDateProductReason(p,'casual','casual')).toBeNull();
});
it('rejects the displayed wool trench / airy linen trouser mismatch while retaining summer tailoring',()=>{
 const cold=refined.map((p,i)=>i===0?piece('Coltra Double-Breasted Wool Trench Coat','outerwear'):i===2?piece('Navy Linen Seaside Trousers','bottom'):p);
 expect(assessOutfitQuality(cold,'old_money','date').dimensions.materialHarmony).toBe(0);
 const summer=cold.map((p,i)=>i===0?piece('Unlined tropical wool blazer','outerwear'):p);
 expect(assessOutfitQuality(summer,'old_money','date').reasons).toEqual([]);
});
it('ranks a coherent refined outfit above one with a single acceptable casual detail, without treating the detail as a global ban',()=>{
 const clean=assessOutfitQuality(refined,'old_money','date');
 const soft=assessOutfitQuality(refined.map((p,i)=>i===1?piece('Cotton shirt with subtle military detail','top'):p),'old_money','date');
 expect(soft.reasons).toEqual([]);expect(soft.score).toBeLessThan(clean.score);
 expect(soft.evidence).toContain('one casual detail balanced by refined companions');
});
it('rejects the observed acid-washed knit and relaxed shirt for this date without banning casual wardrobes',()=>{
 for(const p of [piece('Kit Acid-Washed Cotton Crew-Neck Sweater','outerwear'),piece('Relaxed Button-Down Shirt Light Blue Stripe','top')]){
  expect(refinedDateProductReason(p,'old_money','date')).not.toBeNull();
  expect(refinedDateProductReason(p,'casual','casual')).toBeNull();
 }
});
it('preserves quality-approved family alternatives before a six-product slot trim',()=>{
 const {spreadQualityFamilies}=require('../../backend/supabase/functions/_shared/occasionRules');
 const items=[...Array.from({length:12},(_,i)=>({id:`knit-${i}`,family:'knitwear'})),{id:'blazer',family:'blazer'},{id:'coat',family:'coat'}];
 const spread=spreadQualityFamilies(items,(i:any)=>i.family);
 expect(spread.slice(0,3).map((i:any)=>i.family)).toEqual(['knitwear','blazer','coat']);
 expect(new Set(spread.map((i:any)=>i.id))).toEqual(new Set(items.map(i=>i.id)));
});
it.each([
 ['Dark Blue Denim Leisure Blazer','outerwear'],
 ['Ecru Technical Leisure Blazer','outerwear'],
 ['Brook Regular-Fit Check Western Shirt','top'],
 ['Cotton Ripstop Shirt','top'],
 ['Bonded Stretch Cotton Sports Pants','bottom'],
])('rejects the displayed contextual style mismatch %s without changing casual styling', (name,slot)=>{
 const p=piece(name,slot);expect(refinedDateProductReason(p,'old_money','date')).not.toBeNull();
 expect(refinedDateProductReason(p,'casual','casual')).toBeNull();
});
