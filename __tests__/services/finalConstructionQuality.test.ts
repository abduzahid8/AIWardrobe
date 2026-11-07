import {describeGarment} from '../../src/domain/garmentDescriptor';
import {assessOutfitQuality} from '../../src/domain/occasionRules';
const names=['White long-sleeve cotton poplin shirt','Navy regular wool blazer','Grey tailored wool trousers','Brown leather penny loafers'];
const slots=['top','outerwear','bottom','shoes'];
const look=()=>names.map((name,i)=>({id:`quality-${i}`,name,macroCategory:slots[i],description:''}));
test('recognizes the actual Bond pocket construction while preserving its Casual use',()=>{
 const p=look();p[0]={...p[0],id:'shop_allsaints-usm108sf-2732',name:'Bond Regular-Fit Tencel Shirt'};
 expect(describeGarment(p[0]).utilityConstruction).toBe(true);
 expect(assessOutfitQuality(p,'old_money','Work').acceptance.passed).toBe(false);
 expect(assessOutfitQuality(p,'casual','Casual').acceptance.passed).toBe(true);
});
test.each(['geox-u44q6d00043c6009','geox-u56q6a00022c6372'])('recognizes the observed Moner sole without a generic leather-loafer ban: %s',id=>{
 const p=look();p[3]={...p[3],id:'shop_'+id,name:'Moner W 2Fit Leather Loafers'};
 expect(describeGarment(p[3]).casualSole).toBe(true);
 expect(assessOutfitQuality(p,'old_money','Work').acceptance.passed).toBe(false);
 expect(assessOutfitQuality(p,'casual','Casual').acceptance.passed).toBe(true);
 expect(describeGarment({...p[3],id:'other',name:'Dress penny loafers with slim rubber sole'}).casualSole).toBe(false);
});
test('blocks the actual technical leisure family in refined Work but keeps performance wool and Casual',()=>{
 const p=look();p[1].name='Ecru Technical Leisure Blazer';
 expect(assessOutfitQuality(p,'old_money','Work').acceptance.passed).toBe(false);
 expect(assessOutfitQuality(p,'casual','Casual').acceptance.passed).toBe(true);
 p[1].name='Performance stretch wool tailored blazer';
 expect(assessOutfitQuality(p,'old_money','Work').acceptance.passed).toBe(true);
});
test('drawstring spelling variants retain construction evidence',()=>{
 for(const name of ['Draw-string trousers','Drawstrings linen trousers','Drawstring trousers'])expect(describeGarment({name,macroCategory:'bottom'}).athletic).toBe(true);
 expect(describeGarment({name:'Tailored stretch wool trousers',macroCategory:'bottom'}).athletic).toBe(false);
});
test('known incompatible patterns/fabrics fail the office floor; quiet matching tailoring stays valid',()=>{
 const p=look();p[0].name='Blue fine striped poplin shirt';p[1].name='Navy checked wool blazer';
 expect(assessOutfitQuality(p,'old_money','Work').acceptance.passed).toBe(false);
 p[0].name=names[0];p[1].name='Heavyweight wool blazer';p[2].name='Lightweight tailored linen trousers';
 expect(assessOutfitQuality(p,'old_money','Work').acceptance.passed).toBe(false);
 expect(assessOutfitQuality(look(),'classic','Formal').acceptance.passed).toBe(true);
});

test('the newly observed generic Armani utility shirt and seaside shirts cannot be Formal bases',()=>{
 const p=look();p[0]={...p[0],id:'shop_armani_exchange-xm003247_af28657_ub101',name:'regular fit shirt in cotton poplin'};
 expect(describeGarment(p[0]).utilityConstruction).toBe(true);
 expect(assessOutfitQuality(p,'classic','Formal').acceptance.passed).toBe(false);
 for(const name of ['Coral Linen Seaside Shirt','Sea Green Linen Seaside Shirt']){p[0]={...p[0],id:'seaside',name};expect(assessOutfitQuality(p,'classic','Formal').acceptance.passed).toBe(false);}
 p[0]={...p[0],id:'dress',name:'White long-sleeve linen dress shirt'};
 expect(assessOutfitQuality(p,'classic','Formal').acceptance.passed).toBe(true);
});
test('explicit summer linen tailoring cannot masquerade as a winter flannel suit',()=>{
 const p=look();p[1].name='Double-breasted blazer in pure linen';p[2].name='Anthracite wool flannel regular-fit pants';
 expect(assessOutfitQuality(p,'classic','Formal').acceptance.passed).toBe(false);
 p[2].name='Lightweight tailored linen trousers';expect(assessOutfitQuality(p,'classic','Formal').acceptance.passed).toBe(true);
});
