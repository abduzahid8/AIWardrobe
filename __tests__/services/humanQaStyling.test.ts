import {describeGarment} from '../../src/domain/garmentDescriptor';
import {assessOutfitQuality,styleJudgeReason} from '../../src/domain/occasionRules';
import {scoreGarmentForContext} from '../../src/domain/garmentContext';
import {stylingContext} from '../../src/domain/stylingContext';
const slots=['top','outerwear','bottom','shoes'];
const refined=['White long-sleeve poplin shirt','Navy regular wool blazer','Grey tailored wool trousers','Brown leather penny loafers'];
const look=(names=refined)=>names.map((name,i)=>({id:`qa-${i}`,name,macroCategory:slots[i]}));
const quality=(names:string[],style='old_money',occasion='Work')=>assessOutfitQuality(look(names),style,occasion).acceptance;

test.each(['Short-sleeve shirt with flap chest pockets','Short-sleeve shirt with two front flap pockets','Cotton shirt with utility pockets','Short sleeved shirt with twin patch pockets'])(
 'utility construction cannot hide behind generic shirt labels: %s',name=>{
 const p=[...refined];p[0]=name;
 expect(describeGarment(look(p)[0]).utilityConstruction).toBe(true);
 expect(quality(p).passed).toBe(false);
 });
test('single Oxford chest pocket and unbranded short-sleeve knit polo remain refined',()=>{
 for(const name of ['Oxford cotton shirt with one chest patch pocket','Plain fine merino short-sleeve polo']){
 const p=[...refined];p[0]=name;
 expect(describeGarment(look(p)[0]).utilityConstruction).toBe(false);
 expect(quality(p,'old_money','Date night').passed).toBe(true);
 }
});
test.each([
 ['Black leather Oxford shoes with slim rubber sole','dress','formal'],
 ['Black Oxford hybrid with sneaker sole','sneaker','casual'],
 ['Black derby shoes with chunky rubber soles','comfort','casual'],
 ['Penny loafers with crepe sole','comfort','casual'],
 ['Leather shoes with cushioned insole and rubber sole','unknown','unknown'],
 ['Minimal white leather sneakers','sneaker','casual'],
])('shoe silhouette/sole evidence stays distinct: %s',(name,sole,formality)=>{
 const d=describeGarment({name,macroCategory:'shoes'});
 expect(d.soleProfile).toBe(sole);expect(d.formality).toBe(formality);
});
test.each(['Black Oxford hybrid with sneaker sole','Brown chunky-soled derby shoes','Flextride Plus Leather Loafers - Black'])(
 'refined office floor rejects casual soles: %s',name=>{
 const p=[...refined];p[3]=name;expect(quality(p).passed).toBe(false);
 });
test('actual imported catalog captions carry utility/leisure/comfort evidence',()=>{
 const rows=[
 {id:'shop_allsaints-usm108sf-2732',name:'Bond Regular-Fit Tencel Shirt',macroCategory:'top'},
 {id:'shop_aurelien-10066008899914',name:'Navy Technical Leisure Blazer',macroCategory:'outerwear'},
 {id:'shop_geox-u44q6d00043c6009',name:'Moner W 2Fit Leather Loafers - Coffee',macroCategory:'shoes'}];
 expect(describeGarment(rows[0]).utilityConstruction).toBe(true);
 expect(describeGarment(rows[1]).technicalLeisure).toBe(true);
 expect(describeGarment(rows[2]).soleProfile).toBe('comfort');
 for(const r of rows){const p=look();p[slots.indexOf(r.macroCategory)]=r;expect(assessOutfitQuality(p,'old_money','Work').acceptance.passed).toBe(false);}
});
test.each(['Light wash straight jeans','Heavily faded jeans','Stone-washed jeans'])(
 'Oxford and faded denim fail before the complete-deck shortcut: %s',name=>{
 const p=['Plain merino polo','Suede trucker jacket',name,'Black polished Oxford shoes'];
 expect(quality(p,'casual','Date night').passed).toBe(false);
 expect(quality(p,'casual','Casual').passed).toBe(false);
 });
test('date-night Casual has its own floor while clean denim/suede/trainers remain available',()=>{
 const p=['Plain fine cotton polo','Suede trucker jacket','Light wash straight jeans','Minimal white leather sneakers'];
 expect(quality(p,'casual','Casual').passed).toBe(true);
 expect(quality(p,'casual','Date night').passed).toBe(false);
 p[2]='Dark clean straight denim jeans';
 expect(quality(p,'casual','Date night').passed).toBe(true);
 p[1]='Technical nylon zip jacket';p[3]='Black polished Oxford shoes';
 expect(quality(p,'casual','Date night').passed).toBe(false);
});
test.each(['All-over monogram jacquard polo','Polo with repeated logos','Polo with large logo print','GOODYEAR© GRAPHIC RETRO POLO SHIRT'])(
 'Old Money date requires restrained branding: %s',name=>{
 const p=[...refined];p[0]=name;expect(quality(p,'old_money','Date night').passed).toBe(false);
 });
test('sporty zip layers and draw-string variants cannot pass an Old Money date',()=>{
 for(const name of ['Draw-string trousers','Draw string trousers','Drawstrings linen trousers']){const p=[...refined];p[2]=name;expect(quality(p,'old_money','Date night').passed).toBe(false);}
 const p=[...refined];p[1]='Full-zip track jacket with contrast panels';expect(quality(p,'old_money','Date night').passed).toBe(false);
 p[1]='Fine merino zip cardigan';expect(quality(p,'old_money','Date night').passed).toBe(true);
});
test('quiet pattern, multiple colors and compatible cloth can coordinate without monochrome',()=>{
 const p=['Blue fine striped poplin shirt','Camel wool blazer','Charcoal wool trousers with tailored pleats','Burgundy leather penny loafers'];
 expect(quality(p,'old_money','Date night').passed).toBe(true);
 p[1]='Camel checked wool blazer';expect(quality(p,'old_money','Date night').passed).toBe(false);
 p[0]=refined[0];p[1]='Heavyweight wool blazer';p[2]='Lightweight tailored linen trousers';
 expect(quality(p,'old_money','Date night').passed).toBe(false);
});
test.each(['Denim jacket','Suede trucker jacket','Cotton overshirt','Fine wool cardigan'])(
 'preserves Semi-Classic Casual with %s',layer=>{
 const p=['Plain cotton T-shirt',layer,'Dark straight denim jeans','Minimal white leather sneakers'];
 expect(quality(p,'semi_classic','Casual').passed).toBe(true);
 });
test('utility shirt remains valid with a Casual overshirt and trainers',()=>{
 const p=['Utility shirt with twin chest pockets','Cotton overshirt','Dark straight jeans','Minimal white leather sneakers'];
 expect(quality(p,'semi_classic','Casual').passed).toBe(true);
});
test('pre-cap ranking distinguishes date night from ordinary Casual',()=>{
 const plain={name:'Dark clean straight denim jeans',macroCategory:'bottom'}, faded={...plain,name:'Heavily faded straight denim jeans'};
 const casual=stylingContext({style:'casual',occasion:'Casual'}),date=stylingContext({style:'casual',occasion:'Date night'});
 expect(scoreGarmentForContext(plain,casual)).toBe(scoreGarmentForContext(faded,casual));
 expect(scoreGarmentForContext(plain,date)).toBeGreaterThan(scoreGarmentForContext(faded,date));
});

test.each(['shop_geox-u65hyb00046c6009','shop_geox-u66cjc00047c9999'])(
 'verified Spherica leather model is comfort, not dress or sneaker: %s',id=>{
 const p=look();p[3]={id,name:'Spherica leather dress loafers',macroCategory:'shoes'};
 expect(describeGarment(p[3]).soleProfile).toBe('comfort');
 expect(assessOutfitQuality(p,'old_money','Work').acceptance.passed).toBe(false);
 expect(assessOutfitQuality(p,'semi_classic','Casual').acceptance.passed).toBe(false);
 p[2]={id:'casual-chinos',name:'Cotton chinos',macroCategory:'bottom'};
 expect(assessOutfitQuality(p,'semi_classic','Travel').acceptance.passed).toBe(true);
 expect(describeGarment({...p[3],id:'shop_geox-unverified',name:'Geox leather penny loafers with slim rubber sole'}).soleProfile).toBe('dress');
 });

test.each(['Date night','date','date_night','date-night','Свидание'])(
 'real occasion gate retains clean Casual date garments for %s',occasion=>{
 const p=look(['Plain fine cotton polo','Suede trucker jacket','Dark clean straight denim jeans','Minimal white leather sneakers']);
 expect(styleJudgeReason(p,'casual',occasion)).toBeNull();
 expect(assessOutfitQuality(p,'casual',occasion).acceptance.passed).toBe(true);
 expect(assessOutfitQuality(p,'casual',occasion).acceptance.minimum).toBe(0.7);
 });
