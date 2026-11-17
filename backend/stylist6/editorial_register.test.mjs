import test from 'node:test';
import assert from 'node:assert/strict';
import { editorialRegisterVeto } from './editorial_register.mjs';
const look=(top,outer,bottom,shoes)=>({items:[
  {macroCategory:'top',name:top},
  ...(outer?[{macroCategory:'outerwear',name:outer}]:[]),
  {macroCategory:'bottom',name:bottom},
  {macroCategory:'shoes',name:shoes},
]});
test('classic work vetoes athletic-soled footwear with a complete business suit',()=>{
  assert.equal(editorialRegisterVeto(look('Classic Oxford Shirt','Wool suit blazer','Tailored suit trousers','S LITE HYBRID Leather Sneaker'),{style:'classic',occasion:'Work'}),'athletic_shoe_with_formal_tailoring');
  assert.equal(editorialRegisterVeto(look('Classic Oxford Shirt','Wool suit blazer','Tailored suit trousers','Oxford leather shoes'),{style:'classic',occasion:'Work'}),null);
});
test('dress footwear with a casual t-shirt is not an Old Money work or date direction',()=>{
  assert.equal(editorialRegisterVeto(look('Oversized T-Shirt Navy',null,'Wool tailored pants','Leather Dress Loafers'),{style:'old_money',occasion:'Work'}),'casual_tee_with_dress_footwear');
  assert.equal(editorialRegisterVeto(look('Oversized T-Shirt Navy',null,'Wool tailored pants','Leather Dress Loafers'),{style:'semi_classic',occasion:'Romantic date'}),'casual_tee_with_dress_footwear');
  assert.equal(editorialRegisterVeto(look('Cotton shirt',null,'Wool tailored pants','Leather Dress Loafers'),{style:'semi_classic',occasion:'Romantic date'}),null);
});
test('flannel collision applies to polished work, not casual weekend',()=>{
  const sample=look('Mens checked Flannel Shirt',null,'Pleated wool suit pants','Leather dress loafer');
  assert.equal(editorialRegisterVeto(sample,{style:'old_money',occasion:'Work'}),'flannel_with_office_tailoring');
  assert.equal(editorialRegisterVeto(sample,{style:'casual',occasion:'Weekend'}),null);
  assert.equal(editorialRegisterVeto(sample,{style:'old_money',occasion:'Weekend'}),null);
});
test('observed classic work Henley cannot masquerade as a collared dress shirt',()=>{
  const observed=look('RIBBED LONG SLEEVED HENLEY SHIRT',null,
    'PLEATED TROUSERS WITH METAL HOOK','Redford Brogue Derbies - Suede');
  assert.equal(editorialRegisterVeto(observed,{style:'classic',occasion:'Work'}),
    'henley_with_office_tailoring');
  assert.equal(editorialRegisterVeto(observed,{style:'old_money',occasion:'Business meeting'}),
    'henley_with_office_tailoring');
  assert.equal(editorialRegisterVeto(observed,{style:'semi_classic',occasion:'Work'}),
    'henley_with_office_tailoring');
  assert.equal(editorialRegisterVeto(observed,{style:'classic',occasion:'Casual'}),null);
  assert.equal(editorialRegisterVeto(observed,{style:'semi_classic',occasion:'Romantic date'}),null);
  assert.equal(editorialRegisterVeto(observed,{style:'classic',occasion:'Work',brief:'Please style my Henley'}),null);
  assert.equal(editorialRegisterVeto(observed,{style:'classic',occasion:'Work',request:'Henley top'}),null);
  assert.equal(editorialRegisterVeto(look('Linen Henley',null,'Cotton chino','White Canvas Sneakers'),
    {style:'classic',occasion:'Work'}),null,'no formal collision to veto');
});

test('unknown / sparse catalog descriptions never trigger speculative veto',()=>{
  assert.equal(editorialRegisterVeto(look('item 123','blazer','bottom 123','shoes 123'),{style:'classic',occasion:'Work'}),null);
});
test('animal prints and comfort/suede suit clashes are vetoed on elevated looks',()=>{
  assert.equal(
    editorialRegisterVeto(look('Leo Oversized Leopard-Print Shirt',null,'Moleskin Suit Pants','MELBOURNE Leather Derby Shoe'),{style:'semi_classic',occasion:'Romantic date'}),
    'loud_print_with_dress_tailoring',
  );
  const comfortSuede={items:[
    {macroCategory:'top',name:'Relaxed Oxford Shirt'},
    {macroCategory:'outerwear',name:'100% WOOL SUIT BLAZER'},
    {macroCategory:'bottom',name:'Slim Fit Smart Stretch Trousers'},
    {id:'geox-u65hyb00022c6021',macroCategory:'shoes',name:'Spherica Ec16 Suede Loafers'},
  ]};
  assert.equal(editorialRegisterVeto(comfortSuede,{style:'old_money',occasion:'Work'}),'comfort_sole_with_formal_tailoring');
  assert.equal(
    editorialRegisterVeto(look('Relaxed Oxford Shirt','100% WOOL SUIT BLAZER','Slim Fit Smart Stretch Trousers','Massimiano Suede Loafers'),{style:'old_money',occasion:'Work'}),
    'suede_footwear_with_formal_suit',
  );
  assert.equal(
    editorialRegisterVeto(look('Cream fine-knit polo','Navy cotton blazer','Beige chinos','Brown suede loafers'),{style:'semi_classic',occasion:'Romantic date'}),
    null,
  );
});

test('cool +9C Classic Work does not treat a sleeveless vest as its sole outerwear', () => {
  const candidate=look('Judd Regular-Fit Striped Shirt','Navy Rib Vest',
    'Smart Textured Pants - Steel Blue','Astor Chelsea Boot Black');
  const context={style:'classic',occasion:'Work',sourceMode:'catalog',weather:{temp:9}};
  assert.equal(editorialRegisterVeto(candidate,context),'cool_weather_sleeveless_single_layer');
  assert.equal(editorialRegisterVeto(candidate,{...context,weather:{temp:-4}}),'cool_weather_sleeveless_single_layer');
  assert.equal(editorialRegisterVeto(candidate,{...context,weather:{temp:16}}),null);
  assert.equal(editorialRegisterVeto(candidate,{...context,sourceMode:'wardrobe'}),null);
  assert.equal(editorialRegisterVeto(candidate,{...context,request:'Please wear my navy vest'}),null);
  assert.equal(editorialRegisterVeto(look('Blue shirt','Navy wool coat','Grey wool trousers','Black Chelsea boots'),context),null);
});


test('Classic Work formality is a whole-outfit silhouette check, not a ban on specific brands or knits', () => {
  const easy=look('Fine Navy Merino Knit','Navy Wool Zipper Cardigan',
    'Ultimate Non-Iron Chinos - Navy','Helsinki Leather Penny Loafer');
  const context={style:'classic',occasion:'Work',sourceMode:'catalog',
    brief:'Professional polished classic workplace outfit'};
  assert.equal(editorialRegisterVeto(easy,context),
    'polished_classic_office_requires_tailored_structure');
  assert.equal(editorialRegisterVeto(easy,{...context,occasion:'Weekend'}),null);
  assert.equal(editorialRegisterVeto(easy,{...context,brief:'Relaxed casual office Friday'}),null);
  assert.equal(editorialRegisterVeto(easy,{...context,sourceMode:'wardrobe'}),null);
  assert.equal(editorialRegisterVeto(look('Fine Navy Merino Knit','Navy Tailored Wool Blazer',
    'Beige Chinos','Leather Penny Loafer'),context),null);
  assert.equal(editorialRegisterVeto(look('White Dress Shirt','Fine Merino Cardigan',
    'Charcoal Wool Trousers','Black Oxford Leather Shoes'),context),null);
  assert.equal(editorialRegisterVeto(look('White Dress Shirt','Grey Wool Overcoat',
    'Beige Chinos','Dark Brown Chelsea Boots'),context),null);
});


test('professional Classic Work contract rejects hybrid athletic soles even when bottom is a chino', () => {
  const work={style:'classic',occasion:'Work',sourceMode:'catalog',
    brief:'Classic polished professional office outfit with sophisticated tailoring'};
  const example=look('Merino Wool Button-Through Shirt','Chocolate Wool Blazer',
    'Walnut Velvet Chinos','Herring hybrid-soled brown suede loafers');
  assert.equal(editorialRegisterVeto(example,work),'polished_classic_office_athletic_sole');
  assert.equal(editorialRegisterVeto(example,{...work,occasion:'Weekend'}),null);
  assert.equal(editorialRegisterVeto(example,{...work,brief:'Casual relaxed Friday office outfit'}),null);
  assert.equal(editorialRegisterVeto(example,{...work,sourceMode:'wardrobe'}),null);
  assert.equal(editorialRegisterVeto(look('White Oxford Shirt','Navy Blazer',
    'Beige Chinos','Brown Leather Penny Loafers'),work),null);
});


test('source-backed casual constructions need a tailored anchor for polished Classic Work', () => {
  const office={style:'classic',occasion:'Work',sourceMode:'catalog',brief:'Professional office tailoring'};
  const casual=look('Grey Flannel Shirt','Textured Button Cardigan','Walnut Chinos','Brown Penny Loafers');
  assert.equal(editorialRegisterVeto(casual,office),'polished_classic_office_requires_tailored_structure');
  assert.equal(editorialRegisterVeto(casual,{...office,occasion:'Weekend'}),null);
  assert.equal(editorialRegisterVeto(casual,{...office,brief:'Casual Friday office'}),null);
  assert.equal(editorialRegisterVeto(casual,{...office,sourceMode:'wardrobe'}),null);
  const anchored={items:casual.items.map((item,i)=>({...item,id:'owned-'+i}))};
  assert.equal(editorialRegisterVeto(anchored,{...office,requiredIds:['owned-1']}),null,
    'explicitly user-pinned garment must not be silently replaced');
  assert.equal(editorialRegisterVeto(look('Unknown shirt','Unknown item','Unknown item','Unknown shoes'),office),null,
    'unknown metadata is not evidence of an informal outfit');
  assert.equal(editorialRegisterVeto(look('Blue Dress Shirt','Grey Wool Suit Blazer','Cream Chinos','Dark Brown Leather Oxfords'),office),null);
});


test('polished Semi-Classic romantic date requires intentional dress structure, not only earth-tone matching',()=>{
  const request={style:'semi_classic',occasion:'Date night',sourceMode:'catalog',
    brief:'Refined semi-classic outfit for a nice dinner date with polished footwear'};
  const tooCasual=look('Light brown cotton polo crew neck','Regular-fit Wool Cardigan with Zipper',
    'Walnut comfort cotton velvet regular-fit chinos','Brown suede slip-on loafers');
  assert.equal(editorialRegisterVeto(tooCasual,request),'polished_date_needs_tailored_anchor');
  assert.equal(editorialRegisterVeto(tooCasual,{...request,occasion:'Weekend'}),null);
  assert.equal(editorialRegisterVeto(tooCasual,{...request,brief:'Relaxed casual dinner'}),null);
  assert.equal(editorialRegisterVeto(tooCasual,{...request,sourceMode:'wardrobe'}),null);
  assert.equal(editorialRegisterVeto(look('Cream fine-knit polo','Navy cotton blazer',
    'Walnut chinos','Brown suede loafers'),request),null,
    'polished blazer is the tailored anchor, even with chinos');
  assert.equal(editorialRegisterVeto(look('Cream merino knit','Zip cardigan',
    'Charcoal wool trousers','Brown oxfords'),request),null,
    'real wool tailoring balances a knit layer');
  assert.equal(editorialRegisterVeto(look('White dress shirt','Zip cardigan',
    'Chinos','Dark brown leather derbies'),request),null,
    'proper dress shirt can anchor the dinner outfit');
});

test('unrequested statement prints are not silently treated as understated semi-classic dinner',()=>{
  const request={style:'semi_classic',occasion:'Date night',sourceMode:'catalog',
    brief:'Nice dinner, understated contemporary sophistication'};
  const loud=look('Abstract Polka-Dot Print Regular-Fit Shirt in Black/White',
    'Mahogany Corduroy Blazer','Rich Coffee Corduroy Trousers','Brown Derby');
  assert.equal(editorialRegisterVeto(loud,request),'unrequested_statement_print_on_polished_date');
  assert.equal(editorialRegisterVeto(loud,{...request,brief:'I love polka-dot statement prints'}),null);
  assert.equal(editorialRegisterVeto(loud,{...request,occasion:'Weekend'}),null);
  assert.equal(editorialRegisterVeto(loud,{...request,sourceMode:'wardrobe'}),null);
  assert.equal(editorialRegisterVeto(look('Muted pinstripe shirt','Mahogany Blazer',
    'Wool trousers','Brown loafers'),request),null);
});


test('actual latest cloud dinner rejects casual overshirt-polo-corduroy without sartorial anchor',()=>{
  const dinner={style:'semi_classic',occasion:'Date night',sourceMode:'catalog',
    brief:'Refined semi-classic outfit for a nice dinner date with polished footwear'};
  const actual=look('Burgundy wool and cashmere zippered polo shirt with nubuck details',
    '100% Cotton Overshirt with Pocket Detail','Beige Corduroy Metropolitan Trousers',
    'Herring Mountbatten Chelsea boots in dark brown calf');
  assert.equal(editorialRegisterVeto(actual,dinner),'polished_date_needs_tailored_anchor');
  assert.equal(editorialRegisterVeto(actual,{...dinner,occasion:'Weekend'}),null);
  assert.equal(editorialRegisterVeto(actual,{...dinner,brief:'Casual weekend coffee date'}),null);
  assert.equal(editorialRegisterVeto(actual,{...dinner,sourceMode:'wardrobe'}),null);
  const elevated=look('Burgundy fine knit polo','Chocolate Wool Blazer',
    'Beige Corduroy Metropolitan Trousers','Dark brown Chelsea boots');
  assert.equal(editorialRegisterVeto(elevated,dinner),null);
  const balanced=look('Dark charcoal fine-knit','Brown wool zip cardigan',
    'Tailored wool trousers','Black leather Derbies');
  assert.equal(editorialRegisterVeto(balanced,dinner),null);
});


test('source-evident Oxford cloth button-down counts as the dress shirt anchor on smart-casual dinner',()=>{
  const date={sourceMode:'catalog',style:'semi_classic',occasion:'Date night',
    brief:'Refined semi-classic dinner, modern smart casual elegance'};
  const oxford=look('Striped Brushed Oxford Shirt Crisp Blue',
    'Wool Blend Cable Knit Cardigan Shadow Brown',
    'Sand-colored comfort cotton gabardine chinos','Brown leather loafers');
  assert.equal(editorialRegisterVeto(oxford,date),null);
  const tee=look('ONSMAX T-shirt','100% Cotton Overshirt with Pocket Detail',
    'Tapered Slim-Fit Jeans in Stay Black','Leather Chelsea boots');
  assert.equal(editorialRegisterVeto(tee,date),'polished_date_casual_tee_requires_tailored_layer');
});


test('real 17 Nov photo QA: basic T-shirt with cardigan and tailored-FIT chinos is not polished evening tailoring',()=>{
  const context={sourceMode:'catalog',style:'semi_classic',occasion:'Date night',
    brief:'Refined semi-classic dinner date, modern smart casual elegance'};
  const actual=look('BASIC IRREGULAR FINISH T-SHIRT','100% wool fine-knit cardigan',
    'Cotton-Stretch Tailored-Fit Chinos in Black','Freguiatt');
  assert.equal(editorialRegisterVeto(actual,context),'polished_date_casual_tee_requires_tailored_layer');
  assert.equal(editorialRegisterVeto(actual,{...context,sourceMode:'wardrobe'}),null);
  assert.equal(editorialRegisterVeto(actual,{...context,occasion:'Weekend'}),null);
  assert.equal(editorialRegisterVeto(actual,{...context,brief:'Relaxed casual dinner in my T-shirt'}),null);
  assert.equal(editorialRegisterVeto(look('White premium T-shirt','Chestnut Wool Blazer',
    'Beige cotton chinos','Dark Brown Leather Loafers'),context),null,
    'a blazer and quiet T-shirt can be an intentionally elevated modern date look');
  assert.equal(editorialRegisterVeto(look('Cream Fine-Knit Polo','Chocolate Cardigan',
    'Beige Wool Tailored Trousers','Dark Brown Chelsea Boots'),context),null,
    'genuine wool tailoring makes a fine knit/cardi outfit dress appropriate');
  assert.equal(editorialRegisterVeto(look('White Oxford Shirt','Chocolate Cardigan',
    'Beige Cotton Chinos','Leather Penny Loafers'),context),null,
    'a collared Oxford base anchors the refined outfit');
});
