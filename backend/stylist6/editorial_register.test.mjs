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
