import test from 'node:test';
import assert from 'node:assert/strict';
import { editorialRegisterVeto } from './editorial_register.mjs';
const look=(top,outer,bottom,shoes)=>({items:[
  {macroCategory:'top',name:top},...(outer?[{macroCategory:'outerwear',name:outer}]:[]),
  {macroCategory:'bottom',name:bottom},{macroCategory:'shoes',name:shoes},
]});
test('classic work vetoes athletic footwear on a complete tailored suit',()=>{
  assert.equal(editorialRegisterVeto(look('Oxford shirt','Wool blazer','Tailored suit trousers','S LITE HYBRID Leather Sneaker'),{style:'classic',occasion:'Work'}),'athletic_shoe_with_formal_tailoring');
  assert.equal(editorialRegisterVeto(look('Oxford shirt','Wool blazer','Tailored suit trousers','Leather Oxford Shoes'),{style:'classic',occasion:'Work'}),null);
});
test('old money work and semi classic dates cannot mistake oversized tees and formal dress shoes for polish',()=>{
  const item=look('Oversized T-Shirt Navy',null,'Wool tailored pants','Leather Dress Loafers');
  assert.equal(editorialRegisterVeto(item,{style:'old_money',occasion:'Work'}),'casual_tee_with_dress_footwear');
  assert.equal(editorialRegisterVeto(item,{style:'semi_classic',occasion:'Romantic date'}),'casual_tee_with_dress_footwear');
});
test('work flannel mismatch does not veto casual weekend and sparse evidence',()=>{
  const item=look('Mens checked Flannel Shirt',null,'Pleated wool suit pants','Leather Dress Loafer');
  assert.equal(editorialRegisterVeto(item,{style:'old_money',occasion:'Work'}),'flannel_with_office_tailoring');
  assert.equal(editorialRegisterVeto(item,{style:'casual',occasion:'Weekend'}),null);
  assert.equal(editorialRegisterVeto(look('item 123','blazer','bottom 123','shoes 123'),{style:'classic',occasion:'Work'}),null);
});
