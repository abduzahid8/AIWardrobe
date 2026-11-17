import test from 'node:test';
import assert from 'node:assert/strict';
import {deliberatelyRelaxedIntent} from './editorial_intent.mjs';
import {catalogStylingDiscoveryHints} from './cloud_handler.mjs';
import {fashionStylingContext,polishedEveningLayerRequired} from './fashion_deck.mjs';
import {editorialRegisterVeto} from './editorial_register.mjs';

const realBrief='Refined semi-classic outfit for a nice dinner date with polished footwear, flattering silhouette, modern smart casual elegance';
const realRequest={sourceMode:'catalog',style:'semi_classic',occasion:'Date night',weather:{temp:13},
  brief:realBrief,optionalLayer:true};

test('an actual refined smart-casual dinner keeps the elevated dress code across discovery, layering and editorial admission',()=>{
  assert.equal(deliberatelyRelaxedIntent(realBrief),false);
  assert.equal(catalogStylingDiscoveryHints(realRequest).dressCode,'smart');
  assert.ok(catalogStylingDiscoveryHints(realRequest).preferredSubtypes.includes('blazer'));
  assert.ok(!catalogStylingDiscoveryHints(realRequest).preferredSubtypes.includes('jeans'));
  assert.equal(polishedEveningLayerRequired(fashionStylingContext(realRequest)),true);
  const observed={items:[
    {id:'unpolished-tee',macroCategory:'top',subtype:'t-shirt',name:'ONSMAX T-shirt'},
    {id:'unpolished-overshirt',macroCategory:'outerwear',subtype:'overshirt',name:'Overshirt'},
    {id:'unpolished-jeans',macroCategory:'bottom',subtype:'jeans',name:'Tapered Slim-Fit Jeans in Stay Black'},
    {id:'unpolished-shoes',macroCategory:'shoes',subtype:'chelsea',name:'Barker hybrid-soled Chelsea boots'},
  ]};
  assert.equal(editorialRegisterVeto(observed,{...realRequest}),
    'polished_date_needs_tailored_anchor','real bad outfit must not get an unearned fashion pass');
});

test('a genuinely relaxed date retains personal freedom, but smart-casual dress codes are still refined',()=>{
  for(const text of ['Smart Casual Elegance','business-casual office','polished smart_casual dinner'])
    assert.equal(deliberatelyRelaxedIntent(text),false,text);
  for(const text of ['Relaxed casual date','casual Friday work','Sporty walk and dinner','Weekend brunch','Dinner without jacket — relaxed and casual'])
    assert.equal(deliberatelyRelaxedIntent(text),true,text);
  for(const value of [null,{},[],false])assert.equal(deliberatelyRelaxedIntent(value),false);
  const casual={...realRequest,brief:'Relaxed casual dinner and walk'};
  assert.equal(catalogStylingDiscoveryHints(casual).dressCode,'casual');
  assert.equal(polishedEveningLayerRequired(fashionStylingContext(casual)),false);
  const above={...realRequest,weather:{temp:25}};
  assert.equal(polishedEveningLayerRequired(fashionStylingContext(above)),false);
});
