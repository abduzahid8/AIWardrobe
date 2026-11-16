import test from 'node:test';
import assert from 'node:assert/strict';
import {SOURCE,derivedFootwear,hashCatalogEvidence,searchVector,manufacturerUrlSubtype} from './enrich_imported_footwear.mjs';
const row={id:'test',source:SOURCE,is_active:true,category:'shoes',garment_type:'shoes',
  brand:'Herring',name:'Tan Suede Penny Loafers',description:'Men’s elegant loafers. Product-only photograph.',
  image_url:'https://example.com/shoe.jpg',source_url:'https://shoemaker.example/products/tan-suede-penny-loafers'};
test('metadata is a transparent loafer-derived source snapshot, not invented photos or merchant prices',()=>{
  const x=derivedFootwear(row);
  assert.equal(x.slot,'shoes');
  assert.equal(x.subtype,'loafer');
  assert.ok(['smart','formal','casual'].includes(x.formality));
  assert.equal(x.source_hash,hashCatalogEvidence(row));
  assert.match(x.search_document,/'loafers':/);
  assert.match(x.search_document,/'herring':/);
  assert.ok(!('price' in x));
  assert.ok(!('source_url' in x));
});
test('unknown footwear stays unknown instead of hallucinating a formal shoe type',()=>{
  const x=derivedFootwear({...row,name:'Model Freguiatt',description:'Men’s elegant shoes. Product-only photograph.',source_url:'https://shoemaker.example/products/freguiatt-model'});
  assert.equal(x.subtype,'unknown');
});
test('only active authoritative shoe records in the reviewed source may be enriched',()=>{
  for(const change of [{source:'other'},{category:'tops'},{garment_type:'upper_body'},{is_active:false}])
    assert.throws(()=>derivedFootwear({...row,...change}),/unexpected_source_or_category/);
});
test('search vector safely handles punctuation, unicode, and repeats',()=>{
  const v=searchVector({...row,name:'Men’s Oxford - Oxford, café'});
  assert.match(v,/'oxford':/);
  assert.match(v,/'café':/);
  assert.equal((v.match(/'oxford':/g)||[]).length,1);
});

test('manufacturer URL recovers explicit Oxford evidence for opaque model names',()=>{
  const input={...row,name:'107 ARTISTA',description:'Men’s elegant footwear. Zapato.',
    source_url:'https://www.tlbmallorca.com/products/oxford-picasso-antique-calf-cuero-107'};
  assert.equal(manufacturerUrlSubtype(input),'oxford');
  const derived=derivedFootwear(input);
  assert.equal(derived.subtype,'oxford');
  assert.equal(derived.formality,'formal');
  assert.match(derived.search_document,/'oxford':/);
  assert.ok(derived.intelligence.descriptor.evidence.includes('source:manufacturer-product-url:oxford'));
});
test('manufacturer URL distinguishes Blucher derby while refusing generic shoes and decorative broguing',()=>{
  assert.equal(manufacturerUrlSubtype({...row,source_url:'https://www.blkbrdshoemaker.com/products/raife-longwing-blucher-bourbon'}),'derby');
  assert.equal(manufacturerUrlSubtype({...row,source_url:'https://example.com/products/model-xxx-wingtip-brogue'}),null);
  assert.equal(manufacturerUrlSubtype({...row,source_url:'https://example.com/products/mystery-shoe'}),null);
  assert.equal(manufacturerUrlSubtype({...row,source_url:'http://example.com/products/oxford-black'}),null);
});
