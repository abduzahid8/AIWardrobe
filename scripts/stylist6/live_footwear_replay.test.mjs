import test from 'node:test';
import assert from 'node:assert/strict';
import { verifyLivePreviewRows, comparePilotResults } from './live_footwear_replay.mjs';

const catalog=Array.from({length:40},(_,i)=>({
  id:'item-'+i, name:'Synthetic garment',brand:i===0?'ECCO':'Zara',
  is_active:true, currency:'UZS',
}));
const preview={rows:Array.from({length:4},(_,i)=>({
  id:'fruz-th-'+String(i).padStart(20,'0'),
  brand:'Tommy Hilfiger',is_active:false,currency:'UZS',
  category:'shoes',garment_type:'shoes',
  price:2050300,external_id:'FM0FM12345',source:'frgroup-uz-review-not-active',
  source_url:'https://frgroup.uz/ru/store/product-'+i,
  image_url:'https://media.frgroup.uz/images/'+i+'.webp',
}))};
const scenarioIds=['S6-01','S6-02','S6-04','S6-05','S6-07','S6-08','S6-B01','S6-B03'];
const look=(shoe='shoes-one')=>({items:[
  {id:'top',macroCategory:'top'}, {id:'bottom',macroCategory:'bottom'},
  {id:shoe,macroCategory:'shoes'},
]});
const deck=(override={})=>({
  runs:scenarioIds.map(id=>({
    id, arms:{
      'reference-first':{looks:[look(override[id]??'shoes-one')]},
      'creative-first':{looks:[look('shoes-two')]},
    },
  })),
});

test('hypothetical release audit never edits real source catalog and filters ECCO only as requested',()=>{
  const original=JSON.stringify(catalog);
  const result=verifyLivePreviewRows(catalog,preview,{excludedBrands:['ECCO']});
  assert.equal(catalog.length,40);
  assert.equal(JSON.stringify(catalog),original);
  assert.equal(result.base.length,39);
  assert.equal(result.shadow.length,43);
  assert.equal(result.shadow.filter(x=>x.id.startsWith('fruz-th-')).length,4);
  assert.ok(result.shadow.filter(x=>x.id.startsWith('fruz-th-')).every(x=>x.is_active===true));
  assert.ok(preview.rows.every(x=>x.is_active===false),'preview remains inactive');
});

test('source duplicate ID, changed category, non-reviewed brand or currency refuses replay',()=>{
  assert.throws(()=>verifyLivePreviewRows([...catalog,catalog[1]],preview),/invalid_frozen_live_catalog/);
  assert.throws(()=>verifyLivePreviewRows(catalog,{rows:[...preview.rows.slice(1),{...preview.rows[0],brand:'ECCO'}]}),/untrusted_candidate/);
  assert.throws(()=>verifyLivePreviewRows(catalog,{rows:[...preview.rows.slice(1),{...preview.rows[0],is_active:true}]}),/untrusted_candidate/);
  assert.throws(()=>verifyLivePreviewRows(catalog,{rows:[...preview.rows.slice(1),{...preview.rows[0],currency:'USD'}]}),/untrusted_candidate/);
  assert.throws(()=>verifyLivePreviewRows(catalog,{rows:[...preview.rows.slice(1),{...preview.rows[0],source_url:'https://frgroup.uz.evil/ru/store/x'}]}),/untrusted_candidate/);
});

test('one new selected shoe is detected even when total looks stay the same',()=>{
  const before=deck(),after=deck({'S6-01':preview.rows[0].id});
  const result=comparePilotResults(before,after,new Set(preview.rows.map(x=>x.id)));
  assert.equal(result.before.totalLooks,16);
  assert.equal(result.after.totalLooks,16);
  assert.equal(result.previewShoeSelections,1);
  assert.equal(result.changedLooks,1);
  assert.equal(result.nvidiaCalls,0);
  assert.equal(result.productionChanges,0);
});

test('malformed output, duplicate missing scenarios or missing physical shoe fail closed',()=>{
  const good=deck();
  assert.throws(()=>comparePilotResults({...good,runs:good.runs.slice(1)},good,new Set(preview.rows.map(x=>x.id))),/pilot_runs_missing/);
  const bad=deck();
  bad.runs[0].arms['reference-first'].looks[0].items[2].macroCategory='outerwear';
  assert.throws(()=>comparePilotResults(good,bad,new Set(preview.rows.map(x=>x.id))),/pilot_shoe_slot_mismatch/);
  assert.throws(()=>comparePilotResults(good,deck(),new Set(['only-one'])),/preview_ids_invalid/);
});
