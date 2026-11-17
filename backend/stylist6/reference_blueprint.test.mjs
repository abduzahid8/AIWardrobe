import test from 'node:test';
import assert from 'node:assert/strict';
import index from '../../benchmarks/stylist6/references/image_teaste_46.json' with {type:'json'};
import { buildReferenceBlueprints } from './reference_blueprint.mjs';
test('all 46 user-curated references expose only textual architecture, never photo bytes or URLs',()=>{
  const blueprints=buildReferenceBlueprints(index);
  assert.equal(blueprints.size,46);
  for(const [id,brief] of blueprints){
    assert.match(id,/^rt_\d{3}$/);
    assert.equal(brief.evidenceMode,'textual_editorial_brief_not_visual_reference');
    assert.ok(brief.palette.length);
    assert.ok(brief.garments.top);
    assert.ok(brief.garments.bottom);
    assert.ok(brief.garments.shoes);
    assert.doesNotMatch(JSON.stringify(brief),/https?:\/\/|\.jpg|data:image|license granted|sha256/i);
  }
});
test('unsupported or corrupted index cannot be used as a trusted blueprint source',()=>{
  assert.throws(()=>buildReferenceBlueprints({source:'random',images:[]}),/invalid_reference_blueprint_index/);
  assert.equal(buildReferenceBlueprints({source:index.source,images:[{id:'bad',complete_outfit_collage:true,palette:[],garments:{}}]}).size,0);
});
