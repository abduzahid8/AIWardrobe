import test from 'node:test';
import assert from 'node:assert/strict';
import index from '../../benchmarks/stylist6/references/image_teaste_46.json' with {type:'json'};
import { buildReferenceBlueprints } from './reference_blueprint.mjs';
test('46 metadata references yield textual blueprints without image URLs, binary or inferred licensing',()=>{
  const b=buildReferenceBlueprints(index);
  assert.equal(b.size,46);
  for(const [id,x] of b){
    assert.match(id,/^rt_\d{3}$/);
    assert.equal(x.evidenceMode,'textual_editorial_brief_not_visual_reference');
    assert.ok(x.garments.top&&x.garments.bottom&&x.garments.shoes);
    assert.doesNotMatch(JSON.stringify(x),/https?:\/\/|\.jpg|data:image|license granted|sha256/i);
  }
});
test('untrusted metadata index cannot supply editorial reference text',()=>{
  assert.throws(()=>buildReferenceBlueprints({source:'untrusted',images:[]}),/invalid_reference_blueprint_index/);
});
